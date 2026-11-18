from dataclasses import dataclass
import time

import torch

from gaussian_renderer import render_fastgs
from .detail_densification_utils import (
    DetailDensifyConfig,
    build_detail_need_mask,
    quantile_mask,
    sobel_magnitude,
)
from .fast_utils import compute_photometric_loss, get_loss


@dataclass(frozen=True)
class DetailScoreResult:
    importance_score: torch.Tensor
    pruning_score: torch.Tensor
    detail_score: torch.Tensor
    detail_view_hits: torch.Tensor
    visible_view_count: torch.Tensor
    detail_support_ratio: torch.Tensor
    detail_candidate_mask: torch.Tensor
    statistics: dict


def _require_matching_vectors(named_values, validate_values):
    if type(validate_values) is not bool:
        raise TypeError("validate_values must be a bool")
    reference_shape = None
    reference_device = None
    for name, value in named_values:
        if not torch.is_tensor(value):
            raise TypeError("{} must be a torch.Tensor".format(name))
        if value.ndim != 1:
            raise ValueError("{} must be one-dimensional".format(name))
        if value.numel() == 0:
            raise ValueError("{} must not be empty".format(name))
        if reference_shape is None:
            reference_shape = value.shape
            reference_device = value.device
        elif value.shape != reference_shape:
            raise ValueError("all score vectors must have matching shapes")
        elif value.device != reference_device:
            raise ValueError("all score vectors must be on the same device")
        if validate_values:
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    "{} must contain only finite values".format(name)
                )


def build_detail_view_statistics(
    detail_counts,
    total_counts,
    detail_config,
    validate_values=True,
):
    if not isinstance(detail_config, DetailDensifyConfig):
        raise TypeError("detail_config must be a DetailDensifyConfig")
    _require_matching_vectors(
        (
            ("detail_counts", detail_counts),
            ("total_counts", total_counts),
        ),
        validate_values=validate_values,
    )
    if validate_values:
        if bool((detail_counts < 0).any().item()):
            raise ValueError("detail_counts must be non-negative")
        if bool((total_counts < 0).any().item()):
            raise ValueError("total_counts must be non-negative")
        if bool((detail_counts > total_counts).any().item()):
            raise ValueError("detail_counts must not exceed total_counts")

    # The total/detail maps are rendered independently. FastGS CUDA counting can
    # occasionally differ by a few pixel hits between those passes, so enforce
    # the mathematical subset relation before forming a probability-like ratio.
    # Strict callers still receive the validation error above; the renderer path
    # deliberately skips per-view synchronizing validation and uses this bound.
    bounded_detail_counts = torch.minimum(detail_counts, total_counts)
    detail_counts_float = bounded_detail_counts.to(torch.float32)
    total_counts_float = total_counts.to(torch.float32)
    visible_mask = total_counts > 0
    overlap_ratio = torch.where(
        visible_mask,
        detail_counts_float / total_counts_float.clamp_min(1.0),
        torch.zeros_like(detail_counts_float),
    )
    support_mask = (
        visible_mask
        & (
            total_counts
            >= detail_config.min_total_pixels_per_view
        )
        & (
            bounded_detail_counts
            >= detail_config.min_detail_pixels_per_view
        )
        & (
            overlap_ratio
            >= detail_config.min_detail_overlap_ratio
        )
    )
    return overlap_ratio, visible_mask, support_mask


def build_detail_candidate_mask(
    overlap_ratio_sum,
    detail_view_hits,
    visible_view_count,
    detail_config,
):
    if not isinstance(detail_config, DetailDensifyConfig):
        raise TypeError("detail_config must be a DetailDensifyConfig")
    _require_matching_vectors(
        (
            ("overlap_ratio_sum", overlap_ratio_sum),
            ("detail_view_hits", detail_view_hits),
            ("visible_view_count", visible_view_count),
        ),
        validate_values=True,
    )
    if bool((overlap_ratio_sum < 0).any().item()):
        raise ValueError("overlap_ratio_sum must be non-negative")
    if bool((detail_view_hits < 0).any().item()):
        raise ValueError("detail_view_hits must be non-negative")
    if bool((visible_view_count < 0).any().item()):
        raise ValueError("visible_view_count must be non-negative")
    if bool((detail_view_hits > visible_view_count).any().item()):
        raise ValueError("detail_view_hits must not exceed visible_view_count")
    if bool(
        (
            overlap_ratio_sum
            > visible_view_count.to(torch.float32) + 1e-6
        ).any().item()
    ):
        raise ValueError(
            "overlap_ratio_sum must not exceed visible_view_count"
        )

    visible_count_float = visible_view_count.to(torch.float32)
    safe_visible_count = visible_count_float.clamp_min(1.0)
    detail_score = overlap_ratio_sum.to(torch.float32) / safe_visible_count
    detail_support_ratio = (
        detail_view_hits.to(torch.float32) / safe_visible_count
    )
    enough_visible_views = (
        visible_view_count >= detail_config.min_visible_views
    )
    enough_views = detail_view_hits >= detail_config.min_detail_views
    enough_support = (
        detail_support_ratio >= detail_config.min_detail_support_ratio
    )
    enough_average_overlap = (
        detail_score
        >= detail_config.min_average_detail_overlap_ratio
    )
    detail_candidate_mask = (
        enough_visible_views
        & enough_views
        & enough_support
        & enough_average_overlap
    )
    return detail_score, detail_support_ratio, detail_candidate_mask


def _get_cached_high_frequency_mask(
    viewpoint_camera,
    target_image,
    detail_config,
):
    cache_name = "_fastgs_detail_high_frequency_mask_cache"
    cache = getattr(viewpoint_camera, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(viewpoint_camera, cache_name, cache)

    cache_key = (
        float(detail_config.high_frequency_quantile),
        tuple(target_image.shape),
        str(target_image.dtype),
        str(target_image.device),
        target_image.data_ptr(),
        target_image._version,
    )
    high_frequency_mask = cache.get(cache_key)
    cache_hit = high_frequency_mask is not None
    if high_frequency_mask is None:
        cache.clear()
        target = target_image.detach().clamp(0.0, 1.0)
        high_frequency_mask = quantile_mask(
            sobel_magnitude(target),
            detail_config.high_frequency_quantile,
        ).to(dtype=torch.bool).contiguous()
        cache[cache_key] = high_frequency_mask
    return high_frequency_mask, cache_hit


@torch.no_grad()
def compute_gaussian_score_fastgs_detail(
    camlist,
    gaussians,
    pipe,
    background,
    args,
    detail_config: DetailDensifyConfig,
) -> DetailScoreResult:
    scoring_started_at = time.perf_counter()
    number_of_views = len(camlist)
    if number_of_views == 0:
        raise ValueError("camlist must contain at least one camera")
    if gaussians.get_xyz.shape[0] == 0:
        raise ValueError("detail scoring requires at least one Gaussian")

    full_metric_counts = None
    full_metric_score = None
    overlap_ratio_sum = None
    detail_view_hits = None
    visible_view_count = None
    high_frequency_ratio_sum = 0.0
    error_ratio_sum = 0.0
    detail_ratio_sum = 0.0
    high_frequency_cache_hits = 0

    for viewpoint_camera in camlist:
        target_image = viewpoint_camera.original_image.cuda()
        total_metric_map = torch.ones_like(
            target_image[0],
            dtype=torch.int32,
        )

        total_render = render_fastgs(
            viewpoint_camera,
            gaussians,
            pipe,
            background,
            args.mult,
            get_flag=True,
            metric_map=total_metric_map,
        )
        render_image = total_render["render"]
        total_counts = total_render["accum_metric_counts"]
        photometric_loss = compute_photometric_loss(
            viewpoint_camera,
            render_image,
        )

        normalized_l1 = get_loss(render_image, target_image)
        metric_map = (normalized_l1 > args.loss_thresh).int()
        metric_render = render_fastgs(
            viewpoint_camera,
            gaussians,
            pipe,
            background,
            args.mult,
            get_flag=True,
            metric_map=metric_map,
        )
        metric_counts = metric_render["accum_metric_counts"]

        (
            high_frequency_mask,
            high_frequency_cache_hit,
        ) = _get_cached_high_frequency_mask(
            viewpoint_camera,
            target_image,
            detail_config,
        )
        high_frequency_cache_hits += int(high_frequency_cache_hit)
        detail_mask, mask_statistics = build_detail_need_mask(
            render_image,
            target_image,
            detail_config,
            high_frequency_mask=high_frequency_mask,
            statistics_as_tensors=True,
        )
        detail_render = render_fastgs(
            viewpoint_camera,
            gaussians,
            pipe,
            background,
            args.mult,
            get_flag=True,
            metric_map=detail_mask,
        )
        detail_counts = detail_render["accum_metric_counts"]
        (
            overlap_ratio,
            visible_mask,
            support_mask,
        ) = build_detail_view_statistics(
            detail_counts,
            total_counts,
            detail_config,
            validate_values=False,
        )

        if full_metric_counts is None:
            full_metric_counts = metric_counts.clone()
            full_metric_score = photometric_loss * metric_counts.clone()
            overlap_ratio_sum = overlap_ratio.clone()
            detail_view_hits = support_mask.to(torch.int32)
            visible_view_count = visible_mask.to(torch.int32)
        else:
            full_metric_counts += metric_counts
            full_metric_score += photometric_loss * metric_counts
            overlap_ratio_sum += overlap_ratio
            detail_view_hits += support_mask.to(torch.int32)
            visible_view_count += visible_mask.to(torch.int32)

        high_frequency_ratio_sum += mask_statistics[
            "high_frequency_ratio"
        ]
        error_ratio_sum += mask_statistics["error_ratio"]
        detail_ratio_sum += mask_statistics["detail_ratio"]

    score_minimum = torch.min(full_metric_score)
    score_denominator = (
        torch.max(full_metric_score) - score_minimum
    ).clamp_min(1e-8)
    pruning_score = (
        full_metric_score - score_minimum
    ) / score_denominator
    importance_score = torch.div(
        full_metric_counts,
        number_of_views,
        rounding_mode="floor",
    )
    (
        detail_score,
        detail_support_ratio,
        detail_candidate_mask,
    ) = build_detail_candidate_mask(
        overlap_ratio_sum,
        detail_view_hits,
        visible_view_count,
        detail_config,
    )

    high_frequency_ratio_sum /= number_of_views
    error_ratio_sum /= number_of_views
    detail_ratio_sum /= number_of_views
    high_frequency_ratio = float(high_frequency_ratio_sum.item())
    error_ratio = float(error_ratio_sum.item())
    detail_pixel_ratio = float(detail_ratio_sum.item())
    statistics = {
        "high_frequency_ratio": high_frequency_ratio,
        "error_ratio": error_ratio,
        "detail_pixel_ratio": detail_pixel_ratio,
        "detail_candidate_count": int(
            detail_candidate_mask.sum().item()
        ),
        "detail_mean_score": float(detail_score.mean().item()),
        "detail_view_hits_min": int(detail_view_hits.min().item()),
        "detail_view_hits_mean": float(
            detail_view_hits.to(torch.float32).mean().item()
        ),
        "detail_view_hits_max": int(detail_view_hits.max().item()),
        "visible_view_count_min": int(visible_view_count.min().item()),
        "visible_view_count_mean": float(
            visible_view_count.to(torch.float32).mean().item()
        ),
        "visible_view_count_max": int(visible_view_count.max().item()),
        "detail_support_ratio_mean": float(
            detail_support_ratio.mean().item()
        ),
        "high_frequency_cache_hits": high_frequency_cache_hits,
    }
    statistics["detail_scoring_seconds"] = (
        time.perf_counter() - scoring_started_at
    )
    return DetailScoreResult(
        importance_score=importance_score,
        pruning_score=pruning_score,
        detail_score=detail_score,
        detail_view_hits=detail_view_hits,
        visible_view_count=visible_view_count,
        detail_support_ratio=detail_support_ratio,
        detail_candidate_mask=detail_candidate_mask,
        statistics=statistics,
    )
