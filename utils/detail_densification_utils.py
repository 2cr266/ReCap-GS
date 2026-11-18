from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _require_finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a finite number".format(name))
    if not math.isfinite(float(value)):
        raise ValueError("{} must be finite".format(name))


def _require_integer(name, value):
    if type(value) is not int:
        raise TypeError("{} must be an integer".format(name))


@dataclass(frozen=True)
class DetailDensifyConfig:
    high_frequency_quantile: float = 0.70
    error_quantile: float = 0.70
    dilation_radius: int = 1
    min_visible_views: int = 3
    min_total_pixels_per_view: int = 8
    min_detail_views: int = 2
    min_detail_pixels_per_view: int = 3
    min_detail_overlap_ratio: float = 0.25
    min_detail_support_ratio: float = 0.50
    min_average_detail_overlap_ratio: float = 0.20
    detail_gradient_scale: float = 0.80
    detail_importance_threshold: int = 4
    debug_interval: int = 1_000

    def __post_init__(self):
        for name in ("high_frequency_quantile", "error_quantile"):
            value = getattr(self, name)
            _require_finite_number(name, value)
            if not 0.0 < value < 1.0:
                raise ValueError("{} must be in (0, 1)".format(name))

        _require_integer("dilation_radius", self.dilation_radius)
        if self.dilation_radius < 0:
            raise ValueError("dilation_radius must be >= 0")

        for name in (
            "min_visible_views",
            "min_total_pixels_per_view",
            "min_detail_views",
            "min_detail_pixels_per_view",
            "detail_importance_threshold",
            "debug_interval",
        ):
            value = getattr(self, name)
            _require_integer(name, value)
            if value <= 0:
                raise ValueError("{} must be > 0".format(name))

        for name in (
            "min_detail_overlap_ratio",
            "min_detail_support_ratio",
            "min_average_detail_overlap_ratio",
        ):
            value = getattr(self, name)
            _require_finite_number(name, value)
            if not 0.0 < value <= 1.0:
                raise ValueError("{} must be in (0, 1]".format(name))

        _require_finite_number(
            "detail_gradient_scale",
            self.detail_gradient_scale,
        )
        if not 0.0 < self.detail_gradient_scale <= 1.0:
            raise ValueError("detail_gradient_scale must be in (0, 1]")


def _validate_image(name, image):
    if not torch.is_tensor(image):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("{} must have shape [C, H, W] with C >= 3".format(name))
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("{} spatial dimensions must be non-empty".format(name))
    if not image.is_floating_point():
        raise TypeError("{} must use a floating-point dtype".format(name))


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    _validate_image("image", image)
    return (
        0.299 * image[0:1]
        + 0.587 * image[1:2]
        + 0.114 * image[2:3]
    ).contiguous()


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image).unsqueeze(0)
    kernel_x = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padding_mode = (
        "reflect"
        if gray.shape[-2] > 1 and gray.shape[-1] > 1
        else "replicate"
    )
    padded_gray = F.pad(gray, (1, 1, 1, 1), mode=padding_mode)
    grad_x = F.conv2d(padded_gray, kernel_x)
    grad_y = F.conv2d(padded_gray, kernel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square())
    return magnitude[0, 0].contiguous()


def quantile_mask(score: torch.Tensor, quantile: float) -> torch.Tensor:
    if not torch.is_tensor(score):
        raise TypeError("score must be a torch.Tensor")
    _require_finite_number("quantile", quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")

    flat = score.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(score, dtype=torch.bool)

    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        return torch.zeros_like(score, dtype=torch.bool)

    threshold = torch.quantile(finite_values.float(), quantile)
    mask = score.float() > threshold
    mask = torch.logical_and(mask, finite.reshape(score.shape))
    return torch.logical_and(mask, torch.max(finite_values) > 1e-8)


@torch.no_grad()
def build_detail_need_mask(
    rendered: torch.Tensor,
    target: torch.Tensor,
    config: DetailDensifyConfig,
    high_frequency_mask: torch.Tensor = None,
    statistics_as_tensors: bool = False,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    _validate_image("rendered", rendered)
    _validate_image("target", target)
    if rendered.shape != target.shape:
        raise ValueError(
            "render/target mismatch: {} vs {}".format(
                tuple(rendered.shape),
                tuple(target.shape),
            )
        )
    if rendered.device != target.device:
        raise ValueError("rendered and target must be on the same device")
    if not isinstance(config, DetailDensifyConfig):
        raise TypeError("config must be a DetailDensifyConfig")
    if type(statistics_as_tensors) is not bool:
        raise TypeError("statistics_as_tensors must be a bool")
    detail_config = config

    rendered = rendered.detach().clamp(0.0, 1.0)
    target = target.detach().clamp(0.0, 1.0)

    if high_frequency_mask is None:
        high_frequency_score = sobel_magnitude(target)
        high_frequency_mask = quantile_mask(
            high_frequency_score,
            detail_config.high_frequency_quantile,
        )
    else:
        if not torch.is_tensor(high_frequency_mask):
            raise TypeError("high_frequency_mask must be a torch.Tensor")
        if high_frequency_mask.shape != target.shape[1:]:
            raise ValueError(
                "high_frequency_mask must match the target spatial shape"
            )
        if high_frequency_mask.device != target.device:
            raise ValueError(
                "high_frequency_mask and target must be on the same device"
            )
        high_frequency_mask = high_frequency_mask.to(
            dtype=torch.bool
        ).contiguous()
    error_score = torch.mean(torch.abs(rendered - target), dim=0)
    error_mask = quantile_mask(error_score, detail_config.error_quantile)
    detail_mask = torch.logical_and(high_frequency_mask, error_mask)

    if detail_config.dilation_radius > 0:
        kernel_size = 2 * detail_config.dilation_radius + 1
        detail_mask = (
            F.max_pool2d(
                detail_mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=detail_config.dilation_radius,
            )[0, 0]
            > 0
        )

    statistic_tensors = {
        "high_frequency_ratio": high_frequency_mask.float().mean(),
        "error_ratio": error_mask.float().mean(),
        "detail_ratio": detail_mask.float().mean(),
    }
    if statistics_as_tensors:
        statistics = statistic_tensors
    else:
        statistics = {
            name: value.item() for name, value in statistic_tensors.items()
        }
    return detail_mask.to(dtype=torch.int32).contiguous(), statistics
