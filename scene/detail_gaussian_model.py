import torch

from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation, inverse_sigmoid


def _require_vector(name, value, expected_length, expected_device):
    if not torch.is_tensor(value):
        raise TypeError("{} must be a tensor".format(name))
    if value.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if value.shape[0] != expected_length:
        raise ValueError(
            "{} length mismatch: {} vs {}".format(
                name,
                value.shape[0],
                expected_length,
            )
        )
    if value.device != expected_device:
        raise ValueError("{} is on the wrong device".format(name))


def build_detail_only_mask(normal_branch, detail_branch):
    """Return candidates introduced only by the relaxed detail branch."""
    if not torch.is_tensor(normal_branch) or normal_branch.ndim != 1:
        raise ValueError("normal_branch must be a one-dimensional tensor")
    point_count = normal_branch.shape[0]
    device = normal_branch.device
    _require_vector("detail_branch", detail_branch, point_count, device)
    if normal_branch.dtype != torch.bool:
        raise TypeError("normal_branch must use torch.bool")
    if detail_branch.dtype != torch.bool:
        raise TypeError("detail_branch must use torch.bool")
    return torch.logical_and(detail_branch, torch.logical_not(normal_branch))


def build_detail_branch_diagnostics(
    normal_clone,
    detail_clone,
    normal_split,
    detail_split,
    selected_clone,
    selected_split,
):
    """Summarize branch overlap and actual post-cap detail-only growth."""
    if not torch.is_tensor(normal_clone) or normal_clone.ndim != 1:
        raise ValueError("normal_clone must be a one-dimensional tensor")
    point_count = normal_clone.shape[0]
    device = normal_clone.device
    values = (
        ("normal_clone", normal_clone),
        ("detail_clone", detail_clone),
        ("normal_split", normal_split),
        ("detail_split", detail_split),
        ("selected_clone", selected_clone),
        ("selected_split", selected_split),
    )
    for name, value in values:
        _require_vector(name, value, point_count, device)
        if value.dtype != torch.bool:
            raise TypeError("{} must use torch.bool".format(name))

    detail_only_clone = build_detail_only_mask(normal_clone, detail_clone)
    detail_only_split = build_detail_only_mask(normal_split, detail_split)
    clone_overlap = torch.logical_and(normal_clone, detail_clone)
    split_overlap = torch.logical_and(normal_split, detail_split)
    selected_detail_only_clone = torch.logical_and(
        selected_clone,
        detail_only_clone,
    )
    selected_detail_only_split = torch.logical_and(
        selected_split,
        detail_only_split,
    )
    selected_detail_only_clone_count = int(
        selected_detail_only_clone.sum().item()
    )
    selected_detail_only_split_count = int(
        selected_detail_only_split.sum().item()
    )

    return {
        "normal_clone_candidates": int(normal_clone.sum().item()),
        "detail_clone_candidates": int(detail_clone.sum().item()),
        "normal_detail_clone_overlap_count": int(clone_overlap.sum().item()),
        "detail_only_clone_candidates": int(detail_only_clone.sum().item()),
        "normal_split_candidates": int(normal_split.sum().item()),
        "detail_split_candidates": int(detail_split.sum().item()),
        "normal_detail_split_overlap_count": int(split_overlap.sum().item()),
        "detail_only_split_candidates": int(detail_only_split.sum().item()),
        "selected_detail_only_clone_count": (
            selected_detail_only_clone_count
        ),
        "selected_detail_only_split_count": (
            selected_detail_only_split_count
        ),
        "detail_extra_children_count": (
            selected_detail_only_clone_count
            + 2 * selected_detail_only_split_count
        ),
        "detail_extra_net_gaussian_count": (
            selected_detail_only_clone_count
            + selected_detail_only_split_count
        ),
    }


def limit_densification_candidates(
    clone_mask,
    split_mask,
    detail_candidate_mask,
    detail_score,
    importance_score,
    remaining_capacity,
):
    """Limit topology growth with deterministic lexicographic ranking.

    Ranking is exact rather than encoded with arbitrary floating-point weights:
    detail candidates come first, followed by descending detail score, then
    descending original FastGS importance. Existing Gaussian index is the
    deterministic final tie breaker.
    """
    if type(remaining_capacity) is not int:
        raise TypeError("remaining_capacity must be an integer")
    if remaining_capacity < 0:
        raise ValueError("remaining_capacity must be >= 0")
    if not torch.is_tensor(clone_mask) or clone_mask.ndim != 1:
        raise ValueError("clone_mask must be a one-dimensional tensor")
    point_count = clone_mask.shape[0]
    device = clone_mask.device
    for name, value in (
        ("split_mask", split_mask),
        ("detail_candidate_mask", detail_candidate_mask),
        ("detail_score", detail_score),
        ("importance_score", importance_score),
    ):
        _require_vector(name, value, point_count, device)
    for name, value in (
        ("clone_mask", clone_mask),
        ("split_mask", split_mask),
        ("detail_candidate_mask", detail_candidate_mask),
    ):
        if value.dtype != torch.bool:
            raise TypeError("{} must use torch.bool".format(name))
    if torch.logical_and(clone_mask, split_mask).any().item():
        raise ValueError("clone and split candidates must be disjoint")
    if not torch.isfinite(detail_score).all().item():
        raise ValueError("detail_score must contain only finite values")
    if not torch.isfinite(importance_score).all().item():
        raise ValueError("importance_score must contain only finite values")

    combined = torch.logical_or(clone_mask, split_mask)
    candidate_indices = torch.nonzero(
        combined,
        as_tuple=False,
    ).squeeze(1)
    if remaining_capacity == 0 or candidate_indices.numel() == 0:
        return torch.zeros_like(clone_mask), torch.zeros_like(split_mask)
    if candidate_indices.numel() <= remaining_capacity:
        return clone_mask, split_mask

    # Stable least-significant-to-most-significant sorts implement a true
    # lexicographic order. candidate_indices starts in ascending point order,
    # which supplies a deterministic final tie breaker.
    ordered = candidate_indices
    order = torch.argsort(
        importance_score[ordered],
        descending=True,
        stable=True,
    )
    ordered = ordered[order]
    order = torch.argsort(
        detail_score[ordered],
        descending=True,
        stable=True,
    )
    ordered = ordered[order]
    order = torch.argsort(
        detail_candidate_mask[ordered].to(torch.int8),
        descending=True,
        stable=True,
    )
    ordered = ordered[order]

    selected_indices = ordered[:remaining_capacity]
    keep_mask = torch.zeros_like(combined)
    keep_mask[selected_indices] = True
    return (
        torch.logical_and(clone_mask, keep_mask),
        torch.logical_and(split_mask, keep_mask),
    )


class DetailGaussianModel(GaussianModel):
    def _validate_detail_tracking(self, expected_count=None):
        field_names = (
            "detail_region_flag",
            "detail_extra_origin_flag",
            "detail_lineage_flag",
            "birth_hr_iteration",
        )
        field_presence = [hasattr(self, name) for name in field_names]
        if any(field_presence) and not all(field_presence):
            raise RuntimeError(
                "Detail metadata fields must either all exist or all be absent"
            )
        if not any(field_presence):
            return False

        region_flags = self.detail_region_flag
        extra_origin_flags = self.detail_extra_origin_flag
        lineage_flags = self.detail_lineage_flag
        births = self.birth_hr_iteration
        fields = (
            ("detail_region_flag", region_flags, torch.bool),
            ("detail_extra_origin_flag", extra_origin_flags, torch.bool),
            ("detail_lineage_flag", lineage_flags, torch.bool),
            ("birth_hr_iteration", births, torch.int32),
        )
        reference_length = None
        model_device = self.get_xyz.device
        for name, value, expected_dtype in fields:
            if not torch.is_tensor(value):
                raise TypeError("{} must be a tensor".format(name))
            if value.ndim != 1:
                raise ValueError("{} must be one-dimensional".format(name))
            if reference_length is None:
                reference_length = value.shape[0]
            elif value.shape[0] != reference_length:
                raise ValueError("Detail metadata field lengths do not match")
            if value.dtype != expected_dtype:
                raise TypeError(
                    "{} must use {}".format(name, expected_dtype)
                )
            if value.device != model_device:
                raise ValueError("{} is on the wrong device".format(name))
        if expected_count is not None and reference_length != expected_count:
            raise ValueError(
                "Detail metadata/Gaussian count mismatch: {} vs {}".format(
                    reference_length,
                    expected_count,
                )
            )

        if births.numel():
            if (births < -1).any().item() or (births == 0).any().item():
                raise ValueError(
                    "birth_hr_iteration must be -1 or a positive HR iteration"
                )
            any_detail_flag = torch.logical_or(
                region_flags,
                torch.logical_or(extra_origin_flags, lineage_flags),
            )
            if torch.logical_and(any_detail_flag, births < 1).any().item():
                raise ValueError(
                    "Detail metadata flags require a positive birth iteration"
                )
            if torch.logical_and(
                extra_origin_flags,
                torch.logical_not(region_flags),
            ).any().item():
                raise ValueError(
                    "detail_extra_origin_flag requires detail_region_flag"
                )
            if torch.logical_and(
                extra_origin_flags,
                torch.logical_not(lineage_flags),
            ).any().item():
                raise ValueError(
                    "detail_extra_origin_flag requires detail_lineage_flag"
                )
        return True

    def validate_detail_tracking(self):
        if not self._validate_detail_tracking(self.get_xyz.shape[0]):
            raise RuntimeError("Detail tracking has not been initialized")
        return True

    def initialize_detail_tracking(self):
        point_count = self.get_xyz.shape[0]
        if self._validate_detail_tracking():
            self._validate_detail_tracking(point_count)
            return
        device = self.get_xyz.device
        self.detail_region_flag = torch.zeros(
            (point_count,),
            dtype=torch.bool,
            device=device,
        )
        self.detail_extra_origin_flag = torch.zeros(
            (point_count,),
            dtype=torch.bool,
            device=device,
        )
        self.detail_lineage_flag = torch.zeros(
            (point_count,),
            dtype=torch.bool,
            device=device,
        )
        self.birth_hr_iteration = torch.full(
            (point_count,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if not self._validate_detail_tracking(point_count):
            raise RuntimeError("Detail tracking has not been initialized")

    def append_detail_metadata(
        self,
        new_region_flags,
        new_extra_origin_flags,
        new_lineage_flags,
        current_hr_iteration,
    ):
        if type(current_hr_iteration) is not int or current_hr_iteration < 1:
            raise ValueError("current_hr_iteration must be a positive integer")
        if not self._validate_detail_tracking():
            raise RuntimeError("Detail tracking must be initialized before append")

        device = self.get_xyz.device
        if not torch.is_tensor(new_region_flags) or new_region_flags.ndim != 1:
            raise ValueError("new_region_flags must be a one-dimensional tensor")
        new_count = new_region_flags.shape[0]
        for name, value in (
            ("new_region_flags", new_region_flags),
            ("new_extra_origin_flags", new_extra_origin_flags),
            ("new_lineage_flags", new_lineage_flags),
        ):
            _require_vector(name, value, new_count, device)
            if value.dtype != torch.bool:
                raise TypeError("{} must use torch.bool".format(name))
        if torch.logical_and(
            new_extra_origin_flags,
            torch.logical_not(new_region_flags),
        ).any().item():
            raise ValueError("Extra-origin children must be detail-region children")
        if torch.logical_and(
            new_extra_origin_flags,
            torch.logical_not(new_lineage_flags),
        ).any().item():
            raise ValueError("Extra-origin children must belong to detail lineage")

        new_births = torch.full(
            (new_count,),
            current_hr_iteration,
            dtype=torch.int32,
            device=device,
        )
        self.detail_region_flag = torch.cat(
            (self.detail_region_flag, new_region_flags),
            dim=0,
        )
        self.detail_extra_origin_flag = torch.cat(
            (self.detail_extra_origin_flag, new_extra_origin_flags),
            dim=0,
        )
        self.detail_lineage_flag = torch.cat(
            (self.detail_lineage_flag, new_lineage_flags),
            dim=0,
        )
        self.birth_hr_iteration = torch.cat(
            (self.birth_hr_iteration, new_births),
            dim=0,
        )
        self._validate_detail_tracking()

    def prune_points(self, mask):
        if not torch.is_tensor(mask) or mask.ndim != 1:
            raise ValueError("prune mask must be a one-dimensional tensor")
        if mask.dtype != torch.bool:
            raise TypeError("prune mask must use torch.bool")

        tracking_initialized = self._validate_detail_tracking()
        if not tracking_initialized:
            super().prune_points(mask)
            return
        if mask.device != self.get_xyz.device:
            raise ValueError("prune mask is on the wrong device")
        self._validate_detail_tracking(mask.shape[0])
        valid_mask = torch.logical_not(mask)
        filtered_region_flags = self.detail_region_flag[valid_mask]
        filtered_extra_origin_flags = self.detail_extra_origin_flag[valid_mask]
        filtered_lineage_flags = self.detail_lineage_flag[valid_mask]
        filtered_births = self.birth_hr_iteration[valid_mask]

        # Commit metadata only after the base topology mutation succeeds.
        super().prune_points(mask)
        self.detail_region_flag = filtered_region_flags
        self.detail_extra_origin_flag = filtered_extra_origin_flags
        self.detail_lineage_flag = filtered_lineage_flags
        self.birth_hr_iteration = filtered_births
        self._validate_detail_tracking(self.get_xyz.shape[0])

    def densify_and_clone_fastgs_detail(
        self,
        selected_mask,
        detail_region_parent_mask,
        detail_extra_origin_parent_mask,
        current_hr_iteration,
    ):
        point_count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        _require_vector("selected_mask", selected_mask, point_count, device)
        _require_vector(
            "detail_region_parent_mask",
            detail_region_parent_mask,
            point_count,
            device,
        )
        _require_vector(
            "detail_extra_origin_parent_mask",
            detail_extra_origin_parent_mask,
            point_count,
            device,
        )
        if selected_mask.dtype != torch.bool:
            raise TypeError("selected_mask must use torch.bool")
        if detail_region_parent_mask.dtype != torch.bool:
            raise TypeError("detail_region_parent_mask must use torch.bool")
        if detail_extra_origin_parent_mask.dtype != torch.bool:
            raise TypeError(
                "detail_extra_origin_parent_mask must use torch.bool"
            )
        if torch.logical_and(
            detail_extra_origin_parent_mask,
            torch.logical_not(detail_region_parent_mask),
        ).any().item():
            raise ValueError(
                "Extra-origin clone parents must be detail-region parents"
            )
        if not self._validate_detail_tracking(point_count):
            raise RuntimeError("Detail tracking has not been initialized")

        new_xyz = self._xyz[selected_mask]
        new_features_dc = self._features_dc[selected_mask]
        new_features_rest = self._features_rest[selected_mask]
        new_opacities = self._opacity[selected_mask]
        new_scaling = self._scaling[selected_mask]
        new_rotation = self._rotation[selected_mask]
        new_tmp_radii = self.tmp_radii[selected_mask]
        new_region_flags = detail_region_parent_mask[selected_mask]
        new_extra_origin_flags = detail_extra_origin_parent_mask[selected_mask]
        new_lineage_flags = torch.logical_or(
            self.detail_lineage_flag[selected_mask],
            new_extra_origin_flags,
        )

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        self.append_detail_metadata(
            new_region_flags=new_region_flags,
            new_extra_origin_flags=new_extra_origin_flags,
            new_lineage_flags=new_lineage_flags,
            current_hr_iteration=current_hr_iteration,
        )
        self._validate_detail_tracking(self.get_xyz.shape[0])

    def densify_and_split_fastgs_detail(
        self,
        selected_mask,
        detail_region_parent_mask,
        detail_extra_origin_parent_mask,
        current_hr_iteration,
        N=2,
    ):
        if type(N) is not int or N != 2:
            raise ValueError("FastGS detail split requires N=2")
        if not torch.is_tensor(selected_mask) or selected_mask.ndim != 1:
            raise ValueError("selected_mask must be a one-dimensional tensor")
        original_count = selected_mask.shape[0]
        device = self.get_xyz.device
        _require_vector(
            "detail_region_parent_mask",
            detail_region_parent_mask,
            original_count,
            device,
        )
        _require_vector(
            "detail_extra_origin_parent_mask",
            detail_extra_origin_parent_mask,
            original_count,
            device,
        )
        if selected_mask.device != device:
            raise ValueError("selected_mask is on the wrong device")
        if selected_mask.dtype != torch.bool:
            raise TypeError("selected_mask must use torch.bool")
        if detail_region_parent_mask.dtype != torch.bool:
            raise TypeError("detail_region_parent_mask must use torch.bool")
        if detail_extra_origin_parent_mask.dtype != torch.bool:
            raise TypeError(
                "detail_extra_origin_parent_mask must use torch.bool"
            )
        if torch.logical_and(
            detail_extra_origin_parent_mask,
            torch.logical_not(detail_region_parent_mask),
        ).any().item():
            raise ValueError(
                "Extra-origin split parents must be detail-region parents"
            )

        n_init_points = self.get_xyz.shape[0]
        if original_count > n_init_points:
            raise ValueError("selected_mask exceeds the current Gaussian count")
        if not self._validate_detail_tracking(n_init_points):
            raise RuntimeError("Detail tracking has not been initialized")

        selected_pts_mask = torch.zeros(
            (n_init_points,),
            dtype=torch.bool,
            device=device,
        )
        selected_pts_mask[:original_count] = selected_mask
        padded_region_parent_mask = torch.zeros(
            (n_init_points,),
            dtype=torch.bool,
            device=device,
        )
        padded_region_parent_mask[:original_count] = detail_region_parent_mask
        padded_extra_origin_parent_mask = torch.zeros(
            (n_init_points,),
            dtype=torch.bool,
            device=device,
        )
        padded_extra_origin_parent_mask[:original_count] = (
            detail_extra_origin_parent_mask
        )
        padded_lineage_parent_mask = torch.zeros(
            (n_init_points,),
            dtype=torch.bool,
            device=device,
        )
        padded_lineage_parent_mask[:original_count] = (
            self.detail_lineage_flag[:original_count]
        )
        new_region_flags = padded_region_parent_mask[selected_pts_mask].repeat(N)
        new_extra_origin_flags = padded_extra_origin_parent_mask[
            selected_pts_mask
        ].repeat(N)
        new_lineage_flags = torch.logical_or(
            padded_lineage_parent_mask[selected_pts_mask],
            padded_extra_origin_parent_mask[selected_pts_mask],
        ).repeat(N)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N,
            1,
            1,
        )
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        self.append_detail_metadata(
            new_region_flags=new_region_flags,
            new_extra_origin_flags=new_extra_origin_flags,
            new_lineage_flags=new_lineage_flags,
            current_hr_iteration=current_hr_iteration,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * int(selected_pts_mask.sum().item()),
                    device=device,
                    dtype=torch.bool,
                ),
            )
        )
        self.prune_points(prune_filter)
        self._validate_detail_tracking(self.get_xyz.shape[0])

    def densify_and_prune_fastgs_detail(
        self,
        max_screen_size,
        min_opacity,
        extent,
        radii,
        args,
        importance_score,
        pruning_score,
        detail_score,
        detail_candidate_mask,
        detail_config,
        current_hr_iteration,
        max_gaussians=0,
    ):
        if type(current_hr_iteration) is not int or current_hr_iteration < 1:
            raise ValueError("current_hr_iteration must be a positive integer")
        if type(max_gaussians) is not int or max_gaussians < 0:
            raise ValueError("max_gaussians must be a non-negative integer")
        point_count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        if not self._validate_detail_tracking(point_count):
            raise RuntimeError("Detail tracking has not been initialized")
        for name, value in (
            ("radii", radii),
            ("importance_score", importance_score),
            ("pruning_score", pruning_score),
            ("detail_score", detail_score),
            ("detail_candidate_mask", detail_candidate_mask),
        ):
            _require_vector(name, value, point_count, device)
        if detail_candidate_mask.dtype != torch.bool:
            raise TypeError("detail_candidate_mask must use torch.bool")
        if not detail_score.is_floating_point():
            raise TypeError("detail_score must use a floating-point dtype")
        if not pruning_score.is_floating_point():
            raise TypeError("pruning_score must use a floating-point dtype")
        for name, value in (
            ("importance_score", importance_score),
            ("pruning_score", pruning_score),
            ("detail_score", detail_score),
        ):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("{} must contain only finite values".format(name))

        grad_variables = self.xyz_gradient_accum / self.denom
        grad_variables[grad_variables.isnan()] = 0.0
        self.tmp_radii = radii

        absolute_gradients = self.xyz_gradient_accum_abs / self.denom
        absolute_gradients[absolute_gradients.isnan()] = 0.0

        position_gradient = torch.norm(grad_variables, dim=-1)
        absolute_gradient = torch.norm(absolute_gradients, dim=-1)
        small_gaussian = (
            torch.max(self.get_scaling, dim=1).values <= args.dense * extent
        )
        large_gaussian = (
            torch.max(self.get_scaling, dim=1).values > args.dense * extent
        )

        normal_metric = importance_score > 5
        normal_clone = torch.logical_and(
            small_gaussian,
            torch.logical_and(
                normal_metric,
                position_gradient >= args.grad_thresh,
            ),
        )
        normal_split = torch.logical_and(
            large_gaussian,
            torch.logical_and(
                normal_metric,
                absolute_gradient >= args.grad_abs_thresh,
            ),
        )

        detail_metric = torch.logical_and(
            detail_candidate_mask,
            importance_score > detail_config.detail_importance_threshold,
        )
        detail_clone = torch.logical_and(
            small_gaussian,
            torch.logical_and(
                detail_metric,
                position_gradient
                >= args.grad_thresh * detail_config.detail_gradient_scale,
            ),
        )
        detail_split = torch.logical_and(
            large_gaussian,
            torch.logical_and(
                detail_metric,
                absolute_gradient
                >= args.grad_abs_thresh * detail_config.detail_gradient_scale,
            ),
        )

        detail_only_clone = build_detail_only_mask(normal_clone, detail_clone)
        detail_only_split = build_detail_only_mask(normal_split, detail_split)
        final_clone = torch.logical_or(normal_clone, detail_clone)
        final_split = torch.logical_or(normal_split, detail_split)
        combined_count = int(
            torch.logical_or(final_clone, final_split).sum().item()
        )
        cap_reached = False
        if max_gaussians > 0:
            remaining_capacity = max(0, max_gaussians - point_count)
            cap_reached = combined_count > remaining_capacity
            final_clone, final_split = limit_densification_candidates(
                final_clone,
                final_split,
                detail_candidate_mask,
                detail_score,
                importance_score,
                remaining_capacity,
            )

        selected_clone_count = int(final_clone.sum().item())
        selected_split_count = int(final_split.sum().item())
        diagnostics = build_detail_branch_diagnostics(
            normal_clone,
            detail_clone,
            normal_split,
            detail_split,
            final_clone,
            final_split,
        )
        diagnostics.update({
            "initial_gaussian_count": point_count,
            "detail_candidate_count": int(detail_candidate_mask.sum().item()),
            "selected_clone_count": selected_clone_count,
            "selected_split_count": selected_split_count,
            "new_gaussian_count": selected_clone_count + selected_split_count,
            "cap_reached": cap_reached,
            "max_gaussians": max_gaussians,
        })

        # All provenance masks retain event-start order. The split helper pads them
        # after clone so freshly cloned points cannot also split in this event.
        self.densify_and_clone_fastgs_detail(
            final_clone,
            detail_candidate_mask,
            detail_only_clone,
            current_hr_iteration,
        )
        self.densify_and_split_fastgs_detail(
            final_split,
            detail_candidate_mask,
            detail_only_split,
            current_hr_iteration,
        )

        # Preserve the original FastGS ordinary pruning path exactly, including
        # its CPU multinomial RNG, inverse-score weights, and 50% budget.
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs),
                big_points_ws,
            )

        scores = 1 - pruning_score
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)
        ordinary_pruned_count = 0
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros(
                (n_init_points),
                dtype=torch.float32,
            )
            padded_importance[:scores.shape[0]] = (
                1 / (1e-6 + scores.squeeze())
            )
            selected_pts_mask = torch.zeros_like(
                padded_importance,
                dtype=torch.bool,
                device=device,
            )
            sampled_indices = torch.multinomial(
                padded_importance,
                remove_budget,
                replacement=False,
            )
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            ordinary_pruned_count = int(final_prune.sum().item())
            self.prune_points(final_prune)

        opacities_new = inverse_sigmoid(
            torch.min(
                self.get_opacity,
                torch.ones_like(self.get_opacity) * 0.8
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new,
            "opacity",
        )
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        self._validate_detail_tracking(self.get_xyz.shape[0])
        diagnostics["ordinary_pruned_count"] = ordinary_pruned_count
        diagnostics["total_gaussian_count"] = self.get_xyz.shape[0]
        diagnostics["detail_region_alive_count"] = int(
            self.detail_region_flag.sum().item()
        )
        diagnostics["detail_extra_origin_alive_count"] = int(
            self.detail_extra_origin_flag.sum().item()
        )
        diagnostics["detail_lineage_alive_count"] = int(
            self.detail_lineage_flag.sum().item()
        )
        diagnostics["remaining_capacity"] = (
            None
            if max_gaussians == 0
            else max(0, max_gaussians - self.get_xyz.shape[0])
        )

        torch.cuda.empty_cache()
        return diagnostics
