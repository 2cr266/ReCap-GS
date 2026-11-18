"""Opt-in paper panels from a real pre-densification scoring event."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_renderer import render_fastgs
from utils.detail_densification_utils import build_detail_need_mask, sobel_magnitude
from utils.frequency_loss_utils import bicubic_downsample, bicubic_upsample


def _save(root, name, value):
    value = value.detach().float().clamp(0, 1).cpu()
    if value.ndim == 3:
        value = value.permute(1, 2, 0)
    Image.fromarray((value * 255).round().to(torch.uint8).numpy()).save(root / name)


def _heat(value, limit):
    t = (value / max(float(limit), 1e-8)).clamp(0, 1)
    return torch.stack((t, (1 - (2 * t - 1).abs()).clamp(0, 1), 1 - t))


def _project(camera, gaussians, radii, values):
    """Mean score at visible projected centers; 5px dilation for legibility.

    This is a point-score projection, not an alpha-composited surface map.
    """
    xyz = gaussians.get_xyz
    clip = xyz @ camera.full_proj_transform[:3, :] + camera.full_proj_transform[3, :]
    w = clip[:, 3]
    valid = torch.isfinite(clip).all(1) & (w > 1e-8) & (radii > 0)
    safe_w = torch.where(valid, w, torch.ones_like(w))
    width, height = camera.image_width, camera.image_height
    x = ((clip[:, 0] / safe_w + 1) * width - 1) * 0.5
    y = ((clip[:, 1] / safe_w + 1) * height - 1) * 0.5
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    idx = y[valid].long() * width + x[valid].long()
    count = torch.zeros(height * width, device=xyz.device)
    total = torch.zeros_like(count)
    count.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    total.scatter_add_(0, idx, values[valid].float())
    field = (total / count.clamp_min(1)).reshape(height, width)
    return F.max_pool2d(field[None, None], 5, 1, 2)[0, 0]


@torch.no_grad()
def maybe_save_framework(model_path, iteration, hr_iteration, cameras,
                         gaussians, pipe, background, opt, detail_config, scores):
    if os.environ.get("FRAMEWORK_VIS", "0") != "1":
        return
    if hr_iteration < int(os.environ.get("FRAMEWORK_HR_ITER", "2000")):
        return
    root = Path(model_path) / "framework_visuals"
    if (root / "manifest.json").exists():
        return
    requested = os.environ.get("FRAMEWORK_CAMERA", "")
    ordered = sorted(cameras, key=lambda c: c.image_name)
    matching = [c for c in ordered if c.image_name == requested]
    if requested and not matching:
        return  # Wait until the requested view participates in actual scoring.
    camera = matching[0] if matching else ordered[0]
    root.mkdir(parents=True, exist_ok=True)
    package = render_fastgs(camera, gaussians, pipe, background, opt.mult)
    rendered = package["render"].detach().clamp(0, 1)
    sr = camera.original_image.to(rendered.device).clamp(0, 1)
    lr = camera.original_lr_image.to(rendered.device).clamp(0, 1)
    rendered_lr = bicubic_downsample(rendered, tuple(lr.shape[1:]))
    base = bicubic_upsample(lr, tuple(sr.shape[1:]))
    residual = sr - base
    structure = sobel_magnitude(sr)
    error = (rendered - sr).abs().mean(0)
    lr_error = (rendered_lr - lr).abs().mean(0)
    demand, _ = build_detail_need_mask(rendered, sr, detail_config)
    residual_limit = max(float(residual.abs().quantile(0.99)), 1e-8)
    structure_limit = max(float(structure.quantile(0.99)), 1e-8)
    for name, value in (
        ("01_lr_view.png", lr), ("02_pseudo_hr.png", sr),
        ("03_lr_consistency.png", _heat(lr_error, 0.1)),
        ("04_detail_residual.png", 0.5 + residual / (2 * residual_limit)),
        ("05_rgb_guidance.png", sr),
        ("06_structure.png", structure / structure_limit),
        ("07_error.png", _heat(error, 0.1)),
        ("09_demand_mask.png", demand.float()),
        ("render_hr.png", rendered), ("render_lr.png", rendered_lr),
        ("lr_upsampled.png", base),
    ):
        _save(root, name, value)
    support = _project(camera, gaussians, package["radii"], scores.detail_support_ratio)
    candidates = _project(camera, gaussians, package["radii"], scores.detail_candidate_mask)
    _save(root, "08_across_views.png", _heat(support, 1.0))
    _save(root, "10_multiview_candidates.png", candidates)
    _save(root, "demand_overlay.png", rendered * 0.65 +
          torch.stack((demand.float(), torch.zeros_like(error), torch.zeros_like(error))) * 0.35)
    # Save the actual contributing input views, not arbitrary neighboring frames.
    for index, view in enumerate(ordered):
        _save(root, "lr_view_{:02d}.png".format(index), view.original_lr_image)
    metadata = {
        "global_iteration": iteration, "hr_iteration": hr_iteration,
        "reference_camera": camera.image_name,
        "scoring_cameras": [c.image_name for c in ordered],
        "snapshot": "before clone/split/prune; actual training scoring result",
        "03_lr_consistency": "mean RGB abs(D(render)-real_LR), heat range [0,0.1]",
        "04_detail_residual": "pseudo_HR-U(real_LR), gray=zero; symmetric p99 scale",
        "05_rgb_guidance": "pseudo-HR RGB supervision target (same as 02)",
        "06_structure": "Sobel magnitude of pseudo-HR, p99 normalization",
        "07_error": "mean RGB abs(render-pseudo_HR), heat range [0,0.1]",
        "08_across_views": "support ratio from actual scoring; visible Gaussian center projection, mean per pixel then 5px max dilation; blue=0 red=1",
        "09_demand_mask": "single-view high-frequency AND error mask, with configured dilation",
        "10_multiview_candidates": "projected multi-view eligibility mask; before gradient/importance/capacity selection; not final clone/split mask",
        "residual_display_limit": residual_limit,
        "structure_display_limit": structure_limit,
        "scoring_statistics": scores.statistics,
    }
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[Framework visualization] saved {}".format(root))
