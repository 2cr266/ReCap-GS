from dataclasses import dataclass
import math
from numbers import Real

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrequencyLossConfig:
    scale: int = 4
    lr_weight: float = 1.0
    hf_weight: float = 0.5
    rgb_weight: float = 0.1
    transition_iterations: int = 1_000
    hf_quantile: float = 0.70
    hf_absolute_threshold: float = 0.01
    hf_min_pixels: int = 256
    hf_border_ignore: int = 2
    charbonnier_epsilon: float = 1e-3

    def __post_init__(self):
        if type(self.scale) is not int or self.scale not in (4, 8):
            raise ValueError("Frequency supervision requires scale=4 or scale=8")
        for name in ("lr_weight", "hf_weight", "rgb_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("{} must be a real number".format(name))
            if not math.isfinite(float(value)):
                raise ValueError("{} must be finite".format(name))
            if value < 0.0:
                raise ValueError("{} must be >= 0".format(name))
        if type(self.transition_iterations) is not int:
            raise TypeError("transition_iterations must be an integer")
        if self.transition_iterations < 1:
            raise ValueError("transition_iterations must be >= 1")
        if isinstance(self.hf_quantile, bool) or not isinstance(
            self.hf_quantile,
            Real,
        ):
            raise TypeError("hf_quantile must be a real number")
        if not 0.0 <= self.hf_quantile <= 1.0:
            raise ValueError("hf_quantile must be in [0, 1]")
        if not math.isfinite(float(self.hf_quantile)):
            raise ValueError("hf_quantile must be finite")
        if isinstance(self.hf_absolute_threshold, bool) or not isinstance(
            self.hf_absolute_threshold,
            Real,
        ):
            raise TypeError("hf_absolute_threshold must be a real number")
        if self.hf_absolute_threshold < 0.0:
            raise ValueError("hf_absolute_threshold must be >= 0")
        if not math.isfinite(float(self.hf_absolute_threshold)):
            raise ValueError("hf_absolute_threshold must be finite")
        if type(self.hf_min_pixels) is not int:
            raise TypeError("hf_min_pixels must be an integer")
        if self.hf_min_pixels < 1:
            raise ValueError("hf_min_pixels must be >= 1")
        if type(self.hf_border_ignore) is not int:
            raise TypeError("hf_border_ignore must be an integer")
        if self.hf_border_ignore < 0:
            raise ValueError("hf_border_ignore must be >= 0")
        if isinstance(self.charbonnier_epsilon, bool) or not isinstance(
            self.charbonnier_epsilon,
            Real,
        ):
            raise TypeError("charbonnier_epsilon must be a real number")
        if self.charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be > 0")
        if not math.isfinite(float(self.charbonnier_epsilon)):
            raise ValueError("charbonnier_epsilon must be finite")


@dataclass(frozen=True)
class FrequencyTarget:
    target_base: torch.Tensor
    target_residual: torch.Tensor
    hf_mask: torch.Tensor
    mask_pixel_count: int
    mask_ratio: float
    target_residual_mean: float


@dataclass(frozen=True)
class FrequencyLossWeights:
    lr: float
    hf: float
    rgb: float
    alpha: float


@dataclass(frozen=True)
class FrequencyLossResult:
    total: torch.Tensor
    raw_lr: torch.Tensor
    raw_hf: torch.Tensor
    raw_rgb: torch.Tensor
    weighted_lr: torch.Tensor
    weighted_hf: torch.Tensor
    weighted_rgb: torch.Tensor
    lr_l1: torch.Tensor
    lr_dssim: torch.Tensor
    rgb_l1: torch.Tensor
    rgb_dssim: torch.Tensor
    render_lr: torch.Tensor
    target_base: torch.Tensor
    target_residual: torch.Tensor
    render_base: torch.Tensor
    render_residual: torch.Tensor
    target_lr: torch.Tensor
    hf_mask: torch.Tensor
    alpha: float
    mask_ratio: float
    target_residual_mean: float

    @property
    def render_residual_mean(self):
        return self.render_residual.detach().abs().mean()

    @property
    def lr_psnr(self):
        lr_mse = torch.mean(
            (self.render_lr.detach() - self.target_lr.detach()) ** 2
        )
        return -10.0 * torch.log10(lr_mse.clamp_min(1e-12))


def _validate_chw_image(image, label):
    if not torch.is_tensor(image):
        raise TypeError("{} must be a Tensor".format(label))
    if image.ndim != 3:
        raise ValueError("{} must have shape [C,H,W]".format(label))
    if image.shape[0] != 3:
        raise ValueError("{} must have exactly three RGB channels".format(label))
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError("{} spatial dimensions must be positive".format(label))
    if not image.is_floating_point():
        raise TypeError("{} must use a floating dtype".format(label))


def _normalize_size(size):
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise TypeError("size must be a two-item (height, width) sequence")
    height, width = size
    if type(height) is not int or type(width) is not int:
        raise TypeError("size values must be integers")
    if height < 1 or width < 1:
        raise ValueError("size values must be positive")
    return height, width


def bicubic_downsample(image, size):
    _validate_chw_image(image, "image")
    height, width = _normalize_size(size)
    return F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


def bicubic_upsample(image, size):
    _validate_chw_image(image, "image")
    height, width = _normalize_size(size)
    return F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)


def _clear_mask_border(mask, border):
    if border == 0:
        return mask
    result = mask.clone()
    result[:border, :] = False
    result[-border:, :] = False
    result[:, :border] = False
    result[:, -border:] = False
    return result


@torch.no_grad()
def build_frequency_target(target_lr, target_sr, config):
    if not isinstance(config, FrequencyLossConfig):
        raise TypeError("config must be a FrequencyLossConfig")
    _validate_chw_image(target_lr, "target_lr")
    _validate_chw_image(target_sr, "target_sr")
    expected_hr_size = (
        target_lr.shape[1] * config.scale,
        target_lr.shape[2] * config.scale,
    )
    if tuple(target_sr.shape[1:]) != expected_hr_size:
        raise ValueError(
            "target_sr must be exactly {}x target_lr: expected={} actual={}".format(
                config.scale,
                expected_hr_size,
                tuple(target_sr.shape[1:]),
            )
        )
    if target_lr.device != target_sr.device:
        raise ValueError("target_lr and target_sr must be on the same device")
    if target_lr.dtype != target_sr.dtype:
        raise ValueError("target_lr and target_sr must use the same dtype")
    if not bool(torch.isfinite(target_lr).all().item()) or not bool(
        torch.isfinite(target_sr).all().item()
    ):
        raise ValueError("frequency targets must contain only finite values")
    if float(target_lr.min().item()) < 0.0 or float(target_lr.max().item()) > 1.0:
        raise ValueError("target_lr must be in [0,1]")
    if float(target_sr.min().item()) < 0.0 or float(target_sr.max().item()) > 1.0:
        raise ValueError("target_sr must be in [0,1]")

    target_base = bicubic_upsample(target_lr, tuple(target_sr.shape[1:]))
    target_residual = target_sr - target_base
    strength = target_residual.abs().mean(dim=0)
    quantile_threshold = torch.quantile(strength, config.hf_quantile)
    hf_mask = (strength > quantile_threshold) & (
        strength > config.hf_absolute_threshold
    )
    hf_mask = _clear_mask_border(hf_mask, config.hf_border_ignore).contiguous()
    mask_pixel_count = int(hf_mask.sum().item())
    return FrequencyTarget(
        target_base=target_base.detach().contiguous(),
        target_residual=target_residual.detach().contiguous(),
        hf_mask=hf_mask.detach(),
        mask_pixel_count=mask_pixel_count,
        mask_ratio=float(hf_mask.float().mean().item()),
        target_residual_mean=float(target_residual.abs().mean().item()),
    )


class FrequencyTargetCache:
    def __init__(self, config):
        if not isinstance(config, FrequencyLossConfig):
            raise TypeError("config must be a FrequencyLossConfig")
        self._config = config
        self._entries = {}
        self._sources = {}

    @staticmethod
    def _source_signature(target_lr, target_sr):
        return (
            id(target_lr),
            tuple(target_lr.shape),
            target_lr.dtype,
            target_lr.device,
            id(target_sr),
            tuple(target_sr.shape),
            target_sr.dtype,
            target_sr.device,
        )

    def get(self, image_name, target_lr, target_sr):
        if not isinstance(image_name, str) or not image_name:
            raise ValueError("image_name must be a non-empty string")
        signature = self._source_signature(target_lr, target_sr)
        if image_name in self._entries:
            if self._sources[image_name] != signature:
                raise ValueError(
                    "Frequency target cache source changed for {}".format(
                        image_name
                    )
                )
            return self._entries[image_name]
        cache_target_lr = target_lr.detach().to(
            device="cpu",
            non_blocking=False,
        )
        cache_target_sr = target_sr.detach().to(
            device="cpu",
            non_blocking=False,
        )
        entry = build_frequency_target(
            cache_target_lr,
            cache_target_sr,
            self._config,
        )
        entry = FrequencyTarget(
            target_base=entry.target_base.to(
                device="cpu",
                non_blocking=False,
            ).contiguous(),
            target_residual=entry.target_residual.to(
                device="cpu",
                non_blocking=False,
            ).contiguous(),
            hf_mask=entry.hf_mask.to(
                device="cpu",
                non_blocking=False,
            ).contiguous(),
            mask_pixel_count=entry.mask_pixel_count,
            mask_ratio=entry.mask_ratio,
            target_residual_mean=entry.target_residual_mean,
        )
        self._entries[image_name] = entry
        self._sources[image_name] = signature
        return entry

    def __len__(self):
        return len(self._entries)


def masked_charbonnier(
    prediction,
    target,
    mask,
    epsilon=1e-3,
    min_pixels=256,
    mask_pixel_count=None,
):
    _validate_chw_image(prediction, "prediction")
    _validate_chw_image(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if prediction.dtype != target.dtype:
        raise ValueError("prediction and target must use the same dtype")
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a Tensor")
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or tuple(mask.shape) != tuple(prediction.shape[1:]):
        raise ValueError("mask must have shape [H,W] or [1,H,W]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must use torch.bool")
    if mask.device != prediction.device:
        raise ValueError("mask and prediction must be on the same device")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if type(min_pixels) is not int:
        raise TypeError("min_pixels must be an integer")
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")

    if mask_pixel_count is None:
        mask_pixel_count = int(mask.sum().item())
    elif type(mask_pixel_count) is not int:
        raise TypeError("mask_pixel_count must be an integer or None")
    elif mask_pixel_count < 0 or mask_pixel_count > mask.numel():
        raise ValueError("mask_pixel_count is out of range")
    if mask_pixel_count < min_pixels:
        return prediction.sum() * 0.0
    charbonnier = torch.sqrt((prediction - target) ** 2 + epsilon ** 2)
    masked_sum = (charbonnier * mask.unsqueeze(0)).sum()
    denominator = prediction.shape[0] * mask_pixel_count + epsilon
    return masked_sum / denominator


def frequency_loss_weights(hr_iteration, config):
    if not isinstance(config, FrequencyLossConfig):
        raise TypeError("config must be a FrequencyLossConfig")
    if type(hr_iteration) is not int:
        raise TypeError("hr_iteration must be an integer")
    if hr_iteration < 1:
        raise ValueError("hr_iteration must be >= 1")
    alpha = min(max(hr_iteration / float(config.transition_iterations), 0.0), 1.0)
    return FrequencyLossWeights(
        lr=float(config.lr_weight),
        hf=float(config.hf_weight) * alpha,
        rgb=float(config.rgb_weight) * alpha,
        alpha=alpha,
    )


def _l1_dssim(prediction, target, ssim_fn):
    l1_value = torch.abs(prediction - target).mean()
    ssim_value = ssim_fn(
        prediction.unsqueeze(0),
        target.unsqueeze(0),
    )
    if not torch.is_tensor(ssim_value):
        raise TypeError("ssim_fn must return a Tensor")
    if ssim_value.numel() != 1:
        ssim_value = ssim_value.mean()
    dssim_value = 1.0 - ssim_value
    return 0.8 * l1_value + 0.2 * dssim_value, l1_value, dssim_value


def compute_frequency_loss(
    render_hr,
    target_lr,
    target_sr,
    target,
    hr_iteration,
    config,
    ssim_fn,
):
    if not isinstance(config, FrequencyLossConfig):
        raise TypeError("config must be a FrequencyLossConfig")
    if not isinstance(target, FrequencyTarget):
        raise TypeError("target must be a FrequencyTarget")
    if not callable(ssim_fn):
        raise TypeError("ssim_fn must be callable")
    _validate_chw_image(render_hr, "render_hr")
    _validate_chw_image(target_lr, "target_lr")
    _validate_chw_image(target_sr, "target_sr")
    if render_hr.shape != target_sr.shape:
        raise ValueError("render_hr and target_sr shapes must match")
    if tuple(target_sr.shape[1:]) != (
        target_lr.shape[1] * config.scale,
        target_lr.shape[2] * config.scale,
    ):
        raise ValueError("target_sr must be exactly config.scale times target_lr")
    if render_hr.device != target_lr.device or render_hr.device != target_sr.device:
        raise ValueError("render and target images must be on the same device")
    if render_hr.dtype != target_lr.dtype or render_hr.dtype != target_sr.dtype:
        raise ValueError("render and target images must use the same dtype")

    target_base = target.target_base
    target_residual = target.target_residual.to(
        device=render_hr.device,
        dtype=render_hr.dtype,
        non_blocking=True,
    )
    hf_mask = target.hf_mask.to(device=render_hr.device, non_blocking=True)
    if target_base.shape != render_hr.shape or target_residual.shape != render_hr.shape:
        raise ValueError("cached frequency target shape mismatch")
    if tuple(hf_mask.shape) != tuple(render_hr.shape[1:]):
        raise ValueError("cached frequency mask shape mismatch")

    render_lr = bicubic_downsample(render_hr, tuple(target_lr.shape[1:]))
    raw_lr, lr_l1, lr_dssim = _l1_dssim(render_lr, target_lr, ssim_fn)
    render_base = bicubic_upsample(render_lr, tuple(render_hr.shape[1:]))
    render_residual = render_hr - render_base
    raw_hf = masked_charbonnier(
        render_residual,
        target_residual,
        hf_mask,
        epsilon=config.charbonnier_epsilon,
        min_pixels=config.hf_min_pixels,
        mask_pixel_count=target.mask_pixel_count,
    )
    raw_rgb, rgb_l1, rgb_dssim = _l1_dssim(render_hr, target_sr, ssim_fn)
    weights = frequency_loss_weights(hr_iteration, config)
    weighted_lr = weights.lr * raw_lr
    weighted_hf = weights.hf * raw_hf
    weighted_rgb = weights.rgb * raw_rgb
    total = weighted_lr + weighted_hf + weighted_rgb
    return FrequencyLossResult(
        total=total,
        raw_lr=raw_lr,
        raw_hf=raw_hf,
        raw_rgb=raw_rgb,
        weighted_lr=weighted_lr,
        weighted_hf=weighted_hf,
        weighted_rgb=weighted_rgb,
        lr_l1=lr_l1,
        lr_dssim=lr_dssim,
        rgb_l1=rgb_l1,
        rgb_dssim=rgb_dssim,
        render_lr=render_lr,
        target_base=target_base,
        target_residual=target_residual,
        render_base=render_base,
        render_residual=render_residual,
        target_lr=target_lr,
        hf_mask=hf_mask,
        alpha=weights.alpha,
        mask_ratio=target.mask_ratio,
        target_residual_mean=target.target_residual_mean,
    )
