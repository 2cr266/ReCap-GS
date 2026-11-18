#!/usr/bin/env python3
"""Generate x4 pseudo-HR images with the official classical SwinIR-M model."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from urllib.request import urlretrieve

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used on minimal server envs
    def tqdm(iterable, **_: object):
        return iterable

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}
MODEL_NAME = "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
MODEL_URL = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME}"
NETWORK_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/models/network_swinir.py"
MODEL_MIN_BYTES = 10485760


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix in IMAGE_SUFFIXES)


def default_model_path(swinir_root: Path) -> Path:
    return swinir_root / "model_zoo" / "swinir" / MODEL_NAME


def default_swinir_root() -> Path:
    return Path("third_party") / "SwinIR"


def network_path(swinir_root: Path) -> Path:
    return swinir_root / "models" / "network_swinir.py"


def ensure_network_file(swinir_root: Path) -> None:
    target_path = network_path(swinir_root)
    if target_path.exists():
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SwinIR network definition to {target_path}")
    urlretrieve(NETWORK_URL, target_path)


def ensure_model_file(model_path: Path) -> None:
    if validate_model_file(model_path):
        return

    if model_path.exists():
        raise RuntimeError(format_model_file_error(model_path))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SwinIR classical x4 weights to {model_path}")
    urlretrieve(MODEL_URL, model_path)
    if not validate_model_file(model_path):
        raise RuntimeError(format_model_file_error(model_path))


def validate_model_file(model_path: Path) -> bool:
    if not model_path.exists():
        return False

    if model_path.stat().st_size < MODEL_MIN_BYTES:
        return False

    with model_path.open("rb") as handle:
        header = handle.read(512).lower()

    bad_markers = [
        b"<!doctype html",
        b"<html",
        b"not found",
        b"version https://git-lfs.github.com/spec",
    ]
    return not any(marker in header for marker in bad_markers)


def format_model_file_error(model_path: Path) -> str:
    size_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0.0
    return (
        f"SwinIR weight file looks incomplete or corrupt: {model_path} ({size_mb:.2f} MB).\n"
        "Delete it and download again, for example:\n"
        f"  rm -f {model_path}\n"
        f"  wget -O {model_path} {MODEL_URL}"
    )


def ensure_swinir_files(swinir_root: Path, model_path: Path) -> None:
    ensure_network_file(swinir_root)
    ensure_model_file(model_path)


def require_numpy():
    import numpy as np

    return np


def require_torch():
    import torch

    return torch


def require_image_class():
    from PIL import Image

    return Image


def load_swinir_class(swinir_root: Path):
    source_path = network_path(swinir_root)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {source_path}. Let the script download it, or pass --swinir_root to an existing SwinIR folder."
        )

    spec = importlib.util.spec_from_file_location("swinir_network_swinir", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import SwinIR network from {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def build_swinir_classical(
    swinir_root: Path,
    model_path: Path,
    device,
    scale: int,
):
    torch = require_torch()
    if scale not in (4, 8):
        raise ValueError("SwinIR scale must be 4 or 8")
    ensure_network_file(swinir_root)
    if scale == 4:
        ensure_model_file(model_path)
    elif not validate_model_file(model_path):
        raise RuntimeError(
            "A valid local SwinIR x8 weight is required: {}".format(
                model_path
            )
        )
    SwinIR = load_swinir_class(swinir_root)

    model = SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )

    try:
        checkpoint = torch.load(str(model_path), map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(format_model_file_error(model_path)) from exc
    if isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def build_swinir_classical_x4(swinir_root: Path, model_path: Path, device):
    return build_swinir_classical(
        swinir_root,
        model_path,
        device,
        scale=4,
    )


def image_to_tensor(image, device):
    np = require_numpy()
    torch = require_torch()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    np = require_numpy()
    Image = require_image_class()
    array = tensor.squeeze(0).detach().float().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def pad_to_window(tensor, window_size: int):
    torch = require_torch()
    _, _, height, width = tensor.size()
    height_pad = (window_size - height % window_size) % window_size
    width_pad = (window_size - width % window_size) % window_size
    target_height = height + height_pad
    target_width = width + width_pad

    while tensor.size(2) < target_height:
        tensor = torch.cat([tensor, torch.flip(tensor, [2])], dim=2)
    tensor = tensor[:, :, :target_height, :]

    while tensor.size(3) < target_width:
        tensor = torch.cat([tensor, torch.flip(tensor, [3])], dim=3)
    tensor = tensor[:, :, :, :target_width]
    return tensor, height, width


def forward_tiled(
    model,
    image,
    scale: int,
    tile: int,
    tile_overlap: int,
    window_size: int,
):
    torch = require_torch()
    if tile <= 0:
        return model(image)

    batch, channels, height, width = image.size()
    tile = min(tile, height, width)
    if tile % window_size != 0:
        raise ValueError(f"--tile must be a multiple of {window_size}, got {tile}")

    tile_overlap = min(tile_overlap, tile - 1)
    stride = max(tile - tile_overlap, 1)
    h_idx_list = sorted(set(list(range(0, max(height - tile, 0), stride)) + [height - tile]))
    w_idx_list = sorted(set(list(range(0, max(width - tile, 0), stride)) + [width - tile]))

    output = torch.zeros(batch, channels, height * scale, width * scale, dtype=image.dtype, device=image.device)
    weight = torch.zeros_like(output)
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = image[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            out_weight = torch.ones_like(out_patch)
            h_slice = slice(h_idx * scale, (h_idx + tile) * scale)
            w_slice = slice(w_idx * scale, (w_idx + tile) * scale)
            output[..., h_slice, w_slice].add_(out_patch)
            weight[..., h_slice, w_slice].add_(out_weight)

    return output.div_(weight)


def save_image(image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def resolve_device(device_name: str):
    torch = require_torch()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def run_folder(
    swinir_root: Path,
    input_dir: Path,
    output_dir: Path,
    model_path: Path | None = None,
    device_name: str = "cuda",
    tile: int = 256,
    tile_overlap: int = 32,
    overwrite: bool = False,
) -> None:
    Image = require_image_class()
    torch = require_torch()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    model_path = model_path or default_model_path(swinir_root)
    device = resolve_device(device_name)
    model = build_swinir_classical_x4(swinir_root=swinir_root, model_path=model_path, device=device)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for image_path in tqdm(image_paths, desc=f"SwinIR x4 -> {output_dir.name}"):
        output_path = output_dir / image_path.name
        if output_path.exists() and not overwrite:
            continue

        with Image.open(image_path) as pil_image:
            input_tensor = image_to_tensor(pil_image, device)

        padded, height, width = pad_to_window(input_tensor, window_size=8)
        with torch.no_grad():
            output = forward_tiled(
                model=model,
                image=padded,
                scale=4,
                tile=tile,
                tile_overlap=tile_overlap,
                window_size=8,
            )
        output = output[..., : height * 4, : width * 4]
        save_image(tensor_to_image(output), output_path)
        generated += 1

    print(f"Generated {generated} x4 SR images in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images_sr_x4 with classical SwinIR-M x4.")
    parser.add_argument(
        "--swinir_root",
        default=default_swinir_root(),
        type=Path,
        help="Folder for SwinIR files. Missing network/weight files are downloaded automatically.",
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Folder of LR images.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Folder for pseudo-HR images.")
    parser.add_argument("--model_path", default=None, type=Path, help="Optional path to classical SwinIR-M x4 weights.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--tile", default=256, type=int, help="Tile size. 0 means whole image. Use a multiple of 8.")
    parser.add_argument("--tile_overlap", default=32, type=int, help="Overlap between tiles.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_folder(
        swinir_root=args.swinir_root,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device_name=args.device,
        tile=args.tile,
        tile_overlap=args.tile_overlap,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
