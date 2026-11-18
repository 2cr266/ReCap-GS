#!/usr/bin/env python3
"""Generate white-composited x4 LR and SwinIR targets for NeRF Synthetic."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_srgs_swinir_x4 import (  # noqa: E402
    downsample_like_srgs,
    infer_one,
    save_lossless_rgb,
    validate_saved_image,
)
from scripts.run_swinir_x4 import (  # noqa: E402
    build_swinir_classical,
    default_model_path,
)


DEFAULT_SCENES = (
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare strict white-background NeRF Synthetic x4 pairs."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=ROOT / "data" / "nerf_synthetic",
    )
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--lr_dir", default="train_lr_srgs_x4")
    parser.add_argument("--sr_dir", default="train_srgs_swinir_x4")
    parser.add_argument("--swinir_root", type=Path, default=ROOT / "third_party" / "SwinIR")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=int, choices=(4, 8), default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def source_path(scene_root, frame):
    value = frame.get("file_path")
    if not isinstance(value, str) or not value:
        raise ValueError("Synthetic frame is missing file_path")
    relative = Path(value)
    if not relative.suffix:
        relative = Path(value + ".png")
    path = (scene_root / relative).resolve()
    try:
        path.relative_to(scene_root.resolve())
    except ValueError as error:
        raise ValueError("Frame escapes scene root: {}".format(path)) from error
    return path


def composite_white(path):
    with Image.open(str(path)) as image:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")


def collect_jobs(args):
    jobs = []
    for scene in args.scenes:
        scene_root = (args.data_root / scene).resolve()
        transforms_path = scene_root / "transforms_train.json"
        if not transforms_path.is_file():
            raise FileNotFoundError("Missing {}".format(transforms_path))
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Invalid frames in {}".format(transforms_path))
        lr_root = scene_root / args.lr_dir
        sr_root = scene_root / args.sr_dir
        lr_root.mkdir(parents=True, exist_ok=True)
        sr_root.mkdir(parents=True, exist_ok=True)
        seen = set()
        for frame in frames:
            source = source_path(scene_root, frame)
            if not source.is_file():
                raise FileNotFoundError("Missing source frame: {}".format(source))
            name = source.stem + ".png"
            if name in seen:
                raise ValueError("Duplicate train frame basename: {}".format(name))
            seen.add(name)
            jobs.append(
                {
                    "scene": scene,
                    "source": source,
                    "lr": lr_root / name,
                    "sr": sr_root / name,
                }
            )
    return jobs


def main():
    args = parse_args()
    jobs = collect_jobs(args)
    pending_sr = []
    counts = {scene: {"lr": 0, "sr": 0, "skipped": 0} for scene in args.scenes}
    for job in jobs:
        lr_exists = job["lr"].is_file()
        sr_exists = job["sr"].is_file()
        if lr_exists and sr_exists and not args.overwrite:
            counts[job["scene"]]["skipped"] += 1
            continue
        hr = composite_white(job["source"])
        lr = downsample_like_srgs(hr, scale=args.scale)
        expected_sr = (
            lr.size[0] * args.scale,
            lr.size[1] * args.scale,
        )
        if expected_sr != hr.size:
            raise ValueError(
                "NeRF Synthetic source dimensions must be divisible by {}: {} size={}".format(
                    args.scale, job["source"], hr.size
                )
            )
        save_lossless_rgb(lr, job["lr"])
        validate_saved_image(job["lr"], lr.size, "Synthetic LR")
        counts[job["scene"]]["lr"] += 1
        pending_sr.append((job, expected_sr))

    if pending_sr:
        import torch

        device = torch.device(args.device)
        if args.scale == 8 and args.model_path is None:
            raise ValueError("--model_path is required for SwinIR x8")
        model_path = args.model_path or default_model_path(args.swinir_root)
        model = build_swinir_classical(
            args.swinir_root,
            model_path,
            device,
            scale=args.scale,
        )
        for job, expected_sr in pending_sr:
            output = infer_one(
                model,
                job["lr"],
                expected_sr,
                device,
                scale=args.scale,
            )
            save_lossless_rgb(output, job["sr"])
            validate_saved_image(job["sr"], expected_sr, "Synthetic pseudo-HR")
            counts[job["scene"]]["sr"] += 1

    for scene in args.scenes:
        value = counts[scene]
        print(
            "{}: lr_generated={} sr_generated={} skipped={}".format(
                scene, value["lr"], value["sr"], value["skipped"]
            )
        )


if __name__ == "__main__":
    main()
