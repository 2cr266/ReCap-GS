#!/usr/bin/env python3
"""Reproduce the official SRGS image-size and serialization protocol."""

from __future__ import annotations

import argparse
import os
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_swinir_x4 import (  # noqa: E402
    build_swinir_classical,
    default_model_path,
    image_to_tensor,
    list_images,
    tensor_to_image,
    tqdm,
)


SCALE = 4
WINDOW_SIZE = 8
DEFAULT_SCENES = (
    "bicycle",
    "flowers",
    "garden",
    "stump",
    "treehill",
    "room",
    "counter",
    "kitchen",
    "bonsai",
)


def require_image_class():
    from PIL import Image

    return Image


def require_torch():
    import torch

    return torch


def srgs_output_sizes(width: int, height: int, scale: int = SCALE):
    lr_size = (round(width / scale), round(height / scale))
    if lr_size[0] < 1 or lr_size[1] < 1:
        raise ValueError("SRGS output dimensions must be at least one pixel")

    sr_size = (lr_size[0] * scale, lr_size[1] * scale)
    return lr_size, sr_size


def srgs_padded_length(length: int, window_size: int = WINDOW_SIZE) -> int:
    if length < 1:
        raise ValueError("SRGS input length must be at least one pixel")
    return (length // window_size + 1) * window_size


def pad_like_srgs(tensor, window_size: int = WINDOW_SIZE):
    torch = require_torch()
    _, _, height, width = tensor.size()
    target_height = srgs_padded_length(height, window_size)
    target_width = srgs_padded_length(width, window_size)
    padded = tensor
    while padded.size(2) < target_height:
        padded = torch.cat([padded, torch.flip(padded, [2])], 2)
    padded = padded[:, :, :target_height, :]
    while padded.size(3) < target_width:
        padded = torch.cat([padded, torch.flip(padded, [3])], 3)
    padded = padded[:, :, :, :target_width]
    return padded, height, width


def downsample_like_srgs(image, scale: int = SCALE):
    Image = require_image_class()
    rgb_image = image.convert("RGB")
    lr_size, _ = srgs_output_sizes(*rgb_image.size, scale=scale)
    try:
        bicubic = Image.Resampling.BICUBIC
    except AttributeError:  # pragma: no cover - compatibility with older Pillow
        bicubic = Image.BICUBIC
    return rgb_image.resize(lr_size, resample=bicubic)


def validate_output_target(path, label="output") -> bool:
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError(f"Could not inspect {label} target at {path}") from error

    if stat.S_ISLNK(mode):
        raise ValueError(f"Refusing symlink {label} target at {path}")
    if not stat.S_ISREG(mode):
        raise ValueError(
            f"Invalid {label} target at {path}: expected a regular file"
        )
    return True


def save_lossless_rgb(image, output_path) -> None:
    output_path = Path(output_path)
    validate_output_target(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        image.convert("RGB").save(temporary_path, format="PNG")
        os.replace(str(temporary_path), str(output_path))
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def validate_saved_image(path, expected_size, label) -> None:
    Image = require_image_class()
    try:
        with Image.open(path) as image:
            image.load()
            actual = (image.size, image.mode, image.format)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read {label} image at {path}") from error

    expected = (expected_size, "RGB", "PNG")
    if actual != expected:
        width, height = expected_size
        raise ValueError(
            f"Invalid {label} image at {path}: actual {actual}; "
            f"expected {width}x{height} RGB PNG"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate strict SRGS x4 LR and SwinIR images for all scenes."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("data/mipnerf360"),
    )
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--hr_dir", default="images_2")
    parser.add_argument("--lr_dir", default="images_lr_srgs_x4")
    parser.add_argument("--sr_dir", default="images_srgs_swinir_x4")
    parser.add_argument(
        "--swinir_root",
        type=Path,
        default=Path("third_party/SwinIR"),
    )
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=int, choices=(4, 8), default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validated_scene_names(scenes):
    scene_names = [str(scene) for scene in scenes]
    seen = set()
    for scene in scene_names:
        if scene in seen:
            raise ValueError(f"Duplicate scene requested: {scene}")
        seen.add(scene)

    for scene in scene_names:
        posix_path = PurePosixPath(scene)
        windows_path = PureWindowsPath(scene)
        if (
            not scene
            or scene in {".", ".."}
            or "/" in scene
            or "\\" in scene
            or posix_path.anchor
            or windows_path.anchor
        ):
            raise ValueError(
                f"Scene must be a single relative directory name: {scene}"
            )
    return scene_names


def relative_directory_path(value, label):
    value_text = str(value)
    path = Path(value_text)
    if (
        not value_text
        or path.anchor
        or PurePosixPath(value_text).anchor
        or PureWindowsPath(value_text).anchor
    ):
        raise ValueError(f"{label} must be a relative directory: {value}")
    return path


def path_is_within(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_dataset_layout(data_root, scenes, hr_dir, lr_dir, sr_dir):
    scene_names = validated_scene_names(scenes)
    relative_roots = {
        "HR": relative_directory_path(hr_dir, "hr_dir"),
        "LR": relative_directory_path(lr_dir, "lr_dir"),
        "SR": relative_directory_path(sr_dir, "sr_dir"),
    }
    resolved_data_root = Path(data_root).resolve()

    resolved_scenes = []
    for scene in scene_names:
        scene_root = (resolved_data_root / scene).resolve()
        if scene_root == resolved_data_root or not path_is_within(
            scene_root,
            resolved_data_root,
        ):
            raise ValueError(
                f"Scene resolves outside data root: {scene} -> {scene_root}"
            )
        resolved_scenes.append((scene, scene_root))

    for index, (first_scene, first_root) in enumerate(resolved_scenes):
        for second_scene, second_root in resolved_scenes[index + 1 :]:
            if path_is_within(first_root, second_root) or path_is_within(
                second_root,
                first_root,
            ):
                raise ValueError(
                    "Overlapping scene roots: "
                    f"{first_scene} ({first_root}) and "
                    f"{second_scene} ({second_root})"
                )

    for scene, scene_root in resolved_scenes:
        resolved_roots = {}
        for label, relative_root in relative_roots.items():
            configured_root = scene_root / relative_root
            resolved_root = configured_root.resolve()
            if resolved_root == scene_root or not path_is_within(
                resolved_root,
                scene_root,
            ):
                raise ValueError(
                    f"{label} directory resolves outside scene {scene}: "
                    f"{resolved_root}"
                )
            try:
                configured_mode = configured_root.lstat().st_mode
            except FileNotFoundError:
                configured_mode = None
            except OSError as error:
                raise ValueError(
                    f"Could not inspect {label} root at {configured_root}"
                ) from error
            if configured_mode is not None:
                is_directory = stat.S_ISDIR(configured_mode)
                if stat.S_ISLNK(configured_mode):
                    is_directory = configured_root.is_dir()
                if not is_directory:
                    raise ValueError(
                        f"{label} root must be a directory: {configured_root}"
                    )
            resolved_roots[label] = resolved_root

        labels = list(resolved_roots)
        for index, first_label in enumerate(labels):
            first_root = resolved_roots[first_label]
            for second_label in labels[index + 1 :]:
                second_root = resolved_roots[second_label]
                if path_is_within(first_root, second_root) or path_is_within(
                    second_root,
                    first_root,
                ):
                    raise ValueError(
                        f"Overlapping {first_label}/{second_label} directories "
                        f"for scene {scene}: {first_root} and {second_root}"
                    )


def preflight_scenes(data_root, scenes, hr_dir):
    data_root = Path(data_root)
    records = []
    for scene in validated_scene_names(scenes):
        scene_root = data_root / scene
        hr_root = scene_root / hr_dir
        if not hr_root.is_dir():
            raise FileNotFoundError(f"Missing HR directory: {hr_root}")

        hr_paths = list_images(hr_root)
        if not hr_paths:
            raise FileNotFoundError(f"No HR images found in {hr_root}")

        records.append(
            {
                "scene": scene,
                "scene_root": scene_root,
                "hr_paths": hr_paths,
            }
        )
    return records


def prepare_lr_file(hr_path, lr_path, overwrite=False, scale=SCALE):
    Image = require_image_class()
    hr_path = Path(hr_path)
    lr_path = Path(lr_path)
    with Image.open(hr_path) as hr_image:
        lr_size, sr_size = srgs_output_sizes(*hr_image.size, scale=scale)
        if lr_path.exists() and not overwrite:
            validate_saved_image(lr_path, lr_size, "LR")
            return False, lr_size, sr_size

        lr_image = downsample_like_srgs(hr_image, scale=scale)

    save_lossless_rgb(lr_image, lr_path)
    validate_saved_image(lr_path, lr_size, "LR")
    return True, lr_size, sr_size


def preflight_output_targets(records, lr_dir, sr_dir):
    targets = []
    for record in records:
        scene = record["scene"]
        scene_root = Path(record["scene_root"])
        for hr_path in record["hr_paths"]:
            hr_path = Path(hr_path)
            lr_path = scene_root / lr_dir / hr_path.name
            sr_path = scene_root / sr_dir / hr_path.name
            lr_exists = validate_output_target(lr_path, "LR")
            sr_exists = validate_output_target(sr_path, "SR")
            targets.append(
                {
                    "scene": scene,
                    "hr_path": hr_path,
                    "lr_path": lr_path,
                    "sr_path": sr_path,
                    "lr_exists": lr_exists,
                    "sr_exists": sr_exists,
                }
            )
    return targets


def remove_output_target(path, label):
    path = Path(path)
    if not validate_output_target(path, label):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ValueError(f"Could not remove {label} target at {path}") from error


def prepare_jobs(records, lr_dir, sr_dir, overwrite=False, scale=SCALE):
    jobs = []
    counts = {
        record["scene"]: {
            "lr_generated": 0,
            "lr_skipped": 0,
            "sr_generated": 0,
            "sr_skipped": 0,
        }
        for record in records
    }
    targets = preflight_output_targets(records, lr_dir, sr_dir)
    Image = require_image_class()
    candidates = []

    for target in targets:
        with Image.open(target["hr_path"]) as hr_image:
            lr_size, sr_size = srgs_output_sizes(
                *hr_image.size,
                scale=scale,
            )
        candidate = dict(target)
        candidate["lr_size"] = lr_size
        candidate["sr_size"] = sr_size
        if not overwrite:
            if candidate["lr_exists"]:
                validate_saved_image(candidate["lr_path"], lr_size, "LR")
            if candidate["sr_exists"]:
                validate_saved_image(candidate["sr_path"], sr_size, "SR")
        candidates.append(candidate)

    if overwrite:
        for candidate in candidates:
            remove_output_target(candidate["lr_path"], "LR")
            remove_output_target(candidate["sr_path"], "SR")

    for candidate in candidates:
        scene = candidate["scene"]
        if overwrite or not candidate["lr_exists"]:
            generated, _, _ = prepare_lr_file(
                candidate["hr_path"],
                candidate["lr_path"],
                overwrite=overwrite,
                scale=scale,
            )
            count_key = "lr_generated" if generated else "lr_skipped"
            counts[scene][count_key] += 1
        else:
            counts[scene]["lr_skipped"] += 1

        if not overwrite and candidate["sr_exists"]:
            counts[scene]["sr_skipped"] += 1
        else:
            jobs.append(
                {
                    "scene": scene,
                    "lr_path": candidate["lr_path"],
                    "sr_path": candidate["sr_path"],
                    "sr_size": candidate["sr_size"],
                }
            )

    return jobs, counts


def resolve_strict_device(device_name):
    torch = require_torch()
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device_name}), but CUDA is not available"
        )
    return torch.device(device_name)


def infer_one(model, lr_path, expected_sr_size, device, scale=SCALE):
    Image = require_image_class()
    torch = require_torch()
    lr_path = Path(lr_path)
    with Image.open(lr_path) as lr_image:
        input_tensor = image_to_tensor(lr_image, device)

    padded, height, width = pad_like_srgs(input_tensor)
    with torch.no_grad():
        output = model(padded)
    output = output[..., : height * scale, : width * scale].clamp(0, 1)

    actual_sr_size = (output.size(-1), output.size(-2))
    expected_sr_size = tuple(expected_sr_size)
    if actual_sr_size != expected_sr_size:
        raise RuntimeError(
            f"Unexpected SR output for {lr_path}: actual {actual_sr_size}; "
            f"expected {expected_sr_size}"
        )
    return tensor_to_image(output)


def print_counts(records, counts):
    total = {
        "lr_generated": 0,
        "lr_skipped": 0,
        "sr_generated": 0,
        "sr_skipped": 0,
    }
    for record in records:
        scene = record["scene"]
        scene_counts = counts[scene]
        print(
            f"{scene}: lr_generated={scene_counts['lr_generated']} "
            f"lr_skipped={scene_counts['lr_skipped']} "
            f"sr_generated={scene_counts['sr_generated']} "
            f"sr_skipped={scene_counts['sr_skipped']}"
        )
        for key in total:
            total[key] += scene_counts[key]

    print(
        f"TOTAL: lr_generated={total['lr_generated']} "
        f"lr_skipped={total['lr_skipped']} "
        f"sr_generated={total['sr_generated']} "
        f"sr_skipped={total['sr_skipped']}"
    )


def run_dataset(args):
    validate_dataset_layout(
        args.data_root,
        args.scenes,
        args.hr_dir,
        args.lr_dir,
        args.sr_dir,
    )
    records = preflight_scenes(args.data_root, args.scenes, args.hr_dir)
    jobs, counts = prepare_jobs(
        records,
        args.lr_dir,
        args.sr_dir,
        overwrite=args.overwrite,
        scale=args.scale,
    )

    if jobs:
        device = resolve_strict_device(args.device)
        if args.scale == 8 and args.model_path is None:
            raise ValueError("--model_path is required for SwinIR x8")
        model_path = args.model_path or default_model_path(args.swinir_root)
        model = build_swinir_classical(
            swinir_root=args.swinir_root,
            model_path=model_path,
            device=device,
            scale=args.scale,
        )
        for job in tqdm(
            jobs,
            desc="Strict SRGS SwinIR x{}".format(args.scale),
        ):
            output = infer_one(
                model,
                job["lr_path"],
                job["sr_size"],
                device,
                scale=args.scale,
            )
            save_lossless_rgb(output, job["sr_path"])
            validate_saved_image(job["sr_path"], job["sr_size"], "SR")
            counts[job["scene"]]["sr_generated"] += 1

    print_counts(records, counts)
    return counts


def main():
    run_dataset(parse_args())


if __name__ == "__main__":
    main()
