"""Scale-eight compatibility helpers without changing protected x4 utilities."""

import math
import struct
from pathlib import Path

from PIL import Image

from scripts import check_two_stage_inputs as checker
from utils import two_stage_utils as two_stage


def _read_exact(handle, size, label):
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("Unexpected EOF while reading {}".format(label))
    return value


def _read_camera_ids_x8(path):
    path = Path(path)
    camera_ids = set()
    if path.suffix.lower() == ".bin":
        parameter_counts = {0: 3, 1: 4, 2: 4}
        with path.open("rb") as handle:
            count = struct.unpack("<Q", _read_exact(handle, 8, "camera count"))[0]
            for _ in range(count):
                camera_id, model_id, width, height = struct.unpack(
                    "<iiQQ",
                    _read_exact(handle, 24, "camera record"),
                )
                if model_id not in parameter_counts:
                    raise ValueError(
                        "x8 FastGS supports SIMPLE_PINHOLE, PINHOLE, and "
                        "SIMPLE_RADIAL; camera {} has model id {}".format(
                            camera_id,
                            model_id,
                        )
                    )
                parameter_count = parameter_counts[model_id]
                parameters = struct.unpack(
                    "<{}d".format(parameter_count),
                    _read_exact(
                        handle,
                        8 * parameter_count,
                        "camera parameters",
                    ),
                )
                focal_values = parameters[:2] if model_id == 1 else parameters[:1]
                if (
                    width < 1
                    or height < 1
                    or not all(math.isfinite(value) for value in parameters)
                    or any(value <= 0 for value in focal_values)
                ):
                    raise ValueError("Invalid COLMAP camera {}".format(camera_id))
                if camera_id in camera_ids:
                    raise ValueError("Duplicate COLMAP camera id: {}".format(camera_id))
                camera_ids.add(camera_id)
        return camera_ids

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) < 7 or values[1] not in {
                "PINHOLE",
                "SIMPLE_PINHOLE",
                "SIMPLE_RADIAL",
            }:
                raise ValueError("Unsupported COLMAP camera line: {}".format(stripped))
            camera_id = int(values[0])
            width, height = int(values[2]), int(values[3])
            parameters = tuple(float(value) for value in values[4:])
            expected = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4}
            focal_values = parameters[:2] if values[1] == "PINHOLE" else parameters[:1]
            if (
                len(parameters) != expected[values[1]]
                or width < 1
                or height < 1
                or not all(math.isfinite(value) for value in parameters)
                or any(value <= 0 for value in focal_values)
            ):
                raise ValueError("Invalid COLMAP camera line: {}".format(stripped))
            if camera_id in camera_ids:
                raise ValueError("Duplicate COLMAP camera id: {}".format(camera_id))
            camera_ids.add(camera_id)
    return camera_ids


def pair_multiscale_cameras_x8(lr_cameras, hr_cameras):
    scale = 8
    lr_by_name = two_stage._camera_map(lr_cameras, "LR")
    hr_by_name = two_stage._camera_map(hr_cameras, "HR")
    if set(lr_by_name) != set(hr_by_name):
        missing_lr = sorted(set(hr_by_name) - set(lr_by_name))
        missing_hr = sorted(set(lr_by_name) - set(hr_by_name))
        raise ValueError(
            "LR/HR camera names differ: missing_lr={} missing_hr={}".format(
                missing_lr,
                missing_hr,
            )
        )
    pairs = []
    for hr_camera in hr_cameras:
        lr_camera = lr_by_name[hr_camera.image_name]
        if lr_camera.colmap_id != hr_camera.colmap_id:
            raise ValueError(
                "LR/HR COLMAP id mismatch for {}".format(hr_camera.image_name)
            )
        if lr_camera.uid != hr_camera.uid:
            raise ValueError(
                "LR/HR camera uid mismatch for {}".format(hr_camera.image_name)
            )
        if two_stage._camera_geometry(lr_camera) != two_stage._camera_geometry(
            hr_camera
        ):
            raise ValueError(
                "LR/HR Camera geometry mismatch for {}".format(
                    hr_camera.image_name
                )
            )
        expected = (
            lr_camera.image_width * scale,
            lr_camera.image_height * scale,
        )
        actual = (hr_camera.image_width, hr_camera.image_height)
        if actual != expected:
            raise ValueError(
                "Pseudo-HR must be exactly 8x LR for {}: expected={} actual={}".format(
                    hr_camera.image_name,
                    expected,
                    actual,
                )
            )
        pairs.append(two_stage.MultiScaleCameraPair(lr_camera, hr_camera))
    return pairs


def validate_colmap_scene_x8(
    scene_dir,
    lr_image_dir,
    hr_image_dir,
    scale,
    eval_mode,
):
    if scale != 8:
        raise ValueError("x8 validator requires scale=8")
    scene_dir = Path(scene_dir).resolve()
    lr_image_dir, lr_dir = checker.normalize_scene_relative_directory(
        scene_dir,
        lr_image_dir,
        "LR",
    )
    hr_image_dir, hr_dir = checker.normalize_scene_relative_directory(
        scene_dir,
        hr_image_dir,
        "HR",
    )
    sparse_dir = scene_dir / "sparse" / "0"
    for label, directory in (
        ("COLMAP sparse", sparse_dir),
        ("LR", lr_dir),
        ("HR", hr_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError("Missing {} directory: {}".format(label, directory))

    raw_records, sparse_paths = checker._select_sparse_files(sparse_dir)
    checker._validate_point_sources(sparse_paths[2:])
    camera_ids = _read_camera_ids_x8(sparse_paths[0])
    missing_camera_ids = sorted(
        {record["camera_id"] for record in raw_records} - camera_ids
    )
    if missing_camera_ids:
        raise ValueError(
            "COLMAP image camera reference is missing: {}".format(
                missing_camera_ids
            )
        )
    references = checker.split_references(raw_records, eval_mode)
    if not references:
        raise ValueError("COLMAP sparse model contains no registered images")
    lr_files = checker._image_files(lr_dir)
    hr_files = checker._image_files(hr_dir)
    referenced_names = [Path(item["name"]).name for item in references]
    referenced_set = set(referenced_names)
    lr_name_set = set(lr_files)
    hr_name_set = set(hr_files)
    if lr_name_set == referenced_set and hr_name_set == referenced_set:
        lr_by_reference = lr_files
        hr_by_reference = hr_files
        positional_mapping = False
    elif (
        not (referenced_set & lr_name_set)
        and not (referenced_set & hr_name_set)
        and lr_name_set == hr_name_set
        and len(lr_files) == len(references)
    ):
        ordered_lr = sorted(lr_files.values(), key=lambda path: path.stem)
        ordered_hr = sorted(hr_files.values(), key=lambda path: path.stem)
        lr_by_reference = {
            name: ordered_lr[index]
            for index, name in enumerate(referenced_names)
        }
        hr_by_reference = {
            name: ordered_hr[index]
            for index, name in enumerate(referenced_names)
        }
        positional_mapping = True
    else:
        checker._require_matching_file_sets(references, lr_files, hr_files)
        raise AssertionError("unreachable filename validation branch")

    views = []
    lr_sizes = set()
    hr_sizes = set()
    for reference in references:
        name = Path(reference["name"]).name
        lr_path = lr_by_reference[name]
        hr_path = hr_by_reference[name]
        with Image.open(str(lr_path)) as image:
            image.verify()
        with Image.open(str(hr_path)) as image:
            image.verify()
        with Image.open(str(lr_path)) as image:
            lr_size, lr_mode = tuple(image.size), image.mode
        with Image.open(str(hr_path)) as image:
            hr_size, hr_mode = tuple(image.size), image.mode
        if lr_mode not in {"RGB", "RGBA"} or hr_mode not in {"RGB", "RGBA"}:
            raise ValueError("x8 LR/pseudo-HR images must be RGB or RGBA")
        expected_hr = (lr_size[0] * 8, lr_size[1] * 8)
        if hr_size != expected_hr:
            raise ValueError(
                "Pseudo-HR must be exactly 8x LR for {}: LR={} HR={}".format(
                    name,
                    lr_size,
                    hr_size,
                )
            )
        lr_sizes.add(lr_size)
        hr_sizes.add(hr_size)
        views.append(
            {
                "name": lr_path.name,
                "colmap_name": name,
                "fastgs_name": lr_path.stem,
                "image_id": reference["image_id"],
                "camera_id": reference["camera_id"],
                "split": reference["split"],
                "lr_size": list(lr_size),
                "hr_size": list(hr_size),
                "lr_mode": lr_mode,
                "hr_mode": hr_mode,
                "lr_sha256": checker.sha256_file(lr_path),
                "hr_sha256": checker.sha256_file(hr_path),
            }
        )
    sparse_files = [
        {"name": path.name, "sha256": checker.sha256_file(path)}
        for path in sparse_paths
    ]
    manifest = {
        "schema_version": 1,
        "eval": bool(eval_mode),
        "scale": 8,
        "lr_image_dir": lr_image_dir,
        "hr_image_dir": hr_image_dir,
        "positional_filename_mapping": positional_mapping,
        "sparse_files": sparse_files,
        "views": views,
    }
    train_count = sum(item["split"] == "train" for item in views)
    test_count = sum(item["split"] == "test" for item in views)
    if train_count < 10:
        raise ValueError(
            "FastGS multiview scoring requires at least 10 training views; "
            "found {}".format(train_count)
        )
    return {
        "manifest": manifest,
        "manifest_hash": checker.canonical_sha256(manifest),
        "view_count": len(views),
        "train_count": train_count,
        "test_count": test_count,
        "lr_sizes": sorted(lr_sizes),
        "hr_sizes": sorted(hr_sizes),
    }
