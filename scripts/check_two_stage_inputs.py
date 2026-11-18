#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.two_stage_utils import canonical_sha256


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_BINARY_CAMERA_MODELS = {
    0: 3,  # SIMPLE_PINHOLE
    1: 4,  # PINHOLE
}
PLY_SCALAR_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}
PLY_REQUIRED_VERTEX_PROPERTIES = {
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(handle, size, label):
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("Truncated COLMAP {}".format(label))
    return value


def _read_images_binary(path):
    records = []
    with path.open("rb") as handle:
        image_count = struct.unpack(
            "<Q",
            _read_exact(handle, 8, "image count"),
        )[0]
        for _ in range(image_count):
            values = struct.unpack(
                "<idddddddi",
                _read_exact(handle, 64, "image record"),
            )
            pose_values = values[1:8]
            if (
                not all(math.isfinite(value) for value in pose_values)
                or sum(value * value for value in pose_values[:4]) == 0
            ):
                raise ValueError(
                    "Invalid COLMAP binary image pose for id {}".format(
                        values[0]
                    )
                )
            name_bytes = bytearray()
            while True:
                character = _read_exact(handle, 1, "image name")
                if character == b"\x00":
                    break
                name_bytes.extend(character)
            point_count = struct.unpack(
                "<Q",
                _read_exact(handle, 8, "point count"),
            )[0]
            _read_exact(handle, 24 * point_count, "2D points")
            records.append(
                {
                    "image_id": int(values[0]),
                    "camera_id": int(values[8]),
                    "name": name_bytes.decode("utf-8"),
                }
            )
    return records


def _read_images_text(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            pose_line = handle.readline()
            if pose_line == "":
                break
            stripped = pose_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) < 10:
                raise ValueError(
                    "Invalid COLMAP image pose line: {}".format(stripped)
                )
            if len(values) > 10:
                raise ValueError(
                    "FastGS text loader does not support whitespace in "
                    "COLMAP image names: {}".format(stripped)
                )
            try:
                image_id = int(values[0])
                pose_values = tuple(float(value) for value in values[1:8])
                camera_id = int(values[8])
            except ValueError as error:
                raise ValueError(
                    "Invalid COLMAP image pose line: {}".format(stripped)
                ) from error
            if not all(math.isfinite(value) for value in pose_values):
                raise ValueError(
                    "Invalid COLMAP image pose line: {}".format(stripped)
                )
            if sum(value * value for value in pose_values[:4]) == 0:
                raise ValueError(
                    "Invalid COLMAP image pose quaternion: {}".format(
                        stripped
                    )
                )
            points_line = handle.readline()
            if points_line == "":
                raise ValueError(
                    "Missing COLMAP points line for {}".format(values[9])
                )
            point_values = points_line.split()
            if len(point_values) % 3 != 0:
                raise ValueError(
                    "Invalid COLMAP points2D line for {}".format(values[9])
                )
            try:
                for point_index in range(0, len(point_values), 3):
                    x_value = float(point_values[point_index])
                    y_value = float(point_values[point_index + 1])
                    int(point_values[point_index + 2])
                    if not math.isfinite(x_value) or not math.isfinite(y_value):
                        raise ValueError
            except ValueError as error:
                raise ValueError(
                    "Invalid COLMAP points2D line for {}".format(values[9])
                ) from error
            records.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": values[9],
                }
            )
    return records


def _read_camera_ids_binary(path):
    camera_ids = set()
    with path.open("rb") as handle:
        camera_count = struct.unpack(
            "<Q",
            _read_exact(handle, 8, "camera count"),
        )[0]
        for _ in range(camera_count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ",
                _read_exact(handle, 24, "camera record"),
            )
            if model_id not in SUPPORTED_BINARY_CAMERA_MODELS:
                raise ValueError(
                    "FastGS supports only SIMPLE_PINHOLE/PINHOLE COLMAP "
                    "cameras; camera {} has model id {}".format(
                        camera_id,
                        model_id,
                    )
                )
            parameter_count = SUPPORTED_BINARY_CAMERA_MODELS[model_id]
            parameters = struct.unpack(
                "<{}d".format(parameter_count),
                _read_exact(
                    handle,
                    8 * parameter_count,
                    "camera parameters",
                ),
            )
            focal_values = parameters[:1] if model_id == 0 else parameters[:2]
            if (
                width < 1
                or height < 1
                or not all(math.isfinite(value) for value in parameters)
                or any(value <= 0 for value in focal_values)
            ):
                raise ValueError(
                    "Invalid COLMAP binary camera {}".format(camera_id)
                )
            if camera_id in camera_ids:
                raise ValueError(
                    "Duplicate COLMAP camera id: {}".format(camera_id)
                )
            camera_ids.add(camera_id)
    return camera_ids


def _read_camera_ids_text(path):
    camera_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) != 8:
                raise ValueError(
                    "Invalid COLMAP camera line: {}".format(stripped)
                )
            camera_id = int(values[0])
            if values[1] != "PINHOLE":
                raise ValueError(
                    "FastGS text loader requires PINHOLE cameras; "
                    "camera {} uses {}".format(camera_id, values[1])
                )
            width = int(values[2])
            height = int(values[3])
            if width < 1 or height < 1:
                raise ValueError(
                    "Invalid COLMAP camera line: {}".format(stripped)
                )
            try:
                parameters = tuple(float(value) for value in values[4:8])
            except ValueError as error:
                raise ValueError(
                    "Invalid COLMAP camera line: {}".format(stripped)
                ) from error
            if (
                not all(math.isfinite(value) for value in parameters)
                or parameters[0] <= 0
                or parameters[1] <= 0
            ):
                raise ValueError(
                    "Invalid COLMAP camera line: {}".format(stripped)
                )
            if camera_id in camera_ids:
                raise ValueError(
                    "Duplicate COLMAP camera id: {}".format(camera_id)
                )
            camera_ids.add(camera_id)
    return camera_ids


def _read_camera_ids(path):
    if path.suffix.lower() == ".bin":
        return _read_camera_ids_binary(path)
    return _read_camera_ids_text(path)


def _validate_points3d_text(path):
    point_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.split()
            if len(values) < 8 or (len(values) - 8) % 2 != 0:
                raise ValueError(
                    "Invalid COLMAP points3D line: {}".format(stripped)
                )
            try:
                int(values[0])
                coordinates = tuple(float(value) for value in values[1:4])
                colors = tuple(int(value) for value in values[4:7])
                error_value = float(values[7])
                for index in range(8, len(values), 2):
                    int(values[index])
                    int(values[index + 1])
            except ValueError as error:
                raise ValueError(
                    "Invalid COLMAP points3D line: {}".format(stripped)
                ) from error
            if (
                not all(math.isfinite(value) for value in coordinates)
                or not math.isfinite(error_value)
                or any(value < 0 or value > 255 for value in colors)
            ):
                raise ValueError(
                    "Invalid COLMAP points3D line: {}".format(stripped)
                )
            point_count += 1
    if point_count == 0:
        raise ValueError("COLMAP points3D.txt contains no points")


def _validate_points3d_binary(path):
    with path.open("rb") as handle:
        point_count = struct.unpack(
            "<Q",
            _read_exact(handle, 8, "points3D count"),
        )[0]
        if point_count == 0:
            raise ValueError("COLMAP points3D.bin contains no points")
        for _ in range(point_count):
            values = struct.unpack(
                "<QdddBBBd",
                _read_exact(handle, 43, "points3D record"),
            )
            if not all(
                math.isfinite(value)
                for value in (values[1], values[2], values[3], values[7])
            ):
                raise ValueError("Invalid COLMAP points3D.bin value")
            track_count = struct.unpack(
                "<Q",
                _read_exact(handle, 8, "points3D track count"),
            )[0]
            _read_exact(handle, 8 * track_count, "points3D track")
        if handle.read(1):
            raise ValueError("COLMAP points3D.bin contains trailing data")


def _validate_points3d_ply(path):
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("Invalid COLMAP points3D.ply header")

        ply_format = None
        current_element = None
        data_elements = []
        vertex_count = None
        vertex_properties = []
        header_complete = False
        for _ in range(10_000):
            raw_line = handle.readline()
            if raw_line == b"":
                break
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Invalid COLMAP points3D.ply header encoding"
                ) from error
            if not line or line.startswith("comment ") or line.startswith("obj_info "):
                continue
            values = line.split()
            if values[0] == "format" and len(values) == 3:
                ply_format = values[1]
            elif values[0] == "element" and len(values) == 3:
                try:
                    element_count = int(values[2])
                except ValueError as error:
                    raise ValueError(
                        "Invalid COLMAP points3D.ply element"
                    ) from error
                if element_count < 0:
                    raise ValueError("Invalid COLMAP points3D.ply element")
                current_element = values[1]
                data_elements.append((current_element, element_count))
                if current_element == "vertex":
                    vertex_count = element_count
            elif values[0] == "property":
                if current_element == "vertex":
                    if len(values) != 3 or values[1] not in PLY_SCALAR_TYPES:
                        raise ValueError(
                            "Invalid COLMAP points3D.ply vertex property"
                        )
                    vertex_properties.append((values[1], values[2]))
            elif values[0] == "end_header" and len(values) == 1:
                header_complete = True
                break

        if not header_complete:
            raise ValueError("Invalid COLMAP points3D.ply header")
        if ply_format not in {
            "ascii",
            "binary_little_endian",
            "binary_big_endian",
        }:
            raise ValueError("Unsupported COLMAP points3D.ply format")
        if vertex_count is None or vertex_count < 1:
            raise ValueError("COLMAP points3D.ply contains no vertices")
        if not data_elements or data_elements[0][0] != "vertex":
            raise ValueError(
                "COLMAP points3D.ply must store vertices first"
            )
        property_names = {name for _, name in vertex_properties}
        missing = sorted(PLY_REQUIRED_VERTEX_PROPERTIES - property_names)
        if missing:
            raise ValueError(
                "COLMAP points3D.ply is missing vertex properties: {}".format(
                    missing
                )
            )

        if ply_format == "ascii":
            for _ in range(vertex_count):
                raw_line = handle.readline()
                if raw_line == b"":
                    raise ValueError("Truncated COLMAP points3D.ply vertices")
                values = raw_line.split()
                if len(values) != len(vertex_properties):
                    raise ValueError("Invalid COLMAP points3D.ply vertex")
                try:
                    numeric_values = [float(value) for value in values]
                except ValueError as error:
                    raise ValueError(
                        "Invalid COLMAP points3D.ply vertex"
                    ) from error
                if not all(math.isfinite(value) for value in numeric_values):
                    raise ValueError("Invalid COLMAP points3D.ply vertex")
        else:
            endian = "<" if ply_format == "binary_little_endian" else ">"
            record_format = endian + "".join(
                PLY_SCALAR_TYPES[data_type]
                for data_type, _ in vertex_properties
            )
            record_size = struct.calcsize(record_format)
            for _ in range(vertex_count):
                values = struct.unpack(
                    record_format,
                    _read_exact(handle, record_size, "points3D.ply vertex"),
                )
                if not all(
                    math.isfinite(float(value)) for value in values
                ):
                    raise ValueError("Invalid COLMAP points3D.ply vertex")


def _validate_point_sources(paths):
    for path in paths:
        if path.name == "points3D.txt":
            _validate_points3d_text(path)
        elif path.name == "points3D.bin":
            _validate_points3d_binary(path)
        elif path.name == "points3D.ply":
            _validate_points3d_ply(path)


def _select_sparse_files(sparse_dir):
    binary_images = sparse_dir / "images.bin"
    binary_cameras = sparse_dir / "cameras.bin"
    text_images = sparse_dir / "images.txt"
    text_cameras = sparse_dir / "cameras.txt"

    if binary_images.is_file() and binary_cameras.is_file():
        images_path = binary_images
        cameras_path = binary_cameras
        records = _read_images_binary(images_path)
    elif text_images.is_file() and text_cameras.is_file():
        images_path = text_images
        cameras_path = text_cameras
        records = _read_images_text(images_path)
    else:
        raise FileNotFoundError(
            "COLMAP sparse model needs matching images/cameras .bin or .txt files"
        )

    point_candidates = (
        sparse_dir / "points3D.bin",
        sparse_dir / "points3D.txt",
        sparse_dir / "points3D.ply",
    )
    points_paths = tuple(
        path for path in point_candidates if path.is_file()
    )
    if not points_paths:
        raise FileNotFoundError(
            "COLMAP sparse model is missing points3D data"
        )
    return records, (cameras_path, images_path) + points_paths


def read_colmap_image_references(sparse_dir):
    records, _ = _select_sparse_files(Path(sparse_dir))
    return records


def _fastgs_image_name(name):
    return Path(name).name.split(".")[0]


def split_references(records, eval_mode):
    ordered = sorted(
        records,
        key=lambda item: _fastgs_image_name(item["name"]),
    )
    fastgs_names = set()
    image_ids = set()
    result = []
    for index, record in enumerate(ordered):
        if record["image_id"] in image_ids:
            raise ValueError(
                "Duplicate COLMAP image id: {}".format(
                    record["image_id"]
                )
            )
        image_ids.add(record["image_id"])
        fastgs_name = _fastgs_image_name(record["name"])
        if fastgs_name in fastgs_names:
            raise ValueError(
                "Duplicate FastGS image name: {}".format(fastgs_name)
            )
        fastgs_names.add(fastgs_name)
        copied = dict(record)
        copied["fastgs_name"] = fastgs_name
        copied["split"] = (
            "test" if eval_mode and index % 8 == 0 else "train"
        )
        result.append(copied)
    return result


def _image_files(directory):
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def normalize_scene_relative_directory(scene_dir, value, label):
    scene_dir = Path(scene_dir).resolve()
    raw_path = Path(value)
    if raw_path.is_absolute():
        raise ValueError(
            "{} image directory must be scene-relative".format(label)
        )
    resolved = (scene_dir / raw_path).resolve()
    try:
        relative = resolved.relative_to(scene_dir)
    except ValueError as error:
        raise ValueError(
            "{} image directory must be scene-relative".format(label)
        ) from error
    if relative == Path("."):
        raise ValueError(
            "{} image directory must not be the scene root".format(label)
        )
    return relative.as_posix(), resolved


def _require_matching_file_sets(references, lr_files, hr_files):
    referenced_names = {Path(item["name"]).name for item in references}
    if len(referenced_names) != len(references):
        raise ValueError(
            "Duplicate COLMAP image basenames are not supported"
        )
    for label, files in (("LR", lr_files), ("HR", hr_files)):
        names = set(files)
        if names != referenced_names:
            raise ValueError(
                "{} filenames differ from COLMAP: missing={} extra={}".format(
                    label,
                    sorted(referenced_names - names),
                    sorted(names - referenced_names),
                )
            )


def validate_scene(
    scene_dir,
    lr_image_dir,
    hr_image_dir,
    scale,
    eval_mode,
):
    from PIL import Image

    scene_dir = Path(scene_dir).resolve()
    lr_image_dir, lr_dir = normalize_scene_relative_directory(
        scene_dir,
        lr_image_dir,
        "LR",
    )
    hr_image_dir, hr_dir = normalize_scene_relative_directory(
        scene_dir,
        hr_image_dir,
        "HR",
    )
    sparse_dir = scene_dir / "sparse" / "0"
    if scale != 4:
        raise ValueError("Experiment 2 requires scale=4")
    for label, directory in (
        ("COLMAP sparse", sparse_dir),
        ("LR", lr_dir),
        ("HR", hr_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(
                "Missing {} directory: {}".format(label, directory)
            )

    raw_records, sparse_paths = _select_sparse_files(sparse_dir)
    _validate_point_sources(sparse_paths[2:])
    camera_ids = _read_camera_ids(sparse_paths[0])
    missing_camera_ids = sorted(
        {record["camera_id"] for record in raw_records} - camera_ids
    )
    if missing_camera_ids:
        raise ValueError(
            "COLMAP image camera reference is missing from cameras file: {}".format(
                missing_camera_ids
            )
        )
    references = split_references(raw_records, eval_mode)
    if not references:
        raise ValueError(
            "COLMAP sparse model contains no registered images"
        )
    lr_files = _image_files(lr_dir)
    hr_files = _image_files(hr_dir)
    _require_matching_file_sets(references, lr_files, hr_files)

    views = []
    lr_sizes = set()
    hr_sizes = set()
    for reference in references:
        name = Path(reference["name"]).name
        lr_path = lr_files[name]
        hr_path = hr_files[name]
        with Image.open(str(lr_path)) as image:
            image.verify()
        with Image.open(str(hr_path)) as image:
            image.verify()
        with Image.open(str(lr_path)) as image:
            lr_size = tuple(image.size)
            lr_mode = image.mode
        with Image.open(str(hr_path)) as image:
            hr_size = tuple(image.size)
            hr_mode = image.mode
        if lr_mode not in {"RGB", "RGBA"}:
            raise ValueError(
                "LR image must be RGB or RGBA for {}: mode={}".format(
                    name,
                    lr_mode,
                )
            )
        if hr_mode not in {"RGB", "RGBA"}:
            raise ValueError(
                "Pseudo-HR image must be RGB or RGBA for {}: mode={}".format(
                    name,
                    hr_mode,
                )
            )
        expected_hr = (lr_size[0] * scale, lr_size[1] * scale)
        if hr_size != expected_hr:
            raise ValueError(
                "Pseudo-HR must be exactly {}x LR for {}: "
                "LR={} HR={} expected={}".format(
                    scale,
                    name,
                    lr_size,
                    hr_size,
                    expected_hr,
                )
            )
        lr_sizes.add(lr_size)
        hr_sizes.add(hr_size)
        views.append(
            {
                "name": name,
                "fastgs_name": reference["fastgs_name"],
                "image_id": reference["image_id"],
                "camera_id": reference["camera_id"],
                "split": reference["split"],
                "lr_size": list(lr_size),
                "hr_size": list(hr_size),
                "lr_mode": lr_mode,
                "hr_mode": hr_mode,
                "lr_sha256": sha256_file(lr_path),
                "hr_sha256": sha256_file(hr_path),
            }
        )

    sparse_files = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
        }
        for path in sparse_paths
    ]
    manifest = {
        "schema_version": 1,
        "eval": bool(eval_mode),
        "scale": scale,
        "lr_image_dir": lr_image_dir,
        "hr_image_dir": hr_image_dir,
        "sparse_files": sparse_files,
        "views": views,
    }
    train_count = sum(
        item["split"] == "train" for item in views
    )
    test_count = sum(item["split"] == "test" for item in views)
    if train_count < 10:
        raise ValueError(
            "FastGS multiview scoring requires at least 10 training views; "
            "found {}".format(train_count)
        )
    return {
        "manifest": manifest,
        "manifest_hash": canonical_sha256(manifest),
        "view_count": len(views),
        "train_count": train_count,
        "test_count": test_count,
        "lr_sizes": sorted(lr_sizes),
        "hr_sizes": sorted(hr_sizes),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate two-stage FastGS COLMAP, LR, and pseudo-HR inputs."
        )
    )
    parser.add_argument("--scene_dir", required=True, type=Path)
    parser.add_argument("--lr_image_dir", default="images_8")
    parser.add_argument("--hr_image_dir", default="images_sr_x4")
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--output_json", default=None, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = validate_scene(
        args.scene_dir,
        args.lr_image_dir,
        args.hr_image_dir,
        args.scale,
        args.eval,
    )
    print("views={}".format(result["view_count"]))
    print(
        "train={} test={}".format(
            result["train_count"],
            result["test_count"],
        )
    )
    print("LR sizes={}".format(result["lr_sizes"]))
    print("HR sizes={}".format(result["hr_sizes"]))
    print("manifest_sha256={}".format(result["manifest_hash"]))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                result,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
