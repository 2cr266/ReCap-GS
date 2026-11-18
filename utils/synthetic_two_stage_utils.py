"""NeRF Synthetic adapters for the two-stage FastGS trainers."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from scene.dataset_readers import CameraInfo
from utils.camera_utils import cameraList_from_camInfos
from utils.two_stage_utils import canonical_sha256


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def is_synthetic_scene(scene_dir):
    return (Path(scene_dir) / "transforms_train.json").is_file()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_transforms(scene_dir, split):
    path = Path(scene_dir) / "transforms_{}.json".format(split)
    if not path.is_file():
        raise FileNotFoundError("Missing NeRF Synthetic transforms: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("{} must contain a non-empty frames list".format(path))
    return path, payload


def _source_image_path(scene_dir, frame, extension=".png"):
    file_path = frame.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("Synthetic frame is missing file_path")
    relative = Path(file_path)
    if relative.suffix:
        path = Path(scene_dir) / relative
    else:
        path = Path(scene_dir) / (file_path + extension)
    path = path.resolve()
    scene_root = Path(scene_dir).resolve()
    try:
        path.relative_to(scene_root)
    except ValueError as error:
        raise ValueError("Synthetic frame resolves outside scene: {}".format(path)) from error
    return path


def _frame_name(scene_dir, frame):
    return _source_image_path(scene_dir, frame).name


def _image_files(directory):
    result = {}
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix in IMAGE_SUFFIXES:
            if path.name in result:
                raise ValueError("Duplicate image filename: {}".format(path.name))
            result[path.name] = path
    return result


def validate_synthetic_scene(
    scene_dir,
    lr_image_dir,
    hr_image_dir,
    scale,
    eval_mode,
):
    scene_dir = Path(scene_dir).resolve()
    if scale not in (4, 8):
        raise ValueError(
            "Synthetic two-stage training requires scale=4 or scale=8"
        )
    train_json, train_payload = _load_transforms(scene_dir, "train")
    test_json, test_payload = _load_transforms(scene_dir, "test")
    lr_dir = (scene_dir / lr_image_dir).resolve()
    hr_dir = (scene_dir / hr_image_dir).resolve()
    for label, directory in (("LR", lr_dir), ("pseudo-HR", hr_dir)):
        try:
            directory.relative_to(scene_dir)
        except ValueError as error:
            raise ValueError("{} directory escapes the scene".format(label)) from error
        if not directory.is_dir():
            raise FileNotFoundError("Missing {} directory: {}".format(label, directory))

    train_frames = train_payload["frames"]
    train_names = [_frame_name(scene_dir, frame) for frame in train_frames]
    if len(set(train_names)) != len(train_names):
        raise ValueError("Synthetic train frame basenames must be unique")
    lr_files = _image_files(lr_dir)
    hr_files = _image_files(hr_dir)
    expected_names = set(train_names)
    for label, files in (("LR", lr_files), ("pseudo-HR", hr_files)):
        if set(files) != expected_names:
            raise ValueError(
                "{} filenames differ from transforms_train: missing={} extra={}".format(
                    label,
                    sorted(expected_names - set(files)),
                    sorted(set(files) - expected_names),
                )
            )

    views = []
    lr_sizes = set()
    hr_sizes = set()
    for frame, name in zip(train_frames, train_names):
        source_path = _source_image_path(scene_dir, frame)
        if not source_path.is_file():
            raise FileNotFoundError("Missing Synthetic source image: {}".format(source_path))
        with Image.open(str(source_path)) as image:
            source_size = tuple(image.size)
        with Image.open(str(lr_files[name])) as image:
            image.verify()
        with Image.open(str(hr_files[name])) as image:
            image.verify()
        with Image.open(str(lr_files[name])) as image:
            lr_size, lr_mode = tuple(image.size), image.mode
        with Image.open(str(hr_files[name])) as image:
            hr_size, hr_mode = tuple(image.size), image.mode
        if lr_mode != "RGB" or hr_mode != "RGB":
            raise ValueError("Synthetic LR/pseudo-HR images must be RGB: {}".format(name))
        expected_hr = (lr_size[0] * scale, lr_size[1] * scale)
        if hr_size != expected_hr:
            raise ValueError(
                "Pseudo-HR must be exactly {}x LR for {}: LR={} HR={}".format(
                    scale, name, lr_size, hr_size
                )
            )
        if hr_size != source_size:
            raise ValueError(
                "Synthetic pseudo-HR must preserve source size for {}: source={} HR={}".format(
                    name, source_size, hr_size
                )
            )
        lr_sizes.add(lr_size)
        hr_sizes.add(hr_size)
        views.append(
            {
                "name": Path(name).stem,
                "source_file": str(source_path.relative_to(scene_dir)),
                "split": "train",
                "lr_size": list(lr_size),
                "hr_size": list(hr_size),
                "lr_sha256": _sha256_file(lr_files[name]),
                "hr_sha256": _sha256_file(hr_files[name]),
            }
        )

    test_views = []
    for frame in test_payload["frames"]:
        source_path = _source_image_path(scene_dir, frame)
        if not source_path.is_file():
            raise FileNotFoundError("Missing Synthetic test image: {}".format(source_path))
        test_views.append(
            {
                "name": source_path.stem,
                "source_file": str(source_path.relative_to(scene_dir)),
                "split": "test",
                "sha256": _sha256_file(source_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset_type": "nerf_synthetic",
        "eval": bool(eval_mode),
        "scale": scale,
        "white_background": True,
        "lr_image_dir": str(lr_image_dir),
        "hr_image_dir": str(hr_image_dir),
        "transforms_train_sha256": _sha256_file(train_json),
        "transforms_test_sha256": _sha256_file(test_json),
        "views": views + test_views,
    }
    return {
        "manifest": manifest,
        "manifest_hash": canonical_sha256(manifest),
        "view_count": len(views) + len(test_views),
        "train_count": len(views),
        "test_count": len(test_views) if eval_mode else 0,
        "lr_sizes": sorted(lr_sizes),
        "hr_sizes": sorted(hr_sizes),
    }


def load_synthetic_camera_set(dataset, image_dir, reference_cameras):
    directory = (Path(dataset.source_path) / image_dir).resolve()
    files = _image_files(directory)
    reference_names = [camera.image_name for camera in reference_cameras]
    expected_files = {name + ".png" for name in reference_names}
    if set(files) != expected_files:
        raise ValueError(
            "Synthetic camera directory mismatch: missing={} extra={}".format(
                sorted(expected_files - set(files)),
                sorted(set(files) - expected_files),
            )
        )
    infos = []
    for camera in reference_cameras:
        path = files[camera.image_name + ".png"]
        with Image.open(str(path)) as image:
            rgb = image.convert("RGB").copy()
        infos.append(
            CameraInfo(
                uid=camera.colmap_id,
                R=camera.R,
                T=camera.T,
                FovY=camera.FoVy,
                FovX=camera.FoVx,
                image=rgb,
                image_path=str(path),
                image_name=camera.image_name,
                width=rgb.size[0],
                height=rgb.size[1],
            )
        )
    camera_args = type("SyntheticCameraArgs", (), {})()
    for name, value in vars(dataset).items():
        setattr(camera_args, name, value)
    camera_args.images = image_dir
    return cameraList_from_camInfos(infos, 1.0, camera_args)
