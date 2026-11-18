from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_srgs_swinir_x4.py"


def load_srgs_module():
    spec = importlib.util.spec_from_file_location("run_srgs_swinir_x4", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SrgsProtocolTests(unittest.TestCase):
    def srgs_module(self):
        try:
            return load_srgs_module()
        except FileNotFoundError:
            self.fail(f"Production script is missing: {SCRIPT_PATH}")

    def test_output_sizes_use_python_round_and_exact_x4_super_resolution(self):
        module = self.srgs_module()

        self.assertEqual(
            module.srgs_output_sizes(2473, 1643),
            ((618, 411), (2472, 1644)),
        )
        self.assertEqual(module.srgs_output_sizes(10, 10), ((2, 2), (8, 8)))

    def test_padded_length_always_adds_a_full_window(self):
        module = self.srgs_module()

        self.assertEqual(module.srgs_padded_length(16, 8), 24)
        self.assertEqual(module.srgs_padded_length(17, 8), 24)
        self.assertEqual(module.srgs_padded_length(3, 8), 8)
        with self.assertRaises(ValueError):
            module.srgs_padded_length(0, 8)

    def test_downsample_like_srgs_matches_pillow_rgb_bicubic_bytes(self):
        module = self.srgs_module()
        payload = bytearray()
        for y in range(13):
            for x in range(17):
                payload.extend(
                    (
                        (x * 37 + y * 11) % 256,
                        (x * 13 + y * 41) % 256,
                        (x * 29 + y * 17) % 256,
                        (x * 19 + y * 23) % 256,
                    )
                )
        source = Image.frombytes("RGBA", (17, 13), bytes(payload))
        expected = source.convert("RGB").resize(
            (4, 3),
            resample=Image.Resampling.BICUBIC,
        )

        result = module.downsample_like_srgs(source)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (4, 3))
        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_save_lossless_rgb_writes_png_payload_to_jpg_name(self):
        module = self.srgs_module()
        image = Image.new("RGB", (4, 3), (31, 63, 127))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "frame.JPG"
            module.save_lossless_rgb(image, output_path)

            with Image.open(output_path) as saved:
                self.assertEqual(saved.format, "PNG")
                self.assertEqual(saved.mode, "RGB")
                self.assertEqual(saved.size, (4, 3))

    def test_save_lossless_rgb_rejects_symlink_without_touching_target(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.bin"
            sentinel_payload = b"external-sentinel-payload"
            sentinel.write_bytes(sentinel_payload)
            output_path = root / "frame.JPG"
            try:
                output_path.symlink_to(sentinel)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"File symlinks are unavailable: {error}")

            with self.assertRaises(ValueError) as raised:
                module.save_lossless_rgb(
                    Image.new("RGB", (4, 3), (31, 63, 127)),
                    output_path,
                )

            self.assertIn(str(output_path), str(raised.exception))
            self.assertEqual(sentinel.read_bytes(), sentinel_payload)
            self.assertTrue(output_path.is_symlink())

    def test_save_lossless_rgb_atomically_replaces_hardlink_entry(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.bin"
            sentinel_payload = b"shared-inode-sentinel"
            sentinel.write_bytes(sentinel_payload)
            output_path = root / "frame.JPG"
            try:
                os.link(str(sentinel), str(output_path))
            except OSError as error:
                self.skipTest(f"Hardlinks are unavailable: {error}")

            module.save_lossless_rgb(
                Image.new("RGB", (4, 3), (31, 63, 127)),
                output_path,
            )

            self.assertEqual(sentinel.read_bytes(), sentinel_payload)
            with Image.open(output_path) as saved:
                self.assertEqual((saved.format, saved.mode, saved.size), ("PNG", "RGB", (4, 3)))

    def test_validate_saved_image_rejects_wrong_size(self):
        module = self.srgs_module()
        image = Image.new("RGB", (4, 3), (31, 63, 127))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "frame.JPG"
            module.save_lossless_rgb(image, output_path)

            with self.assertRaisesRegex(ValueError, "expected 8x6"):
                module.validate_saved_image(output_path, (8, 6), "SR")

    def test_validate_saved_image_accepts_expected_rgb_png(self):
        module = self.srgs_module()
        image = Image.new("RGB", (4, 3), (31, 63, 127))

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "frame.JPG"
            module.save_lossless_rgb(image, output_path)

            module.validate_saved_image(output_path, (4, 3), "SR")

    def test_validate_saved_image_rejects_non_png_or_non_rgb_payload(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            invalid_images = (
                ("jpeg_payload.png", Image.new("RGB", (4, 3)), "JPEG"),
                ("grayscale.png", Image.new("L", (4, 3)), "PNG"),
            )
            for filename, image, image_format in invalid_images:
                with self.subTest(filename=filename):
                    output_path = directory_path / filename
                    image.save(output_path, format=image_format)

                    with self.assertRaisesRegex(
                        ValueError,
                        "expected 4x3 RGB PNG",
                    ):
                        module.validate_saved_image(output_path, (4, 3), "SR")

    def test_srgs_padding_matches_official_extra_window_rule(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        module = self.srgs_module()
        tensor = torch.arange(3 * 16 * 17).reshape(1, 3, 16, 17)

        padded, original_height, original_width = module.pad_like_srgs(tensor)
        vertically_extended = torch.cat(
            [tensor, torch.flip(tensor, [2])],
            2,
        )[:, :, :24, :]

        self.assertEqual((original_height, original_width), (16, 17))
        self.assertEqual(tuple(padded.shape[-2:]), (24, 24))
        self.assertTrue(torch.equal(padded[:, :, :16, :17], tensor))
        self.assertTrue(
            torch.equal(
                padded[:, :, 16:24, :17],
                torch.flip(tensor, [2])[:, :, :8, :],
            )
        )
        self.assertTrue(
            torch.equal(
                padded[:, :, :, 17:24],
                torch.flip(vertically_extended, [3])[:, :, :, :7],
            )
        )
        self.assertTrue(
            torch.equal(padded[:, :, 16, 17], tensor[:, :, -1, -1])
        )

        small = torch.arange(3 * 3 * 5).reshape(1, 3, 3, 5)
        small_padded, small_height, small_width = module.pad_like_srgs(small)
        expected_small = small[:, :, [0, 1, 2, 2, 1, 0, 0, 1], :]
        expected_small = expected_small[:, :, :, [0, 1, 2, 3, 4, 4, 3, 2]]
        self.assertEqual((small_height, small_width), (3, 5))
        self.assertEqual(tuple(small_padded.shape[-2:]), (8, 8))
        self.assertTrue(torch.equal(small_padded, expected_small))


class SrgsDatasetRunnerTests(unittest.TestCase):
    def srgs_module(self):
        try:
            return load_srgs_module()
        except FileNotFoundError:
            self.fail(f"Production script is missing: {SCRIPT_PATH}")

    def test_parse_args_has_strict_nine_scene_defaults_without_tile_or_tta(self):
        module = self.srgs_module()

        args = module.parse_args([])

        self.assertEqual(args.data_root, Path("data/mipnerf360"))
        self.assertEqual(
            args.scenes,
            [
                "bicycle",
                "flowers",
                "garden",
                "stump",
                "treehill",
                "room",
                "counter",
                "kitchen",
                "bonsai",
            ],
        )
        self.assertEqual(args.hr_dir, "images_2")
        self.assertEqual(args.lr_dir, "images_lr_srgs_x4")
        self.assertEqual(args.sr_dir, "images_srgs_swinir_x4")
        self.assertEqual(args.swinir_root, Path("third_party/SwinIR"))
        self.assertEqual(args.device, "cuda")
        self.assertIsNone(args.model_path)
        self.assertFalse(args.overwrite)
        self.assertFalse(hasattr(args, "tile"))
        self.assertFalse(hasattr(args, "tta"))

    def test_import_does_not_attempt_to_import_torch(self):
        code = textwrap.dedent(
            f"""
            import builtins
            import importlib.util
            import pathlib

            script_path = pathlib.Path({str(SCRIPT_PATH)!r})
            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "torch" or name.startswith("torch."):
                    raise AssertionError(f"unexpected torch import: {{name}}")
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            spec = importlib.util.spec_from_file_location(
                "isolated_run_srgs_swinir_x4",
                script_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("could not load SRGS module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_main_help_does_not_load_torch_or_model(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--data_root", result.stdout)
        self.assertIn("--scenes", result.stdout)
        self.assertNotIn("--tile", result.stdout)
        self.assertNotIn("--tta", result.stdout)

    def test_run_dataset_rejects_unsafe_layout_before_preflight_or_writes(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            scene_root = data_root / "room"
            source_path = scene_root / "images_2" / "frame.png"
            source_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13), (31, 63, 127)).save(source_path)
            source_payload = source_path.read_bytes()
            outside = root / "outside"
            base = {
                "data_root": data_root,
                "scenes": ["room"],
                "hr_dir": "images_2",
                "lr_dir": "images_lr_srgs_x4",
                "sr_dir": "images_srgs_swinir_x4",
                "swinir_root": Path("/unused/SwinIR"),
                "model_path": None,
                "device": "cuda",
                "overwrite": True,
            }
            cases = (
                ("lr_is_hr", {"lr_dir": "images_2"}, None),
                (
                    "absolute_lr",
                    {"lr_dir": outside / "absolute_lr"},
                    outside / "absolute_lr" / source_path.name,
                ),
                (
                    "lr_is_sr",
                    {"lr_dir": "shared", "sr_dir": "shared"},
                    scene_root / "shared" / source_path.name,
                ),
                (
                    "lr_escapes_scene",
                    {"lr_dir": "../escape"},
                    data_root / "escape" / source_path.name,
                ),
                (
                    "lr_nested_under_hr",
                    {"lr_dir": "images_2/lr"},
                    scene_root / "images_2" / "lr" / source_path.name,
                ),
                (
                    "sr_nested_under_lr",
                    {"lr_dir": "outputs", "sr_dir": "outputs/sr"},
                    scene_root / "outputs" / source_path.name,
                ),
                ("absolute_hr", {"hr_dir": outside / "absolute_hr"}, None),
                ("absolute_sr", {"sr_dir": outside / "absolute_sr"}, None),
                ("hr_escapes_scene", {"hr_dir": "../images_2"}, None),
                ("sr_escapes_scene", {"sr_dir": "../sr"}, None),
                ("scene_escapes", {"scenes": ["../room"]}, None),
                ("absolute_scene", {"scenes": [str(outside / "room")]}, None),
            )

            for name, overrides, unexpected_path in cases:
                with self.subTest(name=name):
                    values = dict(base)
                    values.update(overrides)
                    args = SimpleNamespace(**values)
                    with ExitStack() as stack:
                        preflight = stack.enter_context(
                            mock.patch.object(
                                module,
                                "preflight_scenes",
                                return_value=[],
                            )
                        )
                        prepare = stack.enter_context(
                            mock.patch.object(
                                module,
                                "prepare_jobs",
                                return_value=([], {}),
                            )
                        )
                        summary = stack.enter_context(
                            mock.patch.object(module, "print_counts")
                        )
                        with self.assertRaises(ValueError):
                            module.run_dataset(args)

                    preflight.assert_not_called()
                    prepare.assert_not_called()
                    summary.assert_not_called()
                    self.assertEqual(source_path.read_bytes(), source_payload)
                    if unexpected_path is not None:
                        self.assertFalse(unexpected_path.exists())

    def test_validate_dataset_layout_rejects_existing_symlink_escape(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            scene_root = data_root / "room"
            (scene_root / "images_2").mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            lr_link = scene_root / "images_lr_srgs_x4"
            try:
                lr_link.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                if sys.platform != "win32":
                    self.skipTest(f"Directory symlinks are unavailable: {error}")
                junction = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(lr_link), str(outside)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if junction.returncode != 0:
                    self.skipTest(
                        "Directory links are unavailable: "
                        f"{error}; {junction.stderr.strip()}"
                    )

            with self.assertRaises(ValueError):
                module.validate_dataset_layout(
                    data_root,
                    ["room"],
                    "images_2",
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                )

    def test_run_dataset_rejects_scene_aliases_before_preflight_or_writes(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            scene_a = data_root / "scene_a"
            scene_a.mkdir(parents=True)
            scene_b = data_root / "scene_b"
            try:
                scene_b.symlink_to(scene_a, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                if sys.platform != "win32":
                    self.skipTest(f"Directory symlinks are unavailable: {error}")
                junction = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(scene_b), str(scene_a)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if junction.returncode != 0:
                    self.skipTest(
                        "Directory links are unavailable: "
                        f"{error}; {junction.stderr.strip()}"
                    )

            args = SimpleNamespace(
                data_root=data_root,
                scenes=["scene_a", "scene_b"],
                hr_dir="images_2",
                lr_dir="images_lr_srgs_x4",
                sr_dir="images_srgs_swinir_x4",
                swinir_root=Path("/unused/SwinIR"),
                model_path=None,
                device="cuda",
                overwrite=True,
                scale=4,
            )
            with ExitStack() as stack:
                preflight = stack.enter_context(
                    mock.patch.object(
                        module,
                        "preflight_scenes",
                        return_value=[],
                    )
                )
                prepare = stack.enter_context(
                    mock.patch.object(
                        module,
                        "prepare_jobs",
                        return_value=([], {}),
                    )
                )
                summary = stack.enter_context(
                    mock.patch.object(module, "print_counts")
                )
                with self.assertRaises(ValueError) as raised:
                    module.run_dataset(args)

            resolved_scene_root = scene_a.resolve()
            message = str(raised.exception)
            self.assertIn(f"scene_a ({resolved_scene_root})", message)
            self.assertIn(f"scene_b ({resolved_scene_root})", message)
            preflight.assert_not_called()
            prepare.assert_not_called()
            summary.assert_not_called()

    def test_validate_dataset_layout_rejects_nested_resolved_scene_roots(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            scene_a = data_root / "scene_a"
            child = scene_a / "nested"
            child.mkdir(parents=True)
            scene_b = data_root / "scene_b"
            try:
                scene_b.symlink_to(child, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                if sys.platform != "win32":
                    self.skipTest(f"Directory symlinks are unavailable: {error}")
                junction = subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(scene_b), str(child)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if junction.returncode != 0:
                    self.skipTest(
                        "Directory links are unavailable: "
                        f"{error}; {junction.stderr.strip()}"
                    )

            with self.assertRaises(ValueError) as raised:
                module.validate_dataset_layout(
                    data_root,
                    ["scene_a", "scene_b"],
                    "images_2",
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                )

            message = str(raised.exception)
            self.assertIn(f"scene_a ({scene_a.resolve()})", message)
            self.assertIn(f"scene_b ({child.resolve()})", message)

    def test_duplicate_scenes_are_rejected_before_listing_or_preparing_jobs(self):
        module = self.srgs_module()

        with mock.patch.object(module, "list_images") as list_images:
            with self.assertRaisesRegex(ValueError, "Duplicate scene.*room"):
                module.preflight_scenes(
                    Path("/unused/data"),
                    ["room", "room"],
                    "images_2",
                )
        list_images.assert_not_called()

        args = SimpleNamespace(
            data_root=Path("/unused/data"),
            scenes=["room", "room"],
            hr_dir="images_2",
            lr_dir="images_lr_srgs_x4",
            sr_dir="images_srgs_swinir_x4",
            swinir_root=Path("/unused/SwinIR"),
            model_path=None,
            device="cuda",
            overwrite=False,
            scale=4,
        )
        with ExitStack() as stack:
            preflight = stack.enter_context(
                mock.patch.object(
                    module,
                    "preflight_scenes",
                    return_value=[],
                )
            )
            prepare = stack.enter_context(
                mock.patch.object(
                    module,
                    "prepare_jobs",
                    return_value=([], {}),
                )
            )
            with self.assertRaisesRegex(ValueError, "Duplicate scene.*room"):
                module.run_dataset(args)
        preflight.assert_not_called()
        prepare.assert_not_called()

    def test_preflight_scenes_preserves_scene_order_and_sorts_images(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            scenes = ["room", "garden"]
            for scene in scenes:
                hr_root = data_root / scene / "images_2"
                hr_root.mkdir(parents=True)
                Image.new("RGB", (17, 13)).save(hr_root / "b.JPG")
                Image.new("RGB", (17, 13)).save(hr_root / "a.JPG")

            with mock.patch.object(
                module,
                "require_torch",
                side_effect=AssertionError("preflight must not load torch"),
            ):
                records = module.preflight_scenes(data_root, scenes, "images_2")

        self.assertEqual([record["scene"] for record in records], scenes)
        self.assertEqual(
            [record["scene_root"] for record in records],
            [data_root / scene for scene in scenes],
        )
        self.assertEqual(
            [[path.name for path in record["hr_paths"]] for record in records],
            [["a.JPG", "b.JPG"], ["a.JPG", "b.JPG"]],
        )

    def test_preflight_scenes_rejects_missing_hr_directory(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "Missing HR directory"):
                module.preflight_scenes(data_root, ["room"], "images_2")

    def test_preflight_scenes_rejects_empty_hr_directory(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            (data_root / "room" / "images_2").mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                module.preflight_scenes(data_root, ["room"], "images_2")

    def test_run_dataset_preflights_every_hr_scene_before_creating_outputs(self):
        module = self.srgs_module()

        for second_scene_state in ("missing", "empty"):
            with self.subTest(second_scene_state=second_scene_state):
                with tempfile.TemporaryDirectory() as directory:
                    data_root = Path(directory)
                    first_hr_root = data_root / "scene1" / "images_2"
                    first_hr_root.mkdir(parents=True)
                    Image.new("RGB", (17, 13)).save(first_hr_root / "frame.JPG")
                    if second_scene_state == "empty":
                        (data_root / "scene2" / "images_2").mkdir(parents=True)
                    args = SimpleNamespace(
                        data_root=data_root,
                        scenes=["scene1", "scene2"],
                        hr_dir="images_2",
                        lr_dir="images_lr_srgs_x4",
                        sr_dir="images_srgs_swinir_x4",
                        swinir_root=Path("/unused/SwinIR"),
                        model_path=None,
                        device="cuda",
                        overwrite=False,
                        scale=4,
                    )

                    with self.assertRaises(FileNotFoundError):
                        module.run_dataset(args)

                    self.assertFalse(
                        (data_root / "scene1" / args.lr_dir).exists()
                    )
                    self.assertFalse(
                        (data_root / "scene1" / args.sr_dir).exists()
                    )

    def test_run_dataset_rejects_later_output_root_file_before_any_write(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            for scene in ("scene1", "scene2"):
                hr_root = data_root / scene / "images_2"
                hr_root.mkdir(parents=True)
                Image.new("RGB", (17, 13)).save(hr_root / "frame.JPG")
            invalid_lr_root = data_root / "scene2" / "images_lr_srgs_x4"
            invalid_payload = b"not-a-directory"
            invalid_lr_root.write_bytes(invalid_payload)
            args = SimpleNamespace(
                data_root=data_root,
                scenes=["scene1", "scene2"],
                hr_dir="images_2",
                lr_dir="images_lr_srgs_x4",
                sr_dir="images_srgs_swinir_x4",
                swinir_root=Path("/unused/SwinIR"),
                model_path=None,
                device="cuda",
                overwrite=False,
                scale=4,
            )

            with self.assertRaises(ValueError) as raised:
                module.run_dataset(args)

            self.assertIn(str(invalid_lr_root), str(raised.exception))
            self.assertEqual(invalid_lr_root.read_bytes(), invalid_payload)
            self.assertFalse((data_root / "scene1" / args.lr_dir).exists())
            self.assertFalse((data_root / "scene1" / args.sr_dir).exists())

    def test_prepare_jobs_preflights_all_output_target_types_before_writing(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            records = []
            for scene in ("scene1", "scene2"):
                scene_root = data_root / scene
                hr_path = scene_root / "images_2" / "frame.JPG"
                hr_path.parent.mkdir(parents=True)
                Image.new("RGB", (17, 13)).save(hr_path)
                records.append(
                    {
                        "scene": scene,
                        "scene_root": scene_root,
                        "hr_paths": [hr_path],
                    }
                )
            invalid_target = (
                data_root
                / "scene2"
                / "images_lr_srgs_x4"
                / "frame.JPG"
            )
            invalid_target.mkdir(parents=True)

            with self.assertRaises(ValueError) as direct_raised:
                module.preflight_output_targets(
                    records,
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                )
            self.assertIn(str(invalid_target), str(direct_raised.exception))

            with self.assertRaises(ValueError) as raised:
                module.prepare_jobs(
                    records,
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                )

            self.assertIn(str(invalid_target), str(raised.exception))
            self.assertFalse(
                (
                    data_root
                    / "scene1"
                    / "images_lr_srgs_x4"
                    / "frame.JPG"
                ).exists()
            )

    def test_prepare_jobs_rejects_output_symlink_without_touching_sentinel(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = root / "room"
            hr_path = scene_root / "images_2" / "frame.JPG"
            hr_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13)).save(hr_path)
            sentinel = root / "sentinel.bin"
            sentinel_payload = b"runner-symlink-sentinel"
            sentinel.write_bytes(sentinel_payload)
            lr_path = scene_root / "images_lr_srgs_x4" / hr_path.name
            lr_path.parent.mkdir(parents=True)
            try:
                lr_path.symlink_to(sentinel)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"File symlinks are unavailable: {error}")
            records = [
                {
                    "scene": "room",
                    "scene_root": scene_root,
                    "hr_paths": [hr_path],
                }
            ]

            with self.assertRaises(ValueError) as raised:
                module.prepare_jobs(
                    records,
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                    overwrite=True,
                )

            self.assertIn(str(lr_path), str(raised.exception))
            self.assertEqual(sentinel.read_bytes(), sentinel_payload)
            self.assertTrue(lr_path.is_symlink())

    def test_prepare_lr_file_generates_reuses_and_overwrites_rgb_png(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr_path = root / "images_2" / "frame.png"
            lr_path = root / "images_lr_srgs_x4" / "frame.JPG"
            hr_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13), (31, 63, 127)).save(hr_path)

            generated, lr_size, sr_size = module.prepare_lr_file(hr_path, lr_path)

            self.assertTrue(generated)
            self.assertEqual(lr_size, (4, 3))
            self.assertEqual(sr_size, (16, 12))
            with Image.open(lr_path) as saved:
                saved.load()
                self.assertEqual((saved.format, saved.mode, saved.size), ("PNG", "RGB", (4, 3)))
                first_payload = saved.tobytes()

            Image.new("RGB", (17, 13), (201, 202, 203)).save(hr_path)
            generated, reused_lr_size, reused_sr_size = module.prepare_lr_file(
                hr_path,
                lr_path,
            )

            self.assertFalse(generated)
            self.assertEqual((reused_lr_size, reused_sr_size), ((4, 3), (16, 12)))
            with Image.open(lr_path) as reused:
                reused.load()
                self.assertEqual(reused.tobytes(), first_payload)

            generated, _, _ = module.prepare_lr_file(
                hr_path,
                lr_path,
                overwrite=True,
            )

            self.assertTrue(generated)
            with Image.open(lr_path) as overwritten:
                overwritten.load()
                self.assertEqual(overwritten.getpixel((0, 0)), (201, 202, 203))

    def test_prepare_lr_file_rejects_invalid_existing_lr_payloads(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr_path = root / "images_2" / "frame.png"
            lr_path = root / "images_lr_srgs_x4" / "frame.JPG"
            hr_path.parent.mkdir(parents=True)
            lr_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13)).save(hr_path)
            invalid_cases = (
                ("wrong_size", Image.new("RGB", (5, 3)), "PNG"),
                ("jpeg_payload", Image.new("RGB", (4, 3)), "JPEG"),
                ("grayscale", Image.new("L", (4, 3)), "PNG"),
            )

            for name, invalid_image, image_format in invalid_cases:
                with self.subTest(name=name):
                    invalid_image.save(lr_path, format=image_format)
                    with self.assertRaises(ValueError):
                        module.prepare_lr_file(hr_path, lr_path)

    def test_prepare_jobs_validates_all_existing_sr_before_generating_lr(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            scene_root = Path(directory) / "room"
            hr_root = scene_root / "images_2"
            hr_root.mkdir(parents=True)
            first_hr = hr_root / "a.JPG"
            second_hr = hr_root / "b.JPG"
            Image.new("RGB", (17, 13)).save(first_hr)
            Image.new("RGB", (17, 13)).save(second_hr)
            first_lr = scene_root / "images_lr_srgs_x4" / first_hr.name
            second_lr = scene_root / "images_lr_srgs_x4" / second_hr.name
            invalid_sr = scene_root / "images_srgs_swinir_x4" / second_hr.name
            module.save_lossless_rgb(Image.new("RGB", (4, 3)), second_lr)
            module.save_lossless_rgb(Image.new("RGB", (15, 12)), invalid_sr)
            records = [
                {
                    "scene": "room",
                    "scene_root": scene_root,
                    "hr_paths": [first_hr, second_hr],
                }
            ]

            with self.assertRaisesRegex(ValueError, "expected 16x12"):
                module.prepare_jobs(
                    records,
                    "images_lr_srgs_x4",
                    "images_srgs_swinir_x4",
                )

            self.assertFalse(first_lr.exists())

    def test_prepare_jobs_skips_complete_outputs_and_overwrite_requeues_sr(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            scene_root = data_root / "room"
            hr_path = scene_root / "images_2" / "frame.JPG"
            lr_path = scene_root / "images_lr_srgs_x4" / "frame.JPG"
            sr_path = scene_root / "images_srgs_swinir_x4" / "frame.JPG"
            hr_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13), (11, 12, 13)).save(hr_path)
            module.save_lossless_rgb(Image.new("RGB", (4, 3)), lr_path)
            module.save_lossless_rgb(Image.new("RGB", (16, 12)), sr_path)
            records = [
                {
                    "scene": "room",
                    "scene_root": scene_root,
                    "hr_paths": [hr_path],
                }
            ]

            jobs, counts = module.prepare_jobs(
                records,
                "images_lr_srgs_x4",
                "images_srgs_swinir_x4",
            )

            self.assertEqual(jobs, [])
            self.assertEqual(
                counts,
                {
                    "room": {
                        "lr_generated": 0,
                        "lr_skipped": 1,
                        "sr_generated": 0,
                        "sr_skipped": 1,
                    }
                },
            )

            sr_path.unlink()
            jobs, counts = module.prepare_jobs(
                records,
                "images_lr_srgs_x4",
                "images_srgs_swinir_x4",
            )

            self.assertEqual(
                jobs,
                [
                    {
                        "scene": "room",
                        "lr_path": lr_path,
                        "sr_path": sr_path,
                        "sr_size": (16, 12),
                    }
                ],
            )
            self.assertEqual(
                counts,
                {
                    "room": {
                        "lr_generated": 0,
                        "lr_skipped": 1,
                        "sr_generated": 0,
                        "sr_skipped": 0,
                    }
                },
            )

            module.save_lossless_rgb(Image.new("RGB", (16, 12)), sr_path)
            self.assertTrue(sr_path.exists())
            jobs, counts = module.prepare_jobs(
                records,
                "images_lr_srgs_x4",
                "images_srgs_swinir_x4",
                overwrite=True,
            )

            self.assertEqual(
                jobs,
                [
                    {
                        "scene": "room",
                        "lr_path": lr_path,
                        "sr_path": sr_path,
                        "sr_size": (16, 12),
                    }
                ],
            )
            self.assertEqual(
                counts,
                {
                    "room": {
                        "lr_generated": 1,
                        "lr_skipped": 0,
                        "sr_generated": 0,
                        "sr_skipped": 0,
                    }
                },
            )
            self.assertFalse(sr_path.exists())

    def test_overwrite_failure_removes_old_sr_and_resume_queues_missing_sr(self):
        module = self.srgs_module()

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            scene_root = data_root / "room"
            hr_path = scene_root / "images_2" / "frame.JPG"
            lr_path = scene_root / "images_lr_srgs_x4" / hr_path.name
            sr_path = scene_root / "images_srgs_swinir_x4" / hr_path.name
            hr_path.parent.mkdir(parents=True)
            Image.new("RGB", (17, 13), (91, 92, 93)).save(hr_path)
            module.save_lossless_rgb(Image.new("RGB", (4, 3)), lr_path)
            module.save_lossless_rgb(Image.new("RGB", (16, 12)), sr_path)
            args = SimpleNamespace(
                data_root=data_root,
                scenes=["room"],
                hr_dir="images_2",
                lr_dir="images_lr_srgs_x4",
                sr_dir="images_srgs_swinir_x4",
                swinir_root=Path("/unused/SwinIR"),
                model_path=None,
                device="cuda",
                overwrite=True,
                scale=4,
            )

            with mock.patch.object(
                module,
                "resolve_strict_device",
                side_effect=RuntimeError("simulated CUDA failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated CUDA"):
                    module.run_dataset(args)

            module.validate_saved_image(lr_path, (4, 3), "LR")
            self.assertFalse(sr_path.exists())

            records = module.preflight_scenes(
                data_root,
                ["room"],
                "images_2",
            )
            jobs, counts = module.prepare_jobs(
                records,
                "images_lr_srgs_x4",
                "images_srgs_swinir_x4",
            )

            self.assertEqual(
                jobs,
                [
                    {
                        "scene": "room",
                        "lr_path": lr_path,
                        "sr_path": sr_path,
                        "sr_size": (16, 12),
                    }
                ],
            )
            self.assertEqual(
                counts,
                {
                    "room": {
                        "lr_generated": 0,
                        "lr_skipped": 1,
                        "sr_generated": 0,
                        "sr_skipped": 0,
                    }
                },
            )

    def test_resolve_strict_device_rejects_unavailable_cuda(self):
        module = self.srgs_module()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            device=mock.Mock(),
        )

        with mock.patch.object(module, "require_torch", return_value=fake_torch):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                module.resolve_strict_device("cuda")

        fake_torch.device.assert_not_called()

    def test_run_dataset_with_no_jobs_does_not_load_torch_or_model(self):
        module = self.srgs_module()
        args = SimpleNamespace(
            data_root=Path("/unused/data"),
            scenes=["room"],
            hr_dir="images_2",
            lr_dir="images_lr_srgs_x4",
            sr_dir="images_srgs_swinir_x4",
            swinir_root=Path("/unused/SwinIR"),
            model_path=None,
            device="cuda",
            overwrite=False,
            scale=4,
        )
        records = [
            {
                "scene": "room",
                "scene_root": Path("/unused/data/room"),
                "hr_paths": [Path("/unused/data/room/images_2/frame.JPG")],
            }
        ]
        counts = {
            "room": {
                "lr_generated": 0,
                "lr_skipped": 1,
                "sr_generated": 0,
                "sr_skipped": 1,
            }
        }

        with ExitStack() as stack:
            preflight = stack.enter_context(
                mock.patch.object(
                    module,
                    "preflight_scenes",
                    return_value=records,
                )
            )
            prepare = stack.enter_context(
                mock.patch.object(
                    module,
                    "prepare_jobs",
                    return_value=([], counts),
                )
            )
            require_torch = stack.enter_context(
                mock.patch.object(module, "require_torch")
            )
            resolve_device = stack.enter_context(
                mock.patch.object(module, "resolve_strict_device")
            )
            default_path = stack.enter_context(
                mock.patch.object(module, "default_model_path")
            )
            build_model = stack.enter_context(
                mock.patch.object(module, "build_swinir_classical")
            )
            module.run_dataset(args)

        preflight.assert_called_once_with(args.data_root, args.scenes, args.hr_dir)
        prepare.assert_called_once_with(
            records,
            args.lr_dir,
            args.sr_dir,
            overwrite=False,
            scale=4,
        )
        require_torch.assert_not_called()
        resolve_device.assert_not_called()
        default_path.assert_not_called()
        build_model.assert_not_called()

    def test_run_dataset_builds_once_and_processes_two_jobs(self):
        module = self.srgs_module()
        args = SimpleNamespace(
            data_root=Path("/unused/data"),
            scenes=["room"],
            hr_dir="images_2",
            lr_dir="images_lr_srgs_x4",
            sr_dir="images_srgs_swinir_x4",
            swinir_root=Path("/unused/SwinIR"),
            model_path=None,
            device="cuda",
            overwrite=False,
            scale=4,
        )
        scene_root = args.data_root / "room"
        records = [
            {
                "scene": "room",
                "scene_root": scene_root,
                "hr_paths": [
                    scene_root / args.hr_dir / "a.JPG",
                    scene_root / args.hr_dir / "b.JPG",
                ],
            }
        ]
        jobs = [
            {
                "scene": "room",
                "lr_path": scene_root / args.lr_dir / "a.JPG",
                "sr_path": scene_root / args.sr_dir / "a.JPG",
                "sr_size": (16, 12),
            },
            {
                "scene": "room",
                "lr_path": scene_root / args.lr_dir / "b.JPG",
                "sr_path": scene_root / args.sr_dir / "b.JPG",
                "sr_size": (20, 16),
            },
        ]
        counts = {
            "room": {
                "lr_generated": 0,
                "lr_skipped": 2,
                "sr_generated": 0,
                "sr_skipped": 0,
            }
        }
        device = mock.sentinel.device
        model_path = Path("/unused/model.pth")
        model = mock.sentinel.model
        outputs = [mock.sentinel.output_a, mock.sentinel.output_b]

        with ExitStack() as stack:
            layout = stack.enter_context(
                mock.patch.object(
                    module,
                    "validate_dataset_layout",
                    create=True,
                )
            )
            preflight = stack.enter_context(
                mock.patch.object(
                    module,
                    "preflight_scenes",
                    return_value=records,
                )
            )
            prepare = stack.enter_context(
                mock.patch.object(
                    module,
                    "prepare_jobs",
                    return_value=(jobs, counts),
                )
            )
            resolve = stack.enter_context(
                mock.patch.object(
                    module,
                    "resolve_strict_device",
                    return_value=device,
                )
            )
            default_path = stack.enter_context(
                mock.patch.object(
                    module,
                    "default_model_path",
                    return_value=model_path,
                )
            )
            build = stack.enter_context(
                mock.patch.object(
                    module,
                    "build_swinir_classical",
                    return_value=model,
                )
            )
            progress = stack.enter_context(
                mock.patch.object(
                    module,
                    "tqdm",
                    side_effect=lambda iterable, **_: iterable,
                )
            )
            infer = stack.enter_context(
                mock.patch.object(module, "infer_one", side_effect=outputs)
            )
            save = stack.enter_context(
                mock.patch.object(module, "save_lossless_rgb")
            )
            validate = stack.enter_context(
                mock.patch.object(module, "validate_saved_image")
            )
            summary = stack.enter_context(
                mock.patch.object(module, "print_counts")
            )

            result = module.run_dataset(args)

        layout.assert_called_once_with(
            args.data_root,
            args.scenes,
            args.hr_dir,
            args.lr_dir,
            args.sr_dir,
        )
        preflight.assert_called_once_with(args.data_root, args.scenes, args.hr_dir)
        prepare.assert_called_once_with(
            records,
            args.lr_dir,
            args.sr_dir,
            overwrite=False,
            scale=4,
        )
        resolve.assert_called_once_with("cuda")
        default_path.assert_called_once_with(args.swinir_root)
        build.assert_called_once_with(
            swinir_root=args.swinir_root,
            model_path=model_path,
            device=device,
            scale=4,
        )
        progress.assert_called_once_with(jobs, desc="Strict SRGS SwinIR x4")
        self.assertEqual(
            infer.call_args_list,
            [
                mock.call(model, jobs[0]["lr_path"], (16, 12), device, scale=4),
                mock.call(model, jobs[1]["lr_path"], (20, 16), device, scale=4),
            ],
        )
        self.assertEqual(
            save.call_args_list,
            [
                mock.call(outputs[0], jobs[0]["sr_path"]),
                mock.call(outputs[1], jobs[1]["sr_path"]),
            ],
        )
        self.assertEqual(
            validate.call_args_list,
            [
                mock.call(jobs[0]["sr_path"], (16, 12), "SR"),
                mock.call(jobs[1]["sr_path"], (20, 16), "SR"),
            ],
        )
        self.assertEqual(counts["room"]["sr_generated"], 2)
        summary.assert_called_once_with(records, counts)
        self.assertIs(result, counts)

    def test_infer_one_runs_one_whole_image_forward_then_crops_and_clamps(self):
        module = self.srgs_module()
        device = mock.sentinel.device
        input_tensor = mock.sentinel.input_tensor
        padded = mock.sentinel.padded
        raw_output = mock.MagicMock(name="raw_output")
        cropped = mock.MagicMock(name="cropped")
        clamped = mock.MagicMock(name="clamped")
        raw_output.__getitem__.return_value = cropped
        cropped.clamp.return_value = clamped
        clamped.size.side_effect = lambda dimension: {-1: 64, -2: 48}[dimension]
        model = mock.Mock(return_value=raw_output)
        no_grad_context = mock.MagicMock(name="no_grad_context")
        no_grad = mock.Mock(return_value=no_grad_context)
        fake_torch = SimpleNamespace(no_grad=no_grad)
        rendered = mock.sentinel.rendered_image

        with tempfile.TemporaryDirectory() as directory:
            lr_path = Path(directory) / "frame.JPG"
            Image.new("RGB", (16, 12)).save(lr_path, format="PNG")
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "require_torch",
                        return_value=fake_torch,
                    )
                )
                image_to_tensor = stack.enter_context(
                    mock.patch.object(
                        module,
                        "image_to_tensor",
                        return_value=input_tensor,
                    )
                )
                pad = stack.enter_context(
                    mock.patch.object(
                        module,
                        "pad_like_srgs",
                        return_value=(padded, 12, 16),
                    )
                )
                tensor_to_image = stack.enter_context(
                    mock.patch.object(
                        module,
                        "tensor_to_image",
                        return_value=rendered,
                    )
                )

                result = module.infer_one(model, lr_path, (64, 48), device)

        image_to_tensor.assert_called_once_with(mock.ANY, device)
        pad.assert_called_once_with(input_tensor)
        no_grad.assert_called_once_with()
        model.assert_called_once_with(padded)
        raw_output.__getitem__.assert_called_once_with(
            (Ellipsis, slice(None, 48), slice(None, 64))
        )
        cropped.clamp.assert_called_once_with(0, 1)
        self.assertEqual(
            clamped.size.call_args_list,
            [mock.call(-1), mock.call(-2)],
        )
        tensor_to_image.assert_called_once_with(clamped)
        self.assertIs(result, rendered)

    def test_infer_one_reports_path_actual_and_expected_for_wrong_size(self):
        module = self.srgs_module()
        device = mock.sentinel.device
        raw_output = mock.MagicMock(name="raw_output")
        cropped = mock.MagicMock(name="cropped")
        clamped = mock.MagicMock(name="clamped")
        raw_output.__getitem__.return_value = cropped
        cropped.clamp.return_value = clamped
        clamped.size.side_effect = lambda dimension: {-1: 63, -2: 48}[dimension]
        model = mock.Mock(return_value=raw_output)
        no_grad = mock.Mock(return_value=mock.MagicMock())
        fake_torch = SimpleNamespace(no_grad=no_grad)

        with tempfile.TemporaryDirectory() as directory:
            lr_path = Path(directory) / "wrong.JPG"
            Image.new("RGB", (16, 12)).save(lr_path, format="PNG")
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "require_torch",
                        return_value=fake_torch,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "image_to_tensor",
                        return_value=mock.sentinel.input_tensor,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        module,
                        "pad_like_srgs",
                        return_value=(mock.sentinel.padded, 12, 16),
                    )
                )
                tensor_to_image = stack.enter_context(
                    mock.patch.object(module, "tensor_to_image")
                )

                with self.assertRaises(RuntimeError) as raised:
                    module.infer_one(model, lr_path, (64, 48), device)

        message = str(raised.exception)
        self.assertIn(str(lr_path), message)
        self.assertIn("actual (63, 48)", message)
        self.assertIn("expected (64, 48)", message)
        model.assert_called_once_with(mock.sentinel.padded)
        tensor_to_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
