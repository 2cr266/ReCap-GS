import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SyntheticTwoStageRunnerContract(unittest.TestCase):
    def test_trainer_routes_synthetic_without_changing_colmap_route(self):
        source = (ROOT / "train_two_stage_frequency_bestof5.py").read_text()
        self.assertIn("is_synthetic_scene", source)
        self.assertIn("validator = validate_synthetic_scene", source)
        self.assertIn("validator = validate_colmap_scene_x8", source)
        self.assertIn("validator = validate_scene", source)
        self.assertIn("scene.train_cameras[1.0] = hr_cameras", source)

    def test_data_preparation_composites_white_before_downsampling(self):
        source = (ROOT / "scripts/run_srgs_swinir_x4_synthetic.py").read_text()
        self.assertIn("Image.alpha_composite(background, rgba)", source)
        self.assertIn("downsample_like_srgs(hr, scale=args.scale)", source)
        self.assertIn("expected_sr != hr.size", source)
        self.assertIn("choices=(4, 8)", source)

    def test_dense_x8_runner_covers_all_requested_datasets(self):
        source = (
            ROOT
            / "scripts/run_two_stage_fastgs_big_frequency_x8_synthetic_llff_tanks.sh"
        ).read_text()
        for token in (
            "DATASET_GROUPS=\"${DATASET_GROUPS:-synthetic llff tanks deepblending}\"",
            "SR_SCALE=8",
            "--sr_scale 8",
            "--scale 8",
            "SWINIR_X8_MODEL_PATH",
            "DEEPBLENDING_SCENES_DEFAULT=(playroom drjohnson)",
            "final_gaussian_count",
            "render_fps",
            "psnr",
            "ssim",
            "lpips",
        ):
            self.assertIn(token, source)

    def test_runner_covers_eight_scenes_and_required_metrics(self):
        source = (
            ROOT / "scripts/run_two_stage_fastgs_big_frequency_synthetic8.sh"
        ).read_text()
        self.assertIn(
            "DEFAULT_SCENES=(chair drums ficus hotdog lego materials mic ship)",
            source,
        )
        for token in (
            "--white_background",
            "Training time: LR=",
            "Gaussian number:",
            "Average FPS:",
            "final_gaussian_count",
            "total_train_time_s",
            "render_fps",
            "psnr",
            "ssim",
            "lpips",
        ):
            self.assertIn(token, source)

if __name__ == "__main__":
    unittest.main()
