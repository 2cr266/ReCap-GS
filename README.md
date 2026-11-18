# ReCap-GS

ReCap-GS reconstructs a Gaussian scene for high-resolution novel-view synthesis from posed low-resolution images. It fits the scene at low resolution first, then continues optimizing the same representation at high resolution with real LR observations and precomputed pseudo-HR images. During refinement, it uses target structure and current reconstruction error across training views to identify Gaussians eligible for relaxed densification thresholds.

## Installation

Use Linux with an NVIDIA GPU, Conda, a C++ compiler, and a CUDA 11.x toolkit with `nvcc`. The environment pins PyTorch 1.12.1 and its CUDA 11.6 runtime, following the experiment environment. If CUDA is installed outside the standard location, set `CUDA_HOME` to the toolkit directory before building.

From the repository root, create the environment and build the three included extensions:

```bash
conda env create -f environment.yml
conda activate recap-gs
python -m pip install --no-build-isolation \
  ./submodules/diff-gaussian-rasterization_fastgs \
  ./submodules/simple-knn \
  ./submodules/fused-ssim
```

Use the bundled rasterizer. The detail aggregation code relies on its per-Gaussian metric counts.

## Mip-NeRF 360 at ×4

Place the [Mip-NeRF 360 data](https://jonbarron.info/mipnerf360/) under `data/mipnerf360` or set an absolute dataset path. Each COLMAP scene should have its reference images and a sparse reconstruction. For example:

```text
data/mipnerf360/bonsai/
├── images_2/                # Real HR reference images for evaluation
├── images_lr_srgs_x4/        # LR observations, prepared below
├── images_srgs_swinir_x4/    # Pseudo-HR targets, prepared below
└── sparse/0/                 # COLMAP cameras, images and point cloud
```

The preparation script downsamples the reference images and applies the classical SwinIR-M ×4 model. It keeps the image filenames aligned with the COLMAP records. On first use it fetches the model definition and weights from [SwinIR](https://github.com/JingyunLiang/SwinIR). You can also supply local weights with `--model_path`.

```bash
export DATA_ROOT="$PWD/data/mipnerf360"
python scripts/run_srgs_swinir_x4.py \
  --data_root "$DATA_ROOT" --scenes bonsai \
  --hr_dir images_2 --lr_dir images_lr_srgs_x4 \
  --sr_dir images_srgs_swinir_x4 --scale 4

LR_IMAGE_DIR=images_lr_srgs_x4 HR_IMAGE_DIR=images_srgs_swinir_x4 \
  RUN_NAME=recap_gs_x4 bash scripts/run_two_stage_fastgs_big_frequency_full9.sh bonsai
```

The runner validates the inputs, applies the scene-specific training settings, fits for 8,000 LR iterations followed by 22,000 HR iterations, then renders held-out views against the real HR references. For COLMAP data, evaluation holds out every eighth image in sorted filename order, starting with the first. Pass several scene names to run a subset, or omit them to run all nine after preparing their inputs.

The example generates a fresh bicubic LR/SwinIR pair. If you already have the experiment inputs in `images_8/` and `images_sr_x4/`, the batch runner uses those directories by default.

By default the runner starts from scratch. To resume from previously generated, compatible LR checkpoints, set `START_CHECKPOINT_ROOT` to an absolute directory containing `<scene>/chkpnt8000.pth` for each requested scene.

For a COLMAP scene with existing paired LR observations and pseudo-HR targets, run the trainer directly. The example assumes the same folder names and uses the default thresholds. Use the batch runner above for the Mip-NeRF 360 scene-specific overrides.

```bash
SCENE="/path/to/colmap-scene"
MODEL="$PWD/runs/custom-scene"
python train_two_stage_frequency.py \
  -s "$SCENE" -i images_srgs_swinir_x4 -m "$MODEL" \
  --lr_image_dir images_lr_srgs_x4 --hr_image_dir images_srgs_swinir_x4 \
  --sr_scale 4 --resolution 1 --data_device cpu --eval \
  --switch_iteration 8000 --iterations 30000 \
  --save_iterations 8000 30000 --checkpoint_iterations 8000 30000

python render.py -m "$MODEL" -i images_2 --resolution 1 --iteration 30000 --skip_train
python metrics.py -m "$MODEL"
```

The batch runner writes models under `runs/output/<run>/<scene>` and logs under `runs/logs/<run>`. The final model is saved to `point_cloud/iteration_30000/point_cloud.ply` in the model directory. Evaluation writes `results.json` and `per_view.json` with PSNR, SSIM, and LPIPS.

For Tanks and Temples and Deep Blending at ×4, place the [COLMAP scenes](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) under `data/tandt` and `data/db`, respectively, and run:

```bash
bash scripts/run_two_stage_fastgs_big_frequency_x4_tanks_deepblending_profiled.sh
```

Set `TANKS_ROOT` and `DEEPBLENDING_ROOT` to use other dataset locations. Set `RUN_PREP=0` to reuse prepared LR and pseudo-HR folders.

## ×8 and NeRF Synthetic

The ×8 runner supports NeRF Synthetic, LLFF, Tanks and Temples, and Deep Blending. It uses the same LR-to-HR schedule with the trainer that supports ×8 and the Synthetic JSON camera format. For example, to run the LLFF Fern scene, provide the classical [SwinIR-M ×8 checkpoint](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0):

```bash
DATASET_GROUPS=llff LLFF_SCENES=fern LLFF_ROOT="$PWD/data/llff" \
  SWINIR_X8_MODEL_PATH="/path/to/001_classicalSR_DF2K_s64w8_SwinIR-M_x8.pth" \
  bash scripts/run_two_stage_fastgs_big_frequency_x8_synthetic_llff_tanks.sh
```

| Dataset group | Dataset root variable | Reference layout |
| --- | --- | --- |
| `synthetic` | `SYNTHETIC_ROOT` | `transforms_train.json`, `transforms_test.json`, and their referenced images |
| `llff` | `LLFF_ROOT` | `images_4/` and `sparse/0/` |
| `tanks` | `TANKS_ROOT` | `images/` and `sparse/0/` |
| `deepblending` | `DEEPBLENDING_ROOT` | `images/` and `sparse/0/` |

The runner prepares LR and pseudo-HR images before training by default. For prepared datasets, set `RUN_PREP=0` and use the input folders expected by the script. To run NeRF Synthetic at ×4 instead, use `scripts/run_two_stage_fastgs_big_frequency_synthetic8.sh` with `DATA_ROOT` pointing to the dataset and optional scene names on the command line. For Synthetic scenes the provided training and test transforms determine the split.

## Code layout

| Path | Role |
| --- | --- |
| `train_two_stage_frequency.py` | Main ×4 training entry point |
| `train_two_stage_frequency_bestof5.py` | Single-run trainer used by the ×8 and Synthetic runners. The filename is historical. |
| `utils/detail_densification_utils.py` and `utils/detail_fast_utils.py` | Image-space detail cues and aggregation across views |
| `scene/detail_gaussian_model.py` | Detail-conditioned density updates |
| `utils/frequency_loss_utils.py` | Observation-anchored residual supervision |
| `render.py` and `metrics.py` | Held-out rendering and image metrics |
| `submodules/` | Source for the CUDA extensions |

Run `python -m unittest discover -s tests` to check the data preparation and runner interfaces. These checks do not require a scene dataset.

## Acknowledgments and licenses

This implementation builds on [FastGS](https://github.com/fastgs/FastGS), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [SwinIR](https://github.com/JingyunLiang/SwinIR). See the [FastGS license](LICENSE), the [original 3D Gaussian Splatting license](LICENSE.md), and the notices in the extension directories for the terms applicable to inherited components.
