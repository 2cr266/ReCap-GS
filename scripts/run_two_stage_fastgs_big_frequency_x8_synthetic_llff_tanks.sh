#!/usr/bin/env bash
set -Eeuo pipefail

FASTGS_ROOT="${FASTGS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${FASTGS_ROOT}/runs}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-${FASTGS_ROOT}/data/nerf_synthetic}"
LLFF_ROOT="${LLFF_ROOT:-${FASTGS_ROOT}/data/llff}"
TANKS_ROOT="${TANKS_ROOT:-${FASTGS_ROOT}/data/tandt}"
DEEPBLENDING_ROOT="${DEEPBLENDING_ROOT:-${FASTGS_ROOT}/data/db}"
SWINIR_ROOT="${SWINIR_ROOT:-${FASTGS_ROOT}/third_party/SwinIR}"
SWINIR_X8_MODEL_PATH="${SWINIR_X8_MODEL_PATH:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-exp4a_dense_swinir_x8}"
GPU_ID="${GPU_ID:-0}"
DATASET_GROUPS="${DATASET_GROUPS:-synthetic llff tanks deepblending}"
RUN_PREP="${RUN_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

SYNTHETIC_SCENES_DEFAULT=(chair drums ficus hotdog lego materials mic ship)
LLFF_SCENES_DEFAULT=(fern flower fortress horns leaves orchids room trex)
TANKS_SCENES_DEFAULT=(truck train)
DEEPBLENDING_SCENES_DEFAULT=(playroom drjohnson)

if [[ -z "$SWINIR_X8_MODEL_PATH" && "$RUN_PREP" == "1" ]]; then
  echo "Set SWINIR_X8_MODEL_PATH to the downloaded SwinIR x8 checkpoint." >&2
  exit 2
fi

timestamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[$(timestamp)] CMD: $*" | tee -a "$log_file"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "[$(timestamp)] FAILED (${status}): $*" | tee -a "$log_file"
    exit "$status"
  fi
}

last_match() {
  sed -nE "$1" "$2" | tail -n 1
}

json_metric() {
  python - "$1" "$2" <<'PY'
import json
import pathlib
import sys

model_dir, metric = sys.argv[1:]
data = json.loads((pathlib.Path(model_dir) / "results.json").read_text())
method = "ours_30000" if "ours_30000" in data else sorted(data)[-1]
print("{:.7f}".format(float(data[method][metric])))
PY
}

append_average() {
  python - "$1" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["scene"] != "AVERAGE"]
columns = (
    "lr_time_s", "hr_time_s", "total_train_time_s",
    "final_gaussian_count", "render_fps", "psnr", "ssim", "lpips",
)
if not rows or any(not row[name] for row in rows for name in columns):
    raise RuntimeError("Incomplete result rows in {}".format(path))
averages = {name: sum(float(row[name]) for row in rows) / len(rows) for name in columns}
with path.open("a", encoding="utf-8") as handle:
    handle.write(
        "AVERAGE\t8000\t22000\t{lr_time_s:.3f}\t{hr_time_s:.3f}\t"
        "{total_train_time_s:.3f}\t{final_gaussian_count:.1f}\t"
        "{render_fps:.2f}\t{psnr:.7f}\t{ssim:.7f}\t{lpips:.7f}\t\n".format(
            **averages
        )
    )
print(path.read_text())
PY
}

run_synthetic() {
  local -a scenes
  if [[ -n "${SYNTHETIC_SCENES:-}" ]]; then
    read -r -a scenes <<< "$SYNTHETIC_SCENES"
  else
    scenes=("${SYNTHETIC_SCENES_DEFAULT[@]}")
  fi
  env \
    FASTGS_ROOT="$FASTGS_ROOT" \
    DATA_ROOT="$SYNTHETIC_ROOT" \
    RUN_ROOT="$RUN_ROOT" \
    RUN_NAME="${RUN_NAME_PREFIX}_synthetic" \
    GPU_ID="$GPU_ID" \
    LR_IMAGE_DIR="train_lr_srgs_x8" \
    HR_IMAGE_DIR="train_srgs_swinir_x8" \
    SWINIR_ROOT="$SWINIR_ROOT" \
    SWINIR_MODEL_PATH="$SWINIR_X8_MODEL_PATH" \
    RUN_PREP="$RUN_PREP" \
    SR_SCALE=8 \
    bash scripts/run_two_stage_fastgs_big_frequency_synthetic8.sh "${scenes[@]}"
}

run_colmap_group() {
  local dataset="$1"
  local data_root hr_dir eval_dir
  local -a scenes
  if [[ "$dataset" == "llff" ]]; then
    data_root="$LLFF_ROOT"
    hr_dir="${LLFF_HR_DIR:-images_4}"
    eval_dir="${LLFF_EVAL_DIR:-images_4}"
    if [[ -n "${LLFF_SCENES:-}" ]]; then
      read -r -a scenes <<< "$LLFF_SCENES"
    else
      scenes=("${LLFF_SCENES_DEFAULT[@]}")
    fi
  elif [[ "$dataset" == "tanks" ]]; then
    data_root="$TANKS_ROOT"
    hr_dir="${TANKS_HR_DIR:-images}"
    eval_dir="${TANKS_EVAL_DIR:-images}"
    if [[ -n "${TANKS_SCENES:-}" ]]; then
      read -r -a scenes <<< "$TANKS_SCENES"
    else
      scenes=("${TANKS_SCENES_DEFAULT[@]}")
    fi
  elif [[ "$dataset" == "deepblending" ]]; then
    data_root="$DEEPBLENDING_ROOT"
    hr_dir="${DEEPBLENDING_HR_DIR:-images}"
    eval_dir="${DEEPBLENDING_EVAL_DIR:-images}"
    if [[ -n "${DEEPBLENDING_SCENES:-}" ]]; then
      read -r -a scenes <<< "$DEEPBLENDING_SCENES"
    else
      scenes=("${DEEPBLENDING_SCENES_DEFAULT[@]}")
    fi
  else
    echo "Unknown COLMAP dataset: $dataset" >&2
    exit 2
  fi

  local lr_dir="images_lr_srgs_x8"
  local sr_dir="images_srgs_swinir_x8"
  local run_name="${RUN_NAME_PREFIX}_${dataset}"
  local output_root="${RUN_ROOT}/output/${run_name}"
  local log_root="${RUN_ROOT}/logs/${run_name}"
  local summary="${log_root}/metrics_summary.txt"
  mkdir -p "$log_root"

  if [[ "$RUN_PREP" == "1" ]]; then
    run_logged "${log_root}/00_prepare_images.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python scripts/run_srgs_swinir_x4.py \
        --data_root "$data_root" --scenes "${scenes[@]}" \
        --hr_dir "$hr_dir" --lr_dir "$lr_dir" --sr_dir "$sr_dir" \
        --swinir_root "$SWINIR_ROOT" \
        --model_path "$SWINIR_X8_MODEL_PATH" --device cuda --scale 8
  fi

  printf "scene\tlr_iterations\thr_iterations\tlr_time_s\thr_time_s\ttotal_train_time_s\tfinal_gaussian_count\trender_fps\tpsnr\tssim\tlpips\tmodel_dir\n" > "$summary"
  for scene in "${scenes[@]}"; do
    local scene_dir="${data_root}/${scene}"
    local model_dir="${output_root}/${scene}"
    local scene_logs="${log_root}/${scene}"
    local train_log="${scene_logs}/01_train.log"
    local render_log="${scene_logs}/02_render.log"
    local metrics_log="${scene_logs}/03_metrics.log"
    mkdir -p "$scene_logs"
    local -a train_scene_args=()
    local -a render_scene_args=()
    case "${dataset}/${scene}" in
      tanks/truck)
        train_scene_args=(--highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7)
        render_scene_args=(--mult 0.7)
        ;;
      tanks/train)
        train_scene_args=(--highfeature_lr 0.042 --grad_abs_thresh 0.0004 --dense 0.015 --mult 0.7)
        render_scene_args=(--mult 0.7)
        ;;
      deepblending/playroom)
        train_scene_args=(--highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005)
        render_scene_args=(--mult 0.7)
        ;;
      deepblending/drjohnson)
        train_scene_args=(--highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0005 --dense 0.005 --mult 0.7)
        render_scene_args=(--mult 0.7)
        ;;
    esac

    run_logged "${scene_logs}/00_preflight.log" \
      python scripts/check_two_stage_inputs_x8.py \
        --scene_dir "$scene_dir" --lr_image_dir "$lr_dir" \
        --hr_image_dir "$sr_dir" --scale 8 --eval

    if [[ "$RUN_TRAIN" == "1" ]]; then
      run_logged "$train_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" OAR_JOB_ID="${run_name}_${scene}" \
        python train_two_stage_frequency_bestof5.py \
          -s "$scene_dir" -i "$sr_dir" -m "$model_dir" \
          --resolution 1 --data_device cpu --eval \
          --iterations 30000 --switch_iteration 8000 --sr_scale 8 \
          --lr_image_dir "$lr_dir" --hr_image_dir "$sr_dir" \
          --save_iterations 8000 30000 --checkpoint_iterations 8000 30000 \
          --densify_from_iter 500 --densify_until_iter 15000 \
          --densification_interval 100 --opacity_reset_interval 3000 \
          --optimizer_type default \
          --detail_hf_quantile 0.70 --detail_error_quantile 0.70 \
          --detail_min_visible_views 3 --detail_min_total_pixels 8 \
          --detail_min_views 2 --detail_min_pixels 3 \
          --detail_min_overlap_ratio 0.25 --detail_min_support_ratio 0.50 \
          --detail_min_average_overlap_ratio 0.20 \
          --detail_grad_scale 0.80 --detail_importance_threshold 4 \
          --detail_dilation 1 --max_gaussians 0 \
          --freq_lr_weight 1.0 --freq_hf_weight 0.5 --freq_rgb_weight 0.1 \
          --freq_transition_iters 1000 --freq_hf_quantile 0.70 \
          --freq_hf_abs_thresh 0.01 --freq_hf_min_pixels 256 \
          --freq_hf_border_ignore 2 --freq_charbonnier_epsilon 0.001 \
          "${train_scene_args[@]}"
    fi
    if [[ "$RUN_RENDER" == "1" ]]; then
      run_logged "$render_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python render.py -m "$model_dir" -i "$eval_dir" \
          --resolution 1 --data_device cpu --iteration 30000 --skip_train \
          "${render_scene_args[@]}"
    fi
    if [[ "$RUN_METRICS" == "1" ]]; then
      run_logged "$metrics_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" python metrics.py -m "$model_dir"
    fi

    local lr_time hr_time total gaussians fps psnr ssim lpips
    lr_time="$(last_match 's/.*Training time: LR=([0-9.]+)s.*/\1/p' "$train_log")"
    hr_time="$(last_match 's/.*Training time: LR=[0-9.]+s HR=([0-9.]+)s.*/\1/p' "$train_log")"
    total="$(last_match 's/.*Training time:.*total=([0-9.]+)s.*/\1/p' "$train_log")"
    gaussians="$(last_match 's/.*Gaussian number:[[:space:]]*([0-9]+).*/\1/p' "$train_log")"
    fps="$(last_match 's/.*Average FPS:[[:space:]]*([0-9.]+).*/\1/p' "$render_log")"
    psnr="$(json_metric "$model_dir" PSNR)"
    ssim="$(json_metric "$model_dir" SSIM)"
    lpips="$(json_metric "$model_dir" LPIPS)"
    printf "%s\t8000\t22000\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$scene" "$lr_time" "$hr_time" "$total" "$gaussians" "$fps" \
      "$psnr" "$ssim" "$lpips" "$model_dir" >> "$summary"
  done
  append_average "$summary"
}

cd "$FASTGS_ROOT"
for group in $DATASET_GROUPS; do
  case "$group" in
    synthetic) run_synthetic ;;
    llff|tanks|deepblending) run_colmap_group "$group" ;;
    *) echo "Unknown group: $group" >&2; exit 2 ;;
  esac
done
