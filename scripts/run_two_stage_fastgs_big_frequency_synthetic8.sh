#!/usr/bin/env bash
set -Eeuo pipefail

FASTGS_ROOT="${FASTGS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${FASTGS_ROOT}/data/nerf_synthetic}"
RUN_ROOT="${RUN_ROOT:-${FASTGS_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-exp4a_nerf_synthetic_swinir_x4_full8}"
GPU_ID="${GPU_ID:-0}"
LR_IMAGE_DIR="${LR_IMAGE_DIR:-train_lr_srgs_x4}"
HR_IMAGE_DIR="${HR_IMAGE_DIR:-train_srgs_swinir_x4}"
SWINIR_ROOT="${SWINIR_ROOT:-${FASTGS_ROOT}/third_party/SwinIR}"
SWINIR_MODEL_PATH="${SWINIR_MODEL_PATH:-}"
RUN_PREP="${RUN_PREP:-1}"
MULT="${MULT:-0.5}"
SR_SCALE="${SR_SCALE:-4}"

OUTPUT_ROOT="${RUN_ROOT}/output/${RUN_NAME}"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"
SUMMARY_TSV="${LOG_ROOT}/metrics_summary.txt"
PIPELINE_LOG="${LOG_ROOT}/pipeline.log"

DEFAULT_SCENES=(chair drums ficus hotdog lego materials mic ship)
if [[ "$#" -gt 0 ]]; then
  SCENES=("$@")
else
  SCENES=("${DEFAULT_SCENES[@]}")
fi

timestamp() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }
log_line() { echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"; }

run_logged() {
  local log_file="$1"
  shift
  log_line "CMD: $*"
  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log_line "FAILED (${status}): $*"
    exit "$status"
  fi
}

last_match() {
  local pattern="$1"
  local file="$2"
  sed -nE "$pattern" "$file" | tail -n 1
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

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$FASTGS_ROOT"

if [[ "$RUN_PREP" == "1" ]]; then
  prep=(
    env CUDA_VISIBLE_DEVICES="$GPU_ID"
    python scripts/run_srgs_swinir_x4_synthetic.py
    --data_root "$DATA_ROOT"
    --scenes "${SCENES[@]}"
    --lr_dir "$LR_IMAGE_DIR"
    --sr_dir "$HR_IMAGE_DIR"
    --swinir_root "$SWINIR_ROOT"
    --device cuda
    --scale "$SR_SCALE"
  )
  if [[ -n "$SWINIR_MODEL_PATH" ]]; then
    prep+=(--model_path "$SWINIR_MODEL_PATH")
  fi
  run_logged "$LOG_ROOT/00_prepare_images.log" "${prep[@]}"
fi

printf "scene\tlr_iterations\thr_iterations\tlr_time_s\thr_time_s\ttotal_train_time_s\tfinal_gaussian_count\trender_fps\tpsnr\tssim\tlpips\tmodel_dir\n" > "$SUMMARY_TSV"

for scene in "${SCENES[@]}"; do
  scene_dir="${DATA_ROOT}/${scene}"
  model_dir="${OUTPUT_ROOT}/${scene}"
  scene_log_dir="${LOG_ROOT}/${scene}"
  train_log="${scene_log_dir}/01_train.log"
  render_log="${scene_log_dir}/02_render.log"
  metrics_log="${scene_log_dir}/03_metrics.log"
  mkdir -p "$scene_log_dir"

  run_logged "${scene_log_dir}/00_preflight.log" \
    python scripts/check_two_stage_synthetic_inputs.py \
      --scene_dir "$scene_dir" \
      --lr_image_dir "$LR_IMAGE_DIR" \
      --hr_image_dir "$HR_IMAGE_DIR" \
      --scale "$SR_SCALE"

  run_logged "$train_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" OAR_JOB_ID="${RUN_NAME}_${scene}" \
    python train_two_stage_frequency_bestof5.py \
      -s "$scene_dir" \
      -i "$HR_IMAGE_DIR" \
      -m "$model_dir" \
      --resolution 1 \
      --data_device cpu \
      --white_background \
      --eval \
      --iterations 30000 \
      --switch_iteration 8000 \
      --sr_scale "$SR_SCALE" \
      --lr_image_dir "$LR_IMAGE_DIR" \
      --hr_image_dir "$HR_IMAGE_DIR" \
      --save_iterations 8000 30000 \
      --checkpoint_iterations 8000 30000 \
      --densify_from_iter 500 \
      --densify_until_iter 15000 \
      --densification_interval 100 \
      --opacity_reset_interval 3000 \
      --optimizer_type default \
      --detail_hf_quantile 0.70 \
      --detail_error_quantile 0.70 \
      --detail_min_visible_views 3 \
      --detail_min_total_pixels 8 \
      --detail_min_views 2 \
      --detail_min_pixels 3 \
      --detail_min_overlap_ratio 0.25 \
      --detail_min_support_ratio 0.50 \
      --detail_min_average_overlap_ratio 0.20 \
      --detail_grad_scale 0.80 \
      --detail_importance_threshold 4 \
      --detail_dilation 1 \
      --max_gaussians 0 \
      --freq_lr_weight 1.0 \
      --freq_hf_weight 0.5 \
      --freq_rgb_weight 0.1 \
      --freq_transition_iters 1000 \
      --freq_hf_quantile 0.70 \
      --freq_hf_abs_thresh 0.01 \
      --freq_hf_min_pixels 256 \
      --freq_hf_border_ignore 2 \
      --freq_charbonnier_epsilon 0.001 \
      --mult "$MULT"

  run_logged "$render_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python render.py \
      -m "$model_dir" \
      --iteration 30000 \
      --skip_train \
      --white_background \
      --mult "$MULT"

  run_logged "$metrics_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python metrics.py -m "$model_dir"

  lr_time="$(last_match 's/.*Training time: LR=([0-9.]+)s.*/\1/p' "$train_log")"
  hr_time="$(last_match 's/.*Training time: LR=[0-9.]+s HR=([0-9.]+)s.*/\1/p' "$train_log")"
  total_time="$(last_match 's/.*Training time:.*total=([0-9.]+)s.*/\1/p' "$train_log")"
  gaussians="$(last_match 's/.*Gaussian number:[[:space:]]*([0-9]+).*/\1/p' "$train_log")"
  fps="$(last_match 's/.*Average FPS:[[:space:]]*([0-9.]+).*/\1/p' "$render_log")"
  psnr="$(json_metric "$model_dir" PSNR)"
  ssim="$(json_metric "$model_dir" SSIM)"
  lpips="$(json_metric "$model_dir" LPIPS)"
  printf "%s\t8000\t22000\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$scene" "$lr_time" "$hr_time" "$total_time" "$gaussians" \
    "$fps" "$psnr" "$ssim" "$lpips" "$model_dir" >> "$SUMMARY_TSV"
  log_line "Completed ${scene}: PSNR=${psnr} SSIM=${ssim} LPIPS=${lpips} Gaussians=${gaussians} FPS=${fps} Train=${total_time}s"
done

python - "$SUMMARY_TSV" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
columns = (
    "lr_time_s", "hr_time_s", "total_train_time_s",
    "final_gaussian_count", "render_fps", "psnr", "ssim", "lpips",
)
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

log_line "Finished Synthetic experiment: ${SUMMARY_TSV}"
