#!/usr/bin/env bash
set -Eeuo pipefail

FASTGS_ROOT="${FASTGS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${FASTGS_ROOT}/runs}"
TANKS_ROOT="${TANKS_ROOT:-${FASTGS_ROOT}/data/tandt}"
DEEPBLENDING_ROOT="${DEEPBLENDING_ROOT:-${FASTGS_ROOT}/data/db}"
RUN_NAME="${RUN_NAME:-exp4a_dense_swinir_x4_tanks_deepblending_profiled}"
GPU_ID="${GPU_ID:-0}"
SWINIR_ROOT="${SWINIR_ROOT:-${FASTGS_ROOT}/third_party/SwinIR}"
SWINIR_MODEL_PATH="${SWINIR_MODEL_PATH:-}"
RUN_PREP="${RUN_PREP:-1}"

LR_IMAGE_DIR="${LR_IMAGE_DIR:-images_lr_srgs_x4}"
HR_IMAGE_DIR="${HR_IMAGE_DIR:-images_srgs_swinir_x4}"
EVAL_IMAGE_DIR="${EVAL_IMAGE_DIR:-images}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

OUTPUT_ROOT="${RUN_ROOT}/output/${RUN_NAME}"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"
SUMMARY="${LOG_ROOT}/metrics_summary.txt"

TANKS_SCENES=(truck train)
DEEPBLENDING_SCENES=(playroom drjohnson)

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

run_train_profiled() {
  local train_log="$1"
  shift
  run_logged "$train_log" "$@"
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

append_dataset_average() {
  python - "$SUMMARY" "$1" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
dataset = sys.argv[2]
with path.open(encoding="utf-8") as handle:
    rows = [
        row for row in csv.DictReader(handle, delimiter="\t")
        if row["dataset"] == dataset and row["scene"] != "AVERAGE"
    ]
columns = (
    "lr_time_s", "hr_time_s", "total_train_time_s",
    "final_gaussian_count", "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib", "render_fps",
    "psnr", "ssim", "lpips",
)
if len(rows) != 2 or any(not row[name] for row in rows for name in columns):
    raise RuntimeError("Incomplete {} result rows in {}".format(dataset, path))
avg = {name: sum(float(row[name]) for row in rows) / len(rows) for name in columns}
with path.open("a", encoding="utf-8") as handle:
    handle.write(
        "{dataset}\tAVERAGE\t8000\t22000\t{lr_time_s:.3f}\t{hr_time_s:.3f}\t"
        "{total_train_time_s:.3f}\t{final_gaussian_count:.1f}\t"
        "{peak_gpu_allocated_mib:.1f}\t{peak_gpu_reserved_mib:.1f}\t"
        "{render_fps:.2f}\t{psnr:.7f}\t"
        "{ssim:.7f}\t{lpips:.7f}\t\n".format(dataset=dataset, **avg)
    )
PY
}

run_scene() {
  local dataset="$1"
  local data_root="$2"
  local scene="$3"
  local scene_dir="${data_root}/${scene}"
  local model_dir="${OUTPUT_ROOT}/${dataset}/${scene}"
  local scene_logs="${LOG_ROOT}/${dataset}/${scene}"
  local train_log="${scene_logs}/01_train.log"
  local render_log="${scene_logs}/02_render.log"
  local metrics_log="${scene_logs}/03_metrics.log"
  local -a train_scene_args=()
  local -a render_scene_args=(--mult 0.7)

  case "${dataset}/${scene}" in
    tanks/truck)
      train_scene_args=(--highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7)
      ;;
    tanks/train)
      train_scene_args=(--highfeature_lr 0.042 --grad_abs_thresh 0.0004 --dense 0.015 --mult 0.7)
      ;;
    deepblending/playroom)
      train_scene_args=(--highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005)
      ;;
    deepblending/drjohnson)
      train_scene_args=(--highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0005 --dense 0.005 --mult 0.7)
      ;;
    *)
      echo "Unsupported scene preset: ${dataset}/${scene}" >&2
      exit 2
      ;;
  esac

  if [[ -e "$model_dir" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
    echo "Output already exists: $model_dir" >&2
    echo "Use a new RUN_NAME, or set ALLOW_EXISTING_OUTPUT=1 intentionally." >&2
    exit 2
  fi
  mkdir -p "$scene_logs"

  run_logged "${scene_logs}/00_preflight.log" \
    python scripts/check_two_stage_inputs.py \
      --scene_dir "$scene_dir" \
      --lr_image_dir "$LR_IMAGE_DIR" \
      --hr_image_dir "$HR_IMAGE_DIR" \
      --scale 4 --eval

  run_train_profiled "$train_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" OAR_JOB_ID="${RUN_NAME}_${dataset}_${scene}" \
    python train_two_stage_frequency.py \
      -s "$scene_dir" -i "$HR_IMAGE_DIR" -m "$model_dir" \
      --resolution 1 --data_device cpu --eval \
      --iterations 30000 --switch_iteration 8000 --sr_scale 4 \
      --lr_image_dir "$LR_IMAGE_DIR" --hr_image_dir "$HR_IMAGE_DIR" \
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

  run_logged "$render_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python render.py -m "$model_dir" -i "$EVAL_IMAGE_DIR" \
      --resolution 1 --data_device cpu --iteration 30000 --skip_train \
      "${render_scene_args[@]}"

  run_logged "$metrics_log" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" python metrics.py -m "$model_dir"

  local lr_time hr_time total gaussians peak_allocated peak_reserved fps psnr ssim lpips
  lr_time="$(last_match 's/.*Training time: LR=([0-9.]+)s.*/\1/p' "$train_log")"
  hr_time="$(last_match 's/.*Training time: LR=[0-9.]+s HR=([0-9.]+)s.*/\1/p' "$train_log")"
  total="$(last_match 's/.*Training time:.*total=([0-9.]+)s.*/\1/p' "$train_log")"
  gaussians="$(last_match 's/.*Gaussian number:[[:space:]]*([0-9]+).*/\1/p' "$train_log")"
  peak_allocated="$(last_match 's/.*Peak CUDA memory: allocated=([0-9.]+) MiB.*/\1/p' "$train_log")"
  peak_reserved="$(last_match 's/.*Peak CUDA memory:.*reserved=([0-9.]+) MiB.*/\1/p' "$train_log")"
  fps="$(last_match 's/.*Average FPS:[[:space:]]*([0-9.]+).*/\1/p' "$render_log")"
  psnr="$(json_metric "$model_dir" PSNR)"
  ssim="$(json_metric "$model_dir" SSIM)"
  lpips="$(json_metric "$model_dir" LPIPS)"

  if [[ -z "$lr_time" || -z "$hr_time" || -z "$total" || -z "$gaussians" || \
        -z "$peak_allocated" || -z "$peak_reserved" || -z "$fps" || \
        -z "$psnr" || -z "$ssim" || -z "$lpips" ]]; then
    echo "Incomplete metrics for ${dataset}/${scene}; inspect $scene_logs" >&2
    exit 1
  fi

  printf "%s\t%s\t8000\t22000\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset" "$scene" "$lr_time" "$hr_time" "$total" "$gaussians" \
    "$peak_allocated" "$peak_reserved" "$fps" "$psnr" "$ssim" "$lpips" \
    "$model_dir" >> "$SUMMARY"
}

main() {
  mkdir -p "$LOG_ROOT"
  cd "$FASTGS_ROOT"

  if [[ "$RUN_PREP" == "1" ]]; then
    local -a sr_model_args=()
    if [[ -n "$SWINIR_MODEL_PATH" ]]; then
      sr_model_args=(--model_path "$SWINIR_MODEL_PATH")
    fi
    run_logged "${LOG_ROOT}/00_prepare_tanks_x4.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python scripts/run_srgs_swinir_x4.py \
        --data_root "$TANKS_ROOT" \
        --scenes "${TANKS_SCENES[@]}" \
        --hr_dir "$EVAL_IMAGE_DIR" \
        --lr_dir "$LR_IMAGE_DIR" \
        --sr_dir "$HR_IMAGE_DIR" \
        --swinir_root "$SWINIR_ROOT" \
        "${sr_model_args[@]}" \
        --device cuda --scale 4

    run_logged "${LOG_ROOT}/00_prepare_deepblending_x4.log" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" \
      python scripts/run_srgs_swinir_x4.py \
        --data_root "$DEEPBLENDING_ROOT" \
        --scenes "${DEEPBLENDING_SCENES[@]}" \
        --hr_dir "$EVAL_IMAGE_DIR" \
        --lr_dir "$LR_IMAGE_DIR" \
        --sr_dir "$HR_IMAGE_DIR" \
        --swinir_root "$SWINIR_ROOT" \
        "${sr_model_args[@]}" \
        --device cuda --scale 4
  fi

  printf "dataset\tscene\tlr_iterations\thr_iterations\tlr_time_s\thr_time_s\t" > "$SUMMARY"
  printf "total_train_time_s\tfinal_gaussian_count\tpeak_gpu_allocated_mib\t" >> "$SUMMARY"
  printf "peak_gpu_reserved_mib\trender_fps\t" >> "$SUMMARY"
  printf "psnr\tssim\tlpips\tmodel_dir\n" >> "$SUMMARY"

  local scene
  for scene in "${TANKS_SCENES[@]}"; do
    run_scene tanks "$TANKS_ROOT" "$scene"
  done
  append_dataset_average tanks

  for scene in "${DEEPBLENDING_SCENES[@]}"; do
    run_scene deepblending "$DEEPBLENDING_ROOT" "$scene"
  done
  append_dataset_average deepblending

  echo "Finished: $SUMMARY"
  cat "$SUMMARY"
}

main "$@"
