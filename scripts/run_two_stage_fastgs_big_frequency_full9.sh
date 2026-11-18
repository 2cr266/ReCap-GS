#!/usr/bin/env bash
set -Eeuo pipefail

FASTGS_ROOT="${FASTGS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${FASTGS_ROOT}/data/mipnerf360}"
RUN_ROOT="${RUN_ROOT:-${FASTGS_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-diag_two_stage_fastgs_big_exp4a_frequency_full9}"
START_CHECKPOINT_ROOT="${START_CHECKPOINT_ROOT:-}"

GPU_ID="${GPU_ID:-0}"
ITERATIONS="${ITERATIONS:-30000}"
SWITCH_ITERATION="${SWITCH_ITERATION:-8000}"
DENSIFY_FROM_ITER="${DENSIFY_FROM_ITER:-500}"
DENSIFY_UNTIL_ITER="${DENSIFY_UNTIL_ITER:-15000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
OPACITY_RESET_INTERVAL="${OPACITY_RESET_INTERVAL:-3000}"
OPTIMIZER_TYPE="${OPTIMIZER_TYPE:-default}"
RESOLUTION="${RESOLUTION:-1}"
DATA_DEVICE="${DATA_DEVICE:-cpu}"
SR_SCALE="${SR_SCALE:-4}"
LR_IMAGE_DIR="${LR_IMAGE_DIR:-images_8}"
HR_IMAGE_DIR="${HR_IMAGE_DIR:-images_sr_x4}"
EVAL_IMAGE_DIR="${EVAL_IMAGE_DIR:-images_2}"
RENDER_METHOD="${RENDER_METHOD:-ours_${ITERATIONS}}"

DETAIL_HF_QUANTILE="${DETAIL_HF_QUANTILE:-0.70}"
DETAIL_ERROR_QUANTILE="${DETAIL_ERROR_QUANTILE:-0.70}"
DETAIL_MIN_VISIBLE_VIEWS="${DETAIL_MIN_VISIBLE_VIEWS:-3}"
DETAIL_MIN_TOTAL_PIXELS="${DETAIL_MIN_TOTAL_PIXELS:-8}"
DETAIL_MIN_VIEWS="${DETAIL_MIN_VIEWS:-2}"
DETAIL_MIN_PIXELS="${DETAIL_MIN_PIXELS:-3}"
DETAIL_MIN_OVERLAP_RATIO="${DETAIL_MIN_OVERLAP_RATIO:-0.25}"
DETAIL_MIN_SUPPORT_RATIO="${DETAIL_MIN_SUPPORT_RATIO:-0.50}"
DETAIL_MIN_AVERAGE_OVERLAP_RATIO="${DETAIL_MIN_AVERAGE_OVERLAP_RATIO:-0.20}"
DETAIL_GRAD_SCALE="${DETAIL_GRAD_SCALE:-0.80}"
DETAIL_IMPORTANCE_THRESHOLD="${DETAIL_IMPORTANCE_THRESHOLD:-4}"
DETAIL_DILATION="${DETAIL_DILATION:-1}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:-0}"

FREQ_LR_WEIGHT="${FREQ_LR_WEIGHT:-1.0}"
FREQ_HF_WEIGHT="${FREQ_HF_WEIGHT:-0.5}"
FREQ_RGB_WEIGHT="${FREQ_RGB_WEIGHT:-0.1}"
FREQ_TRANSITION_ITERS="${FREQ_TRANSITION_ITERS:-1000}"
FREQ_HF_QUANTILE="${FREQ_HF_QUANTILE:-0.70}"
FREQ_HF_ABS_THRESH="${FREQ_HF_ABS_THRESH:-0.01}"
FREQ_HF_MIN_PIXELS="${FREQ_HF_MIN_PIXELS:-256}"
FREQ_HF_BORDER_IGNORE="${FREQ_HF_BORDER_IGNORE:-2}"
FREQ_CHARBONNIER_EPSILON="${FREQ_CHARBONNIER_EPSILON:-0.001}"

SAVE_ITERATIONS="${SAVE_ITERATIONS:-8000 9000 12000 15000 18000 24000 30000}"
CHECKPOINT_ITERATIONS="${CHECKPOINT_ITERATIONS:-8000 9000 12000 15000 18000 24000 30000}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

OUTPUT_ROOT="${RUN_ROOT}/output/${RUN_NAME}"
LOG_ROOT="${RUN_ROOT}/logs/${RUN_NAME}"
SUMMARY_LOG="${LOG_ROOT}/pipeline_summary.txt"
RESULTS_TXT="${LOG_ROOT}/metrics_summary.txt"

DEFAULT_SCENES=(
  bicycle flowers garden stump treehill room counter kitchen bonsai
)
if [[ "$#" -gt 0 ]]; then
  SELECTED_SCENES=("$@")
elif [[ -n "${SCENES:-}" ]]; then
  read -r -a SELECTED_SCENES <<< "$SCENES"
else
  SELECTED_SCENES=("${DEFAULT_SCENES[@]}")
fi

read -r -a SAVE_ITERATIONS_ARRAY <<< "$SAVE_ITERATIONS"
read -r -a CHECKPOINT_ITERATIONS_ARRAY <<< "$CHECKPOINT_ITERATIONS"

TRAIN_EXTRA_ARGS_ARRAY=()
if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  read -r -a TRAIN_EXTRA_ARGS_ARRAY <<< "$TRAIN_EXTRA_ARGS"
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

log_line() {
  local message="$1"
  echo "[$(timestamp)] ${message}" | tee -a "$SUMMARY_LOG"
}

require_dir() {
  local dir_path="$1"
  if [[ ! -d "$dir_path" ]]; then
    echo "Missing directory: $dir_path" >&2
    exit 1
  fi
}

require_file() {
  local file_path="$1"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing file: $file_path" >&2
    exit 1
  fi
}

count_images() {
  local dir_path="$1"
  find "$dir_path" -maxdepth 1 -type f \
    \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \
       -o -name "*.PNG" -o -name "*.JPG" -o -name "*.JPEG" \) \
    | wc -l | tr -d " "
}

run_cmd() {
  local log_file="$1"
  shift

  mkdir -p "$(dirname "$log_file")"
  {
    echo
    echo "===== $(timestamp) ====="
    echo "CMD: $*"
  } | tee -a "$log_file" "$SUMMARY_LOG"

  set +e
  "$@" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log_line "FAILED (${status}): $*"
    exit "$status"
  fi

  log_line "OK: $*"
}

last_match() {
  local pattern="$1"
  local file_path="$2"
  if [[ ! -f "$file_path" ]]; then
    return 0
  fi
  sed -nE "$pattern" "$file_path" | tail -n 1
}

json_metric() {
  local model_dir="$1"
  local method="$2"
  local metric="$3"

  python - "$model_dir" "$method" "$metric" <<'PY'
import json
import pathlib
import sys

model_dir, method, metric = sys.argv[1:4]
results_path = pathlib.Path(model_dir) / "results.json"
if not results_path.exists():
    sys.exit(0)

data = json.loads(results_path.read_text())
value = data.get(method, {}).get(metric)
if value is None:
    sys.exit(0)

print(f"{float(value):.7f}")
PY
}

scene_big_args() {
  local scene="$1"
  SCENE_EXTRA_ARGS=()

  case "$scene" in
    bicycle)
      SCENE_EXTRA_ARGS=(--grad_abs_thresh 0.0008)
      ;;
    flowers)
      SCENE_EXTRA_ARGS=(--dense 0.005 --grad_abs_thresh 0.001)
      ;;
    garden)
      SCENE_EXTRA_ARGS=(--highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0003)
      ;;
    stump)
      SCENE_EXTRA_ARGS=(--dense 0.004 --grad_abs_thresh 0.001)
      ;;
    treehill)
      SCENE_EXTRA_ARGS=(--dense 0.01 --grad_abs_thresh 0.0018)
      ;;
    room)
      SCENE_EXTRA_ARGS=(--highfeature_lr 0.02 --grad_abs_thresh 0.0004)
      ;;
    counter)
      SCENE_EXTRA_ARGS=(--highfeature_lr 0.02 --grad_abs_thresh 0.0004)
      ;;
    kitchen)
      SCENE_EXTRA_ARGS=(--highfeature_lr 0.02 --grad_abs_thresh 0.0002)
      ;;
    bonsai)
      SCENE_EXTRA_ARGS=(--highfeature_lr 0.02 --grad_abs_thresh 0.0002)
      ;;
    *)
      echo "Unsupported scene for FastGS-big preset: ${scene}" >&2
      exit 1
      ;;
  esac
}

check_scene_inputs() {
  local scene="$1"
  local scene_dir="${DATA_ROOT}/${scene}"
  local eval_dir="${scene_dir}/${EVAL_IMAGE_DIR}"
  local scene_log_dir="$2"
  local check_log="${scene_log_dir}/00_check_inputs.txt"

  require_dir "$scene_dir"
  require_dir "$eval_dir"

  local eval_count
  eval_count="$(count_images "$eval_dir")"
  {
    echo "===== $(timestamp) ====="
    echo "scene=${scene}"
    echo "${EVAL_IMAGE_DIR}: ${eval_count}"
  } | tee -a "$check_log" "$SUMMARY_LOG"

  if [[ "$eval_count" == "0" ]]; then
    log_line "FAILED input count check for ${scene}: ${EVAL_IMAGE_DIR}=0"
    exit 1
  fi

  run_cmd "$check_log" \
    python scripts/check_two_stage_inputs.py \
      --scene_dir "$scene_dir" \
      --lr_image_dir "$LR_IMAGE_DIR" \
      --hr_image_dir "$HR_IMAGE_DIR" \
      --scale "$SR_SCALE" \
      --eval
}

ensure_results_header() {
  mkdir -p "$LOG_ROOT"
  if [[ ! -f "$RESULTS_TXT" ]]; then
    printf "scene\tlr_iterations\thr_iterations\tlr_image_dir\thr_image_dir\teval_image_dir\tresolution\tbig_args\tlr_gaussian_count\tfinal_gaussian_count\ttrain_time_s\trender_fps\tpsnr\tssim\tlpips\tcheckpoint_lr\tmodel_dir\n" > "$RESULTS_TXT"
  fi
}

append_result_row() {
  local scene="$1"
  local train_log="$2"
  local render_log="$3"
  local metrics_log="$4"
  local big_args="$5"
  local model_dir="${OUTPUT_ROOT}/${scene}"

  local lr_gaussian_count final_gaussian_count train_time
  local render_fps psnr ssim lpips
  lr_gaussian_count="$(last_match "s/.*\[Two-stage\] global=${SWITCH_ITERATION} .*gaussians=([0-9]+).*/\1/p" "$train_log")"
  final_gaussian_count="$(last_match 's/.*Gaussian number:[[:space:]]*([0-9]+).*/\1/p' "$train_log")"
  train_time="$(last_match 's/.*Training time:.*total=([0-9.]+)s.*/\1/p' "$train_log")"
  render_fps="$(last_match 's/.*Average FPS:[[:space:]]*([0-9.]+).*/\1/p' "$render_log")"
  psnr="$(json_metric "$model_dir" "$RENDER_METHOD" "PSNR")"
  ssim="$(json_metric "$model_dir" "$RENDER_METHOD" "SSIM")"
  lpips="$(json_metric "$model_dir" "$RENDER_METHOD" "LPIPS")"

  if [[ -z "$psnr" ]]; then
    psnr="$(last_match 's/.*PSNR[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p' "$metrics_log")"
  fi
  if [[ -z "$ssim" ]]; then
    ssim="$(last_match 's/.*SSIM[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p' "$metrics_log")"
  fi
  if [[ -z "$lpips" ]]; then
    lpips="$(last_match 's/.*LPIPS[[:space:]]*:[[:space:]]*([0-9.]+).*/\1/p' "$metrics_log")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$scene" "$SWITCH_ITERATION" "$((ITERATIONS - SWITCH_ITERATION))" \
    "$LR_IMAGE_DIR" "$HR_IMAGE_DIR" "$EVAL_IMAGE_DIR" "$RESOLUTION" \
    "$big_args" "$lr_gaussian_count" "$final_gaussian_count" \
    "$train_time" "$render_fps" "$psnr" "$ssim" "$lpips" \
    "$6" "$model_dir" \
    >> "$RESULTS_TXT"
}

main() {
  mkdir -p "$LOG_ROOT"
  ensure_results_header
  require_dir "$FASTGS_ROOT"
  require_dir "$DATA_ROOT"
  cd "$FASTGS_ROOT"

  log_line "Starting experiment 4-A frequency-decoupled two-stage FastGS-big full Mip-NeRF 360 benchmark"
  log_line "FASTGS_ROOT=${FASTGS_ROOT}"
  log_line "DATA_ROOT=${DATA_ROOT}"
  log_line "OUTPUT_ROOT=${OUTPUT_ROOT}"
  log_line "LOG_ROOT=${LOG_ROOT}"
  log_line "START_CHECKPOINT_ROOT=${START_CHECKPOINT_ROOT}"
  log_line "SCENES=${SELECTED_SCENES[*]}"
  log_line "LR_IMAGE_DIR=${LR_IMAGE_DIR}"
  log_line "HR_IMAGE_DIR=${HR_IMAGE_DIR}"
  log_line "EVAL_IMAGE_DIR=${EVAL_IMAGE_DIR}"
  log_line "SWITCH_ITERATION=${SWITCH_ITERATION}"
  log_line "HR_ITERATIONS=$((ITERATIONS - SWITCH_ITERATION))"
  log_line "DENSIFY_FROM_ITER=${DENSIFY_FROM_ITER}"
  log_line "DENSIFY_UNTIL_ITER=${DENSIFY_UNTIL_ITER}"
  log_line "DENSIFICATION_INTERVAL=${DENSIFICATION_INTERVAL}"
  log_line "OPACITY_RESET_INTERVAL=${OPACITY_RESET_INTERVAL}"
  log_line "RESOLUTION=${RESOLUTION}"
  log_line "DATA_DEVICE=${DATA_DEVICE}"
  log_line "DETAIL_HF_QUANTILE=${DETAIL_HF_QUANTILE}"
  log_line "DETAIL_ERROR_QUANTILE=${DETAIL_ERROR_QUANTILE}"
  log_line "DETAIL_MIN_VISIBLE_VIEWS=${DETAIL_MIN_VISIBLE_VIEWS}"
  log_line "DETAIL_MIN_TOTAL_PIXELS=${DETAIL_MIN_TOTAL_PIXELS}"
  log_line "DETAIL_MIN_VIEWS=${DETAIL_MIN_VIEWS}"
  log_line "DETAIL_MIN_PIXELS=${DETAIL_MIN_PIXELS}"
  log_line "DETAIL_MIN_OVERLAP_RATIO=${DETAIL_MIN_OVERLAP_RATIO}"
  log_line "DETAIL_MIN_SUPPORT_RATIO=${DETAIL_MIN_SUPPORT_RATIO}"
  log_line "DETAIL_MIN_AVERAGE_OVERLAP_RATIO=${DETAIL_MIN_AVERAGE_OVERLAP_RATIO}"
  log_line "DETAIL_GRAD_SCALE=${DETAIL_GRAD_SCALE}"
  log_line "DETAIL_IMPORTANCE_THRESHOLD=${DETAIL_IMPORTANCE_THRESHOLD}"
  log_line "DETAIL_DILATION=${DETAIL_DILATION}"
  log_line "MAX_GAUSSIANS=${MAX_GAUSSIANS}"
  log_line "FREQ_LR_WEIGHT=${FREQ_LR_WEIGHT}"
  log_line "FREQ_HF_WEIGHT=${FREQ_HF_WEIGHT}"
  log_line "FREQ_RGB_WEIGHT=${FREQ_RGB_WEIGHT}"
  log_line "FREQ_TRANSITION_ITERS=${FREQ_TRANSITION_ITERS}"
  log_line "FREQ_HF_QUANTILE=${FREQ_HF_QUANTILE}"
  log_line "FREQ_HF_ABS_THRESH=${FREQ_HF_ABS_THRESH}"
  log_line "FREQ_HF_MIN_PIXELS=${FREQ_HF_MIN_PIXELS}"
  log_line "FREQ_HF_BORDER_IGNORE=${FREQ_HF_BORDER_IGNORE}"
  log_line "FREQ_CHARBONNIER_EPSILON=${FREQ_CHARBONNIER_EPSILON}"
  log_line "SAVE_ITERATIONS=${SAVE_ITERATIONS_ARRAY[*]}"
  log_line "CHECKPOINT_ITERATIONS=${CHECKPOINT_ITERATIONS_ARRAY[*]}"

  for scene in "${SELECTED_SCENES[@]}"; do
    local scene_dir="${DATA_ROOT}/${scene}"
    local model_dir="${OUTPUT_ROOT}/${scene}"
    local scene_log_dir="${LOG_ROOT}/${scene}"
    local train_log="${scene_log_dir}/01_train.txt"
    local render_log="${scene_log_dir}/02_render.txt"
    local metrics_log="${scene_log_dir}/03_metrics.txt"
    local start_checkpoint=""
    local -a checkpoint_args=()
    if [[ -n "$START_CHECKPOINT_ROOT" ]]; then
      start_checkpoint="${START_CHECKPOINT_ROOT}/${scene}/chkpnt${SWITCH_ITERATION}.pth"
      checkpoint_args=(--start_checkpoint "$start_checkpoint")
    fi

    mkdir -p "$scene_log_dir"
    check_scene_inputs "$scene" "$scene_log_dir"
    scene_big_args "$scene"
    local big_args_string="${SCENE_EXTRA_ARGS[*]}"
    log_line "Scene ${scene}: model_dir=${model_dir}"
    log_line "Scene ${scene}: big_args=${big_args_string}"
    log_line "Scene ${scene}: start_checkpoint=${start_checkpoint}"

    if [[ "$RUN_TRAIN" == "1" ]]; then
      if [[ -n "$start_checkpoint" ]]; then require_file "$start_checkpoint"; fi
      run_cmd "$train_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" OAR_JOB_ID="${RUN_NAME}_${scene}" \
        python train_two_stage_frequency.py \
          -s "$scene_dir" \
          -i "$HR_IMAGE_DIR" \
          -m "$model_dir" \
          --resolution "$RESOLUTION" \
          --data_device "$DATA_DEVICE" \
          --sr_scale "$SR_SCALE" \
          --switch_iteration "$SWITCH_ITERATION" \
          --lr_image_dir "$LR_IMAGE_DIR" \
          --hr_image_dir "$HR_IMAGE_DIR" \
          --eval \
          --iterations "$ITERATIONS" \
          --save_iterations "${SAVE_ITERATIONS_ARRAY[@]}" \
          --checkpoint_iterations "${CHECKPOINT_ITERATIONS_ARRAY[@]}" \
          --densify_from_iter "$DENSIFY_FROM_ITER" \
          --densify_until_iter "$DENSIFY_UNTIL_ITER" \
          --densification_interval "$DENSIFICATION_INTERVAL" \
          --opacity_reset_interval "$OPACITY_RESET_INTERVAL" \
          --optimizer_type "$OPTIMIZER_TYPE" \
          --detail_hf_quantile "$DETAIL_HF_QUANTILE" \
          --detail_error_quantile "$DETAIL_ERROR_QUANTILE" \
          --detail_min_visible_views "$DETAIL_MIN_VISIBLE_VIEWS" \
          --detail_min_total_pixels "$DETAIL_MIN_TOTAL_PIXELS" \
          --detail_min_views "$DETAIL_MIN_VIEWS" \
          --detail_min_pixels "$DETAIL_MIN_PIXELS" \
          --detail_min_overlap_ratio "$DETAIL_MIN_OVERLAP_RATIO" \
          --detail_min_support_ratio "$DETAIL_MIN_SUPPORT_RATIO" \
          --detail_min_average_overlap_ratio "$DETAIL_MIN_AVERAGE_OVERLAP_RATIO" \
          --detail_grad_scale "$DETAIL_GRAD_SCALE" \
          --detail_importance_threshold "$DETAIL_IMPORTANCE_THRESHOLD" \
          --detail_dilation "$DETAIL_DILATION" \
          --max_gaussians "$MAX_GAUSSIANS" \
          --freq_lr_weight "$FREQ_LR_WEIGHT" \
          --freq_hf_weight "$FREQ_HF_WEIGHT" \
          --freq_rgb_weight "$FREQ_RGB_WEIGHT" \
          --freq_transition_iters "$FREQ_TRANSITION_ITERS" \
          --freq_hf_quantile "$FREQ_HF_QUANTILE" \
          --freq_hf_abs_thresh "$FREQ_HF_ABS_THRESH" \
          --freq_hf_min_pixels "$FREQ_HF_MIN_PIXELS" \
          --freq_hf_border_ignore "$FREQ_HF_BORDER_IGNORE" \
          --freq_charbonnier_epsilon "$FREQ_CHARBONNIER_EPSILON" \
          "${checkpoint_args[@]}" \
          "${SCENE_EXTRA_ARGS[@]}" \
          "${TRAIN_EXTRA_ARGS_ARRAY[@]}"
    fi

    if [[ "$RUN_RENDER" == "1" ]]; then
      run_cmd "$render_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python render.py \
          -m "$model_dir" \
          -i "$EVAL_IMAGE_DIR" \
          --resolution "$RESOLUTION" \
          --data_device "$DATA_DEVICE" \
          --iteration "$ITERATIONS" \
          --skip_train
    fi

    if [[ "$RUN_METRICS" == "1" ]]; then
      run_cmd "$metrics_log" \
        env CUDA_VISIBLE_DEVICES="$GPU_ID" \
        python metrics.py \
          -m "$model_dir"
    fi

    append_result_row "$scene" "$train_log" "$render_log" \
      "$metrics_log" "$big_args_string" "$start_checkpoint"
  done

  log_line "Finished. Summary: ${RESULTS_TXT}"
}

main "$@"
