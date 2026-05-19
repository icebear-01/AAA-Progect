#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmd/elevetor_demo0317/AAA-Progect/src/emplanner/rl_dp"
MAIN_DP="${ROOT}/main_DP"
SCENARIO_DATASET="${MAIN_DP}/scenario_sets/dp_screened_scenarios_2000_mixed_complex_v2_20260408.json"
TEMP_DIR="${ROOT}/temp"
RUNS_DIR="${ROOT}/runs"
CHECKPOINT_DIR="${ROOT}/main/checkpoints"
CONDA_SH="/home/wmd/anaconda3/etc/profile.d/conda.sh"

TRAIN_TAG="dataset2000_from0_15k_env16_lookup_eval500_rt_20260418"
TRAIN_LOG="${TEMP_DIR}/train_${TRAIN_TAG}.log"
TRAIN_LOG_DIR="${RUNS_DIR}/ppo_${TRAIN_TAG}"
TRAIN_CHECKPOINT_PREFIX="${CHECKPOINT_DIR}/ppo_policy_${TRAIN_TAG}.pt"
TRAIN_FINAL_CHECKPOINT="${CHECKPOINT_DIR}/ppo_policy_${TRAIN_TAG}_update_15000.pt"
SUCCESS_PNG="${TEMP_DIR}/success_curve_${TRAIN_TAG}.png"
SUCCESS_CSV="${TEMP_DIR}/success_curve_${TRAIN_TAG}.csv"
TRAINING_CURVES_PNG="${TEMP_DIR}/training_curves_${TRAIN_TAG}.png"

mkdir -p "${TEMP_DIR}" "${RUNS_DIR}" "${CHECKPOINT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${TRAIN_LOG}"
}

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate pytorch

python - <<'PY' >> "${TRAIN_LOG}" 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable, aborting real-time training.")
print("CUDA available, proceeding with real-time success-rate training.")
PY

if [[ -f "${TRAIN_FINAL_CHECKPOINT}" ]]; then
  log "Final checkpoint already exists: ${TRAIN_FINAL_CHECKPOINT}"
else
  log "Starting 15k PPO training with per-update real-time success evaluation."
  python -u "${MAIN_DP}/ppo_trainer.py" \
    --updates 15000 \
    --log-interval 50 \
    --buffer-size 1024 \
    --num-envs 16 \
    --vector-env sync \
    --checkpoint-interval 1000 \
    --device cuda \
    --scenario-dataset "${SCENARIO_DATASET}" \
    --eval-dataset "${SCENARIO_DATASET}" \
    --eval-count 500 \
    --eval-interval 1 \
    --eval-num-envs 16 \
    --checkpoint "${TRAIN_CHECKPOINT_PREFIX}" \
    --log-dir "${TRAIN_LOG_DIR}" >> "${TRAIN_LOG}" 2>&1
  log "Training finished."
fi

log "Plotting training curves."
python "${MAIN_DP}/plot_training_curves.py" \
  --log-dir "${TRAIN_LOG_DIR}" \
  --log-file "${TRAIN_LOG}" \
  --output "${TRAINING_CURVES_PNG}" >> "${TRAIN_LOG}" 2>&1

log "Plotting real-time success-rate curve."
python "${MAIN_DP}/plot_realtime_success_curve.py" \
  --log-dir "${TRAIN_LOG_DIR}" \
  --output-png "${SUCCESS_PNG}" \
  --output-csv "${SUCCESS_CSV}" \
  --smooth-window 51 \
  --title "Real-Time Success Rate (15k PPO Updates)" \
  --label "Full safety constraint" \
  --max-update 15000 >> "${TRAIN_LOG}" 2>&1

log "Real-time success curve saved to ${SUCCESS_PNG}"
log "Real-time success samples saved to ${SUCCESS_CSV}"
