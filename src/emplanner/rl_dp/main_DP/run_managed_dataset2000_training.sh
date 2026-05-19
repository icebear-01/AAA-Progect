#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/wmd/elevetor_demo0317/AAA-Progect/src/emplanner/rl_dp"
MAIN_DP="${ROOT}/main_DP"
SCENARIO_DATASET="${MAIN_DP}/scenario_sets/dp_screened_scenarios_2000_mixed_complex_v2_20260408.json"
TEMP_DIR="${ROOT}/temp"
RUNS_DIR="${ROOT}/runs"
CHECKPOINT_DIR="${ROOT}/main/checkpoints"
CONDA_SH="/home/wmd/anaconda3/etc/profile.d/conda.sh"

CLASSIC_9X23_JSON="${TEMP_DIR}/classic_dp_only_training2000_9x23_20260410.json"
CLASSIC_9X7_JSON="${TEMP_DIR}/classic_dp_only_training2000_9x7_20260410.json"

TRAIN_TAG="dataset2000_from0_20k_env16_lookup_20260410"
TRAIN_LOG="${TEMP_DIR}/train_${TRAIN_TAG}.log"
TRAIN_LOG_DIR="${RUNS_DIR}/ppo_${TRAIN_TAG}"
TRAIN_CHECKPOINT_PREFIX="${CHECKPOINT_DIR}/ppo_policy_${TRAIN_TAG}.pt"
TRAIN_FINAL_CHECKPOINT="${CHECKPOINT_DIR}/ppo_policy_${TRAIN_TAG}_update_20000.pt"
CURVE_PNG="${TEMP_DIR}/training_curves_${TRAIN_TAG}.png"

mkdir -p "${TEMP_DIR}" "${RUNS_DIR}" "${CHECKPOINT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${TRAIN_LOG}"
}

wait_for_file() {
  local target="$1"
  while [[ ! -f "${target}" ]]; do
    sleep 30
  done
}

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found at ${CONDA_SH}" >&2
  exit 1
fi

log "Waiting for classic DP success experiments to finish."
wait_for_file "${CLASSIC_9X23_JSON}"
wait_for_file "${CLASSIC_9X7_JSON}"
log "Classic DP success experiments finished."

source "${CONDA_SH}"
conda activate pytorch

python - <<'PY' >> "${TRAIN_LOG}" 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable, aborting managed training.")
print("CUDA available, proceeding with training.")
PY

if [[ -f "${TRAIN_FINAL_CHECKPOINT}" ]]; then
  log "Final checkpoint already exists: ${TRAIN_FINAL_CHECKPOINT}"
else
  log "Starting PPO training from scratch on the 2000-scenario dataset."
  python -u "${MAIN_DP}/ppo_trainer.py" \
    --updates 20000 \
    --log-interval 10 \
    --buffer-size 1024 \
    --num-envs 16 \
    --vector-env sync \
    --checkpoint-interval 1000 \
    --device cuda \
    --scenario-dataset "${SCENARIO_DATASET}" \
    --checkpoint "${TRAIN_CHECKPOINT_PREFIX}" \
    --log-dir "${TRAIN_LOG_DIR}" >> "${TRAIN_LOG}" 2>&1
  log "Training finished."
fi

log "Plotting training curves."
python "${MAIN_DP}/plot_training_curves.py" \
  --log-dir "${TRAIN_LOG_DIR}" \
  --log-file "${TRAIN_LOG}" \
  --output "${CURVE_PNG}" >> "${TRAIN_LOG}" 2>&1
log "Training curves saved to ${CURVE_PNG}"
