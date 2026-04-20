#!/usr/bin/env bash
set -euo pipefail

# Serial paper-run launcher.
#
# Usage on Linux:
#   cd /home/ubuntu/code_projects/memory_enhanced_act
#   export DATA_ROOT=/home/ubuntu/code_projects/memory_enhanced_act/data_process/data
#   bash scripts/run_paper_experiments.sh
#
# DATA_ROOT must contain task_* folders plus:
#   _phase_pca8/phase_pca8_bank.npz and per-task phase_pca8_targets.npz
#   _phase_pca16/phase_pca16_bank.npz and per-task phase_pca16_targets.npz
#   _phase_pca32/phase_pca32_bank.npz and per-task phase_pca32_targets.npz

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data_process/data}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"

LR="${LR:-1e-5}"
LR_BACKBONE="${LR_BACKBONE:-1e-6}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
RECON_LOSS_WEIGHT="${RECON_LOSS_WEIGHT:-1.0}"
PCA_COORD_LOSS_WEIGHT="${PCA_COORD_LOSS_WEIGHT:-1.0}"
RESIDUAL_LOSS_WEIGHT="${RESIDUAL_LOSS_WEIGHT:-1.0}"
QPOS_NOISE_STD="${QPOS_NOISE_STD:-2.0}"
QPOS_NOISE_CLIP="${QPOS_NOISE_CLIP:-5.0}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --num-epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --lr-backbone "${LR_BACKBONE}"
  --kl-weight "${KL_WEIGHT}"
  --recon-loss-weight "${RECON_LOSS_WEIGHT}"
  --pca-coord-loss-weight "${PCA_COORD_LOSS_WEIGHT}"
  --residual-loss-weight "${RESIDUAL_LOSS_WEIGHT}"
  --qpos-input-noise-std-pulse "${QPOS_NOISE_STD}"
  --qpos-input-noise-clip-std "${QPOS_NOISE_CLIP}"
)

run_baseline() {
  python training.py \
    --method baseline \
    --exp-name "paper_baseline_e${EPOCHS}" \
    "${COMMON_ARGS[@]}"
}

run_pca() {
  local dim="$1"
  local variant="$2"
  local exp_name="$3"
  python training.py \
    --method "${variant}" \
    --phase-pca-dim "${dim}" \
    --phase-bank-path "${DATA_ROOT}/_phase_pca${dim}/phase_pca${dim}_bank.npz" \
    --phase-targets-filename "phase_pca${dim}_targets.npz" \
    --exp-name "${exp_name}" \
    "${COMMON_ARGS[@]}"
}

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"

run_baseline

run_pca 8  pca-residual "paper_pca8res_e${EPOCHS}"
run_pca 16 pca-residual "paper_pca16res_e${EPOCHS}"
run_pca 32 pca-residual "paper_pca32res_e${EPOCHS}"

run_pca 8  pca-only "paper_pca8only_e${EPOCHS}"
run_pca 16 pca-only "paper_pca16only_e${EPOCHS}"
run_pca 32 pca-only "paper_pca32only_e${EPOCHS}"

echo "All paper experiments finished."
