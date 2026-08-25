#!/usr/bin/env bash
#SBATCH --job-name=sam_slice
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=sam_slice_%j.out
#SBATCH --error=sam_slice_%j.err

set -euo pipefail

echo "=== JOB START ==="
date
hostname

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi
cd "${REPO_DIR}"
mkdir -p logs

if [[ -n "${SAM_REPOS:-}" ]]; then
  export PYTHONPATH="${SAM_REPOS}:${PYTHONPATH:-}"
fi

# Override at submission time, for example:
# sbatch --export=ALL,SAM_MODEL=medsam,START_SLICE=12,END_SLICE=30 run_sam_slice.slurm
export SAM_MODEL="${SAM_MODEL:-sam}"
export START_SLICE="${START_SLICE:-}"
export END_SLICE="${END_SLICE:-}"

case "${SAM_MODEL}" in
  sam)
    ;;
  medsam)
    ;;
  sam2)
    export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l}"
    ;;
  medsam2)
    export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_t512}"
    ;;
  *)
    echo "Unsupported SAM_MODEL='${SAM_MODEL}'. Expected sam, medsam, sam2, or medsam2." >&2
    exit 2
    ;;
esac

: "${SAM_CHECKPOINT:?Set SAM_CHECKPOINT to the selected model checkpoint}"
: "${BASE_DIR:?Set BASE_DIR to the subject-data root}"

echo "SAM_MODEL=${SAM_MODEL} SAM_CHECKPOINT=${SAM_CHECKPOINT} START_SLICE=${START_SLICE:-all} END_SLICE=${END_SLICE:-all}"

python -u sam_slice_inference.py

echo "=== JOB END ==="
date
