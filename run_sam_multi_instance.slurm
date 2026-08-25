#!/bin/bash
#SBATCH --job-name=sam_pipeline
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=sam_multi_instance_%j.out
#SBATCH --error=sam_multi_instance_%j.err

set -euo pipefail

echo "=== JOB START ==="
date
hostname

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi
if [[ -n "${SAM_REPOS:-}" ]]; then
  export PYTHONPATH="${SAM_REPOS}:${PYTHONPATH:-}"
fi
cd "${REPO_DIR}"
mkdir -p logs

# Change only this variable when submitting:
# sbatch --export=ALL,SAM_MODEL=medsam run_sam_multi_instance.slurm
SAM_MODEL="${SAM_MODEL:-sam}"

case "${SAM_MODEL}" in
  sam)
    export SAM_MODEL_TYPE="vit_b"
    ;;
  medsam)
    export SAM_MODEL_TYPE="vit_b"
    ;;
  sam2)
    export SAM_MODEL_TYPE="vit_b"
    export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_l}"
    ;;
  medsam2)
    export SAM_MODEL_TYPE="vit_b"
    export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1_hiera_t512}"
    ;;
  *)
    echo "ERROR: Unsupported SAM_MODEL='${SAM_MODEL}'"
    echo "Valid: sam | medsam | sam2 | medsam2"
    exit 1
    ;;
esac

: "${SAM_CHECKPOINT:?Set SAM_CHECKPOINT to the selected model checkpoint}"
: "${BASE_DIR:?Set BASE_DIR to the subject-data root}"

export OUTPUT_DIR="${OUTPUT_DIR:-sam_predictions_${SAM_MODEL}}"
export START_SLICE="${START_SLICE:-}"
export END_SLICE="${END_SLICE:-}"

echo "=== CONFIG ==="
echo "SAM_MODEL=${SAM_MODEL}"
echo "SAM_MODEL_TYPE=${SAM_MODEL_TYPE}"
echo "SAM_CHECKPOINT=${SAM_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "START_SLICE=${START_SLICE}"
echo "END_SLICE=${END_SLICE}"
echo "============="

python -u sam_multi_instance.py

echo "=== JOB END ==="
date
