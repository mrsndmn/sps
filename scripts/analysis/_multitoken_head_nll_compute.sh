#!/bin/bash
#SBATCH --job-name=mtnll_compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# One (scale, variant) probe: 5e7 training tokens through frozen-backbone heads.
# Measured wall-clock at the default budget: ~45 min (XS Standard) to ~13.5 h (XL SPS /
# Delayed State, which forward the interleaved 2T sequence). Sized for the slowest task;
# lower it when submitting only the small scales.
#SBATCH --time=16:00:00
#SBATCH --gpu-bind=closest
#SBATCH --output=logs/slurm/mtnll_compute_%A_%a.out
#SBATCH --error=logs/slurm/mtnll_compute_%A_%a.err
#
# One array task = one (scale, variant), read from the exported MANIFEST (line
# SLURM_ARRAY_TASK_ID). --partition / --account / --gres are supplied by the submitter
# (PARTITION / ACCOUNT / GRES); the --gres above is a neutral fallback for a bare sbatch.
# Submitted by run_multitoken_head_nll.sh.
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

task="${SLURM_ARRAY_TASK_ID:?must run as an array task}"

# Read the task-th manifest line (1-based for sed).
line=$(sed -n "$(( task + 1 ))p" "$MANIFEST")
if [[ -z "${line}" ]]; then
  echo "No manifest line for task=${task}; exiting."
  exit 0
fi
IFS=$'\t' read -r scale variant warp batch_size spec <<<"${line}"

echo "=== task ${task}: ${scale} ${variant} (warp=${warp} bs=${batch_size}) -> ${OUTPUT_ROOT} ==="
echo "    spec: ${spec}"
uv run python -m scripts.analysis.multitoken_head_nll \
  --scale "${scale}" \
  --variant "${variant}" \
  --spec "${spec}" \
  --warp-specialize "${warp}" \
  --batch-size "${batch_size}" \
  --data-root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_ROOT}" \
  ${EXTRA_ARGS:-}
