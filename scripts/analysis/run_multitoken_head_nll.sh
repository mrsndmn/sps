#!/bin/bash
# Submitter + merger for the multi-token-ahead linear-probe NLL analysis.
#
# For each (scale, variant) one single-GPU array task freezes a locally trained checkpoint
# and trains per-offset LM heads (k=1..K) on each stream, writing
# <OUTPUT_ROOT>/<scale>/<variant>.json (see multitoken_head_nll.py). `merge` folds those
# per-variant files into <scale>/summary.json for the figure and prints an eyeball table.
#
# Checkpoints are resolved the same way run_persistent_window_nll.sh resolves them:
# <OUT_ROOT>/<scale>_<family>_<budget>/<CKPT_FILE>, i.e. the .pt files your own training runs
# wrote. Pairs with no checkpoint on disk are skipped with a message.
#
# Usage (run directly; it submits the jobs):
#   PARTITION=gpu DATA_ROOT=/path/to/data bash scripts/analysis/run_multitoken_head_nll.sh submit
#   SCALES=s PARTITION=gpu DATA_ROOT=... bash scripts/analysis/run_multitoken_head_nll.sh submit
#   bash scripts/analysis/run_multitoken_head_nll.sh merge     # after the array finishes
#
# Then render the figure (plain CPU, no scheduler needed):
#   uv run python scripts/figures/plot_multitoken_head_nll.py
#
# Overridable via env: SCALES, VARIANTS, OUTPUT_ROOT, OUT_ROOT (checkpoint root), CKPT_FILE,
# DATA_ROOT, DATASET, TRAIN_TOKENS, EVAL_TOKENS, EVAL_EVERY_TOKENS, SEQLEN, LR, K_MAX,
# BATCH_SIZE (global micro-batch override) / BATCH_<SCALE>, ARRAY_CONCURRENCY, PARTITION,
# ACCOUNT, GRES.

set -euo pipefail

MODE="${1:-submit}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/slurm

SCALES="${SCALES:-xs s m l xl}"
VARIANTS="${VARIANTS:-full-attention sps-w64 delayed-state-w64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/multitoken_head_nll}"

# Where the tokenized corpus lives: <DATA_ROOT>/data/<DATASET>/{train,val}.bin, i.e. the
# same DATA_ROOT used for `src/data/prepare.py` and every train.py / evaluate.py command.
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR}"
DATASET="${DATASET:-fineweb-edu}"

merge_all() {
  # Trivial CPU (reads a handful of small JSON files); safe to run inline.
  uv run python -m scripts.analysis.multitoken_head_nll_io \
    --root "$OUTPUT_ROOT" --scales $SCALES --variants $VARIANTS
}

if [[ "$MODE" == "merge" ]]; then
  merge_all
  exit 0
fi

if [[ "$MODE" != "submit" ]]; then
  echo "Usage: $0 {submit|merge}  (env: SCALES, VARIANTS, OUTPUT_ROOT, DATA_ROOT, PARTITION, ...)"
  exit 1
fi

# ---- SLURM (cluster-specific; override via env, like run_persistent_window_nll.sh) ----
PARTITION="${PARTITION:?Set PARTITION to your GPU partition (e.g. PARTITION=gpu)}"
ACCOUNT="${ACCOUNT:-}"                 # SLURM account; blank => omit --account
GRES="${GRES:-}"                       # gres type, e.g. nvidia_h100_80gb_hbm3; blank => gpu:1
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"

GPU_ARGS=(); if [[ -n "$GRES" ]]; then GPU_ARGS=(--gres=gpu:"$GRES":1); else GPU_ARGS=(--gres=gpu:1); fi
ACCT_ARGS=(); [[ -n "$ACCOUNT" ]] && ACCT_ARGS=(--account="$ACCOUNT")

# Optional probe-budget overrides forwarded to multitoken_head_nll.py (unset => the
# script's own defaults: 5e7 train tokens, 8e6 eval, seqlen 1024, k=1..8, lr 5e-4).
EXTRA_ARGS=()
[[ -n "${TRAIN_TOKENS:-}" ]]      && EXTRA_ARGS+=(--train-tokens "${TRAIN_TOKENS}")
[[ -n "${EVAL_TOKENS:-}" ]]       && EXTRA_ARGS+=(--eval-tokens "${EVAL_TOKENS}")
[[ -n "${EVAL_EVERY_TOKENS:-}" ]] && EXTRA_ARGS+=(--eval-every-tokens "${EVAL_EVERY_TOKENS}")
[[ -n "${SEQLEN:-}" ]]            && EXTRA_ARGS+=(--seqlen "${SEQLEN}")
[[ -n "${K_MAX:-}" ]]             && EXTRA_ARGS+=(--k-max "${K_MAX}")
[[ -n "${LR:-}" ]]                && EXTRA_ARGS+=(--lr "${LR}")

# Per-scale micro-batch (seqlen 1024); known-safe on 80GB. A non-empty global BATCH_SIZE
# overrides all; per-scale BATCH_<SCALE> overrides one.
batch_for_scale() {
  if [[ -n "${BATCH_SIZE:-}" ]]; then echo "${BATCH_SIZE}"; return; fi
  case "$1" in
    xs) echo "${BATCH_XS:-24}" ;;
    s)  echo "${BATCH_S:-16}" ;;
    m)  echo "${BATCH_M:-8}" ;;
    l)  echo "${BATCH_L:-6}" ;;
    xl) echo "${BATCH_XL:-4}" ;;
    *)  echo "8" ;;
  esac
}

# variant -> (canonical kind, label, training-run directory suffix). The directory suffix is
# the experiment name your training run used, so checkpoints land where train.py put them.
variant_kind()  { case "$1" in full-attention) echo "full_attention";; sps-w64|sps-w4096) echo "sps";; delayed-state-w64) echo "delayed_state";; *) echo "";; esac; }
variant_label() { case "$1" in full-attention) echo "Standard";; sps-w64) echo "SPS w64";; sps-w4096) echo "2x Memory";; delayed-state-w64) echo "Delayed State w64";; *) echo "$1";; esac; }
variant_dir()   { case "$1" in full-attention) echo "full_attention";; sps-w64) echo "sps_w64";; sps-w4096) echo "sps_w4096";; delayed-state-w64) echo "delayed_state_w64";; *) echo "$1";; esac; }

# Where the training runs wrote their checkpoints, mirroring run_persistent_window_nll.sh.
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs}"
CKPT_FILE="${CKPT_FILE:-ckpt_tokens_20000145408_final.pt}"
BUDGET="${BUDGET:-20b}"

# Build the task manifest: one line per (scale, variant) that has a checkpoint on disk.
#   scale<TAB>variant<TAB>warp<TAB>batch_size<TAB>spec
MANIFEST="logs/slurm/mtnll_manifest_$(date +%s).tsv"
: > "$MANIFEST"
for scale in $SCALES; do
  bs="$(batch_for_scale "${scale}")"
  # XL's Triton kernels need warp specialization disabled (memory); others keep it.
  warp="${WARP_SPECIALIZE:-}"
  if [[ -z "${warp}" ]]; then
    if [[ "${scale}" == "xl" ]]; then warp="off"; else warp="keep"; fi
  fi
  for variant in $VARIANTS; do
    kind="$(variant_kind "${variant}")"
    if [[ -z "${kind}" ]]; then
      echo "SKIP ${scale}/${variant}: unknown variant"
      continue
    fi
    ckpt="${OUT_ROOT}/${scale}_$(variant_dir "${variant}")_${BUDGET}/${CKPT_FILE}"
    if [[ ! -f "${ckpt}" ]]; then
      echo "SKIP ${scale}/${variant}: checkpoint not found at ${ckpt}"
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${scale}" "${variant}" "${warp}" "${bs}" \
      "${kind}:$(variant_label "${variant}"):${ckpt}" >> "$MANIFEST"
  done
done

NTASKS="$(grep -c . "$MANIFEST" || true)"
if [[ "${NTASKS}" -eq 0 ]]; then
  echo "No (scale, variant) pairs with checkpoints found; nothing to submit."
  exit 1
fi
echo "Manifest: ${MANIFEST} (${NTASKS} tasks)"

export MANIFEST OUTPUT_ROOT DATA_ROOT DATASET
export EXTRA_ARGS="${EXTRA_ARGS[*]:-}"

compute_jid=$(sbatch --parsable \
  --partition="$PARTITION" "${ACCT_ARGS[@]}" "${GPU_ARGS[@]}" \
  --array=0-$((NTASKS - 1))%"${ARRAY_CONCURRENCY}" \
  --export=ALL \
  scripts/analysis/_multitoken_head_nll_compute.sh)
echo "Compute array ($PARTITION, %${ARRAY_CONCURRENCY}): ${compute_jid}"
echo "Outputs -> ${OUTPUT_ROOT}"
echo "When it finishes: bash scripts/analysis/run_multitoken_head_nll.sh merge"
