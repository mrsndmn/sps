# The State-Prediction Separation Hypothesis

Code for the paper **["The State-Prediction Separation Hypothesis"](https://arxiv.org/abs/2607.01218)**
(Giovanni Monea, Nathan Godey, Kianté Brantley, Yoav Artzi).

Transformers use the same forward computation stream both to predict the next token
and to store state that future positions read from the KV cache. We formulate the
**state–prediction separation (SPS) hypothesis**: disentangling these two roles yields
better language modeling. The **SPS Transformer** realizes the separation by inserting a
dedicated `<predict>` token after every input token, giving two interleaved streams — a
persistent *input* stream that carries state forward, and an ephemeral *prediction* stream
(kept only within a sliding window `w`) that emits next-token predictions. Across scales
from 53M to 1.68B parameters, SPS lowers FineWeb-Edu validation loss, improves held-out
generalization, and raises zero-shot accuracy at matched inference cost.

## Installation

The project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

This installs the pinned dependencies (PyTorch 2.9, Transformers, a Triton FlashAttention build,
and the EleutherAI `lm-evaluation-harness`). Prefix commands with `uv run` to use the
environment (e.g. `uv run python scripts/train.py ...`). A CUDA GPU with Triton is required
to run the fused attention kernels; CPU fallbacks exist for the tests.

## Data preparation

We pretrain on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenized with the GPT-2 tokenizer and packed across document boundaries:

```bash
# writes train/val memmap bins under <DATA_ROOT>/data
# (downloads FineWeb-Edu to <DATA_ROOT>/.hf_cache on first run)
uv run python src/data/prepare.py --data_root <DATA_ROOT>
```

## Training

Training uses typed CLI arguments. Paper recipes live in
`src/experiments/paper.py` and follow the naming convention
`{scale}_{family}[_w{window}]_{tokens}b`, where:

- **scale** ∈ `{xs, s, m, l, xl}` (53M / 131M / 379M / 831M / 1.678B parameters),
- **family** ∈ `{full_attention, sps, reverse_sps, delayed_state}`,
- **window** is the sliding `<predict>`-window size `w` (e.g. `w64`; the paper uses `w=64`),
- **tokens** is the pre-decay training budget (`10b` / `20b`).

> **Every `train.py` / `evaluate.py` command needs `--data_root <DATA_ROOT>`** — the same
> directory you used for data prep. It is required; the
> examples show it on the first command of each block, so append it to the variants too.

```bash
# STANDARD baseline (xs, 20B tokens)
uv run python scripts/train.py --recipe xs_full_attention_20b --data_root <DATA_ROOT>

# SPS (s, w=64, 20B tokens)
uv run python scripts/train.py --recipe s_sps_w64_20b

# 2X MEMORY ablation is SPS with a full window (w=4096)
uv run python scripts/train.py --recipe s_sps_w4096_20b

# DELAYED STATE and REVERSE SPS baselines
uv run python scripts/train.py --recipe s_delayed_state_w64_20b
uv run python scripts/train.py --recipe s_reverse_sps_w64_20b
```

The available model families are `standard`, `sps`, `reverse_sps`, and `delayed_state`
(`--architecture`). Run `python scripts/train.py --help` for all fields. For distributed data
parallel:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --recipe xs_full_attention_20b --data_root <DATA_ROOT>
```

### Runtime and logging overrides

- **Weights & Biases** logging is on by default and writes to your default W&B entity. Turn it
  off with `--wandb_log false`, or point it somewhere specific with
  `--wandb_entity <entity> --wandb_project <project>`.
- **Device / compile** — add `--device cpu` or `--compile false` to any command to
  run on CPU or skip `torch.compile`.

## Evaluation

Evaluation runs through the `lm-evaluation-harness` and reports FineWeb-Edu validation
loss, out-of-distribution corpus NLL (WikiText, C4, Pile-Books3, GovReport — see
`lm_eval_tasks/`), and zero-shot accuracy on ARC-Easy, HellaSwag, PIQA, SciQ, and LAMBADA:

```bash
uv run python scripts/evaluate.py --recipe s_sps_w64_20b --checkpoint final --tasks piqa --data_root <DATA_ROOT>
```

## Reproducing the paper's figures and tables

- **Figures** — `scripts/figures/plot_*.py` (e.g. `plot_validation_nll_history_wandb.py` for
  Figure 2, `plot_accuracy_gain_by_scale.py` for Figure 3, `plot_window_ablation.py` for
  Figure 4, and — for Figure 5 — `plot_gradient_params.py` (the future/present gradient-ratio
  panel) and `plot_persistent_window_nll.py` (the persistent-window Δℓ panel)).
  Figures are written to `figures/`.
- **Architecture schematics** — `plot_attention_masks.py` draws the attention mask and the
  prediction slot of each variant. It is the one figure script that needs no data and no W&B
  access: the masks follow from the architecture, so it renders from a bare clone.
- **Compute-matched comparisons** — `plot_loss_vs_compute.py` plots pre-decay validation NLL
  against *training FLOPs* with a per-architecture compute-optimal frontier, and
  `plot_validation_nll_history_wandb.py --x-axis combined-flops` renders the FLOPs-axis variant
  of the validation-history figure. Both take their FLOP counts from
  [`src/plotting/flops.py`](src/plotting/flops.py), the term-by-term (DeepMind-convention)
  per-architecture accounting used everywhere training FLOPs appear on an axis.
- **Multi-token-ahead probes** — `scripts/analysis/multitoken_head_nll.py` freezes one of your
  trained checkpoints and trains one linear LM head per look-ahead `k ∈ 1..8` per stream,
  measuring how far ahead each stream predicts. It takes the same
  `--spec kind:label:/path/to/checkpoint.pt` as the other analyses;
  `scripts/analysis/run_multitoken_head_nll.sh` submits the `(scale × variant)` grid (one GPU
  per task, resolving checkpoints under `OUT_ROOT` and skipping pairs it cannot find) and
  merges the per-run JSONs, and `scripts/figures/plot_multitoken_head_nll.py` renders the panel.
- **Tables** — `scripts/tables/export_main_results_table.py` produces the main results
  table in two forms: the compact headline (validation loss, generalization, and
  inference-efficiency ratios; `tab:main-compact`) and its per-corpus / per-benchmark
  expansion (`tab:main-full`). Both `.tex` files are written to `../sps-artifacts/outputs/tables/`.
- **Inference-efficiency benchmarks** — `scripts/benchmark/`; see
  [`scripts/benchmark/REPRODUCE.md`](scripts/benchmark/REPRODUCE.md) for the throughput /
  peak-memory measurement protocol.

### Figure data

Every figure is produced from artifacts this repository generates — nothing is downloaded.
There are two kinds of input:

- **W&B history.** The validation-history, accuracy-gain, window-ablation, seed-variance and
  loss-vs-compute figures read your own runs. They take `--entity` / `--project` and default to
  `$WANDB_ENTITY` and the project in `src/experiments/worker_args.py`, so runs trained with the
  shipped recipes resolve with no flags.
  `plot_loss_vs_compute.py` caches its pull to JSON, so re-renders need no network.
- **Local analysis outputs.** The Δ-NLL, gradient-ratio and multi-token-probe figures read
  `../sps-artifacts/outputs/…` trees written by the corresponding `scripts/analysis/run_*.sh` sweeps, which
  read the `.pt` checkpoints your training runs wrote.

Figures render with the LaTeX text engine to match the paper's Times; a host without a `latex`
binary falls back to a serif approximation instead of failing.

## Repository layout

Start with the research question in `experiments/`. Each note names the training
function and records results and conclusions. The implementation defines the runs.

| Where to look | What lives there |
| --- | --- |
| `src/experiments/paper.py`, `scripts/train.py`, `scripts/evaluate.py` | Typed recipes and entry points for direct training and evaluation. |
| `src/`, `lm_eval_tasks/` | Model implementations, data preparation, training utilities, and evaluation tasks. |
| `scripts/run_train_jobs.py`, `scripts/run_eval.py` | Shared MLSpace launchers. Use `--list` for the family catalog and `--list FAMILY` for run names. |
| `src/experiments/` | Typed experiment definitions and numeric settings. |
| `scripts/` | Family-specific command-line entry points. For example, `scripts/readout/real_readout_run.py` starts training and `scripts/readout/real_readout_mmlu.py` evaluates its checkpoints. |
| `tests/` | Experiment, worker and orchestration tests. Model tests also live under `src/modeling/tests/`. |
| `docs/paper/`, `figures/` | Manuscript sources and supporting material in the former, figures for the original SPS paper in the latter. |

### Script families

Run programs by file path from the repository root. The shared launchers
live in `scripts/`, with worker entry points grouped by experiment family.
`src/orchestration/` builds requests and `src/workloads/` implements workers.
Previously submitted jobs use the script paths in their own committed snapshots.

| Directory under `scripts/` | Contents |
| --- | --- |
| `pilot/`, `adaptation/` | Pilot data and jobs, pretrained AR-to-SPS adaptation, base training and Qwen adapters. |
| `instruction/`, `evaluation/` | Instruct training, schedules, continuation and IFEval; MMLU generation, teacher probes and paper diagnostics. |
| `speculative/`, `distillation/` | MTP training and verification, speculative baselines, Qwen drafters and DFlash experiments. |
| `readout/`, `provenance/` | Native versus SPS readout experiments and paired response-provenance controls. |
| `serving/` | Qwen3 SPS-MTP training and vLLM evaluation, draft-length sweep, runtime setup, graph benchmarks and sampling diagnostics. |
| `analysis/`, `figures/`, `tables/`, `benchmark/` | Analysis, plots, table exports and performance measurements for the original paper. |

Browse the Qwen3 vLLM probe and training runs with
`python scripts/run_train_jobs.py --list vllm-sps-qwen3`. For evaluation in the isolated
runtime use `python scripts/run_eval.py --list vllm-sps-benchmark`. The draft-length sweep
is in the `vllm-sps-draft-sweep` family. Use `scripts/generate.py` for local checkpoint smoke tests.

SPS-MTP training and its native-readout control use checkpoint metadata and
SHA-256 checksums instead of a hash of the training scripts. Existing checkpoints
can still be used in new evaluations. For archived evaluations that validate
the exact worker source, use the staged checkout from the original submission
receipt. Graph and draft-sweep reports can also be regenerated with the current
scripts, which check agreement with the recorded audit protocol.

### Following an experiment into storage

`experiments.environment.COMMON_ROOT` points to the shared `sps-artifacts/` directory beside
this checkout. Most experiment families have their own directory there. The
original `1b` pilot uses the root itself. Shared infrastructure includes `code/`
for staged worker snapshots and `hf/` for the Hugging Face cache.

For example, follow the real-readout experiment from
`experiments/qwen3-instruct-real-readout-mtp.md` to `scripts/readout/real_readout_run.py`.
Its artifact directory is `qwen3-instruct-real-readout-mtp/` under the shared root:

- `runs/out/qwen3_real_readout_mtp/` holds training outputs.
- `full-mmlu-generation/runs/out/A-nonthinking/` holds one evaluation.
  Read `manifest.json` for its inputs, `answers.jsonl` for per-question outputs,
  and `evaluated.json` for the aggregate result.
- `submissions/` under a launcher’s artifact directory holds its job receipts.

Analysis, benchmark and table scripts use the shared NFS directory
`../sps-artifacts/outputs/` directly. On another machine set `SPS_OUTPUT_ROOT` to an absolute directory before
running these commands. Per-script input and output flags still take precedence.
`run/` also contains local orchestration state, including job observations.
The worker's output paths identify the shared results used downstream.

## Tests

```bash
uv run pytest                 # fast CPU-safe tests
uv run pytest -m cuda         # GPU tests (Triton kernels)
```

## Named MLSpace experiments

Browse families and their named runs from the repository root.

```bash
python scripts/run_train_jobs.py --list
python scripts/run_train_jobs.py --list 1b
python scripts/run_train_jobs.py --experiment 1b/data-and-probes --phase prepare --dry
python scripts/run_train_jobs.py --experiment 1b/s_standard_pilot_1b --dry
python scripts/run_eval.py --experiment 1b/s_standard_pilot_1b --dry
```

Select a run by its full `FAMILY/NAME` key. Listing a family prints each name
and supported phase. Use `--phase` to select a phase explicitly. For example,
to evaluate a published DFlash checkpoint:

```bash
python scripts/run_eval.py --experiment dflash-reproduction/qwen3-4b-dflash-b16-scratch --step 5000 --dry
```

Remove `--dry` to submit. The native launcher resolves the output directory and
builds the worker command shown in the preview. An existing result or an active
reservation under `in_progress/` skips the submission. DFlash evaluation
requires a published training result. A winner evaluation can report a pending
training input during preview.

Add related runs to the family's `experiment_configs` in `src/experiments/`.
For DFlash, add an instance to the list returned by
`HardDFlash.experiment_configs()` or `SoftKD.experiment_configs()`. Set the
run's name and output directory, then override the settings that vary.
Preview the soft-KD temperature comparison with
`python scripts/run_train_jobs.py --experiment dflash-soft-kd/higher-temperature --dry`.
The registry in `src/orchestration/named/registry.py` holds one entry per family.

### Zero-window SPS reproductions

List the scales and prepare the shared FineWeb-Edu corpus once:

```bash
python scripts/run_train_jobs.py --list paper-sps
python scripts/run_train_jobs.py --experiment paper-sps/data --phase prepare
```

Preparation reuses the verified pilot memmaps when available, tokenizes the
additional pinned shards, and checks that each distributed sampler has enough
sequences for the full run. The published corpus lives under
`sps-artifacts/paper-sps-scale-reproduction/data/fineweb-edu/`.

Preview one training job after selecting a scale (`s`, `m`, or `xl`):

```bash
python scripts/run_train_jobs.py --experiment paper-sps/xl --dry
```

Omit `--dry` to submit the selected job once the corpus is ready.
Checkpoints are published under `sps-artifacts/paper-sps-scale-reproduction/out/`.

For multinode XL, submit the probe and let it finish before submitting training.

```bash
python scripts/run_train_jobs.py --experiment paper-sps/xl-4x4-probe --phase probe
python scripts/run_train_jobs.py --experiment paper-sps/xl-4x4
```

The probe records the rank layout and all-reduce timings in `interconnect.json`.
Training requires a published `probe.json` showing a speedup. It starts in a
separate output directory with freshly initialized weights. A retry can resume
from a checkpoint written by that same training run.

## MLSpace pilot artifacts

`python scripts/run_train_jobs.py --dry` and `python scripts/run_eval.py --dry` print plans without
API calls, credentials or writes. Submission requires the read-only shared `mls`
package in the existing environment. Run `python scripts/run_train_jobs.py --phase prepare`
first: this reuses pinned FineWeb-Edu cache files where present, downloads only
missing pilot shards, tokenizes them, and measures both methods with four-GPU DDP.
Inspect `prepared.json`, then `python scripts/run_train_jobs.py` submits training.
`python scripts/run_eval.py` submits the downstream tasks after training succeeds.

Each pilot config declares its final training and evaluation directories. The
launcher skips any existing destination. Before sending a job, it reserves the
path under the shared `in_progress/` tree. The worker writes partial results
there and publishes the final directory by rename. Failed attempts are
preserved under `failed/`. Use a new destination to rerun a completed phase.
A pending submission with an unknown outcome keeps its reservation until
MLSpace confirms its status. Shared input memmaps remain under `runs/data`.

### Longer training

The longer comparison uses `SPS_EXPERIMENT_SUITE=10b` with the same launchers:
`SPS_EXPERIMENT_SUITE=10b python scripts/run_train_jobs.py --phase prepare`, followed by
`SPS_EXPERIMENT_SUITE=10b python scripts/run_train_jobs.py` and, after completion,
`SPS_EXPERIMENT_SUITE=10b python scripts/run_eval.py`. Its artifacts and submission receipts
are isolated from the default pilot; the cache and staged code are shared.

## Citation

```bibtex
@article{monea2026sps,
  title   = {The State-Prediction Separation Hypothesis},
  author  = {Monea, Giovanni and Godey, Nathan and Brantley, Kiant\'e and Artzi, Yoav},
  journal = {arXiv preprint arXiv:2607.01218},
  year    = {2026},
}
```

For parallel FineWeb preparation, submit the four disjoint parts first.
After all four publish, submit the data phase to concatenate them in corpus order.

```bash
for part in part-1 part-2 part-3 part-4; do
  python scripts/run_train_jobs.py --experiment "paper-sps/$part" --phase prepare
done
python scripts/run_train_jobs.py --experiment paper-sps/data --phase prepare
```
