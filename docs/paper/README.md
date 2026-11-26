# SPS adaptation — short ARR working draft

Edit `main.tex` for the short English body, `appendix.tex` for supporting detail.
`results-draft-ru.md` is a short Russian synopsis; `appendix-notes-ru.md` preserves
the earlier detailed notes. This is an internal draft, not a submission-ready paper.

## Format and venue

Unmodified official ACL style, anonymous `review` option; vendored revision in
`styles/UPSTREAM.md`. No spacing, margin or font-size overrides.
Official template: https://github.com/acl-org/acl-style-files
ACL 2027 ARR deadline: **4 January 2027, 23:59 AoE**; commitment date remains TBD
in the first call, checked 10 November 2026:
https://www.aclweb.org/portal/content/acl-2027-first-call-papers

## Local build

Following `../compression_horizon/sps_paper`, use latexmk/pdflatex, a local
`.latexmkrc` and ignored `build/` output:

```sh
make -C docs/paper
```

PDF: `docs/paper/build/main.pdf`. Requires latexmk, pdflatex, BibTeX and the
packages required by the ACL style (available in the current environment).
Build needs no GPU, model weights or network. `make -C docs/paper clean` removes
LaTeX outputs. Do not edit vendored style files to squeeze the text.

## Evidence

Numbers are measured outputs, not new experiment definitions. Configurations
remain in the Python registries. Base results are recorded in
`experiments/pretrained-ar-to-sps*.md` and `experiments/pretrained-qwen3-to-sps.md`.
External artifacts root:
`/mnt/virtual_ai0001053-00054_SR004-nfs2/d.tarasov/sps-artifacts`.

- Paired response replacement: `instruct-provenance-evaluation/reports/nonthinking.json`
  contains validated paired contrasts, bootstrap intervals and crossed held-out NLL;
  `instruct-provenance-evaluation/reports/thinking.json` contains the completed
  thinking contrasts. `python -m scripts.provenance.provenance_report --mode nonthinking|thinking`
  reproduces each report from saved answers. The contrast does not match token
  exposure or answer length.
- IFEval: `qwen3-instruct-ifeval/runs/out/*/evaluated.json`.
- Ablations: `qwen3-instruct-{step384,cosine,extended-2048}/runs/out/`.
- Partial MMLU: `reports/instruct-mmlu-partial-20261109/report.json` and `metrics.csv`;
  accompanying `aggregate.py` records intersection checks.
- SPS-MTP eager serving: `vllm-sps-qwen3/reports/{results.json,summary.csv}`.
- Target decode CUDA graphs: `vllm-sps-qwen3-cudagraphs/reports/{results.json,summary.csv,eager-vs-graphs.csv}`.
- Serving per-arm evidence: each serving root's `runs/out/<arm>-<mode>/` contains
  `manifest.json`, `evaluated.json`, `samples.jsonl` and serving provenance;
  graph arms additionally contain `cudagraph.json` with observed target replay.
- Expanded correctness: `sps-sampling-audit/runs/out/kernel-and-cache/report.json`
  records the failed greedy gate; `sps-sampling-trace/runs/out/divergence-distributions/`
  holds full distributions. Derived stable-tie comparisons and head controls are in
  `sps-sampling-precision/runs/out/matched-slots-and-head/report.json`.
  Pre-sampler observations are in `sps-sampling-cache-trace/runs/out/target-logits/`;
  the FP16 control is in `sps-sampling-backbone-precision/runs/out/fp16-backbone/`.
  Diagnostic `passed` markers indicate completed collection, not passage of the
  original sampling gate. Designs live in the corresponding `scripts/serving/sampling_*.py`.
- Qwen3-4B fixed-weight DFlash mask ablation:
  `dflash-mask-ablation/runs/out/277ae900c11f7e9a/evaluated.json` contains
  paired outputs, greedy references, acceptance counts and timing. Run
  `python -m scripts.dflash_mask_ablation --report` to print the summary.
  The one-GPU worker uses the Qwen3-4B and DFlash-b16 snapshots pinned by the
  script from the shared Hugging Face cache.
- Qwen3-4B trained draft mask and embedding comparison
  `qwen4b-drafter/b0088cb54b3bd646/runs/out/eval-5b067a4174bd18c8/evaluated.json`
  contains all timed outputs, per-position acceptance and exact-output throughput
  comparisons. Run `python -m scripts.qwen4b_drafter --dataset smoltalk-2048 --phase eval --report`
  to reproduce the summary. The four sibling run directories contain the
  trained weights and their training reports.
- DFlash-b16 scratch reproduction on GSM8K
  `dflash-repro/ec224ca5303b0f5b/runs/out/gsm8k-step-*-780d2d2a6375/evaluated.json`
  and `dflash-kd/45d1905c50658e45/runs/out/gsm8k-step-*-5938c51ca9ec/evaluated.json`
  contain hard and soft results for the three checkpoints, per-position
  acceptance, matched-output comparisons and raw timings. Use
  `python -m scripts.dflash_repro --phase eval --step <step> --report`
  and `python -m scripts.dflash_kd --phase eval --step <step> --report`.
  Training cost and job provenance are recorded in
  `experiments/dflash-reproduction.md` and `experiments/dflash-soft-kd.md`.
- SPS-MTP training/export: `vllm-sps-qwen3/runs/out/qwen3_sps_mtp/` contains
  `trained.json`, `metrics.jsonl` and the final weights/export. AR/EAGLE use the
  original Qwen verifier; SPS/SPS-MTP share this final, jointly adapted verifier.

Serving definitions live in `scripts/serving/vllm_sps_config.py`, `instruct_config.py`
(sampling), `scripts/serving/vllm_sps_train.py`, `scripts/serving/vllm_sps_evaluate.py`, `scripts/serving/serving_benchmark.py`
and `scripts/serving/serving_graphs.py`. The report formula is in `scripts/serving/vllm_sps_report.py`.
The 100-question STEM pilot is separate from the ordered partial-MMLU analysis.
Its three timing repeats reuse the same seeds, not independent quality samples.
No training or evaluation job is needed to rebuild the paper.

Pinned provenance for the completed serving runs:

| Input | Revision / SHA-256 |
| --- | --- |
| Original `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `AngelSlim/Qwen3-1.7B_eagle3` | `94441b48acc5804677ae12259617c83323b543a9` |
| MMLU dataset | `c30699e8356da336a370243923dbaf21066bb9fe` |
| vLLM upstream wheel | `3fe0d44219471ee5b178036a953b6f0c0c2bf5f6` |
| vLLM eager implementation | `65ff372c9f6fbdeebfbde635449160af95ba2de9` |
| vLLM graph implementation | `31ba03a874f775372154f0edf3e17d436f2d7462` |
| SPS-MTP final checkpoint SHA-256 | `31f9aa8510baf3642ff7bfbacbf07c79e74276bb686d8c44cf26477bffe3be4f` |

On main, the graph `check_gate` / `completed` guards were checked against all
eight saved arms before transcribing results. They verify the checkpoint,
runtime/source identity, protocol and positive target replay evidence. The
reported graph scope excludes Torch compilation and prefill capture; target
replay evidence does not establish graph capture of the EAGLE drafter.

The paper currently uses manually transcribed summary tables checked against
these reports; it does not yet bundle an offline evidence exporter like the
compression_horizon manuscript. Full bibliography and full benchmark tables
remain draft work. The ordered partial MMLU subset must never be relabelled as
full MMLU or a representative random sample. The serving pilot must not be
presented as a same-verifier comparison of SPS-MTP and EAGLE-3 or as evidence of
quality-preserving speculation.
