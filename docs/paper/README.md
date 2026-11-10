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

- IFEval: `qwen3-instruct-ifeval/runs/out/*/evaluated.json`.
- Ablations: `qwen3-instruct-{step384,cosine,extended-2048}/runs/out/`.
- Partial MMLU: `reports/instruct-mmlu-partial-20261109/report.json` and `metrics.csv`;
  accompanying `aggregate.py` records intersection checks.
- SPS-MTP eager serving: `vllm-sps-qwen3/reports/{results.json,summary.csv}`.
- Target decode CUDA graphs: `vllm-sps-qwen3-cudagraphs/reports/{results.json,summary.csv,eager-vs-graphs.csv}`.
- Serving per-arm evidence: each serving root's `runs/out/<arm>-<mode>/` contains
  `manifest.json`, `evaluated.json`, `samples.jsonl` and serving provenance;
  graph arms additionally contain `cudagraph.json` with observed target replay.
- SPS-MTP training/export: `vllm-sps-qwen3/runs/out/qwen3_sps_mtp/` contains
  `trained.json`, `metrics.jsonl` and the final weights/export. AR/EAGLE use the
  original Qwen verifier; SPS/SPS-MTP share this final, jointly adapted verifier.

Serving definitions live in `scripts/vllm_sps_config.py`, `instruct_config.py`
(sampling), `vllm_sps_train.py`, `vllm_sps_evaluate.py`, `serving_benchmark.py`
and `serving_graphs.py`. The report formula is in `vllm_sps_report.py`.
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
