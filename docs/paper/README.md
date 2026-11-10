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

The paper currently uses manually transcribed summary tables checked against
these reports; it does not yet bundle an offline evidence exporter like the
compression_horizon manuscript. Full bibliography and full benchmark tables
remain draft work. The ordered partial MMLU subset must never be relabelled as
full MMLU or a representative random sample.
