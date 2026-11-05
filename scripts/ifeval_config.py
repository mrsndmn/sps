"""Parallel IFEval design; reuse the existing instruct experiment registry."""
SOURCE_SUITE = 'ar-sps-qwen3-1.7b-instruct'
SOURCE_SIGNATURE = '761505551994ceec0a907401b3ee875ffea744b69bc94c7007b23c58181b466b'
ARTIFACT_NAME = 'qwen3-instruct-ifeval'
MODES = ('nonthinking', 'thinking')
EVAL_HOURS = 12
STARTUP_SECONDS = 300
GPU_HOURS = 125
LEGACY_SIGNATURE = '5336b6a2fc6e9379e4368cd3314a62555546a4156ea2484c19977772335f5896'
