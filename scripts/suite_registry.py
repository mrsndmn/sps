"""Suite names and worker modules used by the shared MLSpace launchers."""

LARGE_AR_SPS_SUITES = {
    'ar-sps-360m-ce', 'ar-sps-360m-kd', 'ar-sps-1.7b-ce', 'ar-sps-1.7b-kd',
    'ar-sps-qwen3-0.6b', 'ar-sps-qwen3-1.7b',
}
AR_SPS_SUITES = LARGE_AR_SPS_SUITES | {'ar-sps', 'ar-sps-kd'}
INSTRUCT_SUITE = 'ar-sps-qwen3-1.7b-instruct'
MMLU_SUITES = {'mmlu-generation', 'mmlu-stem-generation', 'mmlu-full-resume'}
EVAL_ONLY_SUITES = MMLU_SUITES | {'mtp-parallel', 'mtp-unified'}

WORKER_MODULES = {
    '1b': 'scripts.pilot_run',
    '10b': 'scripts.pilot_run',
    'mtp': 'scripts.mtp_run',
    'mtp-unfreeze': 'scripts.mtp_run',
    'mtp-parallel': 'scripts.mtp_parallel_evaluate',
    'mtp-unified': 'scripts.mtp_unified_evaluate',
    'spec-baselines': 'scripts.spec_baseline_run',
    INSTRUCT_SUITE: 'scripts.instruct_run',
    **dict.fromkeys(AR_SPS_SUITES, 'scripts.ar_sps_run'),
    **dict.fromkeys(MMLU_SUITES, 'scripts.mmlu_run'),
}
