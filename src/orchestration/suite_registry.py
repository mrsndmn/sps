"""Suite names and worker paths used by the shared MLSpace launchers."""

LARGE_AR_SPS_SUITES = {
    'ar-sps-360m-ce', 'ar-sps-360m-kd', 'ar-sps-1.7b-ce', 'ar-sps-1.7b-kd',
    'ar-sps-qwen3-0.6b', 'ar-sps-qwen3-1.7b',
}
AR_SPS_SUITES = LARGE_AR_SPS_SUITES | {'ar-sps', 'ar-sps-kd'}
INSTRUCT_SUITE = 'ar-sps-qwen3-1.7b-instruct'
MMLU_SUITES = {'mmlu-generation', 'mmlu-stem-generation', 'mmlu-full-resume'}
EVAL_ONLY_SUITES = MMLU_SUITES | {'mtp-parallel', 'mtp-unified'}

WORKER_SCRIPTS = {
    '1b': 'scripts/pilot/pilot_run.py',
    '10b': 'scripts/pilot/pilot_run.py',
    'mtp': 'scripts/speculative/mtp_run.py',
    'mtp-unfreeze': 'scripts/speculative/mtp_run.py',
    'mtp-parallel': 'scripts/speculative/mtp_parallel_evaluate.py',
    'mtp-unified': 'scripts/speculative/mtp_unified_evaluate.py',
    'spec-baselines': 'scripts/speculative/spec_baseline_run.py',
    INSTRUCT_SUITE: 'scripts/instruction/instruct_run.py',
    **dict.fromkeys(AR_SPS_SUITES, 'scripts/adaptation/ar_sps_run.py'),
    **dict.fromkeys(MMLU_SUITES, 'scripts/evaluation/mmlu_run.py'),
}
