"""Pinned generative MMLU design and checkpoint provenance."""
from dataclasses import dataclass
DATASET = 'cais/mmlu'
REVISION = 'c30699e8356da336a370243923dbaf21066bb9fe'
MAX_NEW_TOKENS = 512
MAX_HOURS = 48
SEED = 1337

@dataclass(frozen=True)
class Evaluation:
    name: str
    model: str
    revision: str
    suite: str
    arm: str
    signature: str
    variant: str
    trainable: bool = False
    num_gpus: int = 1

# Signatures belong to completed source training, never to the new evaluator.
SOURCES = [
 ('smol135', 'HuggingFaceTB/SmolLM2-135M', '93efa2f097d58c2a74874c7e644dbc9b0cee75a2', 'ar-sps', 'e9ed3e07768d5fb918301f5f23af9aa3020740a9578cb18f19543365a228a04d', 'ar-sps-kd', '33982b4d1a76244371649df72a77d0c0913f923c4b069fc5bef5e715f61d536a'),
 ('smol360', 'HuggingFaceTB/SmolLM2-360M', 'f8027fd0eaeea54caa13c31d31b9fdc459c38b49', 'ar-sps-360m-ce', 'a91952db7534868c2d23050e447080d6682191ddb1df4ddd398c1db9bcbf8657', 'ar-sps-360m-kd', 'fc3e05940bc95ebef71e3ce31b8b34a939437f51bf0a506bd9ebd63a59126573'),
 ('smol1.7', 'HuggingFaceTB/SmolLM2-1.7B', 'effd688a12921b4cc83e3312b6feb579f70f9c71', 'ar-sps-1.7b-ce', '555846063de2a68b13a7efa818d5330be55fbf90074453ee5736ba4ad11df82a', 'ar-sps-1.7b-kd', 'a89156378af4954f8d5f18840f0130cb8c3f6af6621fa0f7f0c58a15413acf19'),
 ('qwen0.6', 'Qwen/Qwen3-0.6B-Base', 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd', 'ar-sps-qwen3-0.6b', 'cdd252245886a26acbb339fbb8754c1491de1e5b1348ac26f35adc8d47590f40', 'ar-sps-qwen3-0.6b', 'cdd252245886a26acbb339fbb8754c1491de1e5b1348ac26f35adc8d47590f40'),
 ('qwen1.7', 'Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1', 'ar-sps-qwen3-1.7b', '7dad73e5302bf23ef6bf64cb2b1ce9d7aa6633e7213b2db92fa79bb43d14b1c3', 'ar-sps-qwen3-1.7b', '7dad73e5302bf23ef6bf64cb2b1ce9d7aa6633e7213b2db92fa79bb43d14b1c3'),
]
experiment_configs = []
for name, model, revision, ce_suite, ce_sig, kd_suite, kd_sig in SOURCES:
    prefix = 'qwen3' if name.startswith('qwen') else 'smollm2'
    for variant, suite, signature, arm in [
        ('base', ce_suite, ce_sig, prefix + '_original_ar'),
        ('ce', ce_suite, ce_sig, prefix + ('_staged_sps_ce' if prefix == 'qwen3' else '_staged_sps')),
        ('kd', kd_suite, kd_sig, prefix + '_staged_sps_kd'),
    ]:
        experiment_configs.append(Evaluation(f'{name}_{variant}', model, revision, suite, arm, signature, variant))

# Full-MMLU continuation excludes the slow CE arm explicitly requested by the user.
FULL_RESUME_EXCLUDED = frozenset({'qwen0.6_ce'})
import os
if os.environ.get('SPS_EXPERIMENT_SUITE') == 'mmlu-full-resume':
    experiment_configs = [exp for exp in experiment_configs if exp.name not in FULL_RESUME_EXCLUDED]
