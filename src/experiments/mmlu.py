"""Pinned generative MMLU design and checkpoint provenance."""

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class FamilyConfig:
    DATASET: str = "cais/mmlu"
    REVISION: str = "c30699e8356da336a370243923dbaf21066bb9fe"
    MAX_NEW_TOKENS: int = 512
    MAX_HOURS: int = 48
    SEED: int = 1337
    SOURCES: list[tuple[str, ...]] = field(
        default_factory=lambda: [
            (
                "smol135",
                "HuggingFaceTB/SmolLM2-135M",
                "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                "ar-sps",
                "ar-sps-kd",
            ),
            (
                "smol360",
                "HuggingFaceTB/SmolLM2-360M",
                "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
                "ar-sps-360m-ce",
                "ar-sps-360m-kd",
            ),
            (
                "smol1.7",
                "HuggingFaceTB/SmolLM2-1.7B",
                "effd688a12921b4cc83e3312b6feb579f70f9c71",
                "ar-sps-1.7b-ce",
                "ar-sps-1.7b-kd",
            ),
            (
                "qwen0.6",
                "Qwen/Qwen3-0.6B-Base",
                "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                "ar-sps-qwen3-0.6b",
                "ar-sps-qwen3-0.6b",
            ),
            (
                "qwen1.7",
                "Qwen/Qwen3-1.7B-Base",
                "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
                "ar-sps-qwen3-1.7b",
                "ar-sps-qwen3-1.7b",
            ),
        ]
    )
    FULL_RESUME_EXCLUDED: frozenset[str] = frozenset({"qwen0.6_ce"})


family_config = FamilyConfig()


@dataclass(frozen=True)
class Evaluation:
    name: str
    model: str
    revision: str
    suite: str
    arm: str
    variant: str
    trainable: bool = False
    num_gpus: int = 1


def experiments(family):
    experiment_configs = []
    for (
        name,
        model,
        revision,
        ce_suite,
        kd_suite,
    ) in family_config.SOURCES:
        prefix = "qwen3" if name.startswith("qwen") else "smollm2"
        for variant, suite, arm in [
            ("base", ce_suite, prefix + "_original_ar"),
            (
                "ce",
                ce_suite,
                prefix + ("_staged_sps_ce" if prefix == "qwen3" else "_staged_sps"),
            ),
            ("kd", kd_suite, prefix + "_staged_sps_kd"),
        ]:
            experiment_configs.append(
                Evaluation(f"{name}_{variant}", model, revision, suite, arm, variant)
            )

    # Full-MMLU continuation excludes the slow CE arm explicitly requested by the user.
    if family == "mmlu-full-resume":
        experiment_configs = [
            exp
            for exp in experiment_configs
            if exp.name not in family_config.FULL_RESUME_EXCLUDED
        ]
    return experiment_configs
