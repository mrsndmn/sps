"""Generation throughput and peak memory comparison for the paper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    kind: str
    label: str
    run_name: str


@dataclass(frozen=True)
class SpeedBenchmark:
    name: str = "speed"
    seed: int = 1337
    scales: tuple = ("xs", "s", "m", "l", "xl")
    methods: tuple = (
        Method("full_attention", "Standard", "{scale}_full_attention_20b"),
        Method("sps", "2x Memory", "{scale}_sps_w4096_20b"),
        Method("delayed_state", "Delayed State", "{scale}_delayed_state_w64_20b"),
        Method("sps", "SPS", "{scale}_sps_w64_20b"),
    )
    prompt_lens: tuple = (1024,)
    num_prompts: int = 16
    max_new_tokens: int = 3072
    warmup_iters: int = 1
    timed_iters: int = 2


experiment_configs = [SpeedBenchmark()]
