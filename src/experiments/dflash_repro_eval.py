"""Protocol settings for dflash repro eval."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class GSM8KProtocol:
    gsm8k: str = "openai/gsm8k"
    gsm8k_revision: str = "740312add88f781978c0658806c59bc2815b9866"
    gsm8k_file: str = "main/test-00000-of-00001.parquet"
    limit: int = 2048
    warmup_tokens: int = 32


@dataclass(frozen=True, kw_only=True)
class DFlashEvalArguments(GSM8KProtocol):
    name: str
    checkpoint: Path
    result: Path
    proposals: int
    seed: int
    training_settings_sha256: str

    @property
    def width(self):
        return self.proposals + 1
