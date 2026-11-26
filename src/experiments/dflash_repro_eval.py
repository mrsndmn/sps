"""Protocol settings for dflash repro eval."""

from dataclasses import dataclass


@dataclass(kw_only=True)
class FamilyConfig:
    GSM8K: str = "openai/gsm8k"
    GSM8K_REVISION: str = "740312add88f781978c0658806c59bc2815b9866"
    GSM8K_FILE: str = "main/test-00000-of-00001.parquet"
    LIMIT: int = 2048
    WARMUP_TOKENS: int = 32


family_config = FamilyConfig()
