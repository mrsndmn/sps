"""Protocol settings for qwen4b drafter data."""

from dataclasses import dataclass

from .family_arguments import configure, export_fields


@dataclass(frozen=True, kw_only=True)
class FamilyConfig:
    REPO: str = "jiamingshan/qwen3-4b-dflash-official100k-prepared"
    REVISION: str = "948e343b5ccc06e21794d195f2a34872ed180b9b"
    SOURCE_ROWS: int = 100_000
    MMLU_TEST_FILE: str = "all/test-00000-of-00001.parquet"
    MIN_LOSS_TOKENS: int = 128


family_config = configure(FamilyConfig)
__getattr__ = export_fields(family_config)
