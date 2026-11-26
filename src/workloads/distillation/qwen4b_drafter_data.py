"""Select training and validation rows from the prepared DFlash corpus."""

from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import random

from experiments.environment import HF_HOME
from workloads.distillation import dflash_mask_ablation as reference
from experiments import instruct_config
from data import instruct_data
from workloads.evaluation.report_mmlu_stem import STEM_SUBJECTS

from experiments.qwen4b_drafter_train import family_config as qwen4b_drafter_train_settings
NAME = qwen4b_drafter_train_settings.PREPARED_DATASET
from experiments.qwen4b_drafter_data import family_config as qwen4b_drafter_data_settings
REPO = qwen4b_drafter_data_settings.REPO
REVISION = qwen4b_drafter_data_settings.REVISION
SOURCE_ROWS = qwen4b_drafter_data_settings.SOURCE_ROWS
MMLU_TEST_FILE = qwen4b_drafter_data_settings.MMLU_TEST_FILE
MIN_LOSS_TOKENS = qwen4b_drafter_data_settings.MIN_LOSS_TOKENS


def open_dataset(repo=REPO, revision=REVISION, source_rows=SOURCE_ROWS):
    import pyarrow as pa
    from datasets import Dataset
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo, repo_type="dataset", revision=revision, cache_dir=HF_HOME / "hub",
        local_files_only=True,
    )
    # The snapshot declares a newer HF List feature than this environment.
    # Re-infer features from the native Arrow column types.
    folder = Path(path)
    state = json.loads((folder / "state.json").read_text())
    tables = []
    for entry in state["_data_files"]:
        with pa.memory_map(str(folder / entry["filename"])) as source:
            tables.append(pa.ipc.open_stream(source).read_all().replace_schema_metadata(None))
    rows = Dataset(pa.concat_tables(tables))
    if len(rows) != source_rows or not {"input_ids", "loss_mask", "seq_len"} <= set(rows.column_names):
        raise ValueError("Unexpected prepared DFlash dataset")
    return rows


def select_rows(rows, tokenizer, overlap, *, context, horizon, validation, accumulation, seed):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    train, val = [], []
    val_groups = set()
    rejected = dict(length=0, min_loss_tokens=0, anchors=0, overlap=0,
                    held_out_prompt=0, incomplete_batch=0)
    for index in order:
        row = rows[index]
        ids, mask = row["input_ids"], row["loss_mask"]
        if len(ids) != len(mask) or len(ids) != row["seq_len"]:
            raise ValueError(f"Mismatched prepared token and mask lengths at row {index}")
        if len(ids) > context:
            rejected["length"] += 1
            continue
        if sum(mask) < MIN_LOSS_TOKENS:
            rejected["min_loss_tokens"] += 1
            continue
        if not any(flag and sum(1 for _ in run) > horizon for flag, run in groupby(mask[1:])):
            rejected["anchors"] += 1
            continue
        prompt_ids = ids[: mask.index(1)]
        if overlap.matches(tokenizer.decode(ids, skip_special_tokens=True)):
            rejected["overlap"] += 1
            continue
        group = sha256(str(prompt_ids).encode()).digest()
        if group in val_groups:
            rejected["held_out_prompt"] += 1
            continue
        if len(val) < validation:
            val.append(index)
            val_groups.add(group)
        else:
            train.append(index)
    rejected["incomplete_batch"] = len(train) % accumulation
    train = train[: len(train) - rejected["incomplete_batch"]]
    if len(val) != validation or not train:
        raise ValueError("Insufficient eligible prepared DFlash rows")
    return dict(train=train, val=val, rejected=rejected)


def prepare(path, *, context, horizon, validation, accumulation, seed):
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table
    from transformers import AutoTokenizer

    if path.is_file():
        return json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        reference.TARGET, revision=reference.TARGET_REVISION,
        cache_dir=HF_HOME / "hub", local_files_only=True,
    )
    benchmark_path = hf_hub_download(
        instruct_config.family_config.MMLU_DATASET, MMLU_TEST_FILE, repo_type="dataset",
        revision=instruct_config.family_config.MMLU_REVISION, cache_dir=HF_HOME / "hub",
        local_files_only=True,
    )
    benchmark = read_table(benchmark_path, columns=["question", "subject"]).to_pylist()
    overlap = instruct_data.OverlapFilter(
        row["question"] for row in benchmark if row["subject"] in STEM_SUBJECTS
    )
    result = select_rows(open_dataset(), tokenizer, overlap, context=context, horizon=horizon,
                         validation=validation, accumulation=accumulation, seed=seed)
    instruct_data.atomic_json(path, result)
    return result
