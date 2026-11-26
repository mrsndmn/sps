from workloads.readout.real_readout_gsm8k import answer

from workloads.readout import readout_archive


def test_exact_final_numeric_line():
    assert answer("work\n#### +1,000.000") == "1000"
    assert answer("#### -0.00") == "0"
    assert answer("#### 1.00000000000000000000000000001") != "1"
    assert answer("#### 123\nmore reasoning") is None
    assert answer("#### 1,00") is None


def test_legacy_resume_requires_unchanged_model_sources_and_workload(monkeypatch):
    marker = {"design_signature": "archived"}
    current = {
        "checkpoint": "same",
        "seed": 1,
        "code": {"real_readout_eval": "migrated", "real_readout_gsm8k": "migrated"},
        "model_sources": {"real_readout_model.py": "reviewed"},
    }
    historical = {
        "checkpoint": "same",
        "seed": 1,
        "code": {"real_readout_eval": "original", "real_readout_gsm8k": "original"},
    }
    monkeypatch.setattr(readout_archive, "_MIGRATED", {"gsm8k": {readout_archive.code_fingerprint(current)}})
    monkeypatch.setattr(readout_archive, "_HISTORICAL_LOADER", "original")
    monkeypatch.setattr(readout_archive, "_HISTORICAL_WORKERS", {"gsm8k": "original"})
    assert readout_archive.compatible_identity(historical, current, marker, "gsm8k")
    assert not readout_archive.compatible_identity(historical, dict(current, seed=2), marker, "gsm8k")
    changed = dict(current, model_sources={"real_readout_model.py": "changed"})
    assert not readout_archive.compatible_identity(historical, changed, marker, "gsm8k")
