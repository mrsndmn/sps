import hashlib
import json
import pytest
from workloads.evaluation.report_mmlu_stem import STEM_SUBJECTS, read_samples, summarize


def sample(index, row, answer):
    return dict(index=index, doc_hash=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
                subject=row['subject'], target='ABCD'[row['answer']], answer=answer, stop='answer')


def test_stem_selection_and_weighted_accuracy():
    data = [dict(subject=s, answer=0) for s in ['astronomy', 'astronomy', 'machine_learning', 'anatomy']]
    rows = [sample(i, x, a) for i, (x, a) in enumerate(zip(data, ['A', 'A', 'B', 'A']))]
    result = summarize(rows[:3], data)
    assert result['complete'] and result['expected'] == 3
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['macro_accuracy'] == .5
    assert summarize(rows[:2], data)['accuracy'] is None
    assert len(STEM_SUBJECTS) == 18 and 'anatomy' not in STEM_SUBJECTS
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + rows[:1], data)
    rows[0]['doc_hash'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        summarize(rows, data)


def test_active_file_is_never_modified(tmp_path):
    p = tmp_path / 'samples.jsonl'
    content = b'{"index":0}\n{"index":1}'
    p.write_bytes(content)
    assert read_samples(p) == [{'index': 0}]
    assert p.read_bytes() == content


def test_subset_reuse_noncontiguous_ids(tmp_path):
    from types import SimpleNamespace
    from workloads.evaluation.mmlu_subset import selected_indices, reuse_samples, validate_saved
    data = [dict(subject=s, answer=0) for s in ['anatomy', 'astronomy', 'anatomy', 'machine_learning']]
    indices = selected_indices(data, stem=True)
    assert indices == [1, 3]
    assert selected_indices(data, stem=True, limit=1) == [1]
    exp = SimpleNamespace(name='fixture')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(design_signature='archived', limit=None, experiment=exp.__dict__)))
    rows = [sample(i, row, 'A') | {'correct': True} for i, row in enumerate(data)]
    (tmp_path/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    reused = reuse_samples(tmp_path, exp, data, indices)
    assert [x['index'] for x in reused] == indices
    assert validate_saved(list(reversed(reused)), data, indices) == {1, 3}
    with pytest.raises(ValueError, match='Unexpected'):
        validate_saved(rows, data, indices)
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='manifest'):
        reuse_samples(tmp_path, exp, data, indices)


def test_merge_full_stem_deduplicates_and_rejects_conflicts(tmp_path):
    from types import SimpleNamespace
    from workloads.evaluation.mmlu_subset import reuse_full_and_stem
    data = [dict(subject=s, answer=0) for s in ['anatomy', 'astronomy', 'machine_learning']]
    exp = SimpleNamespace(name='fixture')
    rows = [sample(i, row, 'A') | {'correct': True} for i, row in enumerate(data)]
    for suite, signature, selected in [('mmlu-generation', 'archived', rows[:2]),
                                       ('mmlu-stem-generation', 'archived', rows[1:])]:
        folder = tmp_path/suite/'runs/out'/exp.name/'full'
        folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(design_signature=signature, limit=None, experiment=exp.__dict__)))
        (folder/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
    assert reuse_full_and_stem(tmp_path, exp, data, [0, 1, 2]) == rows
    rows[1]['answer'] = 'B'; rows[1]['correct'] = False
    (folder/'samples.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows[1:]))
    with pytest.raises(ValueError, match='Conflicting'):
        reuse_full_and_stem(tmp_path, exp, data, [0, 1, 2])


def test_historical_stem_report_ignores_current_evaluator_signature(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from orchestration import experiment_suite as e
    from experiments import environment
    from workloads.evaluation import report_mmlu_stem as report
    from experiments.mmlu_config import experiment_configs
    data = [dict(subject='astronomy', answer=0)]
    exp = experiment_configs[0]
    folder = tmp_path/'mmlu-stem-generation/runs/out'/exp.name/'full'
    folder.mkdir(parents=True)
    meta = dict(design_signature='archived', limit=None, experiment=exp.__dict__)
    (folder/'manifest.json').write_text(json.dumps(meta))
    (folder/'samples.jsonl').write_text(json.dumps(sample(0, data[0], 'A'))+'\n')
    monkeypatch.setattr(environment, 'COMMON_ROOT', tmp_path)

    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **kw: data))
    monkeypatch.setattr(sys, 'argv', ['report', '--suite', 'mmlu-stem-generation'])
    monkeypatch.setenv('SPS_EXPERIMENT_SUITE', 'mmlu-stem-generation')
    report.main()
    output = json.loads((tmp_path/'mmlu-stem-generation/reports/mmlu_stem.json').read_text())
    assert output['results'][exp.name]['accuracy'] == 1.
    meta['design_signature'] = 'unexpected-source'
    (folder/'manifest.json').write_text(json.dumps(meta))
    report.main()


def test_partial_manifest_uses_concrete_model_and_generation_settings():
    from experiments.mmlu import experiments, family_config
    from workloads.evaluation.mmlu_results import generation_protocol, matches_manifest

    exp = experiments("mmlu-generation")[0]
    manifest = {"experiment": dict(vars(exp), signature="old-source-hash"), "limit": 4}
    assert matches_manifest(manifest, exp, 4)
    assert not matches_manifest(manifest, exp, None)
    manifest["protocol"] = dict(generation_protocol(family_config), seed=0)
    assert not matches_manifest(manifest, exp, 4)


def test_mmlu_worker_resumes_saved_answers(tmp_path, monkeypatch):
    from experiments.mmlu import experiments
    from workloads.evaluation import mmlu_run as worker

    exp = experiments("mmlu-generation")[0]
    rows = [dict(subject="astronomy", answer=0, question=q) for q in ("first", "second")]
    meta = dict(experiment=dict(vars(exp), signature="archived"), limit=2)
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    first = dict(sample(0, rows[0], "A"), correct=True, generation="saved", tokens=[1])
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(first) + "\n")
    monkeypatch.setattr(worker.e, "SUITE", "mmlu-generation")
    monkeypatch.setattr(worker, "load_dataset", lambda *a, **kw: rows)
    monkeypatch.setattr(worker, "load_model", lambda *a: None)
    monkeypatch.setattr(worker.AutoTokenizer, "from_pretrained", lambda *a, **kw: None)
    monkeypatch.setattr(worker, "prompt", lambda row: row["question"])
    generated = []
    def generate(model, tokenizer, question, limit):
        generated.append(question)
        return "Final answer: A", [2], "answer"
    monkeypatch.setattr(worker, "generate", generate)
    worker.evaluate(exp, limit=2, destination=tmp_path)
    assert generated == ["second"]
    assert [json.loads(line)["index"] for line in samples.read_text().splitlines()] == [0, 1]



def test_legacy_manifest_keeps_original_generation_limit(monkeypatch):
    from dataclasses import dataclass
    from experiments import mmlu
    from workloads.evaluation.mmlu_results import matches_manifest

    @dataclass(kw_only=True)
    class ChangedDefaults(mmlu.FamilyConfig):
        MAX_NEW_TOKENS: int = 128

    exp = mmlu.experiments("mmlu-generation")[0]
    manifest = dict(limit=None, experiment=vars(exp))
    monkeypatch.setattr(mmlu, "FamilyConfig", ChangedDefaults)
    monkeypatch.setattr(mmlu, "family_config", mmlu.FamilyConfig())
    assert not matches_manifest(manifest, exp, None)
