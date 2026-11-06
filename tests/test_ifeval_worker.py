import contextlib
import hashlib
import json
import sys
import types
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('mode', [None, 'thinking', 'nonthinking'])
@pytest.mark.parametrize('trainable', [False, True])
def test_worker_only_evaluates_ifeval_and_isolates_outputs(tmp_path, monkeypatch, trainable, mode):
    import torch
    from scripts import ifeval_run as r, instruct_common, instruct_evaluate, instruct_config
    source = tmp_path/'source'; source.mkdir()
    (source/'eval').mkdir()
    (source/'final.pt').write_bytes(b'test checkpoint')
    gate = tmp_path/'sanity'; gate.mkdir()
    (gate/'generation-protocol.json').write_text(json.dumps(dict(design_signature=r.c.SOURCE_SIGNATURE,
                                                        generation=instruct_config.GENERATION)))
    modules = {}
    names = ('utils','instructions','instructions_registry','instructions_util')
    fingerprints = {}
    for name in names:
        path = tmp_path/(name+'.py'); path.write_text('# pinned evaluator')
        m = types.ModuleType(name); m.__file__ = str(path); modules[name] = m
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    package = types.ModuleType('lm_eval.tasks.ifeval')
    for name,m in modules.items():setattr(package,name,m)
    monkeypatch.setitem(sys.modules,'lm_eval.tasks.ifeval',package)
    monkeypatch.setitem(sys.modules,'langdetect',SimpleNamespace(DetectorFactory=SimpleNamespace(seed=None)))
    calls = []
    def dataset(*args,**kwargs):
        assert args==(instruct_config.IFEVAL_DATASET,)
        assert kwargs['revision']==instruct_config.IFEVAL_REVISION
        return [{'prompt':'one'},{'prompt':'two'}]
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(load_dataset=dataset))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw: object())))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:'version')
    (source/'eval/manifest.json').write_text(json.dumps(dict(design_signature=r.c.SOURCE_SIGNATURE,
        ifeval_code=fingerprints,lm_eval_version='version')))
    monkeypatch.setattr(r.e,'output_dir',lambda exp:source)
    monkeypatch.setattr(r.e,'SANITY_ROOT',gate)
    monkeypatch.setattr(r,'RUN_ROOT',tmp_path/'parallel')
    monkeypatch.setattr(r,'signature',lambda:'parallel-signature')
    monkeypatch.setattr(r,'source_check',lambda *a,**k:None)
    monkeypatch.setattr(instruct_evaluate,'ensure_ifeval_resources',lambda:None)
    class Model:
        def eval(self):return self
        def requires_grad_(self,flag):return self
    monkeypatch.setattr(instruct_common,'load_model',lambda *a:Model())
    restored = []
    monkeypatch.setattr(instruct_common,'restore',lambda path,*a:restored.append(path))
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:contextlib.nullcontext())
    def generate(model,tokenizer,rows,task,mode,settings,destination,sig,seed,*rest):
        calls.append((task,mode,len(rows),seed))
        assert settings==instruct_config.GENERATION[mode]
        assert destination != source/'eval'
        return {'count':len(rows)}
    monkeypatch.setattr(instruct_evaluate,'evaluate_generation',generate)
    exp=SimpleNamespace(name='example',trainable=trainable,recipe='staged_sps' if trainable else 'original_ar')
    r.evaluate(exp, mode=mode)
    assert calls==[('ifeval',m,2,instruct_config.SEED) for m in ([mode] if mode else r.c.MODES)]
    assert bool(restored)==trainable
    assert (r.output_dir(exp, mode)/'evaluated.json').exists()
    assert not (source/'evaluated.json').exists()
    r.evaluate(exp, mode=mode)  # same manifest must permit resuming
    manifest=r.output_dir(exp, mode)/'eval/manifest.json'
    data=json.loads(manifest.read_text());data['source_signature']='wrong'
    manifest.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='resume provenance'):
        r.evaluate(exp, mode=mode)


def test_legacy_import_preserves_answers_and_rejects_protocol_changes(tmp_path, monkeypatch):
    from scripts import ifeval_run as r
    monkeypatch.setattr(r, 'RUN_ROOT', tmp_path)
    exp = SimpleNamespace(name='example')
    legacy = r.output_dir(exp)/'eval'; legacy.mkdir(parents=True)
    dest = r.output_dir(exp, 'nonthinking')/'eval'; dest.mkdir(parents=True)
    current = dict(design_signature='new', modes=['nonthinking'], checkpoint={'mtime':1})
    old = dict(current, design_signature=r.c.LEGACY_SIGNATURE, modes=list(r.c.MODES))
    (legacy/'manifest.json').write_text(json.dumps(old))
    samples = b'{"index": 0}\n{"partial":'
    (legacy/'ifeval-nonthinking.jsonl').write_bytes(samples)
    r.reuse_legacy(exp, 'nonthinking', dest, current)
    assert (dest/'ifeval-nonthinking.jsonl').read_bytes()==samples
    (dest/'ifeval-nonthinking.jsonl').write_text('progress')
    r.reuse_legacy(exp, 'nonthinking', dest, current)
    assert (dest/'ifeval-nonthinking.jsonl').read_text()=='progress'
    with pytest.raises(RuntimeError, match='Legacy'):
        r.reuse_legacy(exp, 'nonthinking', dest, dict(current, checkpoint={'mtime':2}))
