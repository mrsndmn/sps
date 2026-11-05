"""Selected checkpoint/workload execution and immutable resume provenance."""
import contextlib
import hashlib
import json
import sys
import types
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('variant',['step384','cosine'])
@pytest.mark.parametrize('workload',['quality','ifeval-nonthinking','ifeval-thinking','mmlu-nonthinking','mmlu-thinking'])
def test_worker_uses_selected_checkpoint_and_only_selected_workload(tmp_path,monkeypatch,variant,workload):
    import torch
    from scripts import sched_eval as r, instruct_config as c, instruct_common, instruct_data, instruct_evaluate
    sc=r.sc
    exp=SimpleNamespace(name='example',recipe='staged_sps')
    source=tmp_path/'source';(source/'eval').mkdir(parents=True)
    sanity=tmp_path/'sanity';sanity.mkdir()
    (sanity/'generation-protocol.json').write_text(json.dumps(dict(design_signature=r.c.SOURCE_SIGNATURE,generation=c.GENERATION)))
    package=types.ModuleType('lm_eval.tasks.ifeval');fingerprints={}
    for name in ('utils','instructions','instructions_registry','instructions_util'):
        path=tmp_path/(name+'.py');path.write_text('# scorer')
        module=types.ModuleType(name);module.__file__=str(path);setattr(package,name,module)
        fingerprints[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules,'lm_eval.tasks.ifeval',package)
    monkeypatch.setitem(sys.modules,'langdetect',SimpleNamespace(DetectorFactory=SimpleNamespace(seed=None)))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:'version')
    (source/'eval/manifest.json').write_text(json.dumps(dict(design_signature=r.c.SOURCE_SIGNATURE,
        ifeval_code=fingerprints,lm_eval_version='version')))
    monkeypatch.setattr(sc.e,'SANITY_ROOT',sanity)
    monkeypatch.setattr(sc.e,'output_dir',lambda exp:source)
    dest=tmp_path/'output'/variant/workload
    monkeypatch.setattr(sc,'output_dir',lambda *args:dest)
    monkeypatch.setattr(sc,'signature',lambda:'current')
    monkeypatch.setattr(sc,'source_plan',lambda:None)
    checkpoint=dict(path=str(tmp_path/('best.pt' if variant=='step384' else 'final.pt')),
                    step=384 if variant=='step384' else 512,size=10,mtime_ns=100)
    monkeypatch.setattr(sc,'validate_checkpoint',lambda *args:checkpoint)
    loaded=[]
    def load(path,**kwargs):loaded.append(path);return {'model':{'sentinel':variant}}
    monkeypatch.setattr(torch,'load',load)
    class Model:
        def eval(self):return self
        def requires_grad_(self,*args):return self
        def load_state_dict(self,state):assert state=={'sentinel':variant}
    monkeypatch.setattr(instruct_common,'load_model',lambda *args:Model())
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:contextlib.nullcontext())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    dataset_calls=[]
    def dataset(*args,**kwargs):dataset_calls.append((args,kwargs));return [{'prompt':'x'},{'prompt':'y'}]
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(load_dataset=dataset))
    monkeypatch.setattr(instruct_evaluate,'ensure_ifeval_resources',lambda:None)
    calls=[]
    def generate(model,tok,rows,task,mode,settings,root,sig,seed,*args):
        calls.append((task,mode));assert root==dest/'eval' and sig=='current' and seed==c.SEED
        assert settings==c.GENERATION[mode];return {'count':len(rows)}
    monkeypatch.setattr(instruct_evaluate,'evaluate_generation',generate)
    monkeypatch.setattr(instruct_evaluate,'likelihood',lambda *args:calls.append(('quality',None)))
    monkeypatch.setattr(instruct_data,'rows',lambda split:[{'test':1}])
    monkeypatch.setattr(instruct_common,'nll',lambda *args:{'nll':1.})
    r.evaluate(variant,exp,workload)
    assert loaded==[checkpoint['path']]
    assert calls==[('quality',None) if workload=='quality' else tuple(workload.split('-',1))]
    assert (dest/'evaluated.json').exists() and not (source/'evaluated.json').exists()
    if workload!='quality':
        expected=c.IFEVAL_REVISION if workload.startswith('ifeval') else c.MMLU_REVISION
        assert dataset_calls[0][1]['revision']==expected
    checkpoint['mtime_ns']=200
    with pytest.raises(RuntimeError,match='resume provenance'):r.evaluate(variant,exp,workload)

def test_checkpoint_gate_selects_best_and_rejects_wrong_step(tmp_path,monkeypatch):
    import torch
    from scripts import sched_common as sc, sched_config as c
    from scripts import instruct_config as source
    exp=SimpleNamespace(name='example');plan={'steps':512,'stage_steps':102}
    monkeypatch.setattr(sc.e,'output_dir',lambda exp:tmp_path)
    monkeypatch.setattr(sc,'source_plan',lambda:plan)
    state=dict(design_signature=c.SOURCE_SIGNATURE,source=source.MODEL,revision=source.REVISION,
               step=384,stage=False,plan=plan)
    torch.save(state,tmp_path/'best.pt')
    assert sc.checkpoint_path('step384',exp)==tmp_path/'best.pt'
    assert sc.validate_checkpoint('step384',exp)['step']==384
    state['step']=512;torch.save(state,tmp_path/'best.pt')
    with pytest.raises(RuntimeError,match='provenance'):sc.validate_checkpoint('step384',exp)



def test_optimizer_and_model_resume_follow_same_lr_trajectory(tmp_path,monkeypatch):
    import torch
    from scripts import sched_train as r
    exp=SimpleNamespace(name='tiny',recipe='continued_ar')
    plan={'steps':8,'stage_steps':2}
    monkeypatch.setattr(r,'ACTIVE_EXPERIMENT',exp,raising=False)
    monkeypatch.setattr(r.sc,'signature',lambda:'scheduled')
    monkeypatch.setattr(r.sc,'source_plan',lambda:plan)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Linear(2,1)
            self.backbone.gradient_checkpointing_enable=lambda **kwargs:None
        def set_stage(self,frozen):self.requires_grad_(not frozen)
    torch.manual_seed(1);model=Model();opt=r.optimizer_for(model,False)
    x=torch.ones(1,2)
    def advance(m,o,step):
        for group in o.param_groups:group['lr']=r.sc.learning_rate('continued_ar',step,plan)
        o.zero_grad();m.backbone(x).square().sum().backward();o.step()
    for i in range(4):advance(model,opt,i)
    path=tmp_path/'latest.pt';r.save(path,model,opt,4,1.,False,plan)
    resumed=Model();state=r.restore(path,resumed,'cpu');resumed_opt=r.optimizer_for(resumed,False)
    resumed_opt.load_state_dict(state['optimizer'])
    for i in range(4,8):advance(model,opt,i);advance(resumed,resumed_opt,i)
    for a,b in zip(model.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
