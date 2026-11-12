from copy import deepcopy
from pathlib import Path
import pytest
from scripts import provenance_train as t


def row(i, mode='nonthinking', size=4):
    return dict(source_id=str(i), group_id=str(i), mode=mode,
                messages=[dict(role='user',content=str(i)),dict(role='assistant',content='a')],
                input_ids=list(range(size)), loss_mask=[0]+[1]*(size-1))


def fixture_data():
    values = {'corpus':{},'teacher':{}}
    report = {'splits':{}}
    for j, split in enumerate(t.prep.SPLITS):
        a = [row(j*1000+i, 'thinking' if i%3 == 0 else 'nonthinking') for i in range(160)]
        b = deepcopy(a)
        for r in b:
            if r['mode']=='nonthinking':
                r['input_ids'] += [7,8];r['loss_mask'] += [1,1]
                r['messages'][-1]['content'] = 'teacher'
        values['corpus'][split], values['teacher'][split] = a,b
        report['splits'][split] = dict(paired_source_ids_sha256=t.prep.data.digest([r['source_id'] for r in a]),
            conditions={condition:dict(by_mode=t.prep.summarize(values[condition][split])) for condition in values})
    return values,report


def test_pairs_and_plan_match_prompts_not_token_exposure():
    values,report=fixture_data()
    t.validate_pairs(values,report)
    plan=t.make_plan(values)
    assert plan['steps']==10 and plan['stage_steps']==2
    assert plan['exposure']['teacher']['all']['nonthinking']['supervised_tokens'] > plan['exposure']['corpus']['all']['nonthinking']['supervised_tokens']
    assert plan['exposure']['teacher']['all']['thinking']==plan['exposure']['corpus']['all']['thinking']
    assert [r['source_id'] for r in t.ordered_train(values['corpus']['train'])] == [r['source_id'] for r in t.ordered_train(values['teacher']['train'])]


@pytest.mark.parametrize('damage',['thinking','prompt','leak','hash'])
def test_pair_integrity_failures(damage):
    values,report=fixture_data()
    if damage=='thinking':values['teacher']['train'][0]['input_ids'][0]=10
    elif damage=='prompt':values['teacher']['train'][1]['messages'][0]['content']='other'
    elif damage=='leak':
        for condition in values:values[condition]['val'][0]['group_id']=values[condition]['train'][0]['group_id']
    else:report['splits']['train']['paired_source_ids_sha256']='wrong'
    with pytest.raises(RuntimeError):t.validate_pairs(values,report)


def test_scoped_training_wiring_and_per_arm_signature():
    values,report=fixture_data();plan=t.make_plan(values)
    original=t.trainer.rows
    a,b=t.experiment_configs[:2]
    signatures=[]
    for exp in (a,b):
        ident=t.identity(exp,report,plan)
        with t.training_context(exp,values,ident):
            assert t.trainer.e.collect_experiments(exp.name)==[exp]
            assert t.trainer.read_plan()==plan
            assert t.trainer.rows('val')==values[exp.condition]['val']
            signatures.append(t.trainer.e.design_signature())
    assert signatures[0]!=signatures[1]
    assert t.trainer.rows is original


def test_train_payload_and_stage_configs():
    assert {(e.recipe,e.condition) for e in t.experiment_configs}=={
        (r,c) for r in ('continued_ar','staged_sps') for c in ('corpus','teacher')}
    for exp in t.experiment_configs:
        p=t.payload(exp,Path('/staged'))
        assert p['instance_type']==t.launch.INSTANCE_TYPES_BY_NUM_GPUS[exp.num_gpus]
        assert p['env_variables']['SPS_EXPERIMENT_SUITE']==t.c.SUITE
        assert f'--worker --arm {exp.name}' in p['script']
