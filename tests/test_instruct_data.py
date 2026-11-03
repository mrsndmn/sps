"""Offline data-boundary tests; no downloads or GPU required."""
import pytest
from scripts.instruct_data import (OverlapFilter, encode_example, group_id, parse_thinking,
                                   split_for, valid_messages)


class CharacterTokenizer:
    """Native-template shape with one character/token makes masks inspectable."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        result = ''
        for index, msg in enumerate(messages):
            result += f"<|im_start|>{msg['role']}\n"
            if msg['role'] == 'assistant' and index == len(messages)-1:
                result += '<think>\n' + msg.get('reasoning_content', '') + '\n</think>\n\n'
            result += msg['content'] + '<|im_end|>\n'
        if add_generation_prompt:
            result += '<|im_start|>assistant\n'
            if not enable_thinking:
                result += '<think>\n\n</think>\n\n'
        return result

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, return_offsets_mapping=True)
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i,i+1) for i in range(len(text))]}


def conversation(answer='Four.'):
    return [{'role':'user','content':'What is two plus two?'}, {'role':'assistant','content':answer}]


def target_text(row):
    return ''.join(chr(x) for x,m in zip(row['input_ids'],row['loss_mask']) if m)


def test_nonthinking_supervises_only_final_assistant_and_end_boundary():
    messages = [{'role':'system','content':'Be helpful.'}] + conversation() + conversation('4.')
    row = encode_example(CharacterTokenizer(), messages, 'nonthinking', 'a', 1000)
    assert target_text(row) == '4.<|im_end|>\n'
    assert len(row['input_ids']) == len(row['loss_mask'])
    assert row['loss_mask'][0] == 0


def test_thinking_supervises_reasoning_and_final():
    messages = conversation()
    messages[-1]['reasoning_content'] = 'Two plus two equals four.'
    row = encode_example(CharacterTokenizer(), messages, 'thinking', 'a', 1000)
    assert target_text(row) == '<think>\nTwo plus two equals four.\n</think>\n\nFour.<|im_end|>\n'


def test_never_silently_truncate():
    with pytest.raises(ValueError, match='exceeds context'):
        encode_example(CharacterTokenizer(), conversation(), 'nonthinking', 'a', 10)


def test_same_request_stays_in_same_split_across_answer_and_mode():
    a = conversation()
    b = conversation('Other answer.')
    b[0]['content'] = 'WHAT is two plus two?!'
    assert group_id(a) == group_id(b)
    assert split_for(group_id(a),'salt',.8,.9) == split_for(group_id(b),'salt',.8,.9)


def test_invalid_conversations_rejected():
    assert valid_messages(conversation())
    assert not valid_messages(conversation()[:-1])
    assert not valid_messages([{'role':'tool','content':'x'}]+conversation())
    assert not valid_messages([{'role':'user','content':'x'}]*2)
    messages = conversation()
    messages[-1]['tool_calls'] = [{'name':'x'}]
    assert not valid_messages(messages)


@pytest.mark.parametrize('text', ['', '<think>hmm', '<think></think>yes', '<think>x</think>',
                                    '<think>x</think>y</think>', '<think>x<think>y</think>z'])
def test_reject_malformed_thinking(text):
    assert parse_thinking(text) is None


def test_parse_thinking_preserves_both_parts():
    assert parse_thinking('<think>\nReason\n</think>\n\nFinal') == dict(role='assistant', reasoning_content='Reason', content='Final')


def test_overlap_blocks_normalized_benchmark_questions():
    filt = OverlapFilter(['How many moons does Mars have?', ' '.join(str(i) for i in range(20))])
    assert filt.matches('Instruction: HOW many moons does Mars have?! Answer please')
    assert filt.matches('prefix ' + ' '.join(str(i) for i in range(4,18)) + ' suffix')
    assert not filt.matches('How many moons does Venus have?')


def _load_worker(monkeypatch):
    import sys
    import types
    from importlib.util import spec_from_file_location, module_from_spec
    config = types.ModuleType('scripts.instruct_config')
    for name, value in dict(STAGE_FRACTION=.2, TRAIN_HOURS=1., STARTUP_SECONDS=0,
                            MAX_STEPS=10, GLOBAL_BATCH=2, MIN_STEPS=2, SAFETY=1.).items():
        setattr(config, name, value)
    exps = [types.SimpleNamespace(name='ce', recipe='staged_sps', trainable=True),
            types.SimpleNamespace(name='ar', recipe='continued_ar', trainable=True)]
    registry = types.ModuleType('experiments')
    registry.collect_experiments = lambda: exps
    registry.design_signature = lambda: 'sig'
    monkeypatch.setitem(sys.modules, 'experiments', registry)
    monkeypatch.setitem(sys.modules, 'scripts.instruct_config', config)
    import scripts
    monkeypatch.setattr(scripts, 'instruct_config', config, raising=False)
    launcher = types.ModuleType('scripts.mlspace_launch')
    launcher.store_json = lambda *_: None
    monkeypatch.setitem(sys.modules, 'scripts.mlspace_launch', launcher)
    spec = spec_from_file_location('instruct_worker_test', 'scripts/instruct_run.py')
    worker = module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def test_plan_matches_exposure_and_counts_actual_target_tokens(monkeypatch):
    worker = _load_worker(monkeypatch)
    data = [dict(input_ids=[1,2,3,4],loss_mask=[0,0,1,1],mode='thinking' if i%2 else 'nonthinking') for i in range(12)]
    probes = {name:dict(embedding=1.,full=2.,design_signature='sig') for name in ('ce','ar')}
    plan = worker.calibrated_plan(probes, data)
    assert plan['steps'] == 6  # unique data, not 10-step max
    assert plan['examples'] == 12
    assert plan['text_tokens'] == 36
    assert plan['target_tokens'] == 24
    assert plan['mode_counts'] == dict(thinking=6, nonthinking=6)
    assert plan['target_tokens_by_mode'] == dict(thinking=12, nonthinking=12)
    worker.c.TRAIN_HOURS = 8/3600
    assert worker.calibrated_plan(probes,data)['steps'] == 4


def test_plan_rejects_missing_or_stale_probes(monkeypatch):
    worker = _load_worker(monkeypatch)
    with pytest.raises(RuntimeError, match='Missing'):
        worker.calibrated_plan({}, [])
    probes = {name:dict(embedding=1.,full=2.,design_signature='wrong') for name in ('ce','ar')}
    with pytest.raises(RuntimeError, match='Stale'):
        worker.calibrated_plan(probes, [])
