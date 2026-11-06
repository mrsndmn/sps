"""Offline checks for native chat scoring and SPS cache generation."""
import pytest
import torch
from scripts.instruct_generation import (split_response, parse_answer, document_seed,
                                          generate, sample, mmlu_messages)
from scripts.instruct_evaluate import aggregate, validate_saved, digest


class Tokenizer:
    eos_token_id = 2
    def convert_tokens_to_ids(self, token):
        return {'<think>':3, '</think>':4}[token]
    def decode(self, tokens, skip_special_tokens=False):
        vocabulary = {2:'<eos>',3:'<think>',4:'</think>',5:'Final answer: A',6:'Final answer: B',7:'Reasoning'}
        return '\n'.join(vocabulary.get(x,str(x)) for x in tokens if not skip_special_tokens or x not in {2,3,4})
    def apply_chat_template(self, messages, **kwargs):
        self.call = (messages, kwargs)
        return torch.tensor([[8,9]])


def test_reasoning_cannot_supply_answer():
    tok = Tokenizer()
    unfinished = split_response(tok,[3,5], True)
    assert not unfinished['thinking_closed']
    assert parse_answer(unfinished['final_text']) is None
    complete = split_response(tok,[3,5,4,6,2], True)
    assert complete['thinking_closed']
    assert parse_answer(complete['final_text']) == 'B'
    assert '<think>' in complete['raw_text']
    assert parse_answer(split_response(tok,[3,5,4,6],False)['final_text']) is None
    assert parse_answer(split_response(tok,[3,5,4,3,6],True)['final_text']) is None
    assert parse_answer(split_response(tok,[6,2],False)['final_text']) == 'B'


def test_seed_independent_of_checkpoint_and_order():
    assert document_seed(1337,'mmlu','thinking',5) == document_seed(1337,'mmlu','thinking',5)
    assert len({document_seed(1337,t,m,i) for t in ('mmlu','ifeval') for m in ('thinking','nonthinking') for i in range(3)}) == 12


def test_sample_top_k_reproducible_and_filters():
    logits = torch.tensor([[1.,2.,3.,4.]])
    settings = dict(temperature=.6,top_p=.95,top_k=1)
    assert sample(logits,settings,torch.Generator().manual_seed(1)) == 3
    settings['top_k'] = 3
    a = torch.Generator().manual_seed(9)
    b = torch.Generator().manual_seed(9)
    assert [sample(logits,settings,a) for _ in range(15)] == [sample(logits,settings,b) for _ in range(15)]


@pytest.mark.parametrize('architecture', ['llama','qwen3'])
@pytest.mark.parametrize('sps', [False, True])
def test_generation_uses_decoder_and_native_chat(architecture,sps):
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
    from modeling.models.sps.pretrained import PretrainedSPS
    cls,cfg = ((LlamaForCausalLM,LlamaConfig) if architecture=='llama' else (Qwen3ForCausalLM,Qwen3Config))
    config = cfg(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                 num_key_value_heads=1,max_position_embeddings=64,eos_token_id=2)
    model = PretrainedSPS(cls(config),sps=sps).eval()
    def forbidden(*args,**kwargs):
        raise AssertionError('Must not bypass SPS with native generate')
    model.backbone.generate = forbidden
    tok = Tokenizer()
    settings = dict(temperature=.6,top_p=.95,top_k=20,max_new_tokens=3)
    messages = [{'role':'user','content':'Example'}]
    first = generate(model,tok,messages,True,settings,42)
    second = generate(model,tok,messages,True,settings,42)
    assert first == second
    assert 1 <= len(first['tokens']) <= 3
    assert tok.call[0] == messages
    assert tok.call[1]['enable_thinking'] is True
    assert tok.call[1]['add_generation_prompt'] is True


def test_mmlu_native_prompt_and_aggregation():
    row = dict(question='Q?',choices=['a','b','c','d'],answer=1,subject='abstract_algebra')
    messages = mmlu_messages(row)
    assert messages[0]['role'] == 'user'
    assert 'Final answer: X' in messages[0]['content']
    saved = [dict(index=0,doc_hash=digest(row),correct=True,subject=row['subject'],answer='B',stop='eos',thinking_closed=True)]
    assert validate_saved(saved,[row]) == {0}
    with pytest.raises(ValueError):
        validate_saved(saved*2,[row])
    result = aggregate(saved,'mmlu')
    assert result['accuracy'] == result['stem_accuracy'] == 1
    assert result['stem_count'] == 1


def test_ifeval_instruction_micro_average():
    saved = [dict(stop='eos',thinking_closed=True,metrics=dict(prompt_level_strict_acc=True,
             prompt_level_loose_acc=True,inst_level_strict_acc=[True],inst_level_loose_acc=[True])),
             dict(stop='length',thinking_closed=False,metrics=dict(prompt_level_strict_acc=False,
             prompt_level_loose_acc=False,inst_level_strict_acc=[True,False,False],inst_level_loose_acc=[True,False,False]))]
    result = aggregate(saved,'ifeval')
    assert result['inst_level_strict_acc'] == .5
    assert result['prompt_level_strict_acc'] == .5
