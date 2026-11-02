import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from modeling.models.sps.pretrained import PretrainedSPS
from scripts.mmlu_generation import Decoder, parse_answer, prompt

@pytest.mark.parametrize('family', ['llama', 'qwen3'])
@pytest.mark.parametrize('sps', [False, True])
def test_cached_logits_equal_full_prefix(family, sps):
    config_cls, cls = (LlamaConfig, LlamaForCausalLM) if family == 'llama' else (Qwen3Config, Qwen3ForCausalLM)
    torch.manual_seed(42)
    config = config_cls(vocab_size=31, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
        max_position_embeddings=128, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = PretrainedSPS(cls(config), sps=sps).eval()
    decoder = Decoder(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        for step in range(5):
            logits = decoder.step(ids if step == 0 else ids[:, -1:])
            torch.testing.assert_close(logits, model(ids).logits[:, -1], atol=2e-6, rtol=2e-5)
            assert decoder.cache.get_seq_length() == ids.shape[1]
            ids = torch.cat([ids, logits.argmax(-1)[:, None]], dim=1)


def test_prompt_and_parser():
    row = dict(question='Why?', choices=['one', 'two', 'three', 'four'], answer=3)
    assert prompt(row) == prompt(dict(row, answer=0))
    assert parse_answer('I considered A and B.') is None
    assert parse_answer('Reasoning\nFinal answer: C\n') == 'C'
    assert parse_answer('Final answer: Cat') is None
    assert parse_answer('Final answer: B.\n') == 'B'
