"""Greedy generation with real-token-only SPS caches; no chat template required."""
import re
import torch
from transformers import DynamicCache

ANSWER = re.compile(r'(?:^|\n)Final answer:\s*([A-D])(?:[.\s]|$)')


def parse_answer(text):
    answers = ANSWER.findall(text)
    return answers[-1] if answers else None


def prompt(row):
    options = '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', row['choices']))
    return ('Answer the following multiple-choice question. Think step by step, then finish '
            'with a separate line exactly in the form "Final answer: X", where X is A, B, C, or D.\n\n'
            'Question: How many days are there in two weeks?\nA. 7\nB. 10\nC. 14\nD. 21\n\n'
            'Reasoning:\nOne week contains seven days. Two weeks contain 2 times 7, which is 14 days.\nFinal answer: C\n\n'
            'Question: Which animal is a mammal?\nA. Dolphin\nB. Trout\nC. Ant\nD. Lizard\n\n'
            'Reasoning:\nDolphins breathe air and nurse their young with milk. The other animals are a fish, an insect, and a reptile.\nFinal answer: A\n\n'
            f"Question: {row['question']}\n{options}\n\nReasoning:\n")


class Decoder:
    """One unpadded stream; prediction KV is removed after every SPS query."""
    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache(config=model.config)

    @torch.inference_mode()
    def step(self, ids):
        m = self.model
        output = m.backbone.model(input_ids=ids, past_key_values=self.cache, use_cache=True)
        if not m.sps:
            return m.backbone.lm_head(output.last_hidden_state[:, -1])
        length = self.cache.get_seq_length()
        pred = m.prediction.to(m.backbone.get_input_embeddings().weight.dtype)[None, None]
        output = m.backbone.model(inputs_embeds=pred, past_key_values=self.cache,
                                  position_ids=torch.tensor([[length]], device=ids.device), use_cache=True)
        self.cache.crop(length)
        return m.backbone.lm_head(output.last_hidden_state[:, -1])


@torch.inference_mode()
def generate(model, tokenizer, text, max_new_tokens):
    ids = tokenizer(text, return_tensors='pt', add_special_tokens=True).input_ids.to(next(model.parameters()).device)
    if ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt plus generation exceeds native context; refusing silent truncation')
    decoder = Decoder(model)
    tokens = []
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for _ in range(max_new_tokens):
        token = decoder.step(ids).argmax(-1).item()
        tokens.append(token)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if '\nQuestion:' in text:
            return text.split('\nQuestion:', 1)[0], tokens, 'next_question'
        if token in eos:
            return text, tokens, 'eos'
        # Require a terminating newline/period: avoid stopping on a partial token string.
        if re.search(r'(?:^|\n)Final answer:\s*[A-D][.\n]', text):
            return text, tokens, 'answer'
        ids = torch.tensor([[token]], device=ids.device)
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, 'length'
