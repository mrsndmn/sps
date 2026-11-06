"""Native Qwen chat decoding, shared by AR and real-cache SPS evaluation."""
import hashlib
import re
import torch
from scripts.mmlu_generation import Decoder

ANSWER = re.compile(r'(?:^|\n)Final answer:\s*([A-D])(?:[.\s]|$)')


def document_seed(seed, task, mode, index):
    """Arm-independent seed: resume/order changes cannot alter sampled answers."""
    return int.from_bytes(hashlib.sha256(f'{seed}:{task}:{mode}:{index}'.encode()).digest()[:8], 'big') % (2**63 - 1)


def parse_answer(text):
    answers = ANSWER.findall(text)
    return answers[-1] if answers else None


def mmlu_messages(row):
    options = '\n'.join(f'{letter}. {choice}' for letter, choice in zip('ABCD', row['choices']))
    return [{'role': 'user', 'content': ('Answer this multiple-choice question. Finish your final response with a separate '
             'line exactly in the form "Final answer: X", where X is A, B, C, or D.\n\n'
             f"{row['question']}\n{options}")}]


def split_response(tokenizer, tokens, thinking):
    """Use actual special-token boundaries, never letters inside hidden reasoning."""
    close_id = tokenizer.convert_tokens_to_ids('</think>')
    open_id = tokenizer.convert_tokens_to_ids('<think>')
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    if thinking:
        if close_id not in tokens:
            return dict(raw_text=raw, final_text='', reasoning=tokenizer.decode(tokens, skip_special_tokens=True), thinking_closed=False)
        end = tokens.index(close_id)
        final_tokens = tokens[end + 1:]
        # Reopening reasoning after closure is malformed rather than a scored answer.
        valid = open_id not in final_tokens and close_id not in final_tokens
        return dict(raw_text=raw, final_text=tokenizer.decode(final_tokens, skip_special_tokens=True).strip() if valid else '',
                    reasoning=tokenizer.decode(tokens[:end], skip_special_tokens=True).strip(), thinking_closed=valid)
    # Native non-thinking prompt already closes the empty block. Unexpected blocks
    # are not allowed to leak their letters into the final-answer parser.
    valid = open_id not in tokens and close_id not in tokens
    return dict(raw_text=raw, final_text=tokenizer.decode(tokens, skip_special_tokens=True).strip() if valid else '',
                reasoning='', thinking_closed=valid)


def sample(logits, settings, generator):
    if settings['temperature'] <= 0 or not 0 < settings['top_p'] <= 1 or settings['top_k'] < 0:
        raise ValueError('Invalid sampling settings')
    scores = logits.float() / settings['temperature']
    k = min(settings['top_k'], scores.shape[-1])
    if k > 0:
        threshold = torch.topk(scores, k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    sorted_scores, indices = scores.sort(descending=True, dim=-1)
    cumulative = sorted_scores.softmax(-1).cumsum(-1)
    remove = cumulative > settings['top_p']
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    probabilities = torch.zeros_like(scores).scatter(-1, indices, sorted_scores.softmax(-1))
    return torch.multinomial(probabilities, 1, generator=generator).item()


@torch.inference_mode()
def generate(model, tokenizer, messages, thinking, settings, seed):
    if settings['max_new_tokens'] < 1:
        raise ValueError('Generation cap must be positive')
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=thinking, return_tensors='pt')
    device = next(model.parameters()).device
    ids = ids.to(device)
    if ids.shape[1] + settings['max_new_tokens'] > model.config.max_position_embeddings:
        raise ValueError('Chat prompt plus generation exceeds native context')
    generator = torch.Generator(device=device).manual_seed(seed)
    decoder = Decoder(model)
    eos = model.backbone.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if tokenizer.eos_token_id is not None:
        eos.add(tokenizer.eos_token_id)
    tokens, stop = [], 'length'
    for _ in range(settings['max_new_tokens']):
        token = sample(decoder.step(ids), settings, generator)
        tokens.append(token)
        if token in eos:
            stop = 'eos'
            break
        ids = torch.tensor([[token]], device=device)
    return dict(tokens=tokens, stop=stop, **split_response(tokenizer, tokens, thinking))
