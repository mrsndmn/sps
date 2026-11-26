"""Greedy parity gate for the final SPS checkpoint and native vLLM execution."""

from workloads.serving import vllm_sps_common as common
from experiments import vllm_sps_config as c


def reference(trained):
    import gc

    import torch

    state = torch.load(
        common.checked_checkpoint(trained), map_location="cpu", mmap=True, weights_only=False
    )
    if not common.matches_training_input(state) or state["step"] != c.family_config.TRAIN_STEPS:
        raise RuntimeError("Audit checkpoint identity mismatch")
    model = common.load_model("cpu", backend="sdpa")
    model.load_state_dict(state["model"])
    del state
    model = model.cuda().to(torch.bfloat16).eval()
    eos = model.backbone.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    result = []
    with torch.inference_mode():
        for row in common.data("val")[: c.family_config.AUDIT_COUNT]:
            prefix = row["input_ids"][: c.family_config.AUDIT_PREFIX_TOKENS]
            tokens = torch.tensor([prefix], device="cuda")
            output = []
            for _ in range(c.family_config.AUDIT_NEW_TOKENS):
                token = (
                    model(tokens, positions=[tokens.shape[1] - 1])
                    .logits[0, 0, 0]
                    .argmax()
                    .item()
                )
                output.append(token)
                if token in eos:
                    break
                tokens = torch.cat((tokens, tokens.new_tensor([[token]])), 1)
            result.append({"prompt_token_ids": prefix, "expected_token_ids": output})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify(llm, references):
    from vllm import SamplingParams

    result = []
    for row in references:
        output = llm.generate(
            [{"prompt_token_ids": row["prompt_token_ids"]}],
            SamplingParams(temperature=0, max_tokens=c.family_config.AUDIT_NEW_TOKENS),
            use_tqdm=False,
        )[0]
        actual = list(output.outputs[0].token_ids)
        result.append(
            dict(
                row, actual_token_ids=actual, equal=actual == row["expected_token_ids"]
            )
        )
    return result
