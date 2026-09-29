"""EOS handling used by the DeepSeek ER extracted step-100 checkpoint."""

import json

EOS_REFERENCE_REPO = "zjhhhh/er-r1-distill-1.5b-compression-n16-extracted-step_100"
EOS_REFERENCE_REVISION = "8183d5b14fbbce488d3a2fd1891ed37af7135a84"
EOS_POLICY = "deepseek_extracted_raw_length_pool_force_eos_training"


def force_eos_token_ids(prompt_ids, response_ids, eos_token_id, pad_token_id):
    """Port Actor.process_sequences for the checkpoint's rollout microbatch=1.

    The original scatter writes EOS immediately after the last ordinary token,
    capped at the existing final position. With no EOS/padding slot available,
    it replaces the last generated token; it never appends a new token.
    """
    if not prompt_ids or not response_ids or eos_token_id is None:
        raise ValueError("EOS processing requires a prompt, response and EOS token")
    sequence = list(prompt_ids) + list(response_ids)
    last = next((index for index in range(len(sequence) - 1, -1, -1)
                 if sequence[index] not in (eos_token_id, pad_token_id)), len(sequence) - 1)
    target = min(last + 1, len(sequence) - 1)
    if target < len(prompt_ids):
        raise ValueError("EOS replacement must stay inside the generated response")
    sequence[target] = eos_token_id
    return sequence[len(prompt_ids):]


def reference_reward_payload(payload, samples_per_prompt, tokenizer):
    """Keep raw group responses separate from the force-EOS training queries.

    Qwen3 prompts contain EOS already. Carry EOS presence from response token
    IDs so the length pool has the same effective gate as the DeepSeek run.
    Full prompt/response text is still passed unchanged to Math-Verify.
    """
    from qwen3_experiments.er_compression_rollouts import aligned_rollouts

    _, records = aligned_rollouts(payload)
    if len(records) % samples_per_prompt:
        raise ValueError("EOS reward request must contain complete prompt groups")
    result = []
    for start in range(0, len(records), samples_per_prompt):
        group = records[start:start + samples_per_prompt]
        if any(row["label"] != group[0]["label"] or row["input"] != group[0]["input"] for row in group):
            raise ValueError("EOS reward group mixes different dataset rows")
        raw_queries = [tokenizer.decode(row["prompt_token_ids"] + row["response_token_ids"],
                                        skip_special_tokens=False) for row in group]
        raw_eos = [tokenizer.eos_token_id in row["response_token_ids"] for row in group]
        for offset, row in enumerate(group):
            training_eos = tokenizer.eos_token_id in row["training_response_token_ids"]
            if not training_eos:
                raise ValueError("Training response is missing the reference forced EOS")
            result.append({"response": payload["query"][start + offset], "aux_info": {
                **json.loads(payload["labels"][start + offset]),
                "all_responses": raw_queries, "all_responses_have_eos": raw_eos,
                "response_has_eos": training_eos,
            }})
    return {"query": result}
