"""Capture unpadded ER generations and the exact reference reward inputs."""

import gzip
import importlib.util
import json
import os
from pathlib import Path


def rollout_helpers():
    path = Path(__file__).resolve().parents[1] / "verl/utils/rollout_dataset.py"
    spec = importlib.util.spec_from_file_location("er_rollout_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_path(root, step):
    return Path(root) / "raw_rollouts" / f"step_{step:06d}.jsonl.gz"


def read_raw(root, step):
    with gzip.open(raw_path(root, step), "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def capture_generated_rollouts(maker, outputs, prompts, labels):
    """Save raw generations, then apply the DeepSeek reference's forced EOS.

    Trace fields are identical within each response group. Raw token IDs remain
    available for the natural-EOS length pool and the complete rollout archive.
    """
    from qwen3_experiments.er_compression_eos import force_eos_token_ids

    root = Path(os.environ["ER_SHARED_ROOT"])
    args = maker.strategy.args
    n = args.n_samples_per_prompt
    expected = args.rollout_batch_size * n
    if not len(outputs) == len(prompts) == len(labels) == expected:
        raise ValueError("Incomplete ER generation batch; refusing to lose rollouts")
    step = getattr(maker, "_er_rollout_step", 0) + 1
    if step > 100:
        raise ValueError("More than the planned 100 ER rollout steps")
    path = raw_path(root, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Raw ER rollouts already exist; explicit recovery is required")
    tagged, records = [], []
    for index, (output, prompt, label) in enumerate(zip(outputs, prompts, labels)):
        metadata = json.loads(label)
        if index % n and metadata != json.loads(labels[index - index % n]):
            raise ValueError("Mixed row labels in an ER response group")
        if len(output.outputs) != 1:
            raise ValueError("Expected exactly one vLLM output per expanded prompt")
        response = output.outputs[0]
        ids = list(response.token_ids)
        if not ids or len(ids) > args.generate_max_len:
            raise ValueError("Empty generation or response cap exceeded")
        prompt_ids = list(output.prompt_token_ids)
        if prompt_ids != maker.tokenizer.encode(prompt, add_special_tokens=False):
            raise ValueError("ER prompt was truncated or changed")
        training_ids = force_eos_token_ids(prompt_ids, ids, maker.tokenizer.eos_token_id,
                                          maker.tokenizer.pad_token_id)
        trace = {"step": step, "group_index": index // n}
        tagged.append(json.dumps({**metadata, "_er_trace": trace}, ensure_ascii=False))
        records.append({
            "step": step, "rollout_index": index, "sample_index": index % n,
            "input": prompt, "output": maker.tokenizer.decode(ids, skip_special_tokens=False),
            "prompt_token_ids": prompt_ids, "response_token_ids": ids,
            "training_response_token_ids": training_ids, "force_eos_applied": training_ids != ids,
            "generated_tokens": len(ids), "generated_eos": maker.tokenizer.eos_token_id in ids,
            "finish_reason": response.finish_reason, "stop_reason": response.stop_reason,
            "label": metadata,
        })
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    for output, record in zip(outputs, records):
        output.outputs[0].token_ids = record["training_response_token_ids"]
    maker._er_rollout_step = step
    return tagged


def aligned_rollouts(payload):
    """Validate HTTP ordering against the original complete generation batch."""
    root = Path(os.environ["ER_SHARED_ROOT"])
    labels = [json.loads(value) for value in payload["labels"]]
    steps = {value["_er_trace"]["step"] for value in labels}
    if len(steps) != 1:
        raise ValueError("A reward request spans multiple training rollouts")
    step = steps.pop()
    records = read_raw(root, step)
    count = len(records)
    if not count == len(payload["query"]) == len(payload["prompts"]) == len(labels):
        raise ValueError("Reward request does not cover the complete raw generation batch")
    n = count // len({label["_er_trace"]["group_index"] for label in labels})
    for index, (record, label, prompt) in enumerate(zip(records, labels, payload["prompts"])):
        trace = label.pop("_er_trace")
        if (record["rollout_index"] != index or record["label"] != label or record["input"] != prompt
                or trace != {"step": step, "group_index": index // n}):
            raise ValueError("Reward ordering differs from the saved raw generations")
    return step, records


def record_rewards(payload, metrics):
    """Persist raw and trained responses plus the exact reward query and scores."""
    root = Path(os.environ["ER_SHARED_ROOT"])
    step, records = aligned_rollouts(payload)
    count = len(records)
    if any(len(values) != count for values in metrics.values()):
        raise ValueError("Reference reward metrics are not aligned with all rollouts")
    fields = {key: [record[key] for record in records]
              for key in records[0] if key not in {"step", "rollout_index", "input", "output"}}
    fields["reward_query"] = payload["query"]
    fields["reward_metrics"] = [{key: values[index] for key, values in metrics.items()} for index in range(count)]
    rollout_helpers().dump_rollout_step(
        root / "rollout_dataset", step=step, inputs=payload["prompts"],
        outputs=[record["output"] for record in records], scores=metrics["rewards"], extra_fields=fields,
    )
