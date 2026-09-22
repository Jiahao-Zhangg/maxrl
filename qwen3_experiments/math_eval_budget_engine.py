"""Model-independent, tested budget accounting for Eval1/2/3.

Prompt tokens are never charged. Every generated token, including EOS and failed
attempts, is charged. Each point uses fresh requests (no prefix-truncation replay).
"""

from __future__ import annotations

import random
import time
from collections import Counter

from math_eval_matrix_common import rollout_seed


def evaluate_point(*, protocol, budget, seed, rows, prompt_token_ids, engine,
                   sampling_params_type, tokenizer, score_many, emit, sampling,
                   progress=None, stop_on_first_success=False):
    count = len(rows)
    if count < 1 or len(prompt_token_ids) != count or budget < 1:
        raise ValueError("Invalid dataset or budget")
    if protocol not in ("eval1", "eval2", "eval3"):
        raise ValueError(f"Unknown evaluation: {protocol}")
    cap = int(sampling["per_rollout_cap"])
    batch_size = int(sampling["max_batch_size"])
    if cap < 1 or batch_size < 1:
        raise ValueError("Invalid response cap or batch size")
    attempts, costs, successes = [0] * count, [0] * count, [0] * count
    remaining = [budget] * count
    global_remaining = count * budget
    sequence = 0
    generation_seconds = scoring_seconds = 0.0

    def generate(requests, round_index=0):
        nonlocal sequence, global_remaining, generation_seconds, scoring_seconds
        parameters = [sampling_params_type(
            n=1, temperature=sampling["temperature"], top_p=sampling["top_p"],
            top_k=sampling["top_k"], max_tokens=request[2], ignore_eos=False,
            detokenize=False, seed=rollout_seed(seed, request[0], request[1]),
        ) for request in requests]
        started = time.monotonic()
        outputs = engine.generate(prompt_token_ids=[prompt_token_ids[r[0]] for r in requests],
                                  sampling_params=parameters, use_tqdm=False)
        generation_seconds += time.monotonic() - started
        if len(outputs) != len(requests):
            raise ValueError("Generator returned the wrong request count")
        pending = []
        for request, parameters, output in zip(requests, parameters, outputs, strict=True):
            position, attempt, request_cap, rank = request
            if len(output.outputs) != 1:
                raise ValueError("Each request must return exactly one response")
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            if not 0 < len(token_ids) <= request_cap:
                raise ValueError(f"Invalid output length: {len(token_ids)}, cap={request_cap}")
            pending.append({
                "prompt_position": position, "unique_id": rows[position]["unique_id"],
                "rollout_index": attempt, "rollout_seed": parameters.seed,
                "round_index": round_index, "permutation_rank": rank,
                "max_output_tokens": request_cap, "output_tokens": len(token_ids),
                "output_token_ids": token_ids, "finish_reason": completion.finish_reason,
                "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                "ground_truth": rows[position]["ground_truth"],
            })
        started = time.monotonic()
        scores = list(score_many([(record["response"], record["ground_truth"]) for record in pending]))
        scoring_seconds += time.monotonic() - started
        if len(scores) != len(pending):
            raise ValueError("Grader returned the wrong number of scores")
        for record, score in zip(pending, scores, strict=True):
            if score not in (0, 1):
                raise ValueError(f"Expected binary MathVerify score, got {score}")
            position = record["prompt_position"]
            if record["rollout_index"] != attempts[position]:
                raise ValueError("Nonsequential per-question attempt index")
            if (protocol == "eval3" or (protocol == "eval2" and stop_on_first_success)) and successes[position]:
                raise ValueError("A solved question was resampled")
            record["request_sequence_index"] = sequence
            record["score"] = float(score)
            record["solved_before"] = bool(successes[position])
            if protocol == "eval2":
                record["remaining_budget_before"] = remaining[position]
                remaining[position] -= record["output_tokens"]
                record["remaining_budget_after"] = remaining[position]
                if remaining[position] < 0:
                    raise ValueError("Per-question budget exceeded")
            if protocol == "eval3":
                record["global_budget_before"] = global_remaining
                global_remaining -= record["output_tokens"]
                record["global_budget_after"] = global_remaining
                if global_remaining < 0:
                    raise ValueError("Shared budget exceeded")
            attempts[position] += 1
            costs[position] += record["output_tokens"]
            successes[position] += int(score)
            record["solved_after"] = bool(successes[position])
            record["cumulative_output_tokens"] = costs[position]
            sequence += 1
            emit(record)
        if progress:
            progress({"rollouts": sequence, "output_tokens": sum(costs),
                      "num_questions_solved": sum(value > 0 for value in successes)})

    if protocol == "eval1":
        requests = [(position, attempt, budget, None) for position in range(count) for attempt in range(4)]
        for start in range(0, len(requests), batch_size):
            generate(requests[start:start + batch_size])
    elif protocol == "eval2":
        round_index = 0
        while True:
            positions = [position for position in range(count)
                         if remaining[position] > 0 and not (stop_on_first_success and successes[position])]
            if not positions:
                break
            for start in range(0, len(positions), batch_size):
                generate([(p, attempts[p], min(cap, remaining[p]), None) for p in positions[start:start + batch_size]], round_index)
            round_index += 1
    else:
        round_index = 0
        while global_remaining > 0 and not all(successes):
            order = list(range(count))
            random.Random((seed + 2_000_033 * round_index) % (2**31 - 1)).shuffle(order)
            cursor = 0
            while cursor < count and global_remaining > 0:
                candidates = [(rank, order[rank]) for rank in range(cursor, count) if not successes[order[rank]]]
                if not candidates:
                    break
                size = min(len(candidates), batch_size, max(1, global_remaining // cap))
                reserved = global_remaining
                requests = []
                for rank, position in candidates[:size]:
                    request_cap = min(cap, reserved)
                    requests.append((position, attempts[position], request_cap, rank))
                    reserved -= request_cap
                generate(requests, round_index)
                cursor = requests[-1][3] + 1
            round_index += 1

    prompts = [{"prompt_position": p, "unique_id": rows[p]["unique_id"], "attempts": attempts[p],
                "output_tokens": costs[p], "successes": successes[p], "solved": successes[p] > 0}
               for p in range(count)]
    summary = summarize_counts(protocol, budget, prompts, stop_on_first_success=stop_on_first_success)
    summary.update({"generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds})
    return summary, prompts


def summarize_counts(protocol, budget, prompts, *, stop_on_first_success=False):
    count = len(prompts)
    used = sum(item["output_tokens"] for item in prompts)
    solved = sum(item["solved"] for item in prompts)
    responses = sum(item["attempts"] for item in prompts)
    correct_responses = sum(item["successes"] for item in prompts)
    if protocol == "eval1" and any(item["attempts"] != 4 for item in prompts):
        raise ValueError("Incomplete mean@4 evaluation")
    if protocol == "eval2":
        for item in prompts:
            cost = item["output_tokens"]
            if not 0 < cost <= budget:
                raise ValueError("Eval2 exceeded a question's budget or left it unvisited")
            if cost < budget and not (stop_on_first_success and item["solved"]):
                raise ValueError("Eval2 stopped before success or budget exhaustion")
            if stop_on_first_success and item["successes"] > 1:
                raise ValueError("Eval2 continued after its first success")
    if protocol == "eval3" and (used > count * budget or (used < count * budget and solved != count)):
        raise ValueError("Eval3 has invalid stopping/budget accounting")
    result = {"num_prompts": count, "num_questions_solved": solved, "fraction_solved": solved / count,
            "total_rollouts": responses, "correct_rollouts": correct_responses,
            "mean_at_4_accuracy": correct_responses / (4 * count) if protocol == "eval1" else None,
            "total_output_tokens": used, "mean_output_tokens": used / responses if responses else 0.0,
            "allocated_output_budget": count * budget if protocol != "eval1" else 4 * count * budget,
            "all_questions_solved": solved == count}
    if protocol == "eval2":
        result.update({"stop_on_first_success": stop_on_first_success,
                       "unused_output_budget": count * budget - used,
                       "early_stopped_questions": sum(item["solved"] and item["output_tokens"] < budget for item in prompts)})
    return result


def audit_records(records, *, protocol, budget, seed, rows, per_rollout_cap, stop_on_first_success=False):
    """Independently recompute every counter from the saved response ledger."""
    count = len(rows)
    attempts, costs, successes = Counter(), Counter(), Counter()
    global_remaining = count * budget
    permutations, last_rank = {}, {}
    for sequence, record in enumerate(records):
        p = record["prompt_position"]
        if not 0 <= p < count or record["unique_id"] != rows[p]["unique_id"]:
            raise ValueError("Response identity mismatch")
        if record["ground_truth"] != rows[p]["ground_truth"]:
            raise ValueError("Response gold answer mismatch")
        if record["request_sequence_index"] != sequence or record["rollout_index"] != attempts[p]:
            raise ValueError("Response sequence mismatch")
        if record["rollout_seed"] != rollout_seed(seed, p, attempts[p]):
            raise ValueError("Response seed mismatch")
        length, cap, score = record["output_tokens"], record["max_output_tokens"], record["score"]
        if score not in (0, 1) or length != len(record["output_token_ids"]) or not 0 < length <= cap:
            raise ValueError("Invalid token count or score")
        if record["solved_before"] != bool(successes[p]):
            raise ValueError("Invalid pre-response solved state")
        if protocol == "eval1" and cap != budget:
            raise ValueError("Wrong Eval1 response cap")
        if protocol == "eval2":
            if stop_on_first_success and successes[p]:
                raise ValueError("Eval2 resampled a solved question")
            before = budget - costs[p]
            if cap != min(per_rollout_cap, before) or record["remaining_budget_before"] != before or record["remaining_budget_after"] != before - length:
                raise ValueError("Wrong Eval2 budget ledger")
        if protocol == "eval3":
            if successes[p] or cap > min(per_rollout_cap, global_remaining):
                raise ValueError("Wrong Eval3 allocation or repeated solved question")
            round_index, rank = record["round_index"], record["permutation_rank"]
            if round_index not in permutations:
                order = list(range(count))
                random.Random((seed + 2_000_033 * round_index) % (2**31 - 1)).shuffle(order)
                permutations[round_index] = order
            if rank <= last_rank.get(round_index, -1) or permutations[round_index][rank] != p:
                raise ValueError("Eval3 question order mismatch")
            last_rank[round_index] = rank
            if record["global_budget_before"] != global_remaining:
                raise ValueError("Wrong shared-budget opening balance")
            global_remaining -= length
            if global_remaining < 0 or record["global_budget_after"] != global_remaining:
                raise ValueError("Wrong shared-budget closing balance")
        attempts[p] += 1
        costs[p] += length
        successes[p] += int(score)
        if record["solved_after"] != bool(successes[p]) or record["cumulative_output_tokens"] != costs[p]:
            raise ValueError("Invalid post-response state")
    prompts = [{"prompt_position": p, "unique_id": rows[p]["unique_id"], "attempts": attempts[p],
                "output_tokens": costs[p], "successes": successes[p], "solved": successes[p] > 0}
               for p in range(count)]
    return summarize_counts(protocol, budget, prompts, stop_on_first_success=stop_on_first_success), prompts
