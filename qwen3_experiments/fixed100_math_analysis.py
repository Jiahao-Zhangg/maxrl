"""Matched-cohort replay of fixed-100 pools, optionally including the paper models."""

import concurrent.futures
import csv
import json
from pathlib import Path

import numpy as np

from qwen3_experiments import fixed100_math_replay as mc
from qwen3_experiments.fixed100_math_eval import (
    BUDGETS,
    CAPS,
    DATASETS,
    REFERENCE_REPO,
    REFERENCE_REVISION,
    digest,
    load_run,
    read,
    write,
)


def validate_pool(pool, questions):
    lengths, scores, caps = pool["lengths"], pool["scores"], pool["capped"]
    if lengths.shape != scores.shape or lengths.shape != (len(questions), 100):
        raise ValueError("Every question must have exactly 100 complete responses")
    if (
        not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths <= 0)
        or np.any(lengths > 32768)
        or not np.all((scores == 0) | (scores == 1))
    ):
        raise ValueError("Invalid response lengths or binary grades")
    if caps.shape != (len(questions), 100, 6) or not np.all((caps == 0) | (caps == 1)):
        raise ValueError("Missing binary capped-response grades")
    np.testing.assert_array_equal(caps[:, :, -1], scores)


def load_pools(args):
    root, plan, questions = load_run(args.run_dir)
    receipt = read(root / "grading_receipt.json")
    if (
        not receipt["complete"]
        or receipt["plan_sha256"] != plan["plan_sha256"]
        or receipt["samples_sha256"] != digest(root / "samples.npz")
    ):
        raise ValueError("Graded pool receipt differs from this frozen run")
    with np.load(root / "samples.npz") as z:
        np.testing.assert_array_equal(z["original_positions"], [q["position"] for q in questions])
        np.testing.assert_array_equal(z["caps"], CAPS)
        own = {
            "label": plan["model"]["label"],
            "lengths": z["lengths"],
            "scores": z["scores"],
            "capped": z["capped_scores"],
        }
    pools = {}
    reference = None
    if args.reference_dir or args.with_reference:
        if args.reference_dir:
            reference = args.reference_dir.resolve()

            def fetch(name):
                return reference / name
        else:
            from huggingface_hub import hf_hub_download

            reference = root / "reference"

            def fetch(name):
                return Path(
                    hf_hub_download(
                        REFERENCE_REPO, name, repo_type="dataset", revision=REFERENCE_REVISION, local_dir=reference
                    )
                )

        other = read(fetch("manifests/questions.json"))
        lookup = {q["unique_id"]: (i, q) for i, q in enumerate(other)}
        indices = []
        for q in questions:
            i, target = lookup[q["unique_id"]]
            if any(q[k] != target[k] for k in ["position", "dataset", "ground_truth", "prompt", "prompt_token_ids"]):
                raise ValueError("Reference and new model do not use the same prompts and seeds")
            indices.append(i)
        index = read(fetch("manifests/model_index.json"))
        cap_path = fetch("arrays/capped_fixed100_scores.npz")
        with np.load(cap_path) as caps:
            np.testing.assert_array_equal(caps["caps"], CAPS)
            np.testing.assert_array_equal(caps["original_positions"], [q["position"] for q in other])
            for key, item in index.items():
                with np.load(fetch(f"manifests/{key}_fixed100.npz")) as z:
                    np.testing.assert_array_equal(z["original_positions"], [q["position"] for q in other])
                    np.testing.assert_array_equal(z["lengths"], caps[f"{key}_lengths"])
                    pools[key] = {
                        "label": item["label"],
                        "lengths": z["lengths"][indices],
                        "scores": z["scores"][indices],
                        "capped": caps[f"{key}_scores"][indices],
                    }
    key = plan["model"]["key"]
    if key in pools:
        raise ValueError("The new model key duplicates a reference model; set a distinct --model-key in prepare")
    pools[key] = own
    for pool in pools.values():
        validate_pool(pool, questions)
    provenance = {
        "new_model": plan["model"],
        "new_samples_sha256": receipt["samples_sha256"],
        "reference": {"repo": REFERENCE_REPO, "revision": REFERENCE_REVISION}
        if args.with_reference and not args.reference_dir
        else {"local_directory": str(reference)}
        if reference
        else None,
    }
    return questions, pools, provenance


def stop_once(lengths, scores, targets, response_rng, sweep_rng):
    """The paper's active-question sweeps, reporting the first unsupported draw."""
    orders = [response_rng.permutation(100) for _ in lengths]
    cursor = np.zeros(len(lengths), dtype=np.int32)
    solved = np.zeros(len(lengths), dtype=bool)
    active = np.arange(len(lengths))
    spent = 0
    event_l, event_r, event_q = [], [], []
    while len(active) and spent < targets[-1]:
        for q in sweep_rng.permutation(active):
            if cursor[q] == 100:
                return None, int(q)
            a = orders[q][cursor[q]]
            cursor[q] += 1
            length, score = int(lengths[q, a]), int(scores[q, a])
            event_l.append(length)
            event_r.append(score)
            event_q.append(q)
            spent += length
            solved[q] = bool(score)
            if spent >= targets[-1]:
                break
        active = active[~solved[active]]
    return mc.event_path(event_l, event_r, event_q, targets, len(lengths)), None


def replay_group(task):
    key, d, questions, lengths, scores = task
    failures, stop = set(), []
    for trial in range(100):
        response_rng = np.random.default_rng(np.random.SeedSequence([42, 2, d, trial, 0]))
        sweep_rng = np.random.default_rng(np.random.SeedSequence([42, 2, d, trial, 1]))
        row, exhausted = stop_once(lengths, scores, BUDGETS * len(questions), response_rng, sweep_rng)
        if exhausted is not None:
            failures.add(questions[exhausted]["unique_id"])
        else:
            stop.append(row)
    if failures:
        return key, d, {"exhausted": sorted(failures)}
    if int(lengths.sum()) < len(questions) * BUDGETS[-1]:
        raise ValueError(f"Continue-after-success pool cannot cover 64k: {key}/{DATASETS[d]}")
    result = {
        "exhausted": [],
        "stop_after_success": np.asarray(stop),
        "continue_after_success": mc.cross_no_exit(lengths, scores, BUDGETS, d, 100, 42),
    }
    individual = np.zeros((100, 5))
    hits = np.full((len(questions), 100), np.inf)
    exact = np.full(len(questions), np.inf)
    for i, q in enumerate(questions):
        rng = np.random.default_rng(np.random.SeedSequence([42, 1, d, q["position"]]))
        orders = np.asarray([rng.permutation(100) for _ in range(100)])
        individual += mc.individual_orders(lengths[i], scores[i], orders, BUDGETS)
        correct = scores[i].astype(bool)
        if correct.any():
            first = np.argmax(scores[i][orders], axis=1)
            hits[i] = np.cumsum(lengths[i][orders], axis=1)[np.arange(100), first]
            exact[i] = lengths[i, correct].mean() + lengths[i, ~correct].sum() / (correct.sum() + 1)
    result.update(individual=individual / len(questions), hits=hits, exact=exact)
    return key, d, result


def run_replays(questions, pools, workers):
    """Apply every coverage exclusion to every model, budget, cap and cost table."""
    excluded, reasons = set(), []
    for key, pool in pools.items():
        missing = ~pool["scores"].any(axis=1) & (pool["lengths"].sum(axis=1) < BUDGETS[-1])
        for i in np.flatnonzero(missing):
            uid = questions[i]["unique_id"]
            excluded.add(uid)
            reasons.append({"question_id": uid, "model": key, "reason": "individual_64k_pool_exhaustion"})
    for iteration in range(len(questions) + 1):
        selected = [i for i, q in enumerate(questions) if q["unique_id"] not in excluded]
        groups = [[i for i in selected if questions[i]["dataset"] == dataset] for dataset in DATASETS]
        if any(not ids for ids in groups):
            raise ValueError("Coverage exclusions emptied a benchmark; do not report a six-benchmark macro")
        tasks = [
            (key, d, [questions[i] for i in ids], pool["lengths"][ids], pool["scores"][ids])
            for key, pool in pools.items()
            for d, ids in enumerate(groups)
        ]
        results, added = {}, set()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(replay_group, task) for task in tasks]
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                key, d, result = future.result()
                results[key, d] = result
                for uid in result["exhausted"]:
                    added.add(uid)
                    reasons.append(
                        {
                            "question_id": uid,
                            "model": key,
                            "round": iteration,
                            "reason": "shared_stop_64k_pool_exhaustion",
                        }
                    )
                print(
                    json.dumps(
                        {
                            "coverage_round": iteration,
                            "groups_done": done,
                            "groups": len(tasks),
                            "model": key,
                            "dataset": DATASETS[d],
                            "exhausted": result["exhausted"],
                        }
                    ),
                    flush=True,
                )
        if not added:
            return selected, groups, results, reasons
        if added & excluded:
            raise RuntimeError("Coverage exclusion failed to make progress")
        excluded.update(added)
    raise RuntimeError("Coverage exclusions did not stabilize")


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def statistics(values):
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def summarize(questions, pools, selected, groups, results):
    budget_rows, capped_rows, first_rows, arrays = [], [], [], {}
    counts = np.array([len(ids) for ids in groups])
    common = np.ones(len(questions), dtype=bool)
    for pool in pools.values():
        common &= pool["scores"].any(axis=1)
    common_counts = np.array([int(common[ids].sum()) for ids in groups])
    for key, pool in pools.items():
        for agg in DATASETS + ["macro", "weighted"]:
            ds = [DATASETS.index(agg)] if agg in DATASETS else list(range(6))
            weights = np.ones(len(ds)) if agg != "weighted" else counts.astype(float)
            weights /= weights.sum()
            for metric in ["individual", "stop_after_success", "continue_after_success"]:
                values = sum(w * results[key, d][metric] for w, d in zip(weights, ds, strict=True))
                arrays[f"{key}_{agg}_{metric}"] = values
                for b, budget in enumerate(BUDGETS):
                    mean, se = statistics(values[:, b] * 100)
                    budget_rows.append(
                        {
                            "model": key,
                            "model_label": pool["label"],
                            "aggregation": agg,
                            "metric": metric,
                            "questions": int(counts[ds].sum()),
                            "budget_k": int(budget // 1024),
                            "pass_percent": mean,
                            "mc_standard_error_pp": se,
                        }
                    )
            cap_values = sum(w * pool["capped"][groups[d]].mean(axis=(0, 1)) for w, d in zip(weights, ds, strict=True))
            mean_length = sum(w * pool["lengths"][groups[d]].mean() for w, d in zip(weights, ds, strict=True))
            for cap, value in zip(CAPS, cap_values, strict=True):
                capped_rows.append(
                    {
                        "model": key,
                        "model_label": pool["label"],
                        "aggregation": agg,
                        "questions": int(counts[ds].sum()),
                        "cap_k": int(cap // 1024),
                        "accuracy_percent": float(value * 100),
                        "mean_length_tokens": float(mean_length),
                    }
                )
            hit_ds = [d for d in ds if common_counts[d] > 0] if agg == "weighted" else ds
            hit_weights = common_counts[hit_ds].astype(float) if agg == "weighted" else np.ones(len(hit_ds))
            if not hit_ds or any(common_counts[d] == 0 for d in hit_ds):
                first_mean = first_se = first_exact = None
            else:
                hit_weights /= hit_weights.sum()
                hits = sum(
                    w * results[key, d]["hits"][common[groups[d]]].mean(axis=0)
                    for w, d in zip(hit_weights, hit_ds, strict=True)
                )
                first_mean, first_se = statistics(hits)
                first_exact = float(
                    sum(
                        w * results[key, d]["exact"][common[groups[d]]].mean()
                        for w, d in zip(hit_weights, hit_ds, strict=True)
                    )
                )
                arrays[f"{key}_{agg}_first_correct"] = hits
            first_rows.append(
                {
                    "model": key,
                    "model_label": pool["label"],
                    "aggregation": agg,
                    "common_solved_questions": int(common_counts[ds].sum()),
                    "mean_tokens_mc100": first_mean,
                    "mc_standard_error_tokens": first_se,
                    "exact_mean_tokens_all_permutations": first_exact,
                }
            )
    common_ids = [questions[i]["unique_id"] for i in selected if common[i]]
    return budget_rows, capped_rows, first_rows, arrays, common_ids


def markdown_tables(output, pools, budgets, capped, first, count, common_count):
    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *("| " + " | ".join(row) + " |" for row in rows),
            ]
        )

    for agg in DATASETS + ["macro", "weighted"]:
        sections = [
            f"# Four evaluation families — {agg}\n",
            f"{count} common retained questions across six benchmarks; 100 responses per question; "
            "100 shuffles, seed 42. Pass@budget uses fractional boundary credit; capped accuracy is binary. "
            "Budget/cap labels use 1k=1024 tokens; mean length/cost uses k=1000 tokens.\n",
        ]
        for metric, label in [
            ("individual", "Per-question pass@budget"),
            ("stop_after_success", "Shared-budget pass@budget: stop after success"),
            ("continue_after_success", "Shared-budget pass@budget: continue after success"),
        ]:
            rows = []
            for key, pool in pools.items():
                values = [r for r in budgets if r["model"] == key and r["aggregation"] == agg and r["metric"] == metric]
                rows.append([pool["label"], *(f"{r['pass_percent']:.2f}" for r in values)])
            sections += [f"## {label}\n", table(["Model", "4k", "8k", "16k", "32k", "64k"], rows) + "\n"]
        rows = []
        for key, pool in pools.items():
            values = [r for r in capped if r["model"] == key and r["aggregation"] == agg]
            rows.append(
                [
                    pool["label"],
                    *(f"{r['accuracy_percent']:.2f}" for r in values),
                    f"{values[0]['mean_length_tokens'] / 1000:.2f}",
                ]
            )
        sections += [
            "## Capped single-attempt accuracy\n",
            table(["Model", "1k", "2k", "4k", "8k", "16k", "32k", "Mean length (k)"], rows) + "\n",
        ]
        rows = []
        for key, pool in pools.items():
            value = next(r for r in first if r["model"] == key and r["aggregation"] == agg)
            fields = [value["mean_tokens_mc100"], value["exact_mean_tokens_all_permutations"]]
            rows.append([pool["label"], *(f"{v / 1000:.2f}" if v is not None else "undefined" for v in fields)])
        sections += [
            "## Tokens to first correct answer\n\n"
            f"All-model observed-solved intersection: {common_count} questions "
            "across six benchmarks. This intersection changes when a model is added.\n",
            table(["Model", "100-shuffle mean (k)", "Exact finite-pool mean (k)"], rows) + "\n",
        ]
        (output / f"four_evaluations_{agg}.md").write_text("\n".join(sections))


def analyze(args):
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    questions, pools, provenance = load_pools(args)
    selected, groups, results, reasons = run_replays(questions, pools, args.workers)
    budgets, capped, first, arrays, common_ids = summarize(questions, pools, selected, groups, results)
    output = args.run_dir / "analysis"
    output.mkdir(exist_ok=True)
    csv_write(output / "budget_metrics_long.csv", budgets)
    csv_write(output / "capped_accuracy_long.csv", capped)
    csv_write(output / "tokens_to_first_correct.csv", first)
    np.savez_compressed(output / "trial_scores.npz", **arrays)
    from qwen3_experiments.fixed100_math_eval import EXCLUDED

    write(
        output / "cohort.json",
        {
            "input_questions": len(questions),
            "retained_questions": len(selected),
            "included_question_ids": [questions[i]["unique_id"] for i in selected],
            "initial_excluded_question_ids": EXCLUDED,
            "additional_exclusion_reasons": reasons,
            "counts": {d: len(ids) for d, ids in zip(DATASETS, groups, strict=True)},
            "all_100_shuffles_supported": True,
            "common_solved_question_ids": common_ids,
            "common_solved_count": len(common_ids),
            **provenance,
        },
    )
    markdown_tables(output, pools, budgets, capped, first, len(selected), len(common_ids))
    print(
        json.dumps(
            {
                "tables": str(output),
                "models": len(pools),
                "retained_questions": len(selected),
                "common_solved_questions": len(common_ids),
            }
        )
    )
