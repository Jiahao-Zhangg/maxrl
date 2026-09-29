"""Compare two 200-question training samples after the queued L+0 evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATASETS = (("polaris_train", "Polaris-1-8-3200", 200),
            ("compression_train", "compression_dataset", 200))
SCORING = None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def core(root):
    return module("difficulty_generation_core", root / "provenance/eval_polaris_step80.py")


def text_key(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def normalize_question(row, index, spec, base):
    problem, gold = row["problem"], row[spec["answer"]]
    assert isinstance(problem, str) and problem.strip(), f"Empty problem at source row {index}"
    assert isinstance(gold, str) and gold.strip(), f"Empty gold at source row {index}"
    gold = gold.strip()
    if spec["key"] == "compression_train":
        boxed = base.last_box(row["solution"])
        assert boxed and boxed[boxed.index("{") + 1:-1].strip() == gold, "Gold disagrees with solution"
    return {"id": f"{spec['key']}_{index:04d}", "dataset": spec["key"],
            "source_row": index, "source_id": str(index), "problem": problem, "gold": gold,
            "messages": [{"role": "user", "content": problem + base.SUFFIX}]}


def copy_frozen(base, source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert base.digest(source) == base.digest(destination), f"Frozen file changed: {source}"
    else:
        shutil.copy2(source, destination)


def verify_plan(root, base):
    plan = base.read(root / "plan.json")
    for name, checksum in plan["frozen_files"].items():
        assert base.digest(root / name) == checksum, f"Changed input: {name}"
    assert base.digest(Path(plan["predecessor_root"]) / "plan.json") == plan["predecessor_plan_sha256"]
    assert plan["sampling"] == base.SAMPLING and plan["sampling_seed"] == 42
    assert plan["questions_per_dataset"] == 200 and plan["samples_per_question"] == 4
    return plan


def predecessor_ready(root, base):
    paths = [root / "queue_status.json", root / "status.json", root / "report/audit.json"]
    if not all(path.is_file() for path in paths):
        return False
    queue, status, audit = map(base.read, paths)
    return (queue.get("state") == "complete" and status.get("state") == "complete"
            and status.get("completed_responses") == status.get("total_responses") == 7276
            and status.get("completed_questions") == 1819
            and audit.get("complete") is True and audit.get("responses_verified") == 7276
            and audit.get("budget_points") == 54 and not audit.get("grader_errors"))


def prepare(root, request_path):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    request = json.loads(request_path.read_text())
    predecessor = Path(request["predecessor_root"])
    source = predecessor / "provenance/eval_polaris_step80.py"
    base = module("difficulty_preparation_core", source)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            verify_plan(root, core(root))
            return
        previous = base.read(predecessor / "plan.json")
        assert previous["job_id"] == request["job_id"] == "146103"
        assert previous["num_gpus"] == 8
        frozen = {}
        sources = [
            (source, "provenance/eval_polaris_step80.py"),
            (predecessor / "provenance/eval_rloo_final.py", "provenance/eval_rloo_final.py"),
            (predecessor / "provenance/eval_l0_final.py", "provenance/eval_l0_final.py"),
            (Path(__file__), "provenance/eval_training_difficulty.py"),
            (Path(__file__).with_name("run_training_difficulty_eval.sh"), "provenance/run_training_difficulty_eval.sh"),
            (request_path, "provenance/request.json"),
        ]
        for origin, relative in sources:
            copy_frozen(base, origin, root / relative)
            frozen[relative] = base.digest(root / relative)
        questions, inventories, samples, audits, problem_sets = [], [], [], [], {}
        base.init_grader()
        for spec in request["datasets"]:
            directory = root / "datasets" / spec["key"]
            directory.mkdir(parents=True, exist_ok=True)
            path = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"]))
            assert base.digest(path) == spec["file_sha256"]
            local_path = directory / "source.parquet"
            copy_frozen(base, path, local_path)
            frozen[str(local_path.relative_to(root))] = base.digest(local_path)
            card = Path(hf_hub_download(spec["repo"], "README.md", repo_type="dataset", revision=spec["revision"]))
            copy_frozen(base, card, directory / "source_README.md")
            frozen[str((directory / "source_README.md").relative_to(root))] = base.digest(card)
            frame = pd.read_parquet(local_path)
            assert len(frame) == spec["source_rows"] == 3200
            indices = random.Random(42).sample(range(len(frame)), 200)
            subset = frame.iloc[indices]
            normalized = [normalize_question(frame.iloc[index].to_dict(), index, spec, base) for index in indices]
            failures = []
            for question in normalized:
                gold = question["gold"]
                result = base.grade((f"\\boxed{{{gold}}}", gold))
                if result != {"correct": 1.0, "grader_status": "scored"}:
                    failures.append({"question_id": question["id"], "gold": gold, "result": result})
            assert not failures, f"Unusable reference answers; do not resample: {failures}"
            subset.to_parquet(directory / "sample_200.parquet", index=False)
            frozen[str((directory / "sample_200.parquet").relative_to(root))] = base.digest(directory / "sample_200.parquet")
            base.write(directory / "source_indices.json", indices)
            frozen[str((directory / "source_indices.json").relative_to(root))] = base.digest(directory / "source_indices.json")
            problem_sets[spec["key"]] = {text_key(row["problem"]) for row in normalized}
            questions.extend(normalized)
            inventories.append({**spec, "rows": 200, "split": "train", "sampling_seed": 42,
                                "sampling": "random.Random(42).sample(range(3200), 200), without replacement; no filtering",
                                "selected_source_rows": indices})
            samples.append({"dataset": spec["key"], "source_rows": indices, "question_ids": [q["id"] for q in normalized]})
            audits.append({"dataset": spec["key"], "source_rows": len(frame), "sampled_rows": 200,
                           "unique_source_rows": len(set(indices)), "unique_normalized_questions": len(problem_sets[spec["key"]]),
                           "gold_self_checks_passed": 200, "rows_filtered_or_replaced": 0})
        assert len(questions) == len({q["id"] for q in questions}) == 400
        base.write(root / "sampled_questions.json", questions)
        frozen["sampled_questions.json"] = base.digest(root / "sampled_questions.json")
        base.write(root / "sampling_manifest.json", {"seed": 42, "datasets": samples})
        frozen["sampling_manifest.json"] = base.digest(root / "sampling_manifest.json")
        for model in request["models"]:
            run_root = root / "runs" / model["key"]
            run_root.mkdir(parents=True, exist_ok=True)
            copy_frozen(base, source, run_root / "provenance/eval_polaris_step80.py")
            frozen[str((run_root / "provenance/eval_polaris_step80.py").relative_to(root))] = base.digest(source)
            previous_run = Path(model["source_run"])
            receipt = base.read(previous_run / "model_receipt.json")
            assert receipt["repo"] == model["repo"] and receipt["revision"] == model["revision"]
            for name, checksum in receipt["merged_files"].items():
                assert base.digest(previous_run / "model" / name) == checksum
            destination = run_root / "model"
            if not destination.exists():
                destination.symlink_to((previous_run / "model").resolve(), target_is_directory=True)
            assert destination.resolve() == (previous_run / "model").resolve()
            copy_frozen(base, previous_run / "model_receipt.json", run_root / "model_receipt.json")
            frozen[str((run_root / "model_receipt.json").relative_to(root))] = base.digest(run_root / "model_receipt.json")
            tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
            assert base.read(destination / "config.json")["max_position_embeddings"] >= 40960
            records = copy.deepcopy(questions)
            for row in records:
                if model["key"] == "qwen3":
                    ids = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, enable_thinking=True)
                    off = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, enable_thinking=False)
                    assert ids != off
                    rendered = tokenizer.decode(ids, skip_special_tokens=False)
                    assert rendered.endswith("<|im_start|>assistant\n")
                else:
                    rendered = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)
                    assert rendered.endswith("<｜Assistant｜><think>\n")
                    ids = tokenizer.encode(rendered, add_special_tokens=False)
                    assert tokenizer.decode(ids, skip_special_tokens=False) == rendered
                assert len(ids) + 32768 <= 40960
                row["prompt_token_ids"] = ids
            base.write(run_root / "questions.json", records)
            frozen[str((run_root / "questions.json").relative_to(root))] = base.digest(run_root / "questions.json")
            base.write(run_root / "prepared_inputs.json", {
                "model": {"repo": model["repo"], "revision": model["revision"]},
                "datasets": inventories, "questions_sha256": base.digest(run_root / "questions.json"),
                "samples_per_question": 4, "sampling_seed": 42, "sampling": base.SAMPLING,
                "thinking": True, "prompt_suffix": base.SUFFIX, "system_prompt": None,
                "max_model_len": 40960, "longest_prompt_tokens": max(len(q["prompt_token_ids"]) for q in records),
                "chat_template_sha256": base.fingerprint(tokenizer.chat_template),
                "prompt_format": "Pinned native model chat template, same user message; native thinking enabled",
                "grading": {"library": "Math-Verify", "timeout_seconds": 1,
                            "prediction": "Entire nonempty suffix after the last completed </think>",
                            "require_box": False, "require_eos": False,
                            "unfinished_reopened_empty_score": 0,
                            "auxiliary_generation_scores": "Frozen worker's boxed scores are regraded before reporting"},
                "paired_sample_ids_and_seeds": True,
            })
            frozen[str((run_root / "prepared_inputs.json").relative_to(root))] = base.digest(run_root / "prepared_inputs.json")
        base.write(root / "input_audit.json", {
            "complete": True, "datasets": audits, "questions": 400, "total_responses": 3200,
            "sample_overlap_normalized": len(set.intersection(*problem_sets.values())),
            "same_question_ids_and_user_messages_for_both_models": True,
            "native_thinking_templates_checked": True, "system_prompt": None,
            "sampling_is_frozen_before_inference": True,
        })
        plan = {**request, "holder_locks": previous["holder_locks"],
                "predecessor_plan_sha256": base.digest(predecessor / "plan.json"),
                "sampling": base.SAMPLING, "sampling_seed": 42, "questions_per_dataset": 200,
                "samples_per_question": 4, "responses_per_model": 1600, "total_responses": 3200,
                "num_gpus": 8, "frozen_files": frozen, "created_at": time.time()}
        base.write(root / "plan.json", plan)
        (root / "README.md").write_text(
            "# 两个训练集的难度比较\n\n"
            "排在 L+0 final checkpoint 的九数据集评测全部完成之后，在同一个 allocation **146103** 使用全部 **8 张 GPU**，"
            "依次评测 Qwen/Qwen3-1.7B 和 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B。\n\n"
            "从 hi-todayis-jh/Polaris-1-8-3200 与 zjhhhh/compression_dataset 的 train split 各无放回抽取 200 题，seed=42，"
            "不额外筛选或替换题目。两个模型共用这 400 题、相同用户消息与相同每题采样种子；"
            "各使用自己的原生聊天模板开启 thinking，没有 system prompt。每题 4 个回答，每个模型 1,600 个回答，共 3,200 个。\n\n"
            "Temperature=0.6、top-p=0.95、top-k=20、min-p=0，最大输出 32,768 tokens。"
            "只把最后一个已完成 </think> 后的非空文本交给 Math-Verify；未结束或重新开启未结束 thinking 计 0。"
            "不额外要求 boxed 或 EOS。所有 800 个回答均进入各模型、各数据集的 mean@4 分母。\n\n"
            "最终报告给出 2×2 mean@4 表、每个模型的两数据集差值，以及按题目重采样的 95% 区间。"
            "结论针对这批样本及指定模型、输出预算。\n\n"
            "[队列状态](queue_status.json) · [固定抽样](sampling_manifest.json) · [输入核验](input_audit.json) · "
            "[评测计划](plan.json) · [最终报告（完成后）](report/README.md)\n"
        )
        print("Prepared two fixed 200-question samples and both native models: 3200 responses", flush=True)


def initialize_scoring(root):
    global SCORING
    root = Path(root)
    SCORING = module("difficulty_scoring_helpers", root.parents[1] / "provenance/eval_l0_final.py")
    SCORING.initialize_grading(root)
    SCORING.CAPS = (32768,)


def score_question(item):
    return SCORING.score_question(item)


def regrade(root, model, base):
    run_root = root / "runs" / model["key"]
    questions = base.read(run_root / "questions.json")
    manifest_hash = base.digest(run_root / "manifest.json")
    manifest = base.read(run_root / "manifest.json")
    assert manifest["evaluator_sha256"] == base.digest(run_root / "provenance/eval_polaris_step80.py")
    assert manifest["inputs"]["questions_sha256"] == base.digest(run_root / "questions.json")
    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_scoring, initargs=(str(run_root),)) as pool:
        futures = [pool.submit(score_question, (str(run_root), manifest_hash, q)) for q in questions]
        for index, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if index % 25 == 0 or index == len(questions):
                base.write(run_root / "status.json", {"state": "regrading_after_thinking", "graded_questions": index,
                                                     "completed_responses": 1600, "total_responses": 1600,
                                                     "updated_at": time.time()})
    rows.sort(key=lambda row: (row["question_id"], row["sample_index"]))
    for row in rows:
        row["model"] = model["key"]
    helpers = module("difficulty_report_helpers", root / "provenance/eval_l0_final.py")
    metrics, question_rows = helpers.aggregate(rows, datasets=DATASETS, caps=(32768,))
    for row in metrics:
        row["model"] = model["key"]
    directory = run_root / "report"
    directory.mkdir(exist_ok=True)
    base.write(directory / "per_sample.json", rows)
    base.write(directory / "metrics.json", metrics)
    base.write_csv(directory / "per_question.csv", question_rows)
    errors = dict(Counter(row["grader_status"] for row in rows if row["grader_status"] not in
                          ("scored", "unfinished_thinking", "reopened_thinking", "empty_answer_suffix")))
    base.write(directory / "audit.json", {
        "complete": not errors, "responses_verified": len(rows), "questions": len(questions),
        "source_checksums_tokens_text_seeds": "passed", "scores_regraded_after_thinking": True,
        "grader_errors": errors, "metrics_sha256": base.digest(directory / "metrics.json"),
        "per_sample_sha256": base.digest(directory / "per_sample.json"),
    })
    assert not errors and len(rows) == 1600 and len(questions) == 400
    base.write(run_root / "status.json", {"state": "complete", "completed_responses": 1600,
                                         "total_responses": 1600, "completed_questions": 400, "updated_at": time.time()})


def bootstrap_difference(first, second, seed=42):
    """Keep each question's four correlated responses together when resampling."""
    import numpy as np

    first, second = np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(seed)
    differences = 100 * (rng.choice(first, (10000, len(first)), replace=True).mean(axis=1)
                         - rng.choice(second, (10000, len(second)), replace=True).mean(axis=1))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def report(root, base, plan):
    summaries, sources = [], []
    for model in plan["models"]:
        directory = root / "runs" / model["key"] / "report"
        audit = base.read(directory / "audit.json")
        assert audit["complete"] and audit["responses_verified"] == 1600
        assert base.digest(directory / "metrics.json") == audit["metrics_sha256"]
        metrics = {row["dataset"]: row for row in base.read(directory / "metrics.json")}
        with (directory / "per_question.csv").open() as stream:
            questions = list(csv.DictReader(stream))
        scores = [[float(r["mean_at_4"]) for r in questions if r["dataset"] == key] for key, _, _ in DATASETS]
        assert all(len(group) == 200 for group in scores)
        first, second = [metrics[key]["mean_at_4_percent"] for key, _, _ in DATASETS]
        summaries.append({"model": model["label"], "model_key": model["key"],
                          "polaris_mean_at_4_percent": first, "compression_mean_at_4_percent": second,
                          "polaris_minus_compression_pp": first - second,
                          "difference_95_percent_ci": bootstrap_difference(*scores),
                          "harder_in_sample": "Polaris-1-8-3200" if first < second else
                                              "compression_dataset" if second < first else "tie"})
        sources.append({"model": model["key"], "audit_sha256": base.digest(directory / "audit.json"),
                        "metrics_sha256": base.digest(directory / "metrics.json")})
    directory = root / "report"
    directory.mkdir(exist_ok=True)
    base.write(directory / "metrics.json", summaries)
    base.write_csv(directory / "mean_at_4.csv", [
        {k: v for k, v in row.items() if k != "difference_95_percent_ci"} for row in summaries
    ])
    lines = ["# Training dataset difficulty: mean@4", "",
             "各数据集从原 train split 随机抽取 200 题；每题 4 次，两个模型使用同一批题目。"
             "mean@4 = 正确回答数 / 800 × 100%。", "",
             "| Model | Polaris-1-8-3200 | compression_dataset | Polaris − Compression (pp) | 95% 区间 (pp) |",
             "|---|---:|---:|---:|---:|"]
    for row in summaries:
        lo, hi = row["difference_95_percent_ci"]
        lines.append(f"| {row['model']} | {row['polaris_mean_at_4_percent']:.2f}% | "
                     f"{row['compression_mean_at_4_percent']:.2f}% | {row['polaris_minus_compression_pp']:+.2f} | "
                     f"[{lo:+.2f}, {hi:+.2f}] |")
    harder = {row["harder_in_sample"] for row in summaries}
    lines.append("")
    if len(harder) == 1 and "tie" not in harder:
        lines.append(f"在这次抽样及 32k 设置下，两个模型都在 **{next(iter(harder))}** 上得到较低的 mean@4。")
    elif harder == {"tie"}:
        lines.append("这次抽样中，两套题在两个模型上的 mean@4 均相同。")
    else:
        lines.append("两个模型对难度的排序不完全一致，应分别看各模型的结果。")
    lines.extend(["", "区间按题目重采样 10,000 次，保留同题四个回答的关联；差值为负表示 Polaris 更难。"
                  "区间跨 0 的模型尚不能清楚区分两个训练集的总体难度。", "",
                  "Thinking on；temperature 0.6、top-p 0.95、top-k 20、min-p 0、seed 42；最大输出 32,768 tokens。"
                  "没有 system prompt，各模型使用自己的原生聊天模板。"
                  "仅判最后一个已完成 </think> 后的非空文本；未结束或重新开启未结束的 thinking 计 0；"
                  "不额外要求 boxed 或 EOS。", ""])
    for spec in plan["datasets"]:
        lines.append(f"- Dataset: [{spec['repo']}](https://huggingface.co/datasets/{spec['repo']}/tree/{spec['revision']})")
    for spec in plan["models"]:
        lines.append(f"- Model: [{spec['repo']}](https://huggingface.co/{spec['repo']}/tree/{spec['revision']})")
    lines.extend(["", "[CSV](mean_at_4.csv) · [数值与区间](metrics.json) · [固定抽样](../sampling_manifest.json) · "
                  "[输入核验](../input_audit.json) · [完整计划](../plan.json)", ""])
    (directory / "README.md").write_text("\n".join(lines))
    base.write(directory / "audit.json", {"complete": True, "models": 2, "datasets": 2,
                                         "responses_verified": 3200, "questions_per_dataset": 200,
                                         "model_sources": sources,
                                         "metrics_sha256": base.digest(directory / "metrics.json")})
    base.write(root / "status.json", {"state": "complete", "completed_models": 2,
                                     "completed_responses": 3200, "total_responses": 3200, "updated_at": time.time()})


def generation_complete(root, base, partial=False):
    assert not partial
    base.write(root / "status.json", {"state": "generated", "completed_responses": 1600,
                                     "total_responses": 1600, "updated_at": time.time()})


def run(root, base):
    plan = verify_plan(root, base)
    assert predecessor_ready(Path(plan["predecessor_root"]), base)
    shared = module("difficulty_shared_runner", root / "provenance/eval_rloo_final.py")
    shared.report = generation_complete
    scratch = Path(os.environ["TMPDIR"])
    for index, model in enumerate(plan["models"]):
        run_root = root / "runs" / model["key"]
        # Separate staging directories prevent one architecture loading the other's leftover shards.
        local = scratch / model["key"]
        local.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(local)
        base.write(root / "status.json", {"state": "evaluating", "model": model["key"],
                                         "completed_models": index, "total_models": 2, "updated_at": time.time()})
        worker = core(run_root)
        ready = run_root / "report/audit.json"
        if not (ready.exists() and base.read(ready).get("complete")):
            shared.run(run_root, worker)
            regrade(root, model, worker)
    os.environ["TMPDIR"] = str(scratch)
    report(root, base, plan)


def queue(root, base):
    plan = verify_plan(root, base)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "hostname": os.uname().nodename, "created_at": time.time(),
                 "job_id": plan["job_id"], "gpus": 8, "total_responses": 3200,
                 "predecessor_root": plan["predecessor_root"]}

        def update(**values):
            state.update(values, updated_at=time.time())
            base.write(root / "queue_status.json", state)

        predecessor = Path(plan["predecessor_root"])
        try:
            while not predecessor_ready(predecessor, base):
                previous = base.read(predecessor / "queue_status.json")
                status_path = predecessor / "status.json"
                status = base.read(status_path) if status_path.exists() else {}
                update(state="waiting_for_l0_evaluation", predecessor_queue_state=previous.get("state"),
                       predecessor_evaluation_state=status.get("state"),
                       predecessor_completed_responses=status.get("completed_responses", 0))
                time.sleep(30)
            base.write(root / "dependency_receipt.json", {
                "predecessor_manifest_sha256": base.digest(predecessor / "manifest.json"),
                "predecessor_audit_sha256": base.digest(predecessor / "report/audit.json"),
                "verified_complete_at": time.time(),
            })
            while True:
                job = subprocess.run(["scontrol", "show", "job", str(plan["job_id"]), "-o"],
                                     capture_output=True, text=True, timeout=30)
                assert job.returncode == 0 and "JobState=RUNNING" in job.stdout
                assert f"UserId={os.environ['USER']}(" in job.stdout
                command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                           "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1", "--job-name=training-difficulty",
                           "bash", str(root / "provenance/run_training_difficulty_eval.sh"), sys.executable, str(root)]
                with (root / "launch.log").open("a", buffering=1) as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    update(state="waiting_for_available_gpus_or_evaluating", launcher_pid=child.pid)
                    result = child.wait()
                if result == 75:
                    update(state="waiting_for_all_eight_gpus")
                    time.sleep(30)
                    continue
                assert result == 0, f"Evaluation exited {result}; inspect launch.log"
                audit = base.read(root / "report/audit.json")
                assert audit["complete"] and audit["responses_verified"] == 3200
                update(state="complete", completed_responses=3200, report=str(root / "report/README.md"))
                return
        except BaseException as exc:
            update(state="failed", error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "queue", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        assert args.request is not None
        prepare(root, args.request.resolve())
    else:
        base = core(root)
        if args.command == "queue":
            queue(root, base)
        else:
            try:
                run(root, base)
            except BaseException as exc:
                base.write(root / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
                raise


if __name__ == "__main__":
    main()
