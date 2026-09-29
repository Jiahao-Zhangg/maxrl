"""Queue a final L+0, L+4096 or GRPO checkpoint and score nine benchmarks after thinking."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DATASETS = (
    ("math500", "MATH500", 500),
    ("minervamath", "Minerva Math", 272),
    ("olympiadbench", "OlympiadBench", 674),
    ("amc22_23", "AMC22+23", 83),
    ("aime24", "AIME24", 30),
    ("aime25", "AIME25", 30),
    ("aime26", "AIME26", 30),
    ("polaris", "Polaris-Test", 100),
    ("polaris4_8", "Polaris-Test-4-8", 100),
)
CAPS = (1024, 2048, 4096, 8192, 16384, 32768)
BASE = TOKENIZER = None
MODEL_LABEL = "L+0"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluator(root):
    return load_module("l0_generation_core", root / "provenance/eval_polaris_step80.py")


def immutable_copy(base, source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert base.digest(source) == base.digest(destination), f"Frozen source changed: {source}"
    else:
        shutil.copy2(source, destination)


def answer_suffix(text):
    if "</think>" not in text:
        return None, "unfinished_thinking"
    suffix = text.rsplit("</think>", 1)[1]
    if "<think>" in suffix:
        return None, "reopened_thinking"
    suffix = suffix.strip()
    return (suffix, "eligible") if suffix else (None, "empty_answer_suffix")


def checkpoint_ready(status, receipt, repo, training_kind="l0"):
    """Require successful training exit and the verified final checkpoint receipt."""
    if training_kind == "grpo":
        assert status["variant"] == status["adv_estimator"] == "grpo"
        if status.get("archive_exit_code") != 0:
            return False
    else:
        assert training_kind in ("l0", "l4096")
        offset = 4096 if training_kind == "l4096" else 0
        assert status["variant"] == f"per_context_rb_l0_{offset}"
        assert status["adv_estimator"] == "fixed_n_rb_offset_cost_aware_marginrl"
        assert status["cost_offset_tokens"] == offset
    assert status["total_steps"] == 100
    if (status.get("state") != "complete" or status.get("exit_code") != 0
            or status.get("last_completed_step") != 100):
        return False
    if receipt is None:
        return False
    assert receipt["checkpoint"] == "global_step_100", "Refusing an intermediate checkpoint"
    assert receipt["repo_id"] == repo, "Final checkpoint belongs to another run"
    if receipt["state"] not in ("verified", "archived_and_deleted"):
        return False
    assert receipt.get("verified_at") and re.fullmatch(r"[0-9a-f]{40}", receipt["remote_commit"])
    for rank in range(8):
        info = receipt["files"][f"actor/model_world_size_8_rank_{rank}.pt"]
        assert info["size"] > 0 and re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
    return True


def training_status(training, base, training_kind="l0"):
    """Read GRPO's separate plan, supervisor status, and atomic exit record."""
    status = base.read(training / "status.json")
    if training_kind in ("l0", "l4096"):
        return status
    assert training_kind == "grpo"
    plan = base.read(training / "plan.json")
    assert Path(plan["output_root"]).resolve() == training.resolve()
    assert plan["total_steps"] == 100
    exit_path = training / "training_exit.json"
    exit_record = base.read(exit_path) if exit_path.exists() else {}
    exit_code = exit_record.get("exit_code")
    if status.get("training_exit_code") != exit_code:
        exit_code = None
    step = status.get("last_completed_step", 0)
    # Older supervisors also matched timing_s/step: in metric rows. Recover the
    # actual trainer step without changing or restarting the running supervisor.
    if not 0 <= step <= plan["total_steps"]:
        with (training / "train.log").open("rb") as stream:
            stream.seek(max(0, (training / "train.log").stat().st_size - 512 * 1024))
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", stream.read().decode(errors="replace"))
        steps = [int(value) for value in re.findall(
            r"(?m)^(?:\(TaskRunner pid=\d+\)\s*)?step:(\d+)\s+-\s+", text.replace("\r", "\n"))]
        step = max((value for value in steps if 0 <= value <= plan["total_steps"]), default=0)
    return {**status, "variant": "grpo", "adv_estimator": "grpo", "total_steps": plan["total_steps"],
            "hf_repo_prefix": plan["hf_repo_prefix"], "exit_code": exit_code, "last_completed_step": step}


def require_queue_node(plan):
    """A GRPO follow-up must stay on the compute node, including while waiting."""
    expected = plan.get("queue_node")
    if expected is None:
        return
    assert os.uname().nodename.split(".")[0] == expected, f"Run the queue on compute node {expected}"
    result = subprocess.run(["scontrol", "show", "job", str(plan["job_id"]), "-o"],
                            text=True, capture_output=True, timeout=30, check=True)
    fields = dict(word.split("=", 1) for word in result.stdout.split() if "=" in word)
    assert fields["JobState"] == "RUNNING" and fields["BatchHost"] == expected
    assert fields["UserId"].endswith(f"({os.getuid()})"), "Allocation belongs to another user"


def verify_plan(root, base):
    plan = base.read(root / "plan.json")
    for relative, checksum in plan["frozen_files"].items():
        assert base.digest(root / relative) == checksum, f"Changed evaluation input: {relative}"
    assert base.digest(plan["repository"] + "/scripts/model_merger.py") == plan["merger_sha256"]
    assert plan["sampling"] == base.SAMPLING and plan["sampling_seed"] == 42
    assert plan["samples_per_question"] == 4
    return plan


def prepare_plan(root, request_path):
    from transformers import AutoTokenizer

    request = json.loads(request_path.read_text())
    source = Path(request["repository"]) / "qwen3_experiments/eval_polaris_step80.py"
    base = load_module("l0_preparation_core", source)
    with (root / "prepare_plan.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            verify_plan(root, evaluator(root))
            return
        frozen = {}
        sources = [
            (source, "provenance/eval_polaris_step80.py"),
            (Path(__file__), "provenance/eval_l0_final.py"),
            (Path(__file__).with_name("run_l0_final_eval.sh"), "provenance/run_l0_final_eval.sh"),
            (Path(request["shared_runner"]), "provenance/eval_rloo_final.py"),
            (request_path, "provenance/request.json"),
            (Path(request["main_baseline"]), "provenance/main_baseline.csv"),
            (Path(request["budget_baseline"]), "provenance/budget_baseline.csv"),
        ]
        for source_path, relative in sources:
            immutable_copy(base, source_path, root / relative)
            frozen[relative] = base.digest(root / relative)
        training = Path(request["training_root"])
        training_kind = request.get("training_kind", "l0")
        label = request.get("model_label", "L+0")
        require_queue_node(request)
        status = training_status(training, base, training_kind)
        repo = status["hf_repo_prefix"] + "-step_100"
        checkpoint_ready(status, None, repo, training_kind)
        immutable_copy(base, training / "resolved_config.yaml", root / "provenance/training_config.yaml")
        frozen["provenance/training_config.yaml"] = base.digest(root / "provenance/training_config.yaml")
        if training_kind in ("grpo", "l4096"):
            import yaml

            training_config = yaml.safe_load((root / "provenance/training_config.yaml").read_text())
            if training_kind == "grpo":
                assert training_config["algorithm"]["adv_estimator"] == "grpo"
            else:
                assert training_config["algorithm"]["adv_estimator"] == "fixed_n_rb_offset_cost_aware_marginrl"
                assert training_config["algorithm"]["cost_offset_tokens"] == 4096
                grading = training_config["reward_model"]["reward_kwargs"]
                assert grading["check_eos"] is False and grading["score_after_thinking"] is True
            assert training_config["trainer"]["total_training_steps"] == 100
            immutable_copy(base, training / "plan.json", root / "provenance/training_plan.json")
            frozen["provenance/training_plan.json"] = base.digest(root / "provenance/training_plan.json")
        if request.get("after_thinking_reference_root"):
            reference = Path(request["after_thinking_reference_root"])
            for name in ("audit.json", "source_report.md", "main.csv", "budgets.csv"):
                relative = f"report/qwen3_after_thinking/{name}"
                immutable_copy(base, reference / name, root / relative)
                frozen[relative] = base.digest(root / relative)
            after_thinking_reference(root, base)
        records, specs, receipts = [], {}, []
        prepared_reference = None
        if request.get("reference_plan_root"):
            assert not request.get("reference_evaluations"), "Use one reference input route"
            reference = Path(request["reference_plan_root"])
            reference_plan = base.read(reference / "plan.json")
            for name, checksum in reference_plan["frozen_files"].items():
                assert base.digest(reference / name) == checksum, f"Changed reference input: {name}"
            assert reference_plan["frozen_files"]["provenance/eval_polaris_step80.py"] == base.digest(source)
            assert reference_plan["sampling"] == base.SAMPLING and reference_plan["sampling_seed"] == 42
            assert reference_plan["samples_per_question"] == 4
            records = base.read(reference / "questions.json")
            prepared_reference = copy.deepcopy(base.read(reference / "input_plan.json"))
            specs = {item["key"]: item for item in prepared_reference["datasets"]}
            for name in ("plan.json", "input_plan.json", "questions.json", "input_audit.json"):
                destination = f"provenance/reference_plan_{name}"
                immutable_copy(base, reference / name, root / destination)
                frozen[destination] = base.digest(root / destination)
            receipts.append({"root": str(reference), "plan_sha256": base.digest(reference / "plan.json"),
                             "questions_sha256": base.digest(reference / "questions.json")})
        for index, directory in enumerate(map(Path, request.get("reference_evaluations", []))):
            manifest = base.read(directory / "manifest.json")
            assert base.read(directory / "status.json")["state"] == "complete"
            assert manifest["evaluator_sha256"] == base.digest(source)
            assert manifest["inputs"]["sampling"] == base.SAMPLING
            assert manifest["inputs"]["sampling_seed"] == 42
            assert manifest["inputs"]["samples_per_question"] == 4
            assert base.digest(directory / "questions.json") == manifest["inputs"]["questions_sha256"]
            for relative in ("manifest.json", "questions.json"):
                destination = f"provenance/reference_{index}_{relative}"
                immutable_copy(base, directory / relative, root / destination)
                frozen[destination] = base.digest(root / destination)
            records.extend(base.read(directory / "questions.json"))
            for item in manifest["inputs"]["datasets"]:
                assert item["key"] not in specs
                specs[item["key"]] = item
            receipts.append({"root": str(directory), "manifest_sha256": base.digest(directory / "manifest.json"),
                             "questions_sha256": base.digest(directory / "questions.json")})
            if prepared_reference is None:
                prepared_reference = copy.deepcopy(manifest["inputs"])
        expected = Counter({key: count for key, _, count in DATASETS})
        assert Counter(row["dataset"] for row in records) == expected
        assert len({row["id"] for row in records}) == len(records) == 1819
        assert {key: item["rows"] for key, item in specs.items()} == dict(expected)
        tokenizer = AutoTokenizer.from_pretrained(request["reference_model"], local_files_only=True)
        assert base.fingerprint(tokenizer.chat_template) == prepared_reference["chat_template_sha256"]
        for question in records:
            assert question["messages"] == [{"role": "user", "content": question["problem"] + base.SUFFIX}]
            ids = tokenizer.apply_chat_template(question["messages"], add_generation_prompt=True, enable_thinking=True)
            assert ids == question["prompt_token_ids"] and len(ids) + 32768 <= 40960
            no_think = tokenizer.apply_chat_template(question["messages"], add_generation_prompt=True, enable_thinking=False)
            assert ids != no_think
        order = {key: i for i, key in enumerate(
            ("aime24", "aime25", "aime26", "amc22_23", "polaris", "polaris4_8", "math500", "minervamath", "olympiadbench")
        )}
        records.sort(key=lambda row: (order[row["dataset"]], row["source_row"]))
        base.write(root / "questions.json", records)
        frozen["questions.json"] = base.digest(root / "questions.json")
        prepared_reference.update(
            model={"repo": repo, "revision": None}, model_label=label, datasets=[specs[key] for key, _, _ in DATASETS],
            questions_sha256=frozen["questions.json"],
            longest_prompt_tokens=max(len(row["prompt_token_ids"]) for row in records),
            prompt_format="Native Qwen3 chat template, thinking enabled, one user message, no system message",
            question_sources=receipts,
            grading={"library": "Math-Verify", "timeout_seconds": 1,
                     "prediction": "Entire nonempty suffix after the last completed </think>",
                     "unfinished_reopened_or_empty_score": 0, "require_box": False, "require_eos": False,
                     "metrics": ["mean@4", "pass@4", "mean_output_tokens"],
                     "auxiliary_generation_scores": "Frozen worker saves boxed-only scores; final reports regrade every cap"},
            budget_evaluation={"caps": CAPS, "method": "prefix replay of exact generated token IDs",
                               "includes": ["thinking", "answer", "EOS"], "excludes": ["prompt"],
                               "append_tokens": False, "regrade_32k": True},
        )
        base.write(root / "input_plan.json", prepared_reference)
        frozen["input_plan.json"] = base.digest(root / "input_plan.json")
        with (root / "provenance/main_baseline.csv").open() as stream:
            main_rows = list(csv.DictReader(stream))
        with (root / "provenance/budget_baseline.csv").open() as stream:
            budget_rows = list(csv.DictReader(stream))
        assert len(main_rows) == 9 and len(budget_rows) == 27
        assert {row["dataset"] for row in budget_rows} == set(expected)
        assert {row["model"] for row in budget_rows} == {"qwen3-1.7B", "MaxRL", "ER"}
        plan = {**request, "model_repo": repo, "final_step": 100, "num_gpus": 8,
                "questions": len(records), "total_responses": len(records) * 4,
                "sampling": base.SAMPLING, "sampling_seed": 42, "samples_per_question": 4,
                "caps": CAPS, "frozen_files": frozen,
                "merger_sha256": base.digest(Path(request["repository"]) / "scripts/model_merger.py"),
                "created_at": time.time()}
        base.write(root / "plan.json", plan)
        base.write(root / "input_audit.json", {
            "questions": len(records), "responses": len(records) * 4,
            "dataset_counts": dict(expected), "exact_reference_questions_and_prompt_tokens": True,
            "no_system_prompt": True, "native_thinking_enabled": True,
            "source_manifests_and_question_checksums": "passed",
        })
        (root / "README.md").write_text(
            f"# {label} final checkpoint evaluation\n\n"
            f"等待训练成功完成 100/100 步，并取得已验证的 `{repo}` 固定版本。\n\n"
            f"使用 allocation {plan['job_id']} 的全部 8 张 GPU；若有其他任务占用，继续等待。\n\n"
            "九个数据集共 1,819 题，每题 4 个回答，总计 7,276 个回答。"
            "Thinking on；temperature 0.6、top-p 0.95、top-k 20、min-p 0、seed 42；输出最多 32,768 tokens。"
            "没有 system prompt；用户消息末尾为 `Please reason step by step, and put your final answer within \\boxed{}.`\n\n"
            "将同一批回答按原始 token IDs 截取 1k / 2k / 4k / 8k / 16k / 32k；1k=1,024 输出 tokens，"
            "包括 thinking、答案和 EOS，不含输入。仅将最后一个已完成 `</think>` 后的非空文本交给 Math-Verify；"
            "未结束或重新开启未结束的 thinking 计 0。不额外要求 boxed 或 EOS，不补结束标记或继续生成。\n\n"
            "最终报告包含 mean@4、pass@4、平均回答长度与九项预算表；32k 用同样的后缀规则重判。"
            "历史三模型数值原样保留，标明原始采样和判分口径差异。\n\n"
            "[队列状态](queue_status.json) · [评测计划](plan.json) · [输入核验](input_audit.json) · "
            "[最终报告（完成后）](report/README.md)\n"
        )
        print("Prepared and froze 1819 questions, 7276 samples, nine datasets and six budgets", flush=True)


def prepare_model(root, base, plan):
    from transformers import AutoTokenizer

    receipt = base.read(root / "final_checkpoint_receipt.json")
    base.MODEL, base.REVISION = plan["model_repo"], receipt["remote_commit"]
    base.PREFIX, base.REPO = "global_step_100/actor/", Path(plan["repository"])
    storage = Path(plan.get("model_storage", root))
    storage.mkdir(parents=True, exist_ok=True)
    base.prepare_model(storage)
    merged = base.read(storage / "model_receipt.json")
    if storage.resolve() != root.resolve():
        destination = root / "model"
        destination.mkdir(exist_ok=True)
        for name, checksum in merged["merged_files"].items():
            source, target = storage / "model" / name, destination / name
            if name.endswith(".safetensors"):
                if not target.is_symlink():
                    assert not target.exists(), f"Unexpected model file: {target}"
                    target.symlink_to(source)
                assert target.resolve() == source.resolve()
            else:
                shutil.copy2(source, target)
            assert base.digest(target) == checksum
        base.write(root / "model_receipt.json", merged)
    for name, item in merged["source_files"].items():
        assert name.startswith("global_step_100/")
        archived = receipt["files"][name.removeprefix("global_step_100/")]
        assert item == {"size": archived["size"], "sha256": archived["sha256"]}
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    reference = AutoTokenizer.from_pretrained(plan["reference_model"], local_files_only=True)
    assert tokenizer.get_vocab() == reference.get_vocab() and tokenizer.chat_template == reference.chat_template
    assert base.read(root / "model/config.json")["max_position_embeddings"] >= 40960
    assert base.read(root / "model/generation_config.json")["eos_token_id"] == [151645, 151643]
    for question in base.read(root / "questions.json"):
        ids = tokenizer.apply_chat_template(question["messages"], add_generation_prompt=True, enable_thinking=True)
        assert ids == question["prompt_token_ids"]
    prepared = base.read(root / "input_plan.json")
    prepared["model"]["revision"] = receipt["remote_commit"]
    prepared["plan_sha256"] = base.digest(root / "plan.json")
    prepared["checkpoint_receipt_sha256"] = base.digest(root / "final_checkpoint_receipt.json")
    base.write(root / "prepared_inputs.json", prepared)


def initialize_grading(root):
    global BASE, TOKENIZER, MODEL_LABEL
    from transformers import AutoTokenizer

    root = Path(root)
    BASE = evaluator(root)
    MODEL_LABEL = BASE.read(root / "plan.json").get("model_label", "L+0")
    BASE.init_grader()
    TOKENIZER = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)


def score_question(item):
    root, manifest_hash, question = item
    root = Path(root)
    rows, cache = [], {}
    for index in range(4):
        identity = BASE.sample_id(question, index)
        original = BASE.saved_result(root, identity, manifest_hash, full=True)
        assert original is not None and original["id"] == identity
        assert original["question_id"] == question["id"] and original["sample_index"] == index
        assert original["dataset"] == question["dataset"]
        assert original["seed"] == BASE.sample_seed(question["id"], index)
        assert original["prompt_tokens"] == len(question["prompt_token_ids"])
        tokens = original["output_token_ids"]
        assert 0 < len(tokens) <= 32768 and len(tokens) == original["output_tokens"]
        assert original["finish_reason"] in ("stop", "length")
        assert original["finish_reason"] != "length" or len(tokens) == 32768
        decoded = TOKENIZER.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert decoded == original["response"]
        assert original["prediction"] == BASE.prediction(decoded)
        for cap in CAPS:
            used = tokens[:cap]
            text = TOKENIZER.decode(used, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            suffix, status = answer_suffix(text)
            if suffix is None:
                score = {"correct": 0.0, "grader_status": status}
            else:
                if suffix not in cache:
                    cache[suffix] = BASE.grade((suffix, question["gold"]))
                score = cache[suffix]
            assert score["correct"] in (0.0, 1.0)
            rows.append({
                "model": MODEL_LABEL, "dataset": question["dataset"], "question_id": question["id"],
                "sample_id": identity, "sample_index": index, "seed": original["seed"],
                "cap_tokens": cap, "original_output_tokens": len(tokens), "used_output_tokens": len(used),
                "prefix_token_ids_sha256": BASE.fingerprint(used), "suffix_status": status,
                "answer_suffix_sha256": BASE.fingerprint(suffix), "answer_suffix_characters": len(suffix or ""),
                "has_complete_box": suffix is not None and BASE.last_box(suffix) is not None,
                "correct": score["correct"], "grader_status": score["grader_status"],
                "auxiliary_boxed_32k_correct": original["correct"],
            })
    return rows


def aggregate(rows, datasets=DATASETS, caps=CAPS):
    labels = {row.get("model", "L+0") for row in rows}
    assert len(labels) == 1, "Cannot aggregate different models together"
    label = labels.pop()
    groups, identities = defaultdict(list), set()
    for row in rows:
        identity = (row["sample_id"], row["cap_tokens"])
        assert identity not in identities, "Duplicate sample at a budget"
        identities.add(identity)
        groups[(row["dataset"], row["cap_tokens"])].append(row)
    assert set(groups) == {(key, cap) for key, _, _ in datasets for cap in caps}
    metrics, question_rows = [], []
    for key, _, count in datasets:
        for cap in caps:
            group = groups[(key, cap)]
            questions = defaultdict(list)
            for row in group:
                assert row["correct"] in (0.0, 1.0)
                questions[row["question_id"]].append(row)
            assert len(group) == count * 4 and len(questions) == count
            assert all({r["sample_index"] for r in q} == {0, 1, 2, 3} and len(q) == 4 for q in questions.values())
            for identity, samples in sorted(questions.items()):
                question_rows.append({
                    "dataset": key, "question_id": identity, "cap_tokens": cap,
                    "mean_at_4": sum(r["correct"] for r in samples) / 4,
                    "pass_at_4": int(any(r["correct"] for r in samples)),
                })
            metrics.append({
                "model": label, "dataset": key, "cap_tokens": cap, "questions": count,
                "responses": len(group), "correct_responses": int(sum(row["correct"] for row in group)),
                "mean_at_4_percent": 100 * sum(row["correct"] for row in group) / len(group),
                "pass_at_4_percent": 100 * sum(any(r["correct"] for r in q) for q in questions.values()) / count,
                "mean_output_tokens": sum(row["used_output_tokens"] for row in group) / len(group),
                "eligible_suffixes": sum(row["suffix_status"] == "eligible" for row in group),
                "grader_status": dict(Counter(row["grader_status"] for row in group)),
            })
    return metrics, question_rows


def after_thinking_reference(root, base):
    """Read the audited Qwen3 reference used by both comparison tables."""
    folder = root / "report/qwen3_after_thinking"
    audit = base.read(folder / "audit.json")
    assert audit["complete"] and audit["after_thinking_only"]
    assert audit["model_repo"] == "Qwen/Qwen3-1.7B"
    for name in ("source_report.md", "main.csv", "budgets.csv"):
        assert base.digest(folder / name) == audit["files_sha256"][name], f"Changed Qwen3 reference: {name}"
    with (folder / "main.csv").open() as stream:
        main = list(csv.DictReader(stream))
    with (folder / "budgets.csv").open() as stream:
        budgets = list(csv.DictReader(stream))
    assert len(main) == 3 and {row["Model"] for row in main} == {"qwen3-1.7B"}
    assert {row["Metric"] for row in main} == {"mean@4 (%)", "pass@4 (%)", "Mean Response Length (tokens)"}
    assert len(budgets) == len(DATASETS) and {row["model"] for row in budgets} == {"qwen3-1.7B"}
    assert {row["dataset"] for row in budgets} == {key for key, _, _ in DATASETS}
    means = next(row for row in main if row["Metric"] == "mean@4 (%)")
    by_dataset = {row["dataset"]: row for row in budgets}
    assert all(means[title] == by_dataset[key]["32k"] for key, title, _ in DATASETS)
    return main, budgets


def write_report(root, base, metrics):
    plan = base.read(root / "plan.json")
    label = plan.get("model_label", "L+0")
    final_label = f"{label} (final)"
    model = base.read(root / "prepared_inputs.json")["model"]
    with (root / "provenance/main_baseline.csv").open() as stream:
        baseline = list(csv.DictReader(stream))
    with (root / "provenance/budget_baseline.csv").open() as stream:
        budgets = list(csv.DictReader(stream))
    training_plan = Path(plan["training_root"]) / "plan.json" if plan.get("training_root") else None
    training = base.read(training_plan) if training_plan and training_plan.exists() else {}
    compression_l0 = label in ("L+0", "L+4096", "ER", "f_cov", "MaxRL") and training.get("dataset_repo") == "zjhhhh/compression_dataset"
    polaris_l0 = plan.get("report_training_dataset") == "Polaris-1-8-3200"
    suffix_reference = compression_l0 or polaris_l0
    if compression_l0:
        final_label = f"{label} step 100 (compression)"
    if polaris_l0:
        final_label = f"{label} step 100 (Polaris-1-8-3200)"
    if suffix_reference:
        baseline, budgets = after_thinking_reference(root, base)
    comparison_run = f"compression {label}" if compression_l0 else f"Polaris {label}"
    baseline_note = (
        "Qwen3-1.7B mean@4, pass@4 and response length come from the 32k column of the complete "
        "after-thinking reference report. Both models use the same after-thinking-only scoring policy "
        "in the main table and all budget columns. "
        f"Qwen3 used top-k -1 and seed 0; this {comparison_run} run uses top-k 20 and seed 42."
        if suffix_reference else
        "Historical Qwen3, MaxRL and ER rows below are preserved from the user's table. "
        "Their original main-table scoring differs from the after-thinking budget regrading; "
        "compare the budget columns when matching that scoring policy. "
        f"Qwen3/MaxRL used top-k -1 and seed 0; local ER and this {label} run use top-k 20 and seed 42."
    )
    lookup = {(r["dataset"], r["cap_tokens"]): r for r in metrics}
    folder = root / "report"
    folder.mkdir(exist_ok=True)
    lines = [
        (f"# Qwen3-1.7B vs {final_label}: nine-benchmark evaluation" if suffix_reference
         else f"# {label} final checkpoint: nine-benchmark evaluation"), "",
        f"Model: [{model['repo']}](https://huggingface.co/{model['repo']}/tree/{model['revision']}).", "",
        *(["Training data: `zjhhhh/compression_dataset` (3,200 questions); checkpoint: step 100.", ""]
          if compression_l0 else []),
        *(["Training data: Polaris-1-8-3200 (3,200 questions); checkpoint: step 100.", ""]
          if polaris_l0 else []),
        f"{label} evaluation: 1,819 questions; 7,276 responses; thinking on; four samples/question; "
        "temperature 0.6, top-p 0.95, top-k 20, min-p 0, seed 42; "
        "32,768 output tokens and 40,960-token context. No system prompt.",
        "Only the nonempty text after the last completed `</think>` is graded with Math-Verify. "
        "Unfinished or reopened thinking scores zero. No extra boxed-answer or EOS requirement.",
        "All six budgets replay exact prefixes of the same output token IDs; no tokens are appended. "
        "1k = 1,024 output tokens, including thinking, answer and EOS, excluding input. All samples stay in the denominator.",
        ("Both models' main-table mean@4 and 32k budget column use exactly the same regraded scores. "
         if suffix_reference else f"The {label} main table and 32k budget column use exactly the same regraded scores. ") +
        "Mean response length includes all outputs, including incorrect and unfinished responses.", "",
        baseline_note, "",
        "| Metric | Model | " + " | ".join(label for _, label, _ in DATASETS) + " |",
        "|---|---|" + "---:|" * len(DATASETS),
    ]
    combined = []
    for metric_label, field, formatting in (
        ("mean@4 (%)", "mean_at_4_percent", ".2f"),
        ("pass@4 (%)", "pass_at_4_percent", ".2f"),
        ("Mean Response Length (tokens)", "mean_output_tokens", ",.1f"),
    ):
        for row in [r for r in baseline if r["Metric"] == metric_label]:
            values = [row[title] for _, title, _ in DATASETS]
            lines.append("| " + " | ".join([metric_label, row["Model"], *values]) + " |")
            combined.append(row)
        values = [format(lookup[(key, 32768)][field], formatting) for key, _, _ in DATASETS]
        lines.append("| " + " | ".join([metric_label, final_label, *values]) + " |")
        combined.append(dict(zip(["Model", "Metric", *[v[1] for v in DATASETS]], [final_label, metric_label, *values])))
    cap_labels = [f"{cap // 1024}k" for cap in CAPS]
    combined_budgets = []
    for key, title, _ in DATASETS:
        lines += ["", f"## {title}", "", "| Model | " + " | ".join(cap_labels) + " |", "|---|" + "---:|" * 6]
        for row in [r for r in budgets if r["dataset"] == key]:
            values = [row[label] for label in cap_labels]
            lines.append("| " + " | ".join([row["model"], *values]) + " |")
            combined_budgets.append(dict(zip(["dataset", "model", *cap_labels], [key, row["model"], *values])))
        values = [f"{lookup[(key, cap)]['mean_at_4_percent']:.2f}" for cap in CAPS]
        lines.append("| " + " | ".join([final_label, *values]) + " |")
        combined_budgets.append(dict(zip(["dataset", "model", *cap_labels], [key, final_label, *values])))
    lines += ["", "[Main table CSV](comparison.csv) · [Budget table CSV](budget_comparison.csv) · "
              f"[{final_label} metrics at all budgets](metrics.json) · [Audit](audit.json) · [Plan](../plan.json)", ""]
    if suffix_reference:
        lines += ["[Qwen3 after-thinking reference audit](qwen3_after_thinking/audit.json) · "
                  "[Source reference report](qwen3_after_thinking/source_report.md)", ""]
    base.write_csv(folder / "comparison.csv", combined)
    base.write_csv(folder / "budget_comparison.csv", combined_budgets)
    (folder / "README.md").write_text("\n".join(lines))
    assert plan["total_responses"] == 7276


def regrade(root, base, workers=8):
    with (root / "prefix_grading.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(root, base)
        manifest = base.read(root / "manifest.json")
        manifest_hash = base.digest(root / "manifest.json")
        assert manifest["inputs"]["questions_sha256"] == base.digest(root / "questions.json")
        assert manifest["evaluator_sha256"] == base.digest(base.__file__)
        jobs = [(str(root), manifest_hash, q) for q in base.read(root / "questions.json")]
        rows = []
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=initialize_grading, initargs=(str(root),)) as pool:
            futures = [pool.submit(score_question, item) for item in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if index % 25 == 0 or index == len(jobs):
                    base.write(root / "status.json", {
                        "state": "grading_prefixes", "completed_responses": plan["total_responses"],
                        "total_responses": plan["total_responses"], "graded_questions": index,
                        "total_questions": len(jobs), "updated_at": time.time(),
                    })
        rows.sort(key=lambda row: (row["question_id"], row["sample_index"], row["cap_tokens"]))
        assert len(rows) == 7276 * 6
        metrics, question_rows = aggregate(rows)
        folder = root / "report"
        folder.mkdir(exist_ok=True)
        base.write(folder / "per_sample.json", rows)
        base.write(folder / "metrics.json", metrics)
        base.write_csv(folder / "per_question.csv", question_rows)
        base.write_csv(folder / "metrics.csv", [
            {key: value for key, value in row.items() if key != "grader_status"} for row in metrics
        ])
        errors = dict(Counter(row["grader_status"] for row in rows if row["grader_status"] not in
                              ("scored", "unfinished_thinking", "reopened_thinking", "empty_answer_suffix")))
        base.write(folder / "audit.json", {
            "complete": not errors, "questions": len(jobs), "responses_verified": 7276,
            "budget_points": len(metrics), "cap_observations": len(rows),
            "source_checksums_tokens_text_seeds": "passed", "all_budgets_regraded": True,
            "grader_errors": errors, "regenerated_at_smaller_budgets": False,
            "per_sample_sha256": base.digest(folder / "per_sample.json"),
            "metrics_sha256": base.digest(folder / "metrics.json"),
        })
        assert not errors, f"Inspect grading errors before finalizing: {errors}"
        write_report(root, base, metrics)
        base.write(root / "status.json", {
            "state": "complete", "completed_questions": len(jobs), "completed_responses": 7276,
            "total_responses": 7276, "dataset_budget_points": len(metrics), "updated_at": time.time(),
        })


def generation_complete(root, base, partial=False):
    assert not partial
    base.write(root / "status.json", {"state": "generated", "completed_responses": 7276,
                                      "total_responses": 7276, "updated_at": time.time()})


def run(root, base):
    plan = verify_plan(root, base)
    prepare_model(root, base, plan)
    shared = load_module("l0_shared_runner", root / "provenance/eval_rloo_final.py")
    shared.report = generation_complete
    shared.run(root, base)
    regrade(root, base)


def queue(root, base):
    plan = verify_plan(root, base)
    require_queue_node(plan)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "hostname": os.uname().nodename,
                 "job_id": plan["job_id"], "gpus": 8, "model_repo": plan["model_repo"],
                 "total_responses": plan["total_responses"], "created_at": time.time()}

        def update(**values):
            state.update(values, updated_at=time.time())
            base.write(root / "queue_status.json", state)

        training = Path(plan["training_root"])
        training_kind = plan.get("training_kind", "l0")
        receipt_path = training / "hf_checkpoint_archive/receipts/global_step_100.json"
        try:
            while True:
                status = training_status(training, base, training_kind)
                receipt = base.read(receipt_path) if receipt_path.exists() else None
                if checkpoint_ready(status, receipt, plan["model_repo"], training_kind):
                    break
                if status.get("state") in ("failed", "training_failed") or status.get("exit_code") not in (None, 0):
                    raise RuntimeError("Training did not complete successfully; final evaluation will not start")
                update(state="waiting_for_training_and_verified_final_checkpoint",
                       training_state=status["state"],
                       last_completed_step=status["last_completed_step"],
                       expected_final_step=100, checkpoint_receipt_exists=receipt is not None)
                time.sleep(30)
            pinned_path = root / "final_checkpoint_receipt.json"
            if pinned_path.exists():
                old = base.read(pinned_path)
                assert old["repo_id"] == receipt["repo_id"] and old["remote_commit"] == receipt["remote_commit"]
                assert old["files"] == receipt["files"]
            else:
                base.write(pinned_path, receipt)
            base.write(root / "training_completion.json", status)
            while True:
                result = subprocess.run(["scontrol", "show", "job", str(plan["job_id"]), "-o"],
                                        text=True, capture_output=True, timeout=30)
                assert result.returncode == 0 and "JobState=RUNNING" in result.stdout, "Evaluation allocation unavailable"
                assert f"UserId={os.environ['USER']}(" in result.stdout, "Allocation belongs to another user"
                command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                           "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                           f"--job-name={plan.get('evaluation_job_name', 'l0-final-eval')}",
                           "bash", str(root / "provenance/run_l0_final_eval.sh"), sys.executable, str(root)]
                update(state="waiting_for_available_gpus_or_evaluating", revision=receipt["remote_commit"],
                       training_state="complete", last_completed_step=100)
                with (root / "launch.log").open("a", buffering=1) as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    update(launcher_pid=child.pid)
                    returncode = child.wait()
                if returncode == 75:
                    update(state="waiting_for_all_eight_gpus", last_returncode=returncode)
                    time.sleep(30)
                    continue
                assert returncode == 0, f"Evaluation exited {returncode}; inspect launch.log"
                audit = base.read(root / "report/audit.json")
                assert audit["complete"] and audit["responses_verified"] == 7276
                assert base.read(root / "status.json")["state"] == "complete"
                update(state="complete", completed_responses=7276, report=str(root / "report/README.md"))
                return
        except BaseException as exc:
            update(state="failed", error=str(exc))
            raise


def launch_queue(root, base):
    plan = verify_plan(root, base)
    require_queue_node(plan)
    with (root / "queue_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = root / "queue_launch.json"
        if receipt_path.exists():
            receipt = base.read(receipt_path)
            pid = receipt["pid"]
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except FileNotFoundError:
                raise RuntimeError("Previous evaluation queue exited; inspect queue.log before restarting") from None
            assert str(root).encode() in command and b"queue" in command, "Queue PID belongs to another process"
            print(json.dumps(receipt), flush=True)
            return
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        env.pop("PYTHONHOME", None)
        with (root / "queue.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([sys.executable, "-u", str(root / "provenance/eval_l0_final.py"),
                                      "queue", "--output-root", str(root)], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": os.uname().nodename, "job_id": plan["job_id"],
                   "launched_at": time.time(), "model_repo": plan["model_repo"]}
        base.write(receipt_path, receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-plan", "launch-queue", "queue", "run", "regrade"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-plan":
        assert args.request is not None
        prepare_plan(root, args.request.resolve())
    else:
        base = evaluator(root)
        if args.command == "launch-queue":
            launch_queue(root, base)
        elif args.command == "queue":
            queue(root, base)
        elif args.command == "run":
            try:
                run(root, base)
            except BaseException as exc:
                base.write(root / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
                raise
        else:
            regrade(root, base, workers=args.workers)


if __name__ == "__main__":
    main()
