"""Compare raw-response, EOS, and after-thinking grading on the same saved samples."""

import argparse
import ast
import csv
import fcntl
import gzip
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

MODELS = {"qwen3": "Qwen3-1.7B", "deepseek_r1": "DeepSeek-R1-Distill-Qwen-1.5B"}
DATASETS = {"polaris_train": "Polaris-1-8-3200", "compression_train": "compression_dataset"}
POLICIES = ("raw_stored", "raw_stored_eos", "after_thinking", "after_thinking_eos",
            "raw_decoded", "raw_decoded_eos")
BASE = TOKENIZER = HELPER = TRAINING_SUFFIX = ROOT = PREVIOUS = MANIFEST_HASH = None
EXTRACTIONS = None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def load_training_suffix(path):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "response_after_thinking"]
    assert len(functions) == 1
    scope = {"Optional": Optional}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope["response_after_thinking"]


def initialize(source, destination, model):
    global BASE, TOKENIZER, HELPER, TRAINING_SUFFIX, ROOT, PREVIOUS, MANIFEST_HASH
    from transformers import AutoTokenizer

    ROOT = Path(source) / "runs" / model
    output = Path(destination)
    BASE = module("comparison_grading_core", ROOT / "provenance/eval_polaris_step80.py")
    HELPER = module("comparison_suffix_helper", Path(source) / "provenance/eval_l0_final.py")
    TRAINING_SUFFIX = load_training_suffix(output / "provenance/multi_thread_naive.py")
    TOKENIZER = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    assert TOKENIZER.eos_token_id is not None
    BASE.init_grader()
    metric = BASE.METRIC

    def capturing_metric(golds, predictions):
        global EXTRACTIONS
        score, feedback = metric(golds, predictions)
        EXTRACTIONS = feedback
        return score, feedback

    BASE.METRIC = capturing_metric
    PREVIOUS = {row["sample_id"]: row for row in BASE.read(ROOT / "report/per_sample.json")}
    MANIFEST_HASH = BASE.digest(ROOT / "manifest.json")
    manifest = BASE.read(ROOT / "manifest.json")
    assert BASE.digest(ROOT / "questions.json") == manifest["inputs"]["questions_sha256"]
    for name in ("math-verify", "sympy", "latex2sympy2_extended", "transformers"):
        assert importlib.metadata.version(name) == manifest["packages"][name]


def score(text, gold, blocked_status=None):
    global EXTRACTIONS
    if text is None:
        return {"correct": 0.0, "grader_status": blocked_status, "extracted_predictions": None}
    EXTRACTIONS = None
    result = BASE.grade((text, gold))
    result["extracted_predictions"] = EXTRACTIONS[1] if EXTRACTIONS else None
    return result


def evaluate_question(question):
    rows = []
    for index in range(4):
        identity = BASE.sample_id(question, index)
        original = BASE.saved_result(ROOT, identity, MANIFEST_HASH, full=True)
        assert original is not None and original["question_id"] == question["id"]
        assert original["sample_index"] == index and original["dataset"] == question["dataset"]
        assert original["seed"] == BASE.sample_seed(question["id"], index)
        assert original["prompt_tokens"] == len(question["prompt_token_ids"])
        tokens = original["output_token_ids"]
        assert 0 < len(tokens) == original["output_tokens"] <= 32768
        assert original["finish_reason"] in ("stop", "length")
        assert original["finish_reason"] != "length" or len(tokens) == 32768
        raw = TOKENIZER.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert raw == original["response"]
        decoded = TOKENIZER.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Match the current reward manager's actual decoder as well as the saved evaluation.
        assert decoded == TOKENIZER.decode(tokens, skip_special_tokens=True)
        suffix, suffix_status = HELPER.answer_suffix(decoded)
        assert suffix == TRAINING_SUFFIX(decoded)
        previous = PREVIOUS[identity]
        assert previous["prefix_token_ids_sha256"] == BASE.fingerprint(tokens)
        assert previous["answer_suffix_sha256"] == BASE.fingerprint(suffix)
        assert previous["suffix_status"] == suffix_status

        # The response tokens are unpadded; neither prompt EOS nor artificial padding is present.
        contains_eos = TOKENIZER.eos_token_id in tokens
        stop_ids = BASE.read(ROOT / "model/generation_config.json")["eos_token_id"]
        stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids
        contains_stop = any(token in stop_ids for token in tokens)
        raw_result = score(raw, question["gold"])
        decoded_result = raw_result if raw == decoded else score(decoded, question["gold"])
        suffix_result = score(suffix, question["gold"], suffix_status)
        outcomes = {"raw_stored": raw_result, "raw_decoded": decoded_result, "after_thinking": suffix_result}
        scores = {name: outcomes[name]["correct"] for name in outcomes}
        scores.update({name + "_eos": value * int(contains_eos) for name, value in list(scores.items())})
        assert set(scores) == set(POLICIES) and all(value in (0.0, 1.0) for value in scores.values())
        rows.append({"model": ROOT.name, "dataset": question["dataset"], "question_id": question["id"],
                     "sample_id": identity, "sample_index": index, "seed": original["seed"],
                     "output_tokens": len(tokens), "finish_reason": original["finish_reason"],
                     "stop_reason": original["stop_reason"], "last_token_id": tokens[-1],
                     "tokenizer_eos_token_id": TOKENIZER.eos_token_id, "contains_eos": contains_eos,
                     "contains_configured_stop": contains_stop, "suffix_status": suffix_status,
                     "raw_text_sha256": BASE.fingerprint(raw), "raw_decoded_sha256": BASE.fingerprint(decoded),
                     "suffix_sha256": BASE.fingerprint(suffix), "tokens_sha256": BASE.fingerprint(tokens),
                     "baseline_after_thinking_score": previous["correct"],
                     "baseline_reproduced": previous["correct"] == suffix_result["correct"],
                     "scores": scores, "grading": outcomes})
    return rows


def gated_status(row, policy):
    if policy.endswith("_eos") and not row["contains_eos"]:
        return "missing_eos"
    return row["grading"][policy.removesuffix("_eos")]["grader_status"]


def report(source, destination, base, rows):
    metrics, comparisons, transitions, distributions = [], [], [], []
    for model, label in MODELS.items():
        for dataset, dataset_label in DATASETS.items():
            group = [row for row in rows if row["model"] == model and row["dataset"] == dataset]
            by_question = defaultdict(list)
            for row in group:
                by_question[row["question_id"]].append(row)
            assert len(group) == 800 and len(by_question) == 200
            assert all(len(items) == 4 for items in by_question.values())
            summary = {"model": label, "dataset": dataset_label}
            for policy in POLICIES:
                correct = sum(row["scores"][policy] for row in group)
                passed = sum(any(row["scores"][policy] for row in items) for items in by_question.values())
                metric = {"model": model, "dataset": dataset, "policy": policy, "questions": 200,
                          "responses": 800, "correct_responses": int(correct), "mean_at_4_percent": correct / 8,
                          "pass_at_4_percent": passed / 2,
                          "grader_status": dict(Counter(gated_status(row, policy) for row in group))}
                metrics.append(metric)
                summary[policy + "_mean_at_4_percent"] = correct / 8
                bins = Counter(int(sum(row["scores"][policy] for row in items)) for items in by_question.values())
                for k in range(5):
                    distributions.append({"model": model, "dataset": dataset, "policy": policy,
                                          "correct_of_4": k, "question_count": bins[k], "question_percent": bins[k] / 2})
            summary["raw_minus_eos_after_thinking_pp"] = (
                summary["raw_stored_mean_at_4_percent"] - summary["after_thinking_eos_mean_at_4_percent"])
            summary["raw_1_to_strict_0"] = sum(row["scores"]["raw_stored"] == 1 and
                                               row["scores"]["after_thinking_eos"] == 0 for row in group)
            summary["raw_0_to_strict_1"] = sum(row["scores"]["raw_stored"] == 0 and
                                               row["scores"]["after_thinking_eos"] == 1 for row in group)
            summary["missing_eos_responses"] = sum(not row["contains_eos"] for row in group)
            summary["invalid_thinking_responses"] = sum(row["suffix_status"] != "eligible" for row in group)
            summary["raw_correct_missing_eos"] = sum(row["scores"]["raw_stored"] and not row["contains_eos"] for row in group)
            summary["suffix_correct_missing_eos"] = sum(row["scores"]["after_thinking"] and not row["contains_eos"] for row in group)
            summary["raw_special_token_score_differences"] = sum(row["scores"]["raw_stored"] != row["scores"]["raw_decoded"] for row in group)
            comparisons.append(summary)
            for old, new in (("raw_stored", "raw_stored_eos"), ("raw_stored", "after_thinking"),
                             ("after_thinking", "after_thinking_eos"), ("raw_stored_eos", "after_thinking_eos"),
                             ("raw_stored", "after_thinking_eos")):
                pairs = Counter((int(row["scores"][old]), int(row["scores"][new])) for row in group)
                transitions.append({"model": model, "dataset": dataset, "from": old, "to": new,
                                    **{f"{a}_to_{b}": pairs[(a, b)] for a in range(2) for b in range(2)}})
    base.write(destination / "per_sample.json", rows)
    base.write(destination / "metrics.json", metrics)
    base.write_csv(destination / "metrics.csv", [{k: v for k, v in row.items() if k != "grader_status"} for row in metrics])
    base.write_csv(destination / "comparison.csv", comparisons)
    base.write_csv(destination / "transitions.csv", transitions)
    base.write_csv(destination / "question_distributions.csv", distributions)
    changed = [row for row in rows if row["scores"]["raw_stored"] != row["scores"]["after_thinking_eos"]]
    changed_rows = [
        {k: row[k] for k in ("model", "dataset", "question_id", "sample_id", "output_tokens", "finish_reason",
                              "contains_eos", "suffix_status")} | row["scores"] for row in changed]
    if changed_rows:
        base.write_csv(destination / "changed_samples.csv", changed_rows)
    else:
        (destination / "changed_samples.csv").write_text("model,dataset,question_id,sample_id\n")
    errors = Counter(result["grader_status"] for row in rows for result in row["grading"].values()
                     if result["grader_status"] not in ("scored", "unfinished_thinking", "empty_answer_suffix", "reopened_thinking"))
    mismatches = [dict(model=row["model"], sample_id=row["sample_id"], old=row["baseline_after_thinking_score"],
                       new=row["scores"]["after_thinking"]) for row in rows if not row["baseline_reproduced"]]
    audit = {"complete": not errors and not mismatches, "responses_verified": len(rows),
             "generation_reused": True, "new_model_inference": False, "baseline_mismatches": mismatches,
             "grader_errors": dict(errors), "literal_raw_vs_decoded_differences": sum(
                 row["scores"]["raw_stored"] != row["scores"]["raw_decoded"] for row in rows),
             "secondary_stop_without_tokenizer_eos": sum(row["contains_configured_stop"] and not row["contains_eos"] for row in rows),
             "finish_reason_stop_without_eos": sum(row["finish_reason"] == "stop" and not row["contains_eos"] for row in rows),
             "per_sample_sha256": base.digest(destination / "per_sample.json"),
             "metrics_sha256": base.digest(destination / "metrics.json")}
    base.write(destination / "audit.json", audit)
    lines = ["# Same-sample grading comparison", "",
             "同一批 3,200 个回答：两套训练集各 200 题 × 两个模型 × 每题 4 次。没有重新生成或补 EOS；被规则排除的回答计 0，分母始终为每组 800 个回答。", "",
             "四种主要规则仅改变 EOS 门控与送入 Math-Verify 的文本：", "",
             "- **Raw**：保存的完整生成字符串原样送入 Math-Verify，包含 thinking 和生成的特殊标记。",
             "- **Raw + EOS**：同上，但实际生成的 response token 中必须含 `tokenizer.eos_token_id`，与当前 `check_eos=True` 实现一致。",
             "- **After think**：按原评测去除特殊 token，只取最后一个完成的 `</think>` 后的非空回答；未结束、重新开启未结束的 thinking 或空后缀计 0。这是之前报告的规则。",
             "- **After think + EOS**：同时满足后两项条件，即这次请求的严格规则。", "",
             "另算 `raw_decoded` / `raw_decoded_eos`，按训练 reward manager 的 `skip_special_tokens=True` 解码完整 response 后判分，核对特殊标记是否影响 Raw 对比。"
             "所有规则保留相同 Math-Verify 0.9.0 配置、参考答案和 1 秒外层超时，不增加 boxed 要求。", "",
             "| Model | Dataset | Raw | Raw + EOS | After think（旧） | After think + EOS | Raw − 严格 (pp) | 1→0 / 0→1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        values = [f"{row[key + '_mean_at_4_percent']:.2f}" for key in POLICIES[:4]]
        lines.append("| " + " | ".join([row["model"], row["dataset"], *values,
                      f"{row['raw_minus_eos_after_thinking_pp']:+.3f}",
                      f"{row['raw_1_to_strict_0']} / {row['raw_0_to_strict_1']}"]) + " |")
    lines += ["", "以上数值为 mean@4 (%)；每增加或减少 1 个正确回答，对应 0.125 个百分点。两种方向的翻转分别列出，以免净差抵消。", "",
              "| Model | Dataset | 无 EOS | thinking / 后缀无效 | Raw 判对但无 EOS | 最终回答判对但无 EOS |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append("| " + " | ".join(str(row[k]) for k in ("model", "dataset", "missing_eos_responses",
                     "invalid_thinking_responses", "raw_correct_missing_eos", "suffix_correct_missing_eos")) + " |")
    lines += ["", f"重新判分与原 After think 基线不一致：{len(mismatches)} 个回答。"
              f"原样 Raw 与去掉特殊 token 后的完整文本判分不同：{audit['literal_raw_vs_decoded_differences']} 个。", "",
              "注意：EOS 检查只看实际生成的 token，不看 prompt 或 padding，也不把达到 32k 本身当作缺少 EOS。"
              "本比较衡量判分规则对这些样本的影响；Raw 的高分可能来自 thinking 内的数学表达式，需要结合变化样本检查。", "",
              "[CSV 汇总](comparison.csv) · [所有规则的 mean@4 / pass@4](metrics.csv) · [逐样本结果](per_sample.json) · "
              "[变化样本](changed_samples.csv) · [分布 CSV](question_distributions.csv) · [逐项变化归因](transitions.csv) · "
              "[核验](audit.json) · [例子](examples.md)", ""]
    (destination / "README.md").write_text("\n".join(lines))
    save_examples(source, destination, base, changed)
    assert not errors and not mismatches, f"Review errors / baseline mismatches: {dict(errors)}, {mismatches}"
    return comparisons


def save_examples(source, destination, base, changed):
    counts = Counter()
    lines = ["# Samples whose score changes", "", "每个模型、数据集、变化方向最多展示 3 个样本，按 sample ID 排序选择。完整变化清单见 CSV。", ""]
    example_dir = destination / "examples"
    example_dir.mkdir(exist_ok=True)
    questions = {model: {q["id"]: q for q in base.read(source / "runs" / model / "questions.json")} for model in MODELS}
    for row in sorted(changed, key=lambda x: (x["model"], x["dataset"], x["sample_id"])):
        direction = f"{int(row['scores']['raw_stored'])}→{int(row['scores']['after_thinking_eos'])}"
        key = (row["model"], row["dataset"], direction)
        if counts[key] >= 3:
            continue
        counts[key] += 1
        question = questions[row["model"]][row["question_id"]]
        path = source / "runs" / row["model"] / "responses" / (row["sample_id"] + ".json.gz")
        with gzip.open(path, "rt") as stream:
            original = json.load(stream)
        name = row["model"] + "__" + row["sample_id"] + ".txt"
        (example_dir / name).write_text("Question\n" + question["problem"] + "\n\nGold\n" + question["gold"] +
                                       "\n\nScores\n" + json.dumps(row["scores"], indent=2) + "\n\nRaw response\n" + original["response"])
        lines += [f"## {MODELS[row['model']]} · {row['sample_id']} · {direction}", "",
                  f"Gold: `{question['gold']}`; EOS: {row['contains_eos']}; finish: {row['finish_reason']}; "
                  f"suffix: {row['suffix_status']}; tokens: {row['output_tokens']}.", "",
                  "Raw extracted predictions: `" + str(row["grading"]["raw_stored"]["extracted_predictions"])[:1800] + "`", "",
                  "After-thinking extracted predictions: `" + str(row["grading"]["after_thinking"]["extracted_predictions"])[:1800] + "`", "",
                  f"[完整题目及 raw response](examples/{name})", ""]
    (destination / "examples.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = module("comparison_main_core", source / "provenance/eval_polaris_step80.py")
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = base.read(output / "plan.json")
        for name, checksum in plan["sources_sha256"].items():
            assert base.digest(name) == checksum, name
        state = {"state": "grading", "pid": os.getpid(), "hostname": os.uname().nodename,
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                 "workers": args.workers, "total_responses": 3200, "started_at": time.time()}
        base.write(output / "status.json", state)
        try:
            rows = []
            for model in MODELS:
                questions = base.read(source / "runs" / model / "questions.json")
                assert len(questions) == 400
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=initialize, initargs=(str(source), str(output), model)) as pool:
                    futures = [pool.submit(evaluate_question, question) for question in questions]
                    for future in as_completed(futures):
                        rows.extend(future.result())
                        if len(rows) % 100 == 0:
                            state.update(model=model, completed_responses=len(rows), updated_at=time.time())
                            base.write(output / "status.json", state)
                            print(f"Graded {len(rows)}/3200 saved responses", flush=True)
                base.write(output / f"{model}_per_sample.json", [row for row in rows if row["model"] == model])
            assert len(rows) == 3200
            rows.sort(key=lambda row: (row["model"], row["dataset"], row["sample_id"]))
            comparisons = report(source, output, base, rows)
            state.update(state="complete", completed_responses=3200, finished_at=time.time(), updated_at=time.time())
            base.write(output / "status.json", state)
            print(json.dumps(comparisons, indent=2), flush=True)
        except BaseException as exc:
            state.update(state="failed", error=repr(exc), updated_at=time.time())
            base.write(output / "status.json", state)
            raise


if __name__ == "__main__":
    main()
