"""Evaluate the four pinned coding holdouts with the training LCB grader."""

import gzip
import importlib.metadata
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen3_experiments.code_grading import final_code
from qwen3_experiments.lcb_coding_format import LCB_REVISION, validate_truth
from qwen3_experiments.lcb_coding_grading import LiveCodeBenchGrader
from qwen3_experiments.taco_eval import digest, read, write

COUNTS = {"lcb_v6": 175, "taco_test": 782, "usaco": 307, "code_contests": 165}


def taco_io(tests):
    """Translate TACO's argument lists and singleton result wrappers to LCB."""
    name = tests.get("fn_name")
    if name:
        inputs, outputs = [], []
        for arguments, expected in zip(tests["inputs"], tests["outputs"], strict=True):
            if not isinstance(arguments, (list, str)):
                raise ValueError("Unexpected TACO call arguments")
            if not isinstance(expected, list) or len(expected) != 1:
                raise ValueError("Expected TACO's singleton call-result wrapper")
            inputs.append("\n".join(json.dumps(value, ensure_ascii=False) for value in arguments))
            outputs.append(json.dumps(expected[0], ensure_ascii=False))
    else:
        def stdin_text(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list) and all(isinstance(line, str) for line in value):
                return "\n".join(value)
            raise ValueError("Unsupported TACO stdin/stdout format")

        inputs = [stdin_text(value) for value in tests["inputs"]]
        outputs = [stdin_text(value) for value in tests["outputs"]]
    return {"inputs": inputs, "outputs": outputs, "fn_name": name}


def holdout_truth(dataset, question, usaco_tests=None):
    if dataset in ("lcb_v6", "taco_test"):
        tests = json.loads(question["input_output"])
        if dataset == "taco_test":
            tests = taco_io(tests)
    elif dataset == "code_contests":
        tests = {"inputs": [t["input"] for t in question["tests"]],
                 "outputs": [t["output"] for t in question["tests"]], "fn_name": None}
    elif dataset == "usaco":
        root = Path(usaco_tests).resolve()

        def content(relative):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("USACO test path escapes its source directory")
            return path.read_bytes().decode("utf-8")

        tests = {"inputs": [content(t["input_file"]) for t in question["tests"]],
                 "outputs": [content(t["output_file"]) for t in question["tests"]], "fn_name": None}
    else:
        raise ValueError(dataset)
    return validate_truth({
        "grader": "livecodebench", "schema_version": 1, "revision": LCB_REVISION,
        "unit_test_timeout_seconds": 10, "check_eos": False, "score_after_thinking": True,
        "input_output": tests, "code_prelude": "", "adapter": None,
    })


def prepare_holdouts(plan, baseline_root, competition_root):
    """Retain every source question and test; store large test payloads separately."""
    folder = Path(plan["scratch"]) / "evaluation_data"
    folder.mkdir(parents=True, exist_ok=True)
    competition = read(Path(competition_root) / "plan.json")
    receipt = {}
    for dataset, expected in COUNTS.items():
        origin = Path(baseline_root if dataset in ("lcb_v6", "taco_test") else competition_root)
        source = origin / "data" / f"{dataset}.json"
        questions = read(source)
        if len(questions) != expected or len({q["id"] for q in questions}) != expected:
            raise ValueError(f"Wrong holdout population: {dataset}")
        directory = folder / dataset
        directory.mkdir(exist_ok=True)
        prepared = []
        for question in questions:
            truth = holdout_truth(dataset, question, competition["grading"]["usaco_tests"])
            target = directory / f"{question['source_index']}.json.gz"
            # Preparation can resume after a later configuration check fails.
            # Reuse a payload only after comparing its entire decoded content.
            reusable = False
            if target.exists():
                try:
                    reusable = json.loads(gzip.decompress(target.read_bytes())) == truth
                except (OSError, EOFError, ValueError):
                    pass
            if not reusable:
                target.write_bytes(gzip.compress(json.dumps(truth, ensure_ascii=False).encode(),
                                                 compresslevel=3, mtime=0))
            prepared.append({
                "id": question["id"], "source_index": question["source_index"],
                "difficulty": question.get("difficulty", "unknown"),
                "prompt_token_ids": question["prompt_token_ids"],
                "truth_path": str(target), "truth_sha256": digest(target),
                "test_count": len(truth["input_output"]["inputs"]),
            })
        destination = folder / f"{dataset}.json"
        write(destination, prepared)
        receipt[dataset] = {"questions": expected, "source": str(source), "source_sha256": digest(source),
                            "manifest": str(destination), "manifest_sha256": digest(destination),
                            "tests": sum(q["test_count"] for q in prepared),
                            "max_prompt_tokens": max(len(q["prompt_token_ids"]) for q in prepared)}
    return receipt


def truth_for(question):
    path = Path(question["truth_path"])
    if digest(path) != question["truth_sha256"]:
        raise ValueError("Holdout tests changed")
    return validate_truth(json.loads(gzip.decompress(path.read_bytes())))


def grade_response(grader, question, response):
    code, reason = final_code(response["response"])
    if reason != "ok":
        return {"score": 0.0, "reason": reason, "seconds": 0.0, "results": [], "executed_tests": 0}
    return grader(truth_for(question), code)


def verify_response(record, question, model, plan_hash):
    if (record["id"] != question["id"] or record["index"] != question["source_index"]
            or record["model_revision"] != model["revision"] or record["plan_sha256"] != plan_hash):
        raise ValueError("Saved response belongs to another question, checkpoint, or run")


def worker(plan, dataset, rank):
    from vllm import LLM, SamplingParams

    if importlib.metadata.version("vllm") != "0.24.0":
        raise ValueError("Evaluation must use vLLM 0.24.0")
    manifest = plan["holdouts"][dataset]
    if digest(manifest["manifest"]) != manifest["manifest_sha256"]:
        raise ValueError("Holdout manifest changed")
    questions = read(manifest["manifest"])[rank::8]
    directory = Path(plan["scratch"]) / "evaluation" / dataset
    for subdir in ("responses", "grades"):
        (directory / subdir).mkdir(parents=True, exist_ok=True)
    model = read(Path(plan["scratch"]) / "control_mirrors/final_model.json")
    plan_hash = plan["plan_sha256"]
    grader = LiveCodeBenchGrader(read(plan["grading_plan"]))

    def score_one(question):
        source = directory / "responses" / f"{question['source_index']}.json"
        response = read(source)
        verify_response(response, question, model, plan_hash)
        target = directory / "grades" / source.name
        if target.exists():
            old = read(target)
            if old["response_sha256"] == digest(source):
                return
        result = grade_response(grader, question, response)
        write(target, {"id": question["id"], "index": question["source_index"],
                       "response_sha256": digest(source), "result": result,
                       "tokens": len(response["token_ids"]), "difficulty": question["difficulty"]})

    with ThreadPoolExecutor(max_workers=16) as pool:
        pending, futures = [], []
        for question in questions:
            path = directory / "responses" / f"{question['source_index']}.json"
            if path.exists():
                verify_response(read(path), question, model, plan_hash)
                futures.append(pool.submit(score_one, question))
            else:
                pending.append(question)
        if pending:
            engine = LLM(model=model["path"], tokenizer=model["path"], **plan["evaluation_engine"])
            for offset in range(0, len(pending), 32):
                batch = pending[offset:offset + 32]
                outputs = engine.generate(
                    [{"prompt_token_ids": q["prompt_token_ids"]} for q in batch],
                    [SamplingParams(**plan["sampling"], seed=q["source_index"]) for q in batch], use_tqdm=False,
                )
                if len(outputs) != len(batch):
                    raise ValueError("Missing evaluation generations")
                for question, output in zip(batch, outputs, strict=True):
                    if len(output.outputs) != 1:
                        raise ValueError("Expected pass@1 evaluation")
                    sample = output.outputs[0]
                    write(directory / "responses" / f"{question['source_index']}.json", {
                        "id": question["id"], "index": question["source_index"], "response": sample.text,
                        "token_ids": list(sample.token_ids), "finish_reason": sample.finish_reason,
                        "model_revision": model["revision"], "plan_sha256": plan_hash,
                    })
                    futures.append(pool.submit(score_one, question))
                # Surface grader infrastructure errors during generation, too.
                for future in futures:
                    if future.done():
                        future.result()
        for future in futures:
            future.result()


def summarize(plan, dataset):
    directory = Path(plan["scratch"]) / "evaluation" / dataset
    questions = read(plan["holdouts"][dataset]["manifest"])
    model = read(Path(plan["scratch"]) / "control_mirrors/final_model.json")
    records = []
    for question in questions:
        source = directory / "responses" / f"{question['source_index']}.json"
        response = read(source)
        verify_response(response, question, model, plan["plan_sha256"])
        record = read(directory / "grades" / source.name)
        if record["id"] != question["id"] or record["response_sha256"] != digest(source):
            raise ValueError("Grade/response mismatch")
        if record["result"]["score"] not in (0, 1):
            raise ValueError("Expected binary scores")
        records.append(record)

    def metrics(rows):
        return {"questions": len(rows), "correct": sum(r["result"]["score"] for r in rows),
                "pass_at_1_percent": 100 * sum(r["result"]["score"] for r in rows) / len(rows),
                "mean_response_tokens": sum(r["tokens"] for r in rows) / len(rows)}

    result = {**metrics(records), "grader": "livecodebench", "grader_revision": LCB_REVISION,
              "unit_test_timeout_seconds": 10, "vllm": "0.24.0", "check_eos": False,
              "score_after_thinking": True, "model_revision": model["revision"],
              "by_difficulty": {d: metrics([r for r in records if r["difficulty"] == d])
                                for d in sorted({r["difficulty"] for r in records})}}
    write(directory / "metrics.json", result)
    return result
