"""Prepare immutable TACO training inputs and final code-generation evaluations."""

import argparse
import base64
import importlib.util
import io
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import zlib

from qwen3_experiments.taco_eval import digest, test_case_issue, write

DATASET_REVISION = "7ef9b8ac1260cefe2c03ee054f1a44d13e37c6a5"
DATASET_SHA256 = "2ade74c3ab6cdab2ac9804c1b49ff4cfb06c1d1309b258c59f6933e8b13dedec"
TACO_TEST_SHA256 = "5d99adc603500c05751aff9f61bed7bbd54a9ad5ea569d108b645346655aae44"
TACO_PRETOKENIZING_SHA256 = "4c3ba6b5d95d72b057474f6a5025344f8ef0b0233d00fbff417434bb4920ceb5"


def official_taco_messages(rows, official_root, tokenizer):
    """Call the pinned, unmodified TACO training formatter for each question."""
    source = Path(official_root) / "pretokenizing.py"
    if digest(source) != TACO_PRETOKENIZING_SHA256:
        raise ValueError("TACO official training formatter changed")
    name = "_maxrl_official_taco_pretokenizing"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(official_root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    # initialize() emits one source per reference solution. RL needs one prompt
    # per question, so supply one empty target and discard the target column.
    # Reference solutions never enter the model's prompt.
    inputs = [{"question": row["question"], "starter_code": row["starter_code"],
               "input_output": row["input_output"], "solutions": '[""]'} for row in rows]
    formatted = module.initialize(inputs, tokenizer)
    if len(formatted) != len(rows):
        raise ValueError("The official formatter dropped training questions")
    return [[{"role": "user", "content": prompt}] for prompt in formatted["source"]]


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive benchmark test data are permitted")


def private_tests(value):
    try:
        return json.loads(value)
    except ValueError:
        raw = zlib.decompress(base64.b64decode(value))
        return json.loads(PrimitiveUnpickler(io.BytesIO(raw)).load())


def lcb_messages(row):
    # Official generic code-generation prompt, wrapped in Qwen3's thinking template.
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += ("### Format: You will use the following starter code to write the solution to the problem "
                   "and enclose your code within delimiters.\n"
                   f"```python\n{row['starter_code']}\n```\n\n")
    else:
        prompt += ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
                   "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
                   "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
                   "output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": "You are an expert Python programmer. You will be given a question "
             "(problem specification) and will generate a correct Python program that matches the specification "
             "and passes all tests."}, {"role": "user", "content": prompt}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--previous-taco", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    root, scratch = args.root, args.scratch
    root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    prior = json.loads((args.previous_taco / "plan.json").read_text())
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    official = provenance / "taco_official"
    shutil.copytree(prior["official"], official, dirs_exist_ok=True)
    source = Path(prior["model"]["path"])
    model = scratch / "base_model"
    model.mkdir(exist_ok=True)
    for name, checksum in prior["model"]["files_sha256"].items():
        target = model / name
        if not target.exists():
            shutil.copy2(source / name, target)
        assert digest(target) == checksum, name
    write(provenance / "base_model.json", {**prior["model"], "path": str(model)})
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    subset = args.subset / "data/train-00000-of-00001.parquet"
    assert digest(subset) == DATASET_SHA256
    rows = pq.read_table(subset).to_pylist()
    assert len(rows) == 3200 and len({r["taco_id"] for r in rows}) == 3200
    assert all(r["difficulty"] == "EASY" and r["taco_id"].startswith("taco_train_") for r in rows)
    assert all(test_case_issue(r) is None for r in rows)
    prepared, lengths = [], []
    training_messages = official_taco_messages(rows, official, tokenizer)
    for row, messages in zip(rows, training_messages):
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True)
        lengths.append(len(tokens))
        prepared.append({"data_source": "taco", "prompt": messages, "ability": "code",
                         "reward_model": {"style": "rule", "ground_truth": row["input_output"]},
                         "extra_info": {"split": "train", "index": row["taco_source_index"],
                                        "taco_id": row["taco_id"], "difficulty": "EASY"}})
    data = root / "data"
    data.mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(prepared), data / "train.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(prepared[:1]), data / "unused_validation.parquet", compression="zstd")
    write(provenance / "training_dataset.json", {
        "repo": "hi-todayis-jh/TACO-easy-subset", "revision": DATASET_REVISION,
        "source_sha256": DATASET_SHA256, "split": "train", "rows": 3200, "filtered_rows": 0,
        "max_prompt_tokens": max(lengths), "prompt_cap": math.ceil(max(lengths) / 256) * 256,
        "p50_prompt_tokens": sorted(lengths)[len(lengths) // 2], "thinking": True,
        "spj_filter": False, "validation_enabled": False,
        "prompt_template": {"repo": "FlagOpen/TACO", "revision": prior["official_revision"],
                            "file": "pretokenizing.py", "function": "initialize",
                            "sha256": TACO_PRETOKENIZING_SHA256,
                            "source_content": "verbatim official formatter output",
                            "chat_wrapper": "Qwen3 native chat template; enable_thinking=True"},
    })
    print("Training prompts:", max(lengths), "maximum tokens; all 3200 retained", flush=True)

    test_source = Path(prior["scratch"]) / "dataset_source/ALL/test-00000-of-00001.parquet"
    assert digest(test_source) == TACO_TEST_SHA256
    test = pq.read_table(test_source).to_pylist()
    spj = [json.loads(line) for line in (official / "output_spj.jsonl").read_text().splitlines()]
    assert len(test) == len(spj) == 1000
    assert {r["idx"] for r in spj} == set(range(1000))
    excluded = {r["idx"] for r in spj if r["special_judge"]}
    assert len(excluded) == 218
    questions = []
    test_messages = official_taco_messages(test, official, tokenizer)
    for index, (row, messages) in enumerate(zip(test, test_messages)):
        if index in excluded:
            continue
        questions.append({"id": f"taco_test_{index:04d}", "source_index": index,
                          "difficulty": row["difficulty"], "input_output": row["input_output"],
                          "prompt_token_ids": tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                                             enable_thinking=True),
                          "test_case_issue": test_case_issue(row)})
    write(data / "taco_test.json", questions)
    write(provenance / "taco_test_filter.json", {
        "repo": "BAAI/TACO", "revision": prior["dataset"]["revision"], "source_sha256": TACO_TEST_SHA256,
        "split": "test", "before": 1000, "excluded_spj": sorted(excluded), "after": len(questions),
        "official_revision": prior["official_revision"], "all_difficulties": True,
        "invalid_tests_count": sum(q["test_case_issue"] is not None for q in questions),
    })
    print("TACO non-SPJ test:", len(questions), flush=True)

    api = HfApi(token=False)
    identity_path = provenance / "lcb_dataset.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
    else:
        info = api.dataset_info("livecodebench/code_generation_lite", files_metadata=True)
        file = next(f for f in info.siblings if f.rfilename == "test6.jsonl")
        identity = {"repo": info.id, "revision": info.sha, "version_tag": "v6", "file": "test6.jsonl",
                    "expected_sha256": file.lfs.sha256 if file.lfs else None, "expected_size": file.size}
        write(identity_path, identity)
    path = Path(hf_hub_download(identity["repo"], identity["file"], repo_type="dataset", token=False,
                                revision=identity["revision"], cache_dir=scratch / "dataset_cache"))
    assert path.stat().st_size == identity["expected_size"]
    if identity["expected_sha256"]:
        assert digest(path) == identity["expected_sha256"]
    lcb = [json.loads(line) for line in path.open()]
    assert len(lcb) == 175
    questions = []
    for index, row in enumerate(lcb):
        tests = json.loads(row["public_test_cases"]) + private_tests(row["private_test_cases"])
        sample = {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
                  "fn_name": json.loads(row["metadata"]).get("func_name")}
        assert sample["inputs"]
        questions.append({"id": row["question_id"], "source_index": index, "difficulty": row["difficulty"],
                          "platform": row["platform"], "contest_date": row["contest_date"],
                          "input_output": json.dumps(sample), "prompt_token_ids": tokenizer.apply_chat_template(
                              lcb_messages(row), add_generation_prompt=True, enable_thinking=True),
                          "test_case_issue": None})
    assert len({q["id"] for q in questions}) == len(questions)
    write(data / "lcb_v6.json", questions)
    identity.update(rows=len(questions), sha256=digest(path))
    write(identity_path, identity)
    print("LCB v6:", len(questions), flush=True)


if __name__ == "__main__":
    main()
