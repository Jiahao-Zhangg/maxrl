"""Queue the original Qwen3-1.7B through the unchanged step-80 evaluator."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eval_polaris_step80 as evaluator

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ORIGINAL_REPORT = evaluator.report


def immutable_copy(source, destination):
    if destination.exists():
        assert evaluator.digest(destination) == evaluator.digest(source), "Existing copied input differs"
    else:
        shutil.copy2(source, destination)


def prepare(root, predecessor):
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    source_manifest = evaluator.read(predecessor / "manifest.json")
    assert source_manifest["evaluator_sha256"] == evaluator.digest(evaluator.__file__)
    original_inputs = source_manifest["inputs"]
    assert evaluator.digest(predecessor / "questions.json") == original_inputs["questions_sha256"]
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model_path = root / "model"
        model_path.mkdir(exist_ok=True)
        api = HfApi()
        info = api.model_info(MODEL, revision=REVISION, files_metadata=True)
        assets = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "model.safetensors.index.json"}
        available = {item.rfilename: item for item in info.siblings
                     if item.rfilename in assets or item.rfilename.endswith(".safetensors")}

        def download(name):
            path = Path(hf_hub_download(MODEL, name, revision=REVISION, local_dir=model_path))
            metadata = available[name]
            checksum = evaluator.digest(path)
            assert path.stat().st_size == metadata.size
            if metadata.lfs:
                assert checksum == metadata.lfs.sha256
            else:
                body = path.read_bytes()
                assert hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest() == metadata.blob_id
            print(f"Verified Qwen3 reference file {name}", flush=True)
            return name, {"sha256": checksum, "size": path.stat().st_size}

        with ThreadPoolExecutor(max_workers=4) as pool:
            files = dict(pool.map(download, sorted(available)))
        inventory = {}
        for path in model_path.glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    item = stream.get_slice(key)
                    assert key not in inventory and item.get_dtype() == "BF16"
                    inventory[key] = item.get_shape()
        assert len(inventory) == source_manifest["model"]["tensor_count"]
        assert evaluator.fingerprint(inventory) == source_manifest["model"]["tensor_inventory_sha256"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        comparison_tokenizer = AutoTokenizer.from_pretrained(predecessor / "model", local_files_only=True)
        assert tokenizer.get_vocab() == comparison_tokenizer.get_vocab()
        assert tokenizer.chat_template == comparison_tokenizer.chat_template
        assert evaluator.fingerprint(tokenizer.chat_template) == original_inputs["chat_template_sha256"]
        questions = evaluator.read(predecessor / "questions.json")
        for question in questions:
            ids = tokenizer.apply_chat_template(question["messages"], add_generation_prompt=True, enable_thinking=True)
            assert ids == question["prompt_token_ids"], "Reference prompt tokens differ from step-80"
        config = evaluator.read(model_path / "config.json")
        assert config["max_position_embeddings"] >= original_inputs["max_model_len"]
        generation = evaluator.read(model_path / "generation_config.json")
        original_generation = evaluator.read(predecessor / "model/generation_config.json")
        assert generation["eos_token_id"] == original_generation["eos_token_id"]
        receipt = {"repo": MODEL, "revision": REVISION, "format": "Native Hugging Face BF16 safetensors",
                   "source_files": files, "tensor_count": len(inventory),
                   "tensor_inventory_sha256": evaluator.fingerprint(inventory),
                   "merged_files": {name: entry["sha256"] for name, entry in files.items()}}
        evaluator.write(root / "model_receipt.json", receipt)
        immutable_copy(predecessor / "questions.json", root / "questions.json")
        immutable_copy(predecessor / "input_audit.json", root / "input_audit.json")
        prepared = copy.deepcopy(original_inputs)
        prepared["model"] = {"repo": MODEL, "revision": REVISION}
        prepared["paired_comparison"] = {
            "step80_manifest_sha256": evaluator.digest(predecessor / "manifest.json"),
            "evaluator_sha256": source_manifest["evaluator_sha256"],
            "reference_wrapper_sha256": evaluator.digest(__file__),
            "identical_questions_and_token_ids": True, "identical_seed_function": True,
            "identical_sampling_and_grader": True,
        }
        evaluator.write(root / "prepared_inputs.json", prepared)
        evaluator.write(root / "comparison_plan.json", {
            "predecessor_root": str(predecessor), "question_count": len(questions),
            "response_count": len(questions) * prepared["samples_per_question"],
            "model": prepared["model"], "same_engine_settings": source_manifest["engine"],
        })
        (root / "README.md").write_text(
            "# Qwen3-1.7B reference evaluation\n\n"
            f"Model: [{MODEL}](https://huggingface.co/{MODEL}/tree/{REVISION}).\n\n"
            "Queued after the step-80 evaluation on allocation 146102, using all eight H100 GPUs.\n\n"
            "The same 864 questions, prompt token IDs, four seeds per question, Math-Verify grader, "
            "and inference engine settings are reused. Temperature 0.6, top-p 0.95, top-k 20, "
            "thinking enabled, 32,768 output tokens; 3,456 responses in total. "
            "Datasets: AIME 2024, 2025, 2026, the 674-question Olympiad-Bench subset, and the "
            "published 100-question Polaris-Test.\n\n"
            "Queue progress is in [queue_status.json](queue_status.json). Once started, inference "
            "progress is in `status.json`; final metrics will be written to `report/README.md`, "
            "with a paired comparison in `comparison/README.md`. Raw outputs and token IDs "
            "are saved under `responses/` with checksums.\n"
        )
        print(f"Reference prepared: {len(questions)} identical questions, 3456 identical sampling seeds", flush=True)


def predecessor_ready(root):
    status_file, audit_file = root / "status.json", root / "report/audit.json"
    if not status_file.exists() or not audit_file.exists():
        return False
    status, audit = evaluator.read(status_file), evaluator.read(audit_file)
    return (status.get("state") == "complete" and status.get("completed_responses") == 3456
            and audit.get("complete") is True and audit.get("responses_verified") == 3456
            and audit.get("complete_questions") == 864)


def reference_report(root, partial=False):
    ORIGINAL_REPORT(root, partial=partial)
    path = root / ("partial_report" if partial else "report") / "README.md"
    text = path.read_text().replace("# Step-80 math evaluation", "# Qwen3-1.7B reference evaluation", 1)
    path.write_text(text.replace("model merge provenance", "model file provenance"))


def compare(root, predecessor):
    assert predecessor_ready(predecessor) and predecessor_ready(root)
    before = evaluator.read(predecessor / "manifest.json")
    after = evaluator.read(root / "manifest.json")
    assert before["inputs"]["questions_sha256"] == after["inputs"]["questions_sha256"]
    assert before["engine"] == after["engine"] and before["packages"] == after["packages"]
    assert before["evaluator_sha256"] == after["evaluator_sha256"]
    for field in ("datasets", "sampling", "sampling_seed", "samples_per_question", "grading", "thinking"):
        assert before["inputs"][field] == after["inputs"][field]
    tables = []
    for directory in (predecessor, root):
        with (directory / "report/accuracy.csv").open() as stream:
            tables.append({row["dataset"]: row for row in csv.DictReader(stream)})
    trained, reference = tables
    assert set(trained) == set(reference)
    rows = []
    for dataset, row in trained.items():
        other = reference[dataset]
        assert row["responses"] == other["responses"]
        rows.append({"dataset": dataset, "questions": int(row["questions_total"]),
                     "responses_per_model": int(row["responses"]),
                     "step80_accuracy_percent": float(row["accuracy_percent"]),
                     "qwen3_accuracy_percent": float(other["accuracy_percent"]),
                     "step80_minus_qwen3_percentage_points": float(row["accuracy_percent"]) - float(other["accuracy_percent"]),
                     "step80_mean_output_tokens": float(row["mean_output_tokens"]),
                     "qwen3_mean_output_tokens": float(other["mean_output_tokens"])})
    directory = root / "comparison"
    directory.mkdir(exist_ok=True)
    evaluator.write_csv(directory / "accuracy_comparison.csv", rows)
    lines = ["# Step-80 versus original Qwen3-1.7B", "",
             "Mean@4 accuracy with identical questions, prompt tokens, seeds, generation settings, and grading.", "",
             "| Dataset | Step-80 | Qwen3-1.7B | Difference (pp) |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['step80_accuracy_percent']:.2f}% | {row['qwen3_accuracy_percent']:.2f}% | {row['step80_minus_qwen3_percentage_points']:+.2f} |")
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    evaluator.write(directory / "provenance.json", {"step80_manifest_sha256": evaluator.digest(predecessor / "manifest.json"),
                    "qwen3_manifest_sha256": evaluator.digest(root / "manifest.json"), "paired_protocol_verified": True})


def run(root, predecessor):
    prepared = evaluator.read(root / "prepared_inputs.json")
    paired = prepared["paired_comparison"]
    assert prepared["model"] == {"repo": MODEL, "revision": REVISION}
    assert paired["reference_wrapper_sha256"] == evaluator.digest(__file__)
    assert paired["step80_manifest_sha256"] == evaluator.digest(predecessor / "manifest.json")
    assert paired["evaluator_sha256"] == evaluator.digest(evaluator.__file__)
    assert predecessor_ready(predecessor)
    evaluator.MODEL, evaluator.REVISION = MODEL, REVISION
    evaluator.report = reference_report
    evaluator.run(root)
    compare(root, predecessor)


def queue(root, predecessor, job_id, holder_lock):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "waiting_for_step80", "job_id": job_id, "gpus": 8,
                 "predecessor_root": str(predecessor), "pid": os.getpid(), "created_at": time.time(),
                 "model": MODEL, "total_responses": 3456}

        def update(**values):
            state.update(values, updated_at=time.time())
            evaluator.write(root / "queue_status.json", state)

        update()
        while not predecessor_ready(predecessor):
            prior = evaluator.read(predecessor / "status.json")
            update(predecessor_state=prior.get("state"),
                   predecessor_completed_responses=prior.get("completed_responses"))
            if prior.get("state") == "failed":
                update(state="waiting_for_step80_recovery", predecessor_error=prior.get("error"))
            time.sleep(30)
        description = subprocess.check_output(["scontrol", "show", "job", str(job_id), "-o"], text=True)
        assert "JobState=RUNNING" in description and f"UserId={os.environ['USER']}(" in description
        update(state="launching_reference")
        command = ["srun", f"--jobid={job_id}", "--overlap", "--nodes=1", "--ntasks=1",
                   "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1", "--job-name=qwen3-reference-eval",
                   "bash", str(Path(__file__).with_name("run_qwen3_reference_eval.sh")),
                   sys.executable, str(root), str(predecessor), str(holder_lock)]
        with (root / "launch.log").open("a", buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            update(state="running_reference", launcher_pid=child.pid)
            returncode = child.wait()
        if returncode != 0:
            update(state="failed", returncode=returncode)
            raise RuntimeError(f"Reference evaluation exited {returncode}")
        assert predecessor_ready(root)
        update(state="complete", completed_responses=3456)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "queue", "run", "compare"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--predecessor-root", type=Path, required=True)
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--holder-lock", type=Path)
    args = parser.parse_args()
    root, predecessor = args.output_root.resolve(), args.predecessor_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(root, predecessor)
    elif args.command == "run":
        run(root, predecessor)
    elif args.command == "compare":
        compare(root, predecessor)
    else:
        assert args.job_id is not None and args.holder_lock is not None
        queue(root, predecessor, args.job_id, args.holder_lock.resolve())


if __name__ == "__main__":
    main()
