"""Queue Polaris L+0 nine-benchmark evaluation and three step-100 Minerva sweeps."""

import argparse
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from qwen3_experiments import er_compression_evaluation as shared
from qwen3_experiments import eval_l0_final as nine
from qwen3_experiments import minerva_individual_budget as mini
from qwen3_experiments.grpo_compute_control import digest, ensure_public_repository, now, read, require_compute, write
from qwen3_experiments.math_eval_matrix_common import fingerprint

MODULE = "qwen3_experiments.polaris_checkpoint_evaluation"
MODELS = [
    {"key": "polaris_l0_step100", "label": "Polaris L+0 step 100", "format": "fsdp8",
     "repo": "hi-todayis-jh/per-context-rb-l0-0-qwen3-1.7b-polaris-1-8-3200-bs32-32k-146103-step_100"},
    {"key": "polaris_er_step100", "label": "Polaris ER step 100", "format": "zero3",
     "repo": "hi-todayis-jh/rloo-qwen3-1.7b-polaris-official-hybrid8-bs32-n8-b128-32k-146102-step_100"},
    {"key": "polaris_maxrl_step100", "label": "Polaris MaxRL step 100", "format": "fsdp8",
     "repo": "hi-todayis-jh/maxrl-qwen3-1.7b-polaris-1-8-3200-bs32-32k-145514-step_100"},
]
L0_KEY = MODELS[0]["key"]


def verify_plan(root):
    plan = read(root / "plan.json")
    if digest(Path(plan["predecessor_root"]) / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("The preceding ER evaluation plan changed")
    if (root / "launch.json").exists() and read(root / "launch.json")["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Polaris evaluation plan changed after launch")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen Polaris evaluation input changed: {relative}")
    return plan


def predecessor_ready(plan):
    previous = Path(plan["predecessor_root"])
    status = read(previous / "queue_status.json")
    if status.get("state") == "failed":
        raise RuntimeError("The preceding ER evaluation failed")
    if status.get("state") != "complete":
        return False
    old = read(previous / "plan.json")
    if status.get("nine_responses") != 7276 or status.get("minerva_points") != 5:
        raise ValueError("Predecessor queue has incomplete evaluation counts")
    if not shared.nine_complete(Path(old["nine_root"])):
        raise ValueError("ER nine-benchmark evaluation is incomplete")
    folder = Path(old["minerva_root"])
    audit = read(folder / "report/audit.json")
    if (read(folder / "status.json").get("state") != "complete" or not audit.get("complete")
            or audit.get("points") != 5 or audit.get("questions_per_point") != 272
            or not audit.get("all_rollout_ledgers_verified")
            or digest(folder / "report/metrics.json") != audit.get("metrics_sha256")):
        raise ValueError("ER Minerva evaluation is incomplete or changed")
    return True


def prepare(args):
    from huggingface_hub import HfApi

    source = Path(__file__).resolve().parents[1]
    previous = args.after_evaluation.resolve()
    predecessor = shared.verify_plan(previous)
    node = require_compute(predecessor["job_id"])
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise RuntimeError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = verify_plan(root)
            if plan["predecessor_root"] != str(previous):
                raise ValueError("Existing queue has a different predecessor")
            return plan
        runtime = root / "runtime"
        for name in ("polaris_checkpoint_evaluation.py", "er_compression_evaluation.py", "eval_l0_final.py",
                     "eval_polaris_step80.py", "minerva_individual_budget.py", "math_eval_budget_engine.py",
                     "math_eval_matrix_common.py", "grpo_compute_control.py", "prepare_math_eval_matrix.py"):
            shared.copy_file(source / "qwen3_experiments" / name, runtime / "qwen3_experiments" / name)
        shared.copy_file(source / "scripts/model_merger.py", runtime / "scripts/model_merger.py")
        api, models = HfApi(), []
        for spec in MODELS:
            ensure_public_repository(api, spec["repo"])
            info = api.model_info(spec["repo"], files_metadata=True)
            if info.private:
                raise ValueError("Project evaluation checkpoints must be public")
            prefix = "global_step_100/actor/"
            names = {f.rfilename for f in info.siblings}
            required = ([prefix + f"model_world_size_8_rank_{rank}.pt" for rank in range(8)] if spec["format"] == "fsdp8"
                        else [prefix + f"{kind}_pp_rank_{rank}_mp_rank_00_{suffix}_states.pt"
                              for rank in range(8) for kind, suffix in (("zero", "model"), ("bf16_zero", "optim"))])
            if not set(required) <= names:
                raise ValueError(f"Incomplete step-100 checkpoint: {spec['repo']}")
            models.append({**spec, "revision": info.sha, "step": 100})
        reference = Path(predecessor["nine_root"])
        nroot = root / "nine_datasets"
        for name in ("questions.json", "provenance/main_baseline.csv", "provenance/budget_baseline.csv", "provenance/eval_rloo_final.py"):
            shared.copy_file(reference / name, nroot / name)
        for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
            shared.copy_file(runtime / "qwen3_experiments" / name, nroot / "provenance" / name)
        for name in ("main.csv", "budgets.csv", "source_report.md", "audit.json"):
            shared.copy_file(reference / "report/qwen3_after_thinking" / name, nroot / "report/qwen3_after_thinking" / name)
        inputs = read(reference / "input_plan.json")
        inputs["model"] = {key: models[0][key] for key in ("repo", "revision")}
        if digest(nroot / "questions.json") != inputs["questions_sha256"]:
            raise ValueError("Nine-benchmark questions differ from the current ER evaluation")
        write(nroot / "input_plan.json", inputs)
        nplan = {**read(reference / "plan.json"), "training_root": None, "training_kind": "archived_checkpoint",
                 "model_label": "L+0", "model_repo": models[0]["repo"], "model_format": "fsdp8",
                 "report_training_dataset": "Polaris-1-8-3200", "repository": str(runtime),
                 "merger_sha256": digest(runtime / "scripts/model_merger.py"),
                 "frozen_files": {p.relative_to(nroot).as_posix(): digest(p) for p in nroot.rglob("*")
                                  if p.is_file() and p != nroot / "plan.json"}}
        write(nroot / "plan.json", nplan)
        mroot = root / "minerva_individual_budget_seed0"
        reference_minerva = Path(predecessor["minerva_root"])
        rows = read(reference_minerva / "questions.json")
        if len(rows) != 272:
            raise ValueError("Expected all 272 Minerva questions")
        original = read(reference_minerva / "plan.json")
        for spec in models:
            folder = mroot / spec["key"]
            write(folder / "questions.json", rows)
            mplan = {**original, "parent_eval_root": str(nroot), "parent_eval_plan_sha256": digest(nroot / "plan.json"),
                     "models": {spec["key"]: {**spec, "family": "qwen3", "questions_file": "questions.json"}},
                     "frozen_files": {"questions.json": digest(folder / "questions.json")}}
            write(folder / "plan.json", mplan)
            for rank in range(8):
                shard = folder / "shards" / str(rank)
                write(shard / "questions.json", [{**row, "seed_position": i} for i, row in enumerate(rows) if i % 8 == rank])
                for name in ("eval_l0_final.py", "eval_polaris_step80.py"):
                    shared.copy_file(nroot / "provenance" / name, shard / "provenance" / name)
                write(shard / "plan.json", {**mplan, "num_questions": 34, "budgets": sorted(mini.BUDGETS, reverse=True),
                      "frozen_files": {p.relative_to(shard).as_posix(): digest(p) for p in shard.rglob("*")
                                       if p.is_file() and p != shard / "plan.json"}})
        training = read(Path(predecessor["training_root"]) / "plan.json")
        plan = {"job_id": predecessor["job_id"], "node": node, "python_bin": predecessor["python_bin"],
                "zero3_python": training["python_bin"], "base_model": training["model_path"],
                "base_model_revision": training["model_revision"], "base_files_sha256": training["model_files_sha256"],
                "runtime": str(runtime), "scratch": str(Path("/tmp") / f"polariseval{predecessor['job_id']}"),
                "holder_locks": predecessor["holder_locks"], "predecessor_root": str(previous),
                "predecessor_plan_sha256": digest(previous / "plan.json"), "models": models,
                "cleanup_each_model_after_evaluation": True,
                "nine_root": str(nroot), "minerva_root": str(mroot), "created_at": now(),
                "frozen_files": {p.relative_to(root).as_posix(): digest(p) for base in (runtime, nroot, mroot)
                                 for p in base.rglob("*") if p.is_file()}}
        write(root / "plan.json", plan)
        (root / "README.md").write_text(
            "# Polaris step-100 evaluations after compression ER\n\n"
            "Wait for the current compression ER nine-benchmark evaluation and all five Minerva budgets. "
            "All supervisors and workers remain on the allocated compute node.\n\n"
            "1. Polaris L+0 step100: the same nine benchmarks, 1,819 questions / 7,276 responses, "
            "four samples, seed 42, 32k response cap, and exact 1k/2k/4k/8k/16k/32k prefix regrading.\n"
            "2. Polaris L+0, ER and MaxRL step100: Minerva Math, 272 questions, seed 0, "
            "individual budgets 8k/16k/32k/48k/64k. Each attempt is capped at 32k or the remaining "
            "budget; stop at first success. Eight GPUs process disjoint question shards with unchanged seeds.\n\n"
            "Both stages: thinking on, temperature 0.6 / top-p 0.95 / top-k 20; grade only the nonempty "
            "answer after completed thinking, with no extra EOS/boxed requirement. Full response text and "
            "token IDs are retained. This report contains Polaris-trained models.\n\n"
            + "\n".join(f"- {s['label']}: `{s['repo']}@{s['revision']}`" for s in models)
            + "\n\n[Queue](queue_status.json) · [Nine datasets](nine_datasets/report/README.md) · "
            "[Three-model Minerva](minerva_individual_budget_seed0/report/README.md)\n")
        return plan


def convert_zero3(source, destination, base):
    import torch
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    from safetensors.torch import save_file
    from qwen3_experiments.prepare_math_eval_matrix import ASSETS

    for rank in range(8):
        state = torch.load(source / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt", map_location="cpu", mmap=True, weights_only=False)
        if state.get("global_steps") != 200:
            raise ValueError("ER step100 must contain 200 optimizer updates")
        del state
    state = get_fp32_state_dict_from_zero_checkpoint(str(source.parent), tag=source.name, lazy_mode=True)
    converted = {name: tensor.contiguous().to(torch.bfloat16).clone() for name, tensor in state.items()}
    if read(base / "config.json").get("tie_word_embeddings") and "lm_head.weight" in converted:
        if not torch.equal(converted["lm_head.weight"], converted["model.embed_tokens.weight"]):
            raise ValueError("Converted ER tied embeddings disagree")
    destination.mkdir(parents=True, exist_ok=True)
    save_file(converted, destination / "model.safetensors", metadata={"format": "pt"})
    for name in ASSETS:
        if (base / name).is_file():
            shutil.copy2(base / name, destination / name)


def prepared_model(root, plan, spec):
    from transformers import AutoTokenizer
    from qwen3_experiments.prepare_math_eval_matrix import download_files, tensor_inventory, validate_tensor_inventory

    folder = Path(plan["scratch"]) / "models" / spec["key"]
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "model_receipt.json"
    if not receipt_path.exists():
        if spec["format"] == "fsdp8":
            base = nine.evaluator(Path(plan["nine_root"]))
            base.MODEL, base.REVISION = spec["repo"], spec["revision"]
            base.PREFIX, base.REPO = "global_step_100/actor/", Path(plan["runtime"])
            base.prepare_model(folder)
        else:
            prefix = "global_step_100/actor/"
            names = [prefix + f"{kind}_pp_rank_{rank}_mp_rank_00_{suffix}_states.pt"
                     for rank in range(8) for kind, suffix in (("zero", "model"), ("bf16_zero", "optim"))]
            names.append(prefix + "train_config.json")
            inventory = download_files(spec, names, folder / "source_model")
            actor = folder / "source_model" / prefix
            command = read(actor / "train_config.json")["command"]
            if command[command.index("--zero_stage") + 1] != "3":
                raise ValueError("Expected the user's ZeRO-3 ER checkpoint")
            for name, checksum in plan["base_files_sha256"].items():
                if digest(Path(plan["base_model"]) / name) != checksum:
                    raise ValueError("Pinned Qwen3 base assets changed")
            destination = folder / "model_in_progress"
            subprocess.run([plan["zero3_python"], "-u", "-m", MODULE, "convert-zero3", "--source", str(actor),
                            "--destination", str(destination), "--base-model", plan["base_model"]],
                           cwd=plan["runtime"], env={**shared.environment(plan), "CUDA_VISIBLE_DEVICES": "", "DS_ACCELERATOR": "cpu"}, check=True)
            validate_tensor_inventory(tensor_inventory(destination), Path(plan["base_model"]))
            destination.rename(folder / "model")
            write(receipt_path, {"repo": spec["repo"], "revision": spec["revision"], "format": "ZeRO-3 step100 to BF16",
                                 "source_files": inventory, "optimizer_updates": 200,
                                 "merged_files": {p.name: digest(p) for p in (folder / "model").iterdir() if p.is_file()}})
    receipt = read(receipt_path)
    if receipt["repo"] != spec["repo"] or receipt["revision"] != spec["revision"]:
        raise ValueError("Prepared model belongs to another checkpoint")
    for name, checksum in receipt["merged_files"].items():
        if digest(folder / "model" / name) != checksum:
            raise ValueError("Prepared model checksum changed")
    tokenizer = AutoTokenizer.from_pretrained(folder / "model", local_files_only=True)
    for q in read(Path(plan["nine_root"]) / "questions.json"):
        if tokenizer.apply_chat_template(q["messages"], add_generation_prompt=True, enable_thinking=True) != q["prompt_token_ids"]:
            raise ValueError("Prepared checkpoint changed a frozen prompt")
    shared.immutable_json(root / "prepared_models" / f"{spec['key']}.json", receipt)
    return folder / "model", receipt


def install_nine_model(plan, path, receipt):
    root = Path(plan["nine_root"])
    destination = root / "model"
    if not destination.exists():
        destination.symlink_to(path, target_is_directory=True)
    if destination.resolve() != path.resolve():
        raise ValueError("Nine-benchmark model path differs from the pinned L+0 checkpoint")
    shared.immutable_json(root / "model_receipt.json", receipt)
    inputs = read(root / "input_plan.json")
    if inputs["model"] != {"repo": receipt["repo"], "revision": receipt["revision"]}:
        raise ValueError("Nine-benchmark checkpoint identity mismatch")
    inputs["plan_sha256"] = digest(root / "plan.json")
    inputs["checkpoint_receipt_sha256"] = digest(root / "model_receipt.json")
    shared.immutable_json(root / "prepared_inputs.json", inputs)


def install_minerva_model(plan, spec, path, receipt):
    root = Path(plan["minerva_root"]) / spec["key"]
    for folder in [root, *(root / "shards" / str(rank) for rank in range(8))]:
        mplan = read(folder / "plan.json")
        model = {**mplan["models"][spec["key"]], "path": str(path), "files_sha256": receipt["merged_files"]}
        if model["repo"] != receipt["repo"] or model["revision"] != receipt["revision"]:
            raise ValueError("Minerva model differs from the pinned checkpoint")
        manifest = {"plan_sha256": digest(folder / "plan.json"), "models": {spec["key"]: model},
                    "checkpoint_receipt_fingerprint": fingerprint(receipt)}
        manifest["fingerprint"] = fingerprint(manifest)
        shared.immutable_json(folder / "execution_manifest.json", manifest)


def model_evaluation_complete(plan, spec):
    """Validate saved ledgers without requiring the model cache to still exist."""
    root = Path(plan["minerva_root"]) / spec["key"]
    if not (root / "status.json").exists() or read(root / "status.json").get("state") != "complete":
        return False
    manifest = read(root / "execution_manifest.json")
    model = manifest["models"][spec["key"]]
    if model["repo"] != spec["repo"] or model["revision"] != spec["revision"]:
        raise ValueError("Completed Minerva results belong to another checkpoint")
    if manifest["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("Completed Minerva plan changed")
    for budget in mini.BUDGETS:
        point = mini.completed_point(root, spec["key"], budget, manifest)
        if (point is None or point.get("num_prompts") != 272 or point.get("seed") != 0
                or point.get("ledger_audit") != "passed" or point.get("question_shards") != 8):
            raise ValueError("Cannot remove a model before all five Minerva budgets are verified")
    return True


def cleanup_completed_model(root, plan, spec):
    """Caller holds the evaluation/allocation locks and has joined all workers."""
    if not model_evaluation_complete(plan, spec):
        raise ValueError("Minerva evaluation is incomplete")
    if spec["key"] == L0_KEY and not shared.nine_complete(Path(plan["nine_root"])):
        raise ValueError("L+0 nine-dataset evaluation is incomplete")
    if not re.fullmatch(r"[a-z0-9_]+", spec["key"]) or spec not in plan["models"]:
        raise ValueError("Unknown model cache")
    scratch = Path(plan["scratch"])
    folder = scratch / "models" / spec["key"]
    if not scratch.is_absolute() or scratch == Path("/") or folder.resolve() != folder:
        raise ValueError("Model cache must be a direct node-local path")
    protected = [root, Path(plan["base_model"]), *map(Path, plan.get("protected_cache_paths", []))]
    if any(folder == p.resolve() or folder.is_relative_to(p.resolve()) or p.resolve().is_relative_to(folder)
           for p in protected):
        raise ValueError("Model cache is required by another task")
    prepared = read(root / "prepared_models" / f"{spec['key']}.json")
    if prepared["repo"] != spec["repo"] or prepared["revision"] != spec["revision"]:
        raise ValueError("Prepared cache has another checkpoint identity")
    targets = [folder]
    # The nine-dataset runner also stages one local inference copy. Minerva
    # uses the prepared model directly, so this copy is obsolete with L+0.
    if spec["key"] == L0_KEY:
        cache_receipt = root / "nine_inference_cache.json"
        if cache_receipt.exists():
            cached = read(cache_receipt)
            if any(cached.get(key) != value for key, value in {
                    "repo": spec["repo"], "revision": spec["revision"],
                    "job_id": plan["job_id"], "node": plan["node"]}.items()):
                raise ValueError("Nine-dataset inference cache belongs to another allocation or model")
            cache_path = Path(cached["path"])
            if not cache_path.is_absolute() or cache_path.name != "rloo_final_model":
                raise ValueError("Unexpected inference cache path")
            targets.append(cache_path)
        else:
            targets.append(scratch / "tmp/rloo_final_model")
    size = 0
    for target in targets:
        if target.resolve() != target:
            raise ValueError("Model cache path traverses a symlink")
        if any(target == p.resolve() or target.is_relative_to(p.resolve()) or p.resolve().is_relative_to(target)
               for p in protected):
            raise ValueError("Model cache is required by another task")
        if not target.exists():
            continue
        if not target.is_dir() or target.stat().st_uid != os.getuid():
            raise ValueError("Model cache has unexpected ownership or type")
        for path in target.rglob("*"):
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                raise ValueError("Model cache contains a symlink or another user's files")
            if path.is_file():
                size += path.stat().st_size
        if target != folder:
            for name, checksum in prepared["merged_files"].items():
                if digest(target / name) != checksum:
                    raise ValueError("Nine-dataset inference cache has another model's weights")
    receipt_path = root / "model_cache_cleanup" / f"{spec['key']}.json"
    if receipt_path.exists() and read(receipt_path).get("state") == "complete" and not any(p.exists() for p in targets):
        return read(receipt_path)
    receipt = {"state": "cleaning", "key": spec["key"], "repo": spec["repo"], "revision": spec["revision"],
               "path": str(folder), "paths": list(map(str, targets)), "bytes": size,
               "plan_sha256": digest(root / "plan.json"), "started_at": now()}
    write(receipt_path, receipt)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
    receipt.update(state="complete", finished_at=now())
    write(receipt_path, receipt)
    return receipt


def combined_report(plan):
    root, rows = Path(plan["minerva_root"]), []
    for spec in plan["models"]:
        folder = root / spec["key"]
        mplan, manifest = read(folder / "plan.json"), read(folder / "execution_manifest.json")
        mini.report(folder, mplan, manifest)
        rows.extend(read(folder / "report/metrics.json"))
    if len(rows) != 15 or any(r["questions"] != 272 or r["seed"] != 0 for r in rows):
        raise ValueError("Incomplete three-model Minerva comparison")
    report = root / "report"
    write(report / "metrics.json", rows)
    base = nine.evaluator(Path(plan["nine_root"]))
    base.write_csv(report / "metrics.csv", rows)
    lines = ["# Polaris step100: Minerva individual pass@budget", "",
             "272 questions; seed 0; temperature 0.6 / top-p 0.95 / top-k 20; thinking on. "
             "Grade only the answer after completed thinking; no extra EOS/box requirement. "
             "Each attempt uses at most 32k or the remaining per-question allowance; stop on first success.", "",
             "| Model | 8k | 16k | 32k | 48k | 64k |", "|---|---:|---:|---:|---:|---:|"]
    for spec in plan["models"]:
        selected = {r["budget_tokens"]: r for r in rows if r["model_key"] == spec["key"]}
        lines.append("| " + spec["label"] + " | " + " | ".join(f"{selected[b]['pass_at_budget_percent']:.2f}%" for b in mini.BUDGETS) + " |")
    (report / "README.md").write_text("\n".join(lines) + "\n")
    write(report / "audit.json", {"complete": True, "points": 15, "questions_per_point": 272,
                                   "all_rollout_ledgers_verified": True, "metrics_sha256": digest(report / "metrics.json")})


def run_phase(root, plan, phase):
    require_compute(plan["job_id"])
    if not predecessor_ready(plan):
        return 75
    with ExitStack() as stack:
        try:
            for path in [root / "run.lock", *map(Path, plan["holder_locks"])]:
                lock = stack.enter_context(path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            return 75
        if phase == "nine":
            nroot = Path(plan["nine_root"])
            if not shared.nine_complete(nroot):
                path, receipt = prepared_model(root, plan, plan["models"][0])
                install_nine_model(plan, path, receipt)
                write(root / "nine_inference_cache.json", {
                    "path": str(Path(os.environ["TMPDIR"]) / "rloo_final_model"),
                    "repo": plan["models"][0]["repo"], "revision": plan["models"][0]["revision"],
                    "job_id": plan["job_id"], "node": plan["node"],
                })
                base = nine.evaluator(nroot)
                runner = nine.load_module("polaris_nine_runner", nroot / "provenance/eval_rloo_final.py")
                runner.report = nine.generation_complete
                runner.run(nroot, base)
                nine.regrade(nroot, base)
        else:
            if not shared.nine_complete(Path(plan["nine_root"])):
                raise ValueError("Minerva must follow the completed Polaris L+0 nine-benchmark evaluation")
            for spec in plan["models"]:
                if not model_evaluation_complete(plan, spec):
                    path, receipt = prepared_model(root, plan, spec)
                    install_minerva_model(plan, spec, path, receipt)
                    write(root / "progress.json", {"phase": "minerva", "model": spec["key"], "updated_at": now()})
                    shared.run_minerva({**plan, "minerva_root": str(Path(plan["minerva_root"]) / spec["key"])}, prepared=True)
                if plan.get("cleanup_each_model_after_evaluation", False):
                    cleanup_completed_model(root, plan, spec)
            combined_report(plan)
    return 0


def queue(root, plan):
    require_compute(plan["job_id"])
    state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "started_at": now()}

    def update(state_name, **values):
        state.update(state=state_name, **values, updated_at=now())
        write(root / "queue_status.json", state)

    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not predecessor_ready(plan):
                require_compute(plan["job_id"])
                update("waiting_for_compression_er_evaluations")
                time.sleep(30)
            verify_plan(root)
            for phase in ("nine", "minerva"):
                failures = 0
                while True:
                    require_compute(plan["job_id"])
                    command = ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                               f"--nodelist={plan['node']}", "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1",
                               f"--job-name=polaris-{phase}-eval", plan["python_bin"], "-u", "-m", MODULE,
                               "run", "--output-root", str(root), "--phase", phase]
                    with (root / f"{phase}.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(command, cwd=plan["runtime"], env=shared.environment(plan),
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            update("running_or_waiting_for_gpus", phase=phase, launcher_pid=child.pid)
                            time.sleep(15)
                    if child.returncode == 0:
                        if phase == "nine" and not shared.nine_complete(Path(plan["nine_root"])):
                            raise ValueError("Missing completed Polaris nine-dataset audit")
                        if phase == "minerva":
                            audit = read(Path(plan["minerva_root"]) / "report/audit.json")
                            if not audit["complete"] or audit["points"] != 15:
                                raise ValueError("Missing complete three-model Minerva audit")
                        break
                    failures += int(child.returncode != 75)
                    if failures >= 3:
                        raise RuntimeError(f"Polaris {phase} evaluation failed three times")
                    update("retrying", phase=phase, last_exit_code=child.returncode)
                    time.sleep(30)
            update("complete", nine_responses=7276, minerva_points=15, finished_at=now())
        except BaseException as exc:
            update("failed", error=str(exc), finished_at=now())
            raise


def launch(root, plan):
    require_compute(plan["job_id"])
    for name in ("tmp", "triton"):
        (Path(plan["scratch"]) / name).mkdir(parents=True, exist_ok=True)
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise RuntimeError("Polaris queue already launched")
        with (root / "queue.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "queue", "--output-root", str(root)],
                                     cwd=plan["runtime"], env=shared.environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "prepare-models", "launch", "queue", "run", "convert-zero3"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--after-evaluation", type=Path)
    parser.add_argument("--phase", choices=("nine", "minerva"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--base-model", type=Path)
    args = parser.parse_args()
    if args.command == "convert-zero3":
        convert_zero3(args.source, args.destination, args.base_model)
        return
    root = args.output_root.resolve()
    if args.command == "prepare":
        prepare(args)
        return
    plan = verify_plan(root)
    if args.command == "prepare-models":
        require_compute(plan["job_id"])
        for spec in plan["models"]:
            prepared_model(root, plan, spec)
    elif args.command == "launch":
        launch(root, plan)
    elif args.command == "queue":
        queue(root, plan)
    else:
        sys.exit(run_phase(root, plan, args.phase))


if __name__ == "__main__":
    main()
