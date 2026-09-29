"""Queue compression RB/MaxRL/f_cov training and archival on one compute node."""

import argparse
from contextlib import ExitStack, contextmanager
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from qwen3_experiments import eval_l0_final as evaluation
from qwen3_experiments.grpo_compute_control import (
    checkpoint_complete,
    digest,
    ensure_public_repository,
    monitor,
    now,
    read,
    require_compute,
    write,
)

LAUNCHER = "qwen3_experiments/run_qwen3_1_7b_compression_per_context_rb_l0_0.sh"
L4096_LAUNCHER = "qwen3_experiments/run_qwen3_1_7b_compression_per_context_rb_l0_4096.sh"
MAXRL_LAUNCHER = "qwen3_experiments/run_qwen3_1_7b_compression_maxrl.sh"
FCOV_LAUNCHER = "qwen3_experiments/run_qwen3_1_7b_compression_f_cov_l0_0.sh"
RB_ESTIMATOR = "fixed_n_rb_offset_cost_aware_marginrl"
MODULE = "qwen3_experiments.compression_l0_compute_control"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def environment(plan):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("L0_", "MAXRL_", "FCOV_", "GRPO_", "RAY_", "VLLM_", "SLURM_", "WANDB_")):
            if key != "WANDB_API_KEY":
                env.pop(key)
    for key in ("PYTHONHOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "DRY_RUN", "PREPARE_ONLY"):
        env.pop(key, None)
    env.update({
        "PATH": str(Path(plan["python_bin"]).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHON_BIN": plan["python_bin"], "PYTHONPATH": plan["runtime"],
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "WANDB_MODE": "online", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "L0_RUN_DIR": plan["output_root"], "L0_DATA_DIR": plan["data_dir"],
        "L0_CHECKPOINT_DIR": plan["checkpoint_dir"], "L0_RAY_DIR": plan["ray_dir"],
        "L0_ROLLOUT_DIR": plan["rollout_dir"], "L0_ROLLOUT_HF_REPO": plan["rollout_hf_repo"],
        "L0_ROLLOUT_PRIVATE": "false",
        "L0_COST_OFFSET_TOKENS": str(plan.get("cost_offset_tokens", 0)),
        "L0_CHECK_EOS": str(plan.get("grading", {}).get("check_eos", True)).lower(),
        "L0_ADV_ESTIMATOR": plan.get("adv_estimator", RB_ESTIMATOR),
    })
    if plan.get("experiment_name"):
        env["L0_EXPERIMENT_NAME"] = plan["experiment_name"]
    return env


def snapshot(source, runtime):
    """Keep run artifacts inside maxrl; this is a code snapshot, not a checkout."""
    if runtime.exists():
        verify_runtime(runtime)
        return
    temporary = runtime.with_name(runtime.name + f".{os.getpid()}.tmp")
    temporary.mkdir()
    try:
        shutil.copytree(source / "verl", temporary / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        files = [LAUNCHER, L4096_LAUNCHER, MAXRL_LAUNCHER, FCOV_LAUNCHER,
                 "examples/maxrl_data_preprocess/compression.py", "scripts/model_merger.py"]
        files += [f"qwen3_experiments/{name}" for name in (
            "compression_l0_compute_control.py", "grpo_compute_control.py", "verify_checkpoint_upload.py",
            "eval_l0_final.py", "eval_polaris_step80.py", "run_l0_final_eval.sh",
            "evaluation_model_cache_cleanup.py", "verified_rollout_cleanup.py",
        )]
        for relative in files:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
        write(temporary / "runtime_manifest.json", {
            "source": str(source), "created_at": now(),
            "files": {p.relative_to(temporary).as_posix(): digest(p)
                      for p in sorted(temporary.rglob("*")) if p.is_file()},
        })
        temporary.rename(runtime)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def verify_runtime(runtime):
    for relative, checksum in read(runtime / "runtime_manifest.json")["files"].items():
        if digest(runtime / relative) != checksum:
            raise ValueError(f"Prepared runtime changed: {relative}")


def verify_training_inputs(plan):
    root = Path(plan["output_root"])
    if (root / "launch.json").exists():
        receipt = read(root / "launch.json")
        if receipt.get("plan_sha256") and receipt["plan_sha256"] != digest(root / "plan.json"):
            raise ValueError("Training plan changed after supervisor launch")
    verify_runtime(Path(plan["runtime"]))
    for path, checksum in plan["input_hashes"].items():
        if digest(path) != checksum:
            raise ValueError(f"Prepared input changed: {path}")


def prepare_evaluation(plan, template):
    """Reuse the frozen nine-benchmark questions, sampling and grading rules."""
    original = evaluation.verify_plan(template, evaluation.evaluator(template))
    root = Path(plan["evaluation_root"])
    root.mkdir(exist_ok=True)
    for relative in original["frozen_files"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template / relative, target)
    for name in ("eval_l0_final.py", "eval_polaris_step80.py", "run_l0_final_eval.sh"):
        shutil.copy2(Path(plan["runtime"]) / "qwen3_experiments" / name, root / "provenance" / name)
    shutil.copy2(Path(plan["output_root"]) / "resolved_config.yaml", root / "provenance/training_config.yaml")
    request = read(root / "provenance/request.json")
    request.update(repository=plan["runtime"], training_root=plan["output_root"], job_id=plan["job_id"],
                   holder_locks=plan["holder_locks"], reference_model=plan["model_path"],
                   shared_runner=str(root / "provenance/eval_rloo_final.py"),
                   main_baseline=str(root / "provenance/main_baseline.csv"),
                   budget_baseline=str(root / "provenance/budget_baseline.csv"))
    write(root / "provenance/request.json", request)
    model_repo = plan["hf_repo_prefix"] + "-step_100"
    inputs = read(root / "input_plan.json")
    inputs["model"] = {"repo": model_repo, "revision": None}
    write(root / "input_plan.json", inputs)
    prepared = {**copy.deepcopy(original), **request, "model_repo": model_repo, "created_at": time.time(),
                "template_plan_sha256": digest(template / "plan.json"),
                "merger_sha256": digest(Path(plan["runtime"]) / "scripts/model_merger.py")}
    prepared["frozen_files"] = {relative: digest(root / relative) for relative in original["frozen_files"]}
    write(root / "plan.json", prepared)
    evaluation.verify_plan(root, evaluation.evaluator(root))


def prepare(args):
    import yaml
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import validate_repo_id

    node = require_compute(args.job_id)
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != \
            "agent/add-math12k-maxrl-launcher":
        raise RuntimeError("Use the primary maxrl launcher branch")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = read(root / "plan.json")
            if (plan["job_id"] != args.job_id or plan["hf_repo_prefix"] != args.hf_prefix
                    or plan.get("cost_offset_tokens", 0) != args.cost_offset_tokens
                    or plan.get("grading", {}).get("check_eos", True) != args.check_eos
                    or plan.get("evaluate_after_training", True) != (not args.skip_final_evaluation)
                    or plan.get("adv_estimator", RB_ESTIMATOR) != args.adv_estimator):
                raise ValueError("Run directory already belongs to another plan")
            if args.predecessor_evaluation and plan.get("predecessor_evaluation_root") != str(args.predecessor_evaluation.resolve()):
                raise ValueError("Run directory already has another predecessor evaluation")
            if (args.predecessor_polaris_evaluation
                    and plan.get("predecessor_evaluation_root") != str(args.predecessor_polaris_evaluation.resolve())):
                raise ValueError("Run directory already has another Polaris predecessor")
            if (args.predecessor_training
                    and plan.get("predecessor_training_root") != str(args.predecessor_training.resolve())):
                raise ValueError("Run directory already has another training predecessor")
            verify_runtime(Path(plan["runtime"]))
            return plan
        if args.predecessor_training:
            predecessor_root = args.predecessor_training.resolve()
            predecessor = read(predecessor_root / "plan.json")
            verify_training_inputs(predecessor)
            if (predecessor["job_id"] != args.job_id or predecessor["node"] != node
                    or predecessor.get("adv_estimator") != "maxrl" or args.adv_estimator != "f_cov"
                    or predecessor["total_steps"] != 100 or predecessor["rows_per_step"] != 512
                    or predecessor["checkpoint_steps"] != list(range(10, 101, 10))
                    or predecessor.get("evaluate_after_training", True)
                    or predecessor["model_revision"] != MODEL_REVISION
                    or predecessor["dataset_repo"] != "zjhhhh/compression_dataset"
                    or predecessor["dataset_revision"] != "bfdd7af1633ecc6db191a9f28f76449165a4ee06"):
                raise ValueError("Expected this allocation's compression MaxRL training before f_cov")
            predecessor_fields = {
                "predecessor_kind": "training", "predecessor_training_root": str(predecessor_root),
                "predecessor_plan_sha256": digest(predecessor_root / "plan.json"),
                "predecessor_status": str(predecessor_root / "status.json"),
                "predecessor_rollout_data_root": str(Path(predecessor["rollout_dir"]).resolve()),
            }
        elif args.predecessor_polaris_evaluation:
            predecessor_root = args.predecessor_polaris_evaluation.resolve()
            predecessor = verify_polaris_evaluation_plan(predecessor_root)
            if (predecessor["job_id"] != args.job_id or predecessor["node"] != node
                    or len(predecessor["models"]) != 3 or any(s["step"] != 100 for s in predecessor["models"])):
                raise ValueError("Expected this allocation's three-model Polaris step100 evaluation")
            predecessor_fields = {
                "predecessor_kind": "polaris_evaluation", "predecessor_evaluation_root": str(predecessor_root),
                "predecessor_plan_sha256": digest(predecessor_root / "plan.json"),
                "predecessor_status": str(predecessor_root / "queue_status.json"),
            }
        elif args.predecessor_evaluation:
            predecessor_root = args.predecessor_evaluation.resolve()
            predecessor = evaluation.verify_plan(predecessor_root, evaluation.evaluator(predecessor_root))
            if (predecessor["job_id"] != args.job_id or predecessor["queue_node"] != node
                    or predecessor["final_step"] != 100 or predecessor["training_kind"] != "grpo"):
                raise ValueError("Expected this allocation's final GRPO evaluation")
            predecessor_fields = {
                "predecessor_kind": "evaluation", "predecessor_evaluation_root": str(predecessor_root),
                "predecessor_plan_sha256": digest(predecessor_root / "plan.json"),
                "predecessor_status": str(predecessor_root / "queue_status.json"),
                "predecessor_repo": predecessor["model_repo"],
            }
        else:
            predecessor = read(args.predecessor_config)
            if predecessor["job_id"] != args.job_id or predecessor["node"] != node:
                raise ValueError("Predecessor must run in this same allocation")
            previous_root = Path(predecessor["training_root"])
            previous_status = read(previous_root / "status.json")
            predecessor_fields = {
                "predecessor_status": str(previous_root / "status.json"),
                "predecessor_pipeline_status": str(Path(predecessor["control_root"]) / "status.json"),
                "predecessor_receipt": str(previous_root / "hf_checkpoint_archive/receipts/global_step_100.json"),
                "predecessor_repo": previous_status["hf_repo_prefix"] + "-step_100",
            }
        prefix = args.hf_prefix
        rollout_repo = args.rollout_hf_repo or prefix + "-rollouts"
        validate_repo_id(prefix + "-step_100")
        validate_repo_id(rollout_repo)
        variant = "maxrl" if args.adv_estimator == "maxrl" else f"per_context_rb_l0_{args.cost_offset_tokens}"
        if args.adv_estimator == "f_cov":
            variant = "f_cov_l0_0"
        no_eos = "_no_eos" if not args.check_eos else ""
        scratch_name = (f"compressionl0{args.job_id}" if args.cost_offset_tokens == 0 and args.check_eos
                        else f"compressionl0_{args.cost_offset_tokens}{no_eos}_{args.job_id}")
        if args.adv_estimator == "maxrl":
            scratch_name = f"compressionmaxrl{args.job_id}"
        elif args.adv_estimator == "f_cov":
            scratch_name = f"compressionfcov{args.job_id}"
        scratch = Path("/tmp") / scratch_name
        if (scratch / "checkpoints").exists() and any((scratch / "checkpoints").iterdir()):
            raise RuntimeError("Checkpoint directory is already occupied")
        plan = {
            "job_id": args.job_id, "node": node, "python_bin": sys.executable, "source_repo": str(source),
            "output_root": str(root), "runtime": str(root / "runtime"), "data_dir": str(root / "data"),
            "checkpoint_dir": str(scratch / "checkpoints"), "ray_dir": str(scratch / "ray"),
            "rollout_dir": str(root / "rollout_dataset"), "rollout_hf_repo": rollout_repo,
            "hf_repo_prefix": prefix, "new_hf_repositories_private": False,
            "model_revision": MODEL_REVISION, "evaluation_root": str(root / "evaluation"),
            "variant": variant, "cost_offset_tokens": args.cost_offset_tokens,
            "adv_estimator": args.adv_estimator,
            "grading": {"check_eos": args.check_eos, "score_after_thinking": True, "force_eos": False},
            "experiment_name": f"{variant}{no_eos}_Qwen3-1.7B_compression_bs32_n16_32k_1epoch",
            "launcher": (MAXRL_LAUNCHER if args.adv_estimator == "maxrl" else
                         FCOV_LAUNCHER if args.adv_estimator == "f_cov" else
                         L4096_LAUNCHER if args.cost_offset_tokens == 4096 and not args.check_eos else LAUNCHER),
            "evaluate_after_training": not args.skip_final_evaluation,
            "cleanup_predecessor_models_before_training": bool(
                args.predecessor_polaris_evaluation and args.adv_estimator == "maxrl"),
            "cleanup_predecessor_rollouts_before_training": bool(args.predecessor_training),
            "handoff_local_receipt_dir": str(scratch / "handoff"),
            "initialization": "Pinned initial Qwen3-1.7B weights; new compression experiment",
            **predecessor_fields,
            "holder_locks": list(dict.fromkeys([
                *predecessor["holder_locks"], str(source / f"outputs/logs/gpu_holder_{args.job_id}.launch.lock"),
                *([str(predecessor_root / "supervisor.lock")] if args.predecessor_training else []),
            ])),
            "total_steps": 100, "rows_per_step": 512, "checkpoint_steps": list(range(10, 101, 10)),
            "dataset_repo": "zjhhhh/compression_dataset", "dataset_revision": "bfdd7af1633ecc6db191a9f28f76449165a4ee06",
            "created_at": now(),
        }
        snapshot(source, Path(plan["runtime"]))
        env = environment(plan)
        launcher = str(Path(plan["runtime"]) / plan["launcher"])
        with (root / "prepare.log").open("a") as log:
            subprocess.run(["bash", launcher], cwd=plan["runtime"],
                           env={**env, "PREPARE_ONLY": "1"}, stdout=log, stderr=subprocess.STDOUT, check=True)
        plan["model_path"] = snapshot_download("Qwen/Qwen3-1.7B", revision=MODEL_REVISION,
                                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        preview = subprocess.run(["bash", launcher, "--cfg", "job", "--resolve"],
                                 cwd=plan["runtime"], env=env, text=True, capture_output=True, check=True)
        (root / "resolved_config.yaml").write_text(preview.stdout)
        (root / "config_preview.stderr").write_text(preview.stderr)
        config = yaml.safe_load(preview.stdout)
        if (config["algorithm"]["adv_estimator"] != args.adv_estimator
                or config["algorithm"]["cost_offset_tokens"] != args.cost_offset_tokens
                or config["reward_model"]["reward_kwargs"] != {
                    "check_eos": args.check_eos, "score_after_thinking": True,
                }
                or config["trainer"]["rollout_dataset"] != {
                    "enabled": True, "local_dir": plan["rollout_dir"], "hub_repo_id": rollout_repo,
                    "private": False, "upload_num_workers": 4,
                }):
            raise ValueError("Unexpected cost offset, grading or rollout configuration")
        if args.cost_offset_tokens != 0 or not args.check_eos or args.adv_estimator != RB_ESTIMATOR:
            baseline_env = {**env, "L0_COST_OFFSET_TOKENS": "0", "L0_CHECK_EOS": "true",
                            "L0_ADV_ESTIMATOR": RB_ESTIMATOR}
            baseline_env.pop("L0_EXPERIMENT_NAME", None)
            baseline = subprocess.run(["bash", str(Path(plan["runtime"]) / LAUNCHER), "--cfg", "job", "--resolve"],
                                      cwd=plan["runtime"], env=baseline_env, text=True, capture_output=True, check=True)
            (root / "reference_l0_config.yaml").write_text(baseline.stdout)
            changes = compare_training_recipe(yaml.safe_load(baseline.stdout), config, args.adv_estimator)
            write(root / "config_comparison.json", changes)
        inputs = [root / "resolved_config.yaml", *sorted((root / "data").glob("*")),
                  *sorted(Path(plan["model_path"]).glob("*"))]
        plan["input_hashes"] = {str(p): digest(p) for p in inputs if p.is_file()}
        if plan["evaluate_after_training"]:
            prepare_evaluation(plan, args.eval_template.resolve())
        api = HfApi()
        # Establish public storage and write access before a long training run.
        ensure_public_repository(api, rollout_repo, "dataset")
        info = api.repo_info(repo_id=rollout_repo, repo_type="dataset")
        if info.private or any(item.rfilename.startswith("data/") for item in info.siblings):
            raise ValueError("Expected an empty public rollout dataset repository")
        write(root / "hf_auth_check.json", {"account": api.whoami()["name"], "hostname": node, "checked_at": now()})
        write(root / "plan.json", plan)
        return plan


def compare_training_recipe(before, after, adv_estimator=RB_ESTIMATOR):
    """Permit exactly the requested algorithm/grading changes and run label."""
    allowed = {"algorithm.cost_offset_tokens", "reward_model.reward_kwargs.check_eos", "trainer.experiment_name"}
    if adv_estimator == "maxrl":
        if (before["algorithm"]["adv_estimator"] != RB_ESTIMATOR
                or after["algorithm"]["adv_estimator"] != "maxrl"
                or after["algorithm"]["cost_offset_tokens"] != 0):
            raise ValueError("Expected plain MaxRL with no length cost")
        allowed.add("algorithm.adv_estimator")
    elif adv_estimator == "f_cov":
        if (before["algorithm"]["adv_estimator"] != RB_ESTIMATOR
                or after["algorithm"]["adv_estimator"] != "f_cov"
                or after["algorithm"]["cost_offset_tokens"] != 0
                or after["algorithm"]["f_cov_num_prompts"] != after["data"]["train_batch_size"]
                or after["reward_model"]["reward_kwargs"]["check_eos"]):
            raise ValueError("Expected f_cov with L_0=0, full-batch prompt count and no EOS gate")
        allowed.update(("algorithm.adv_estimator", "algorithm.f_cov_num_prompts"))
    changes = []

    def compare(left, right, key=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(set(left) | set(right)):
                compare(left.get(name), right.get(name), f"{key}.{name}" if key else name)
        elif left != right:
            if key not in allowed:
                raise ValueError(f"Unexpected change from the compression L+0 recipe: {key}")
            changes.append({"path": key, "l0": left, "new": right})

    compare(before, after)
    return changes


def verify_polaris_evaluation_plan(root):
    plan = read(root / "plan.json")
    if ((root / "launch.json").exists()
            and read(root / "launch.json")["plan_sha256"] != digest(root / "plan.json")):
        raise ValueError("Polaris evaluation plan changed after launch")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen Polaris evaluation input changed: {relative}")
    if digest(Path(plan["predecessor_root"]) / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("The preceding compression ER evaluation plan changed")
    return plan


def polaris_evaluation_predecessor_ready(plan):
    root = Path(plan["predecessor_evaluation_root"])
    if digest(root / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor Polaris evaluation plan changed")
    queue = read(root / "queue_status.json")
    if queue.get("state") == "failed":
        raise RuntimeError("Polaris evaluation failed; MaxRL has not been launched")
    if queue.get("state") != "complete":
        return False
    previous = verify_polaris_evaluation_plan(root)
    if (previous["job_id"] != plan["job_id"] or previous["node"] != plan["node"]
            or queue.get("nine_responses") != 7276 or queue.get("minerva_points") != 15):
        raise ValueError("Expected the complete preceding Polaris evaluation queue")
    nine = Path(previous["nine_root"])
    audit = read(nine / "report/audit.json")
    if (read(nine / "status.json").get("state") != "complete" or not audit.get("complete")
            or audit.get("questions") != 1819 or audit.get("responses_verified") != 7276
            or audit.get("budget_points") != 54 or not audit.get("all_budgets_regraded")
            or audit.get("grader_errors")):
        raise ValueError("Polaris L+0 nine-dataset evaluation is incomplete")
    for name in ("per_sample", "metrics"):
        if digest(nine / f"report/{name}.json") != audit[f"{name}_sha256"]:
            raise ValueError("Polaris nine-dataset results changed")
    minerva = Path(previous["minerva_root"])
    audit = read(minerva / "report/audit.json")
    if (not audit.get("complete") or audit.get("points") != 15 or audit.get("questions_per_point") != 272
            or not audit.get("all_rollout_ledgers_verified")
            or digest(minerva / "report/metrics.json") != audit.get("metrics_sha256")):
        raise ValueError("The three-model Minerva comparison is incomplete or changed")
    metrics = read(minerva / "report/metrics.json")
    budgets = [8192, 16384, 32768, 49152, 65536]
    expected = {(spec["key"], budget) for spec in previous["models"] for budget in budgets}
    if (len(metrics) != 15 or {(r["model_key"], r["budget_tokens"]) for r in metrics} != expected
            or any(r["questions"] != 272 or r["seed"] != 0 for r in metrics)):
        raise ValueError("All fifteen full-dataset Minerva points must finish before MaxRL")
    for spec in previous["models"]:
        folder = minerva / spec["key"]
        if read(folder / "status.json").get("state") != "complete":
            raise ValueError("A preceding Polaris model is not finished")
        manifest = read(folder / "execution_manifest.json")
        model = manifest["models"][spec["key"]]
        if model["repo"] != spec["repo"] or model["revision"] != spec["revision"]:
            raise ValueError("Predecessor Minerva checkpoint identity changed")
        for budget in budgets:
            directory = folder / "results" / spec["key"] / f"budget_{budget}"
            point = read(directory / "summary.json")
            if (point.get("state") != "complete" or point.get("num_prompts") != 272
                    or point.get("ledger_audit") != "passed" or point.get("seed") != 0
                    or set(point.get("artifacts", {})) != {"rollouts", "prompts"}
                    or point.get("identity") != {"manifest": manifest["fingerprint"], "model": spec["key"], "budget": budget}):
                raise ValueError("Missing fully audited predecessor Minerva point")
            for artifact in point["artifacts"].values():
                relative = Path(artifact["file"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid predecessor Minerva artifact path")
                path = directory / relative
                if path.stat().st_size != artifact["size"] or digest(path) != artifact["sha256"]:
                    raise ValueError("Predecessor Minerva rollout artifact changed")
    return True


def evaluation_predecessor_ready(plan):
    root = Path(plan["predecessor_evaluation_root"])
    if digest(root / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor evaluation plan changed")
    queue = read(root / "queue_status.json") if (root / "queue_status.json").exists() else {}
    status = read(root / "status.json") if (root / "status.json").exists() else {}
    if queue.get("state") == "failed" or status.get("state") == "failed":
        raise RuntimeError("Predecessor evaluation failed; new training has not been launched")
    if queue.get("state") != "complete" or status.get("state") != "complete":
        return False
    previous = evaluation.verify_plan(root, evaluation.evaluator(root))
    if (previous["job_id"] != plan["job_id"] or previous["model_repo"] != plan["predecessor_repo"]
            or previous["final_step"] != 100):
        raise ValueError("Predecessor evaluation belongs to a different run")
    audit = read(root / "report/audit.json")
    if (not audit["complete"] or audit.get("grader_errors") or not audit["all_budgets_regraded"]
            or audit["questions"] != previous["questions"]
            or audit["responses_verified"] != previous["total_responses"]
            or status["completed_responses"] != previous["total_responses"]
            or queue["completed_responses"] != previous["total_responses"]
            or audit["budget_points"] != len(previous["caps"]) * 9
            or (root / "exit_status").read_text().strip() != "0"):
        raise ValueError("Predecessor evaluation is incomplete or unsuccessful")
    for name in ("per_sample", "metrics"):
        if digest(root / f"report/{name}.json") != audit[f"{name}_sha256"]:
            raise ValueError(f"Predecessor evaluation report changed: {name}")
    if not (root / "report/README.md").is_file():
        raise ValueError("Predecessor evaluation report is missing")
    receipt = read(root / "final_checkpoint_receipt.json")
    training = read(root / "training_completion.json")
    return evaluation.checkpoint_ready(training, receipt, plan["predecessor_repo"], previous["training_kind"])


def training_predecessor_ready(plan):
    """Require successful training and every upload before releasing the GPUs."""
    root = Path(plan["predecessor_training_root"])
    if digest(root / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor training plan changed")
    previous, status = read(root / "plan.json"), read(root / "status.json")
    if previous["job_id"] != plan["job_id"] or previous["node"] != plan["node"]:
        raise ValueError("Predecessor training belongs to another allocation")
    if status.get("state") == "failed":
        raise RuntimeError("Predecessor training failed; waiting for its recovery")
    if status.get("state") != "complete":
        return False
    if (status.get("exit_code") != 0 or status.get("last_completed_step") != previous["total_steps"]
            or read(root / "training_exit.json").get("exit_code") != 0):
        raise ValueError("Predecessor training did not finish all steps successfully")
    archive = read(root / "hf_checkpoint_archive/status.json")
    if archive.get("state") != "complete" or archive.get("archived_steps") != previous["checkpoint_steps"]:
        raise ValueError("Predecessor checkpoint archive is incomplete")
    for step in previous["checkpoint_steps"]:
        receipt = read(root / f"hf_checkpoint_archive/receipts/global_step_{step}.json")
        if (receipt.get("state") != "archived_and_deleted"
                or receipt.get("checkpoint") != f"global_step_{step}"
                or receipt.get("repo_id") != previous["hf_repo_prefix"] + f"-step_{step}"
                or not re.fullmatch(r"[0-9a-f]{40}", receipt.get("remote_commit", ""))):
            raise ValueError("Predecessor checkpoint upload is not verified")
    upload = read(root / "rollout_upload.json")
    if (upload.get("state") != "verified" or upload.get("repo_id") != previous["rollout_hf_repo"]
            or upload.get("num_steps") != previous["total_steps"]
            or upload.get("num_rollouts") != previous["total_steps"] * previous["rows_per_step"]
            or not re.fullmatch(r"[0-9a-f]{40}", upload.get("remote_commit", ""))):
        raise ValueError("Predecessor rollout upload is not fully verified")
    return True


def predecessor_ready(plan):
    if plan.get("predecessor_kind") == "training":
        return training_predecessor_ready(plan)
    if plan.get("predecessor_kind") == "polaris_evaluation":
        return polaris_evaluation_predecessor_ready(plan)
    if plan.get("predecessor_kind") == "evaluation":
        return evaluation_predecessor_ready(plan)
    status = read(plan["predecessor_status"])
    pipeline = read(plan["predecessor_pipeline_status"])
    if status.get("state") == "failed" or pipeline.get("state") == "failed":
        raise RuntimeError("Predecessor failed; new training has not been launched")
    receipt_path = Path(plan["predecessor_receipt"])
    receipt = read(receipt_path) if receipt_path.exists() else None
    return (pipeline.get("state") == "complete"
            and evaluation.checkpoint_ready(status, receipt, plan["predecessor_repo"]))


def cleanup_before_training(plan):
    if plan.get("cleanup_predecessor_rollouts_before_training", False):
        return cleanup_predecessor_rollouts(plan)
    if not plan.get("cleanup_predecessor_models_before_training", False):
        return None
    if plan.get("predecessor_kind") != "polaris_evaluation":
        raise ValueError("Model cache cleanup requires the completed Polaris evaluation queue")
    root = Path(plan["predecessor_evaluation_root"])
    with holder_locks([root / "queue.lock", root / "run.lock"]):
        if not predecessor_ready(plan):
            raise RuntimeError("All preceding evaluations must finish before cleaning their model caches")
        from qwen3_experiments.evaluation_model_cache_cleanup import cleanup_model_caches

        receipt = cleanup_model_caches(plan, verify_polaris_evaluation_plan(root))
        if receipt["state"] != "complete":
            raise RuntimeError("Model cache cleanup must finish before training")
        return receipt


def cleanup_predecessor_rollouts(plan):
    """Delete only the completed predecessor's shards, checked against its Hub commit."""
    from huggingface_hub import HfApi
    from qwen3_experiments.verified_rollout_cleanup import cleanup

    if plan.get("predecessor_kind") != "training" or not training_predecessor_ready(plan):
        raise RuntimeError("Predecessor training and uploads must finish before rollout cleanup")
    root = Path(plan["predecessor_training_root"])
    previous, upload = read(root / "plan.json"), read(root / "rollout_upload.json")
    source = Path(previous["rollout_dir"])
    if source.resolve() != Path(plan["predecessor_rollout_data_root"]):
        raise ValueError("Predecessor rollout data moved after preparation")
    destination = Path(plan["output_root"]) / "predecessor_rollout_cleanup"
    local = Path(plan["handoff_local_receipt_dir"])
    with holder_locks([local / "cleanup.lock"]):
        for folder in (destination, local):
            receipt_path = folder / "cleanup_receipt.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                if (receipt.get("state") == "uploaded_verified_and_deleted"
                        and receipt.get("repo_id") == upload["repo_id"]
                        and receipt.get("revision") == upload["remote_commit"]):
                    return receipt
        return cleanup({
            "rollout_directory": str(source), "repo_id": upload["repo_id"],
            "revision": upload["remote_commit"], "steps": previous["total_steps"],
            "rollouts": previous["total_steps"] * previous["rows_per_step"],
            "allowed_data_roots": [plan["predecessor_rollout_data_root"]],
            "receipt_directories": [str(local), str(destination)],
        }, HfApi())


@contextmanager
def holder_locks(paths):
    with ExitStack() as stack:
        for path in map(Path, paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def gpu_idle():
    return not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=30,
    ).strip()


def training_command(plan):
    return ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
            f"--nodelist={plan['node']}", "--cpus-per-task=128", "--gres=gpu:8", "--kill-on-bad-exit=1",
            "--job-name=compression-maxrl" if plan.get("adv_estimator") == "maxrl"
            else "--job-name=compression-f-cov" if plan.get("adv_estimator") == "f_cov"
            else f"--job-name=compression-l{plan.get('cost_offset_tokens', 0)}", "bash",
            str(Path(plan["runtime"]) / plan.get("launcher", LAUNCHER))]


def audit_rollouts(plan, api):
    root = Path(plan["rollout_dir"])
    manifest = read(root / "rollout_manifest.json")
    expected_steps = {str(step) for step in range(1, plan["total_steps"] + 1)}
    if set(manifest["steps"]) != expected_steps:
        raise ValueError("Missing or unexpected rollout steps")
    info = api.repo_info(repo_id=plan["rollout_hf_repo"], repo_type="dataset", files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    count = 0
    for step, metadata in manifest["steps"].items():
        relative = f"data/step_{int(step):06d}.jsonl.gz"
        if metadata != {"file": relative, "num_rollouts": plan["rows_per_step"]}:
            raise ValueError(f"Unexpected rollout shard metadata: {step}")
        path = root / relative
        item = remote.get(relative)
        if item is None or item.size != path.stat().st_size:
            raise ValueError(f"Missing or wrong-sized Hub rollout shard: {relative}")
        if item.lfs:
            expected = item.lfs["sha256"] if isinstance(item.lfs, dict) else item.lfs.sha256
            if digest(path) != expected:
                raise ValueError(f"Hub rollout hash mismatch: {relative}")
        else:
            body = path.read_bytes()
            if hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest() != item.blob_id:
                raise ValueError(f"Hub rollout blob mismatch: {relative}")
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row["step"] != int(step) or row["rollout_index"] != index:
                    raise ValueError(f"Rollout identity mismatch: {relative}")
                rows += 1
        if rows != plan["rows_per_step"]:
            raise ValueError(f"Missing rollout records: {relative}")
        count += rows
    if manifest["num_steps"] != plan["total_steps"] or manifest["num_rollouts"] != count:
        raise ValueError("Rollout manifest counts disagree with actual records")
    return {"state": "verified", "num_steps": len(expected_steps), "num_rollouts": count,
            "repo_id": plan["rollout_hf_repo"], "remote_commit": info.sha, "verified_at": now(),
            "dataset_url": f"https://huggingface.co/datasets/{plan['rollout_hf_repo']}"}


def spawn(plan, command, log):
    with Path(log).open("ab", buffering=0) as stream:
        return subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, command,
                                 "--plan", str(Path(plan["output_root"]) / "plan.json")],
                                cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)


def evaluate(plan):
    require_compute(plan["job_id"])
    root, training = Path(plan["evaluation_root"]), Path(plan["output_root"])
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            status = read(training / "status.json")
            state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"],
                     "state": "waiting_for_training_and_verified_archives", "updated_at": now(),
                     "training_state": status["state"], "last_completed_step": status["last_completed_step"]}
            if status["state"] == "failed":
                write(root / "queue_status.json", {**state, "state": "blocked_by_training_failure"})
                return 1
            write(root / "queue_status.json", state)
            if status["state"] == "complete":
                break
            time.sleep(30)
        evaluation.queue(root, evaluation.evaluator(root))
    return 0


def supervise(plan):
    require_compute(plan["job_id"])
    root = Path(plan["output_root"])
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "training_started.json").exists():
            raise RuntimeError("Refusing to launch a second training process for this run")
        status = {"state": "waiting_for_predecessor", "job_id": plan["job_id"], "last_completed_step": 0,
                  "variant": plan.get("variant", "per_context_rb_l0_0"), "adv_estimator": plan.get("adv_estimator", RB_ESTIMATOR),
                  "cost_offset_tokens": plan.get("cost_offset_tokens", 0), "total_steps": 100, "hf_repo_prefix": plan["hf_repo_prefix"],
                  "grading": plan.get("grading", {"check_eos": True, "score_after_thinking": True}),
                  "pid": os.getpid(), "hostname": socket.gethostname(), "started_at": now()}
        state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"]}

        def update(phase, **values):
            state.update(state=phase, updated_at=now(), **values)
            write(root / "supervisor_status.json", state)
            status["updated_at"] = now()
            write(root / "status.json", status)

        update("verifying_inputs")
        archiver = queue = None
        try:
            verify_training_inputs(plan)
            (root / "hf_checkpoint_archive").mkdir(exist_ok=True)
            archiver = spawn(plan, "monitor", root / "hf_checkpoint_archive/upload.log")
            if plan.get("evaluate_after_training", True):
                queue = spawn(plan, "evaluate", Path(plan["evaluation_root"]) / "queue.log")

            def maintain_children():
                nonlocal archiver, queue
                if archiver.poll() is not None:
                    archiver = spawn(plan, "monitor", root / "hf_checkpoint_archive/upload.log")
                if queue is not None and queue.poll() is not None:
                    queue = spawn(plan, "evaluate", Path(plan["evaluation_root"]) / "queue.log")
                state.update(checkpoint_monitor_pid=archiver.pid, evaluation_queue_pid=queue.pid if queue is not None else None)

            while True:
                require_compute(plan["job_id"])
                maintain_children()
                previous = read(plan["predecessor_status"])
                update("waiting_for_predecessor", predecessor_state=previous["state"],
                       predecessor_step=previous.get("last_completed_step"))
                if predecessor_ready(plan):
                    try:
                        with holder_locks(plan["holder_locks"]):
                            if predecessor_ready(plan) and gpu_idle():
                                verify_training_inputs(plan)
                                if (plan.get("cleanup_predecessor_models_before_training", False)
                                        or plan.get("cleanup_predecessor_rollouts_before_training", False)):
                                    update("cleaning_predecessor_outputs")
                                    cleanup_before_training(plan)
                                    verify_training_inputs(plan)
                                    if not predecessor_ready(plan):
                                        raise RuntimeError("Predecessor changed during cleanup")
                                status["state"] = "training"
                                update("launching_training")
                                with (root / "train.log").open("ab", buffering=0) as log:
                                    child = subprocess.Popen(training_command(plan), cwd=plan["runtime"],
                                                             env=environment(plan), stdin=subprocess.DEVNULL,
                                                             stdout=log, stderr=subprocess.STDOUT)
                                write(root / "training_started.json", {
                                    "pid": child.pid, "hostname": socket.gethostname(), "started_at": now(),
                                })
                                with (root / "train.log").open(errors="replace") as log:
                                    while True:
                                        content = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log.read())
                                        steps = [int(s) for s in re.findall(r"(?<![/\w])step:\s*(\d+)\b", content)]
                                        if steps:
                                            status["last_completed_step"] = max(status["last_completed_step"], *steps)
                                        maintain_children()
                                        update("training", training_launcher_pid=child.pid)
                                        if child.poll() is not None:
                                            break
                                        time.sleep(10)
                                final = Path(plan["checkpoint_dir"]) / "global_step_100"
                                success = child.returncode == 0 and checkpoint_complete(final, final.parent)
                                code = child.returncode or (0 if success else 1)
                                write(root / "training_exit.json", {
                                    "exit_code": code, "launcher_exit_code": child.returncode, "finished_at": now(),
                                })
                                if not success:
                                    raise RuntimeError(f"Training failed or final checkpoint missing: exit {code}")
                                status["last_completed_step"] = 100
                                update("verifying_rollout_upload")
                                from huggingface_hub import HfApi

                                write(root / "rollout_upload.json", audit_rollouts(plan, HfApi()))
                                update("waiting_for_checkpoint_archive")
                                while archiver.poll() is None:
                                    update("waiting_for_checkpoint_archive")
                                    time.sleep(15)
                                if archiver.returncode:
                                    raise RuntimeError("Checkpoint archive did not finish successfully")
                                status.update(state="complete", exit_code=0, finished_at=now())
                                update("training_complete")
                                break
                    except BlockingIOError:
                        update("waiting_for_holder_locks")
                time.sleep(30)
            # Holder locks are released before the queued evaluation takes them.
            if queue is not None:
                while queue.poll() is None:
                    update("evaluating")
                    time.sleep(15)
                if queue.returncode:
                    raise RuntimeError("Final evaluation failed; inspect evaluation/queue.log")
            update("complete", finished_at=now())
            return 0
        except BaseException as exc:
            # A completed training result remains complete even if evaluation fails.
            if status["state"] != "complete":
                status.update(state="failed", error=str(exc))
                if not (root / "training_exit.json").exists():
                    write(root / "training_exit.json", {"exit_code": 1, "error": str(exc), "finished_at": now()})
            update("failed", error=str(exc), finished_at=now())
            raise


def launch(plan):
    require_compute(plan["job_id"])
    root = Path(plan["output_root"])
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise RuntimeError("Persistent supervisor already launched; inspect its status")
        with (root / "supervisor.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "supervise",
                                      "--plan", str(root / "plan.json")],
                                     cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"], "launched_at": now(),
                   "plan_sha256": digest(root / "plan.json")}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "supervise", "monitor", "evaluate"))
    parser.add_argument("--job-id", default="146103")
    parser.add_argument("--output-root", type=Path)
    predecessor = parser.add_mutually_exclusive_group()
    predecessor.add_argument("--predecessor-config", type=Path)
    predecessor.add_argument("--predecessor-evaluation", type=Path)
    predecessor.add_argument("--predecessor-polaris-evaluation", type=Path)
    predecessor.add_argument("--predecessor-training", type=Path)
    parser.add_argument("--eval-template", type=Path)
    parser.add_argument("--hf-prefix")
    parser.add_argument("--rollout-hf-repo")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--cost-offset-tokens", type=int, default=0)
    parser.add_argument("--adv-estimator", choices=(RB_ESTIMATOR, "maxrl", "f_cov"), default=RB_ESTIMATOR)
    parser.add_argument("--check-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-final-evaluation", action="store_true")
    args = parser.parse_args()
    if args.command in ("prepare", "launch"):
        if not all((args.output_root, args.predecessor_config or args.predecessor_evaluation
                    or args.predecessor_polaris_evaluation or args.predecessor_training, args.hf_prefix)):
            parser.error("prepare/launch needs --output-root, a predecessor and --hf-prefix")
        if not args.skip_final_evaluation and (not args.eval_template or args.cost_offset_tokens != 0
                                              or args.adv_estimator != RB_ESTIMATOR):
            parser.error("Automatic final evaluation currently needs L+0 and --eval-template; otherwise use --skip-final-evaluation")
        if args.cost_offset_tokens < 0:
            parser.error("--cost-offset-tokens must be nonnegative")
        if args.adv_estimator in ("maxrl", "f_cov") and (args.cost_offset_tokens != 0 or args.check_eos):
            parser.error("The compression MaxRL/f_cov recipes need --cost-offset-tokens 0 --no-check-eos")
        plan = prepare(args)
        if args.command == "launch":
            launch(plan)
        else:
            print(json.dumps({"state": "prepared", "hostname": plan["node"], "output_root": plan["output_root"]}))
    else:
        if not args.plan:
            parser.error("--plan is required")
        plan = read(args.plan)
        require_compute(plan["job_id"])
        return {"supervise": supervise, "monitor": monitor, "evaluate": evaluate}[args.command](plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
