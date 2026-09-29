"""Queue the reference OpenRLHF ER run after all five-model Minerva evaluations."""

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

from qwen3_experiments.grpo_compute_control import digest, ensure_public_repository, now, read, require_compute, write
from qwen3_experiments.er_compression_eos import EOS_POLICY, EOS_REFERENCE_REPO, EOS_REFERENCE_REVISION

MODULE = "qwen3_experiments.er_compression_compute"
LAUNCHER = "qwen3_experiments/run_qwen3_1_7b_compression_er.sh"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATASET_REVISION = "bfdd7af1633ecc6db191a9f28f76449165a4ee06"
UPSTREAM_COMMIT = "4a8683d3d5266494d20cea7f77abe044653639cb"
DATASET_KEY = "datasets/compression_dataset"


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f"Reference source changed; cannot apply integration hook: {path}")
    path.write_text(text.replace(before, after))


def environment(plan):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("L0_", "GRPO_", "MAXRL_", "ER_", "RAY_", "VLLM_", "SLURM_", "WANDB_")) and key != "WANDB_API_KEY":
            env.pop(key)
    for key in ("PYTHONHOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "MASTER_ADDR",
                "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "DRY_RUN", "PREPARE_ONLY", "RESUME"):
        env.pop(key, None)
    runtime, root = Path(plan["runtime"]), Path(plan["output_root"])
    env.update({
        "PATH": str(Path(plan["python_bin"]).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join(map(str, [runtime, runtime / "upstream", runtime / "reference",
                                               runtime / "reference/utils/latex2sympy"])),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        "ER_PLAN": str(root / "plan.json"), "ER_SHARED_ROOT": str(root), "ER_RUNTIME_ROOT": str(runtime),
        "ER_TRAIN_DIR": plan["train_dir"], "ER_MODEL_PATH": plan["model_path"],
        "RUN_NAME": root.name, "HF_REPO_ID": plan["hf_repo_prefix"], "RM_PORT": str(plan["reward_port"]),
        "ROLLOUT_BATCH_SIZE": "32", "N_SAMPLES_PER_PROMPT": "8", "TRAIN_BATCH_SIZE": "128",
        "MAX_SAMPLES": "3200", "GENERATE_MAX_LEN": "32768", "PROMPT_MAX_LEN": "1536", "SAVE_STEPS": "20",
        "USE_WANDB": "1", "ARCHIVE_CHECKPOINTS": "1", "RESUME": "0", "VLLM_GPU_MEMORY_UTILIZATION": "0.5",
        "RAY_TMPDIR": str(Path(plan["scratch"]) / "ray"), "WANDB_MODE": "online",
        "CONDA_BASE": str(Path(plan["python_bin"]).parents[3]),
        "CONDA_ENV": Path(plan["python_bin"]).parents[1].name,
    })
    return env


def adapt_reward_eos(path):
    """Separate natural-EOS length-pool entries from forced-EOS single queries."""
    replace_once(path, '                    "all_responses": all_responses,',
                 '                    "all_responses": all_responses,\n'
                 '                    "response_has_eos": aux_info["response_has_eos"],\n'
                 '                    "all_responses_have_eos": aux_info["all_responses_have_eos"],')
    replace_once(path, '            for candidate_response in [response, *all_responses]:',
                 '            for candidate_response, contains_eos in zip(\n'
                 '                [response, *all_responses],\n'
                 '                [aux_info["response_has_eos"], *aux_info["all_responses_have_eos"]],\n'
                 '            ):')
    replace_once(path, '                cache_key = (dataset_name, question, reference, candidate_response)\n'
                       '                _, contains_eos = response_info[candidate_response]',
                 '                cache_key = (dataset_name, question, reference, candidate_response, bool(contains_eos))')
    replace_once(path, '            for response in context["all_responses"]:\n'
                       '                cache_key = (*group_key, response)',
                 '            for response, contains_eos in zip(context["all_responses"], context["all_responses_have_eos"]):\n'
                 '                cache_key = (*group_key, response, bool(contains_eos))')
    replace_once(path, '            accuracy = accuracy_by_key[(*group_key, response)]',
                 '            accuracy = accuracy_by_key[(*group_key, response, bool(context["response_has_eos"]))]')


def snapshot(source, reference, root, model_path, eos_reference, checkpoint_config):
    runtime = root / "runtime"
    runtime.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.egg-info", "*.jar")
    shutil.copytree(reference / "upstream/openrlhf", runtime / "upstream/openrlhf", ignore=ignore)
    for name in ("integration", "reward_server", "utils"):
        shutil.copytree(reference / name, runtime / "reference" / name, ignore=ignore)
    for relative in (LAUNCHER, "qwen3_experiments/er_compression_compute.py", "qwen3_experiments/er_compression_rollouts.py",
                     "qwen3_experiments/er_compression_eos.py",
                     "qwen3_experiments/grpo_compute_control.py", "verl/utils/rollout_dataset.py"):
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    (runtime / "environment").mkdir()
    for name in ("source.json", "version_diff.json"):
        shutil.copy2(reference / "environment" / name, runtime / "environment" / name)
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=reference / "upstream")
    (runtime / "environment/upstream.patch").write_bytes(patch)
    shutil.copy2(reference / "run_polaris.sh", runtime / "environment/reference_run_polaris.sh")
    for relative in ("run_rloo_deepseek_1.5B_compression.sh", "openrlhf/models/actor.py",
                     "openrlhf/trainer/ppo_utils/experience_maker.py", "reward_server/math_server.py"):
        target = runtime / "eos_reference" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eos_reference / relative, target)
    shutil.copy2(checkpoint_config, runtime / "eos_reference/train_config.json")
    original_hashes = {p.relative_to(runtime).as_posix(): digest(p) for p in runtime.rglob("*") if p.is_file()}
    replace_once(runtime / "reference/integration/prepare_data.py", 'DATASET = "hi-todayis-jh/Polaris-1-8-3200"',
                 f"DATASET = {DATASET_KEY!r}")
    replace_once(runtime / "reference/integration/prepare_data.py", 'MODEL = "Qwen/Qwen3-1.7B"', f"MODEL = {model_path!r}")
    replace_once(runtime / "upstream/openrlhf/trainer/ppo_utils/experience_maker.py",
                 "        all_outputs = sum(ray.get(all_output_refs), [])",
                 "        all_outputs = sum(ray.get(all_output_refs), [])\n"
                 "        from qwen3_experiments.er_compression_rollouts import capture_generated_rollouts\n"
                 "        all_labels = capture_generated_rollouts(self, all_outputs, all_prompts, all_labels)")
    replace_once(runtime / "reference/integration/reward_bridge.py", "        metrics = response.get_json()",
                 "        metrics = response.get_json()\n"
                 "        from qwen3_experiments.er_compression_rollouts import record_rewards\n"
                 "        record_rewards(request.get_json(), metrics)")
    replace_once(runtime / "reference/integration/reward_bridge.py",
                 '    payload = convert_request(request.get_json(), app.config["samples_per_prompt"])',
                 '    from qwen3_experiments.er_compression_eos import reference_reward_payload\n'
                 '    payload = reference_reward_payload(request.get_json(), app.config["samples_per_prompt"], app.config["tokenizer"])')
    adapt_reward_eos(runtime / "reference/reward_server/math_server.py")
    replace_once(runtime / "reference/integration/archive_watch.py", "api.create_repo(repo_id, private=False, exist_ok=True)",
                 "from qwen3_experiments.grpo_compute_control import ensure_public_repository\n"
                 "                    ensure_public_repository(api, repo_id)")
    # Full-text mathematical verification stays identical; only EOS routing changes.
    for relative in ("utils/math_verifier.py",):
        if digest(runtime / "reference" / relative) != digest(reference / relative):
            raise ValueError("Reference ER reward changed")
    write(root / "source_audit.json", {
        "upstream_commit": UPSTREAM_COMMIT, "reference_project": str(reference),
        "original_files_sha256": original_hashes,
        "adapted_files": [p.relative_to(runtime).as_posix() for p in runtime.rglob("*")
                          if p.is_file() and digest(p) != original_hashes[p.relative_to(runtime).as_posix()]],
        "reference_math_verify_byte_identical": True,
        "eos_policy": EOS_POLICY, "eos_reference_project": str(eos_reference),
        "eos_reference_checkpoint": {"repo": EOS_REFERENCE_REPO, "revision": EOS_REFERENCE_REVISION,
                                     "train_config_sha256": digest(checkpoint_config)},
    })


def render_data(frame, tokenizer):
    if len(frame) != 3200:
        raise ValueError("Expected the complete frozen 3200-row compression training split")
    rows, lengths = [], []
    for index, row in enumerate(frame.to_dict("records")):
        gold = str(row["extracted"]).strip()
        if not gold or gold != row["reward_model"]["ground_truth"]:
            raise ValueError("Compression row-level extracted gold changed")
        messages = [{"role": "user", "content": row["problem"] +
                     "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=True, tokenize=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt.endswith("<|im_start|>assistant\n") or len(ids) > 1536:
            raise ValueError("Thinking prompt changed or exceeds the audited limit; do not truncate rows")
        label = {"row_id": index, "dataset_name": DATASET_KEY, "problem": row["problem"], "extracted": gold}
        rows.append({"prompt": prompt, "label": json.dumps(label, ensure_ascii=False), "datasource": DATASET_KEY})
        lengths.append(len(ids))
    return rows, max(lengths)


def verify_plan(root):
    plan = read(root / "plan.json")
    if (root / "launch.json").exists() and read(root / "launch.json")["plan_sha256"] != digest(root / "plan.json"):
        raise ValueError("ER plan changed after queue launch")
    for relative, checksum in plan["frozen_files"].items():
        if digest(root / relative) != checksum:
            raise ValueError(f"Frozen ER input changed: {relative}")
    if digest(Path(plan["predecessor_root"]) / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("The preceding five-model evaluation plan changed")
    return plan


def check_waiting_refresh(root):
    """A queued configuration can only be replaced before training ever starts."""
    plan = verify_plan(root)
    state, receipt = read(root / "status.json"), read(root / "launch.json")
    if state.get("state") != "waiting_for_all_evaluations" or state.get("pid") != receipt.get("pid"):
        raise ValueError("Only the identified waiting ER supervisor can be refreshed")
    if any((root / name).exists() for name in ("training_started.json", "raw_rollouts", "rollout_dataset")):
        raise ValueError("ER execution already began; its frozen inputs cannot be replaced")
    return plan


def prepare(args):
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import validate_repo_id
    from transformers import AutoTokenizer

    previous = read(args.after_eval / "plan.json")
    node = require_compute(previous["job_id"])
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() != "agent/add-math12k-maxrl-launcher":
        raise RuntimeError("Use the primary maxrl launcher branch")
    root, reference = args.output_root.resolve(), args.reference_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "plan.json").exists():
            plan = verify_plan(root)
            if plan["reward"].get("eos_policy") != EOS_POLICY:
                raise ValueError("The waiting ER plan needs an explicit EOS-policy refresh")
            if plan.get("private") is not False:
                raise ValueError("The waiting ER plan needs a public-archive refresh")
            if plan["hf_repo_prefix"] != args.hf_prefix or plan["predecessor_root"] != str(args.after_eval.resolve()):
                raise ValueError("Prepared directory belongs to a different ER run")
            return plan
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference / "upstream", text=True).strip() != UPSTREAM_COMMIT:
            raise ValueError("Expected the pinned official OpenRLHF v0.7.3 source")
        if len(previous["models"]) != 5 or previous["budgets"] != [8192, 16384, 32768, 49152, 65536]:
            raise ValueError("ER must follow the complete five-model Minerva budget comparison")
        parent = read(Path(previous["parent_run"]) / "plan.json")
        if parent["model_revision"] != MODEL_REVISION or parent["dataset_revision"] != DATASET_REVISION:
            raise ValueError("Initial model or compression dataset differs from the current run")
        data_source = Path(parent["data_dir"]) / "train.parquet"
        if digest(data_source) != parent["input_hashes"][str(data_source)]:
            raise ValueError("Frozen compression data changed")
        initial = Path(parent["model_path"])
        weights = {Path(p).name: value for p, value in parent["input_hashes"].items() if Path(p).parent == initial}
        for name, checksum in weights.items():
            if digest(initial / name) != checksum:
                raise ValueError(f"Initial Qwen3 model file changed: {name}")
        scratch = Path(f"/tmp/erc{previous['job_id']}")
        if (scratch / "train/checkpoints").exists() and any((scratch / "train/checkpoints").iterdir()):
            raise ValueError("ER checkpoint destination already contains a run")
        plan = {
            "job_id": previous["job_id"], "node": node, "output_root": str(root), "runtime": str(root / "runtime"),
            "python_bin": sys.executable, "scratch": str(scratch), "train_dir": str(scratch / "train"),
            "source_repo": str(source), "reference_project": str(reference), "reference_upstream_commit": UPSTREAM_COMMIT,
            "predecessor_root": str(args.after_eval.resolve()), "predecessor_plan_sha256": digest(args.after_eval / "plan.json"),
            "holder_locks": previous["holder_locks"], "model_repo": "Qwen/Qwen3-1.7B", "model_revision": MODEL_REVISION,
            "model_path": str(initial), "model_files_sha256": weights,
            "dataset_repo": "zjhhhh/compression_dataset", "dataset_revision": DATASET_REVISION,
            "dataset_source_sha256": digest(data_source), "num_questions": 3200,
            "hf_repo_prefix": args.hf_prefix, "rollout_hf_repo": args.hf_prefix + "-rollouts", "private": False,
            "reward_port": 24378, "total_steps": 100, "rows_per_step": 256, "optimizer_updates": 200,
            "checkpoint_steps": [20, 40, 60, 80, 100], "seed": 79, "advantage_estimator": "rloo",
            "reward": {"type": "sigmoid", "alpha": 0.1, "check_eos": True, "score_after_thinking": False,
                       "scope": "full query Math-Verify; raw group responses and forced-EOS training queries",
                       "eos_policy": EOS_POLICY, "force_eos": True, "eos_presence_scope": "response tokens only",
                       "eos_reference_repo": EOS_REFERENCE_REPO, "eos_reference_revision": EOS_REFERENCE_REVISION,
                       "length_statistics": "raw responses with natural EOS and correct full-text grading; population std + 1e-7",
                       "empty_correct_length_pool": "reference fallback to current forced-response length"},
            "created_at": now(),
        }
        validate_repo_id(plan["rollout_hf_repo"])
        validate_repo_id(args.hf_prefix + "-step_100")
        checkpoint_config = Path(hf_hub_download(EOS_REFERENCE_REPO, "train_config.json", revision=EOS_REFERENCE_REVISION))
        eos_config = read(checkpoint_config)
        if (eos_config["pretrain"] != "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
                or eos_config["micro_rollout_batch_size"] != 1 or eos_config["packing_samples"]):
            raise ValueError("The reference checkpoint does not use the audited EOS processing path")
        snapshot(source, reference, root, str(initial), reference.parent / "efficient-reasoning", checkpoint_config)
        tokenizer = AutoTokenizer.from_pretrained(initial, local_files_only=True)
        plan["eos_token_id"], plan["pad_token_id"] = tokenizer.eos_token_id, tokenizer.pad_token_id
        rows, longest = render_data(pd.read_parquet(data_source), tokenizer)
        data = root / "runtime/datasets/compression_thinking.jsonl"
        data.parent.mkdir(parents=True)
        data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        preview = subprocess.check_output(["bash", str(root / "runtime" / LAUNCHER)], cwd=root / "runtime",
                                          env={**environment(plan), "DRY_RUN": "1"}, text=True)
        (root / "training_command.txt").write_text(preview)
        command = shlex.split(preview)
        settings = {key: command[command.index(key) + 1] for key in (
            "--advantage_estimator", "--rollout_batch_size", "--n_samples_per_prompt", "--train_batch_size",
            "--generate_max_len", "--prompt_max_len", "--temperature", "--top_p", "--actor_learning_rate",
            "--lr_warmup_ratio", "--init_kl_coef", "--zero_stage", "--seed", "--vllm_gpu_memory_utilization")}
        expected = ["rloo", "32", "8", "128", "32768", "1536", "1.0", "1.0", "1e-6", "0", "0.0", "3", "79", "0.5"]
        if list(settings.values()) != expected:
            raise ValueError("ER launcher no longer matches the requested reference settings")
        plan["training_settings"] = settings
        plan["versions"] = {name: importlib.metadata.version(name) for name in
                            ("openrlhf", "torch", "vllm", "deepspeed", "transformers", "ray", "math-verify")}
        write(root / "input_audit.json", {"rows": len(rows), "longest_prompt_tokens": longest, "prompt_limit": 1536,
                                         "truncated_or_dropped_rows": 0, "expected_rollouts": 25600, "node": node,
                                         "reference_full_query_reward": True})
        plan["frozen_files"] = {p.relative_to(root).as_posix(): digest(p)
                                for p in [*sorted((root / "runtime").rglob("*")), root / "training_command.txt",
                                          root / "source_audit.json", root / "input_audit.json"] if p.is_file()}
        api = HfApi()
        account = api.whoami()["name"]
        if account != args.hf_prefix.split("/")[0]:
            raise ValueError("Archive owner differs from the authenticated HF account")
        ensure_public_repository(api, plan["rollout_hf_repo"], "dataset")
        info = api.repo_info(repo_id=plan["rollout_hf_repo"], repo_type="dataset")
        if info.private or any(item.rfilename.startswith("data/") for item in info.siblings):
            raise ValueError("Expected an empty public training-rollout repository")
        write(root / "hf_auth_check.json", {"account": account, "checked_on": node, "checked_at": now()})
        write(root / "plan.json", plan)
        return plan


def predecessor_ready(plan):
    previous = Path(plan["predecessor_root"])
    if digest(previous / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Preceding evaluation identity changed")
    paths = [previous / "queue_status.json", previous / "status.json", previous / "report/audit.json"]
    queue, status, audit = [read(path) if path.exists() else {} for path in paths]
    if queue.get("state") == "failed" or status.get("state") == "failed":
        raise RuntimeError("The preceding Minerva comparison failed; ER has not been launched")
    if queue.get("state") != "complete" or status.get("state") != "complete":
        return False
    if (queue.get("points") != 25 or status.get("points") != 25 or not audit.get("complete")
            or audit.get("points") != 25 or audit.get("questions_per_point") != 272
            or not audit.get("all_rollout_ledgers_verified")):
        raise ValueError("ER requires all 25 fully audited Minerva points")
    if digest(previous / "report/metrics.json") != audit["metrics_sha256"]:
        raise ValueError("Preceding evaluation report changed")
    old_plan = read(previous / "plan.json")
    manifest = read(previous / "execution_manifest.json")
    for key in old_plan["models"]:
        for budget in old_plan["budgets"]:
            directory = previous / "results" / key / f"budget_{budget}"
            point = read(directory / "summary.json")
            if point.get("state") != "complete" or point.get("identity") != {
                "manifest": manifest["fingerprint"], "model": key, "budget": budget,
            }:
                raise ValueError("Missing completed model-budget point before ER")
            for artifact in point["artifacts"].values():
                path = directory / artifact["file"]
                if path.stat().st_size != artifact["size"] or digest(path) != artifact["sha256"]:
                    raise ValueError("Preceding rollout artifact changed")
    return True


def audit_local_rollouts(root, plan):
    from qwen3_experiments.er_compression_eos import force_eos_token_ids

    manifest = read(root / "rollout_dataset/rollout_manifest.json")
    if set(manifest["steps"]) != {str(step) for step in range(1, plan["total_steps"] + 1)}:
        raise ValueError("Training did not save every ER rollout step")
    visits, count = Counter(), 0
    for step in range(1, plan["total_steps"] + 1):
        info = manifest["steps"][str(step)]
        if info != {"file": f"data/step_{step:06d}.jsonl.gz", "num_rollouts": plan["rows_per_step"]}:
            raise ValueError("Unexpected ER rollout shard metadata")
        with gzip.open(root / "rollout_dataset" / info["file"], "rt") as stream:
            records = [json.loads(line) for line in stream]
        if len(records) != plan["rows_per_step"]:
            raise ValueError("Incomplete ER rollout shard")
        for index, record in enumerate(records):
            if (record["step"] != step or record["rollout_index"] != index
                    or record["generated_tokens"] != len(record["response_token_ids"])
                    or not record["response_token_ids"] or not record["reward_query"]):
                raise ValueError("Corrupt ER rollout token or reward record")
            raw = record["response_token_ids"]
            trained = force_eos_token_ids(record["prompt_token_ids"], raw, plan["eos_token_id"], plan["pad_token_id"])
            if (record["training_response_token_ids"] != trained
                    or record["generated_eos"] != (plan["eos_token_id"] in raw)
                    or record["force_eos_applied"] != (raw != trained)):
                raise ValueError("Saved ER tokens do not match the reference EOS policy")
            visits[record["label"]["row_id"]] += 1
        count += len(records)
    if visits != Counter({i: 8 for i in range(plan["num_questions"])}):
        raise ValueError("Expected exactly eight saved responses for every compression row")
    if manifest["num_steps"] != plan["total_steps"] or manifest["num_rollouts"] != count:
        raise ValueError("ER rollout manifest totals disagree")
    return count


def upload_rollouts(root, plan):
    from huggingface_hub import HfApi
    from qwen3_experiments.er_compression_rollouts import rollout_helpers

    require_compute(plan["job_id"])
    count = audit_local_rollouts(root, plan)
    helper, api = rollout_helpers(), HfApi()
    metadata = {"experiment_name": root.name, "model": plan["model_repo"], "model_revision": MODEL_REVISION,
                "dataset": plan["dataset_repo"], "dataset_revision": DATASET_REVISION,
                "seed": plan["seed"], "reward": plan["reward"], "expected_rollouts": count}
    # A completed dataset is immutable. Keep retrying uploads on the compute node.
    while True:
        try:
            write(root / "rollout_upload_status.json", {"state": "uploading", "num_rollouts": count, "updated_at": now()})
            helper._finalize_manifest(root / "rollout_dataset", metadata)
            repo = plan["rollout_hf_repo"]
            ensure_public_repository(api, repo, "dataset")
            commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root / "rollout_dataset"),
                                       allow_patterns=["README.md", "rollout_manifest.json", "data/*.jsonl.gz"],
                                       commit_message="Archive every ER compression training rollout")
            remote = {f.rfilename: f for f in api.repo_info(repo_id=repo, repo_type="dataset", revision=commit.oid,
                                                         files_metadata=True).siblings}
            from integration.checkpoint_archive import digests

            for path in (root / "rollout_dataset").rglob("*"):
                if not path.is_file() or ".cache" in path.parts:
                    continue
                name = path.relative_to(root / "rollout_dataset").as_posix()
                sha256, git_sha1 = digests(path)
                item = remote[name]
                actual = item.lfs.sha256 if item.lfs else item.blob_id
                if item.size != path.stat().st_size or actual != (sha256 if item.lfs else git_sha1):
                    raise ValueError(f"Remote training-rollout hash mismatch: {name}")
            receipt = {"state": "verified", "repo_id": repo, "revision": commit.oid, "num_steps": plan["total_steps"],
                       "num_rollouts": count, "verified_at": now()}
            write(root / "rollout_upload.json", receipt)
            write(root / "rollout_upload_status.json", receipt)
            return 0
        except Exception as exc:
            write(root / "rollout_upload_status.json", {"state": "retrying", "error": str(exc), "updated_at": now()})
            time.sleep(30)


def initialize_training_directories(root, plan):
    train = Path(plan["train_dir"])
    train.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "provenance", "archive_receipts"):
        destination = root / name
        destination.mkdir(exist_ok=True)
        link = train / name
        if not link.exists():
            link.symlink_to(destination, target_is_directory=True)
        if link.resolve() != destination:
            raise ValueError("Training log/archive directory points at another run")
    for name in ("source_audit.json", "input_audit.json", "plan.json", "training_command.txt"):
        shutil.copy2(root / name, root / "provenance" / name)


def training_command(plan):
    return ["srun", f"--jobid={plan['job_id']}", "--overlap", "--nodes=1", "--ntasks=1", f"--nodelist={plan['node']}",
            "--cpus-per-task=128", "--gres=gpu:8", "--kill-on-bad-exit=1", "--job-name=er-compression",
            "bash", str(Path(plan["runtime"]) / LAUNCHER)]


def supervise(root, plan):
    require_compute(plan["job_id"])
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "training_started.json").exists():
            raise RuntimeError("This ER run already launched training; refusing a duplicate")
        state = {"pid": os.getpid(), "hostname": socket.gethostname(), "job_id": plan["job_id"], "started_at": now()}

        def update(phase, **values):
            state.update(state=phase, updated_at=now(), **values)
            write(root / "status.json", state)

        try:
            while True:
                require_compute(plan["job_id"])
                previous = read(Path(plan["predecessor_root"]) / "queue_status.json")
                update("waiting_for_all_evaluations", predecessor_state=previous.get("state"), expected_minerva_points=25)
                if predecessor_ready(plan):
                    with ExitStack() as stack:
                        try:
                            for name in plan["holder_locks"]:
                                path = Path(name)
                                path.parent.mkdir(parents=True, exist_ok=True)
                                held = stack.enter_context(path.open("a"))
                                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            update("waiting_for_gpu_locks")
                        else:
                            busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
                            if not busy.strip():
                                verify_plan(root)
                                for name, checksum in plan["model_files_sha256"].items():
                                    if digest(Path(plan["model_path"]) / name) != checksum:
                                        raise ValueError("Pinned initial Qwen3 model changed before ER launch")
                                initialize_training_directories(root, plan)
                                with (root / "launcher.log").open("ab", buffering=0) as log:
                                    child = subprocess.Popen(training_command(plan), cwd=plan["runtime"], env=environment(plan),
                                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                                write(root / "training_started.json", {"pid": child.pid, "hostname": socket.gethostname(),
                                                                        "started_at": now(), "command": training_command(plan)})
                                while child.poll() is None:
                                    update("training_or_archiving", launcher_pid=child.pid)
                                    time.sleep(15)
                                if child.returncode != 0:
                                    raise RuntimeError(f"ER training/archival failed: exit {child.returncode}")
                                if read(root / "rollout_upload.json")["num_rollouts"] != 25600:
                                    raise ValueError("Incomplete final training-rollout upload")
                                for name in [f"global_step{step}" for step in plan["checkpoint_steps"]] + ["final_model"]:
                                    receipt = read(root / "archive_receipts" / f"{name}.json")
                                    suffix = "final" if name == "final_model" else "step_" + name.removeprefix("global_step")
                                    if receipt["repo_id"] != plan["hf_repo_prefix"] + "-" + suffix or not receipt["sha256"]:
                                        raise ValueError("Wrong or incomplete checkpoint archive receipt")
                                update("complete", exit_code=0, completed_rollout_steps=100, optimizer_updates=200,
                                       saved_training_rollouts=25600, finished_at=now())
                                return 0
                time.sleep(30)
        except BaseException as exc:
            update("failed", error=str(exc), finished_at=now())
            raise


def launch(root, plan):
    require_compute(plan["job_id"])
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("ER supervisor already launched")
        with (root / "supervisor.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([plan["python_bin"], "-u", "-m", MODULE, "supervise", "--plan", str(root / "plan.json")],
                                     cwd=plan["runtime"], env=environment(plan), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": socket.gethostname(), "job_id": plan["job_id"],
                   "plan_sha256": digest(root / "plan.json"), "launched_at": now()}
        write(root / "launch.json", receipt)
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "supervise", "upload-rollouts"))
    parser.add_argument("--after-eval", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--hf-prefix")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command in ("prepare", "launch"):
        if not all((args.after_eval, args.output_root, args.reference_root, args.hf_prefix)):
            parser.error("prepare/launch requires --after-eval, --output-root, --reference-root, --hf-prefix")
        plan = prepare(args)
        root = args.output_root.resolve()
        if args.command == "launch":
            launch(root, plan)
        else:
            print(json.dumps({"state": "prepared", "node": plan["node"], "output_root": str(root)}))
        return 0
    if not args.plan:
        parser.error("--plan is required")
    root = args.plan.resolve().parent
    plan = verify_plan(root)
    require_compute(plan["job_id"])
    return {"supervise": supervise, "upload-rollouts": upload_rollouts}[args.command](root, plan)


if __name__ == "__main__":
    raise SystemExit(main())
