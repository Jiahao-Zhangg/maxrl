"""Evaluate L+0 step 80, compare training sets, then resume the interrupted training."""

import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def prepare_step80(config):
    original = Path(config["original_eval_root"])
    root = Path(config["step80_eval_root"])
    root.mkdir(parents=True, exist_ok=True)
    helper = module("l0_step80_prepare", original / "provenance/eval_l0_final.py")
    base = helper.evaluator(original)
    old = helper.verify_plan(original, base)
    assert old["job_id"] == config["job_id"] == "146103"
    with (root / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / "manifest.json").exists(), "Do not change an evaluation that has started"
        for name in old["frozen_files"]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original / name, target)
        receipt = base.read(Path(config["training_root"]) / "hf_checkpoint_archive/receipts/global_step_80.json")
        assert receipt["checkpoint"] == "global_step_80" and receipt["state"] == "archived_and_deleted"
        assert receipt["remote_commit"] == config["checkpoint_revision"]
        base.write(root / "checkpoint_receipt.json", receipt)
        base.write(root / "provenance/original_final_eval_plan.json", old)
        base.write(root / "provenance/request.json", {
            "user_revision": "Evaluate L+0 step 80 first, then both training datasets, then resume training",
            "model_repo": receipt["repo_id"], "revision": receipt["remote_commit"],
            "job_id": "146103", "same_questions_prompts_sampling_and_after_thinking_budgets": True,
        })
        inputs = base.read(root / "input_plan.json")
        inputs["model"] = {"repo": receipt["repo_id"], "revision": receipt["remote_commit"]}
        base.write(root / "input_plan.json", inputs)
        shutil.copy2(Path(__file__), root / "provenance/l0_recovery_pipeline.py")
        # Spawned grading processes must import the helper under its parent-process name.
        shutil.copy2(root / "provenance/eval_l0_final.py", root / "provenance/l0_step80_helpers.py")
        launcher = (original / "provenance/run_l0_final_eval.sh").read_text()
        launcher = launcher.replace('"${EVAL_ROOT}/provenance/eval_l0_final.py" run',
                                    '"${EVAL_ROOT}/provenance/l0_recovery_pipeline.py" step80')
        launcher = launcher.replace("/tmp/l0-final-eval-", "/tmp/l0-step80-eval-")
        (root / "provenance/run_l0_step80_eval.sh").write_text(launcher)
        names = set(old["frozen_files"]) | {
            "checkpoint_receipt.json", "provenance/original_final_eval_plan.json",
            "provenance/l0_recovery_pipeline.py", "provenance/run_l0_step80_eval.sh",
            "provenance/l0_step80_helpers.py",
        }
        plan = {**old, "model_repo": receipt["repo_id"], "final_step": 80, "checkpoint_step": 80,
                "model_storage": config["model_storage"], "created_at": time.time(),
                "user_choices": {**old["user_choices"], "checkpoint": "step 80 before resuming training",
                                 "allocation": "Use all eight GPUs on 146103 for both evaluations before resuming training"},
                "frozen_files": {name: base.digest(root / name) for name in sorted(names)}}
        base.write(root / "plan.json", plan)
        shutil.copy2(original / "input_audit.json", root / "input_audit.json")
        shutil.copy2(original / "prefix_validation.json", root / "prefix_validation.json")
        (root / "README.md").write_text(
            "# L+0 step 80 evaluation\n\n"
            "按用户更新的顺序：先评测 L+0 step 80，再进行两个训练集的难度比较，最后从 step 80 恢复训练至 100。"
            "全部使用 allocation 146103 的 8 张 GPU。\n\n"
            "本项沿用已核验的 1,819 题、每题 4 个回答，共 7,276 个回答。"
            "Thinking on；temperature 0.6、top-p 0.95、top-k 20、min-p 0、seed 42；32k 输出。"
            "总表和 1k–32k 预算表只判最后一个完成的 </think> 后的文本，不额外要求 boxed 或 EOS。\n\n"
            f"Model: [{receipt['repo_id']}](https://huggingface.co/{receipt['repo_id']}/tree/{receipt['remote_commit']}).\n\n"
            "模型权重暂存在 146103 的节点本地磁盘；分词器、回答和报告保留在共享目录。\n\n"
            "[状态](status.json) · [计划](plan.json) · [最终报告（完成后）](report/README.md)\n"
        )
        helper.verify_plan(root, helper.evaluator(root))
        print("Prepared the step-80 evaluation with unchanged nine-dataset questions and grading", flush=True)


def prepare_step80_model(root, base, plan):
    from transformers import AutoTokenizer

    receipt = base.read(root / "checkpoint_receipt.json")
    storage = Path(plan["model_storage"])
    storage.mkdir(parents=True, exist_ok=True)
    base.MODEL, base.REVISION = receipt["repo_id"], receipt["remote_commit"]
    base.PREFIX, base.REPO = "global_step_80/actor/", Path(plan["repository"])
    base.prepare_model(storage)
    merged = base.read(storage / "model_receipt.json")
    for name, item in merged["source_files"].items():
        original = receipt["files"][name.removeprefix("global_step_80/")]
        assert item == {"size": original["size"], "sha256": original["sha256"]}
    destination = root / "model"
    destination.mkdir(exist_ok=True)
    for name, checksum in merged["merged_files"].items():
        source, target = storage / "model" / name, destination / name
        if name.endswith(".safetensors"):
            if not target.is_symlink():
                assert not target.exists()
                target.symlink_to(source)
            assert target.resolve() == source.resolve()
        else:
            shutil.copy2(source, target)
        assert base.digest(target) == checksum
    base.write(root / "model_receipt.json", merged)
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    reference = AutoTokenizer.from_pretrained(plan["reference_model"], local_files_only=True)
    assert tokenizer.get_vocab() == reference.get_vocab() and tokenizer.chat_template == reference.chat_template
    for question in base.read(root / "questions.json"):
        assert tokenizer.apply_chat_template(question["messages"], add_generation_prompt=True,
                                             enable_thinking=True) == question["prompt_token_ids"]
    inputs = base.read(root / "input_plan.json")
    inputs["plan_sha256"] = base.digest(root / "plan.json")
    inputs["checkpoint_receipt_sha256"] = base.digest(root / "checkpoint_receipt.json")
    base.write(root / "prepared_inputs.json", inputs)


def evaluate_step80(root):
    helper = module("l0_step80_helpers", root / "provenance/eval_l0_final.py")
    base = helper.evaluator(root)
    helper.prepare_model = prepare_step80_model
    original_report = helper.write_report

    def labeled_report(directory, evaluator, metrics):
        original_report(directory, evaluator, metrics)
        for name in ("README.md", "comparison.csv", "budget_comparison.csv"):
            path = directory / "report" / name
            text = path.read_text().replace("L+0 final checkpoint", "L+0 step-80 checkpoint")
            path.write_text(text.replace("L+0 (final)", "L+0 (step 80)"))

    helper.write_report = labeled_report
    try:
        helper.run(root, base)
    except BaseException as exc:
        base.write(root / "status.json", {"state": "failed", "error": str(exc), "updated_at": time.time()})
        raise


def train_environment(config, original):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("MAXRL_", "RAY_", "VLLM_", "SLURM_", "WANDB_")) and key != "WANDB_API_KEY":
            env.pop(key)
    for key in ("PYTHONHOME", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        env.pop(key, None)
    scratch = Path(config["training_scratch"])
    env.update({
        "PATH": str(Path(original["python_bin"]).parent) + os.pathsep + env["PATH"],
        "PYTHONPATH": original["repo_root"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NCCL_DEBUG": "WARN", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "SEED": "79", "MAXRL_TRAIN_RUN_DIR": config["training_root"], "MAXRL_MODEL_PATH": original["model_path"],
        "MAXRL_MAX_PROMPT_LENGTH": str(config["max_prompt_length"]), "MAXRL_RAY_DIR": str(scratch / "ray"),
        "TMPDIR": str(scratch / "tmp"), "TRITON_CACHE_DIR": str(scratch / "triton"),
        "WANDB_DIR": str(Path(config["training_root"]) / "wandb"), "WANDB_MODE": "online", "WANDB_INIT_TIMEOUT": "60",
    })
    return env


def resume_command(config, original):
    source = Path(config["model_storage"]) / "source_model/global_step_80"
    output = Path(config["training_scratch"]) / "checkpoints"
    return ["bash", str(Path(original["repo_root"]) / "qwen3_experiments/run_qwen3_1_7b_polaris_1_8_3200_per_context_rb_l0_0.sh"),
            "trainer.resume_mode=resume_path", f"trainer.resume_from_path={source}",
            f"trainer.default_local_dir={output}"]


def slurm_training_command(config, command):
    # Slurm injects visibility variables after the controller's environment cleanup.
    # Remove ROCm variables inside the job step, before Python and Ray are started.
    return ["srun", f"--jobid={config['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=128", "--gres=gpu:8", "--kill-on-bad-exit=1", "--job-name=l0-resume80",
            "env", "-u", "ROCR_VISIBLE_DEVICES", "-u", "HIP_VISIBLE_DEVICES", *command]


def completed_evaluation(root, base, expected):
    paths = [root / "queue_status.json", root / "status.json", root / "report/audit.json"]
    if not all(path.is_file() for path in paths):
        return False
    queue, status, audit = map(base.read, paths)
    if queue.get("state") != "complete" or status.get("state") != "complete":
        return False
    assert status["completed_responses"] == status["total_responses"] == expected
    assert audit["complete"] and audit["responses_verified"] == expected
    assert not audit.get("grader_errors")
    for filename, checksum in (("metrics.json", "metrics_sha256"), ("per_sample.json", "per_sample_sha256")):
        if checksum in audit:
            assert base.digest(root / "report" / filename) == audit[checksum]
    if expected == 7276:
        helper = module("completed_l0_validation", root / "provenance/eval_l0_final.py")
        assert audit["budget_points"] == 54 and status["completed_questions"] == 1819
        helper.verify_plan(root, base)
    else:
        helper = module("completed_difficulty_validation", root / "provenance/eval_training_difficulty.py")
        helper.verify_plan(root, base)
    return True


def verify_resume_configuration(before, after):
    expected_changes = {("trainer", "resume_mode"), ("trainer", "resume_from_path"),
                        ("trainer", "default_local_dir"), ("ray_init", "ray_dir")}
    differences = []

    def compare(left, right, prefix=()):
        if isinstance(left, dict) and isinstance(right, dict):
            assert set(left) == set(right), f"Config keys changed: {prefix}"
            for key in left:
                compare(left[key], right[key], prefix + (key,))
        elif left != right:
            differences.append({"path": ".".join(prefix), "before": left, "after": right})
            assert prefix in expected_changes, f"Unexpected training change: {prefix}"

    compare(before, after)
    assert after["trainer"]["resume_mode"] == "resume_path"
    assert after["trainer"]["resume_from_path"].endswith("/global_step_80")
    assert after["trainer"]["total_training_steps"] == 100
    assert after["actor_rollout_ref"]["actor"]["checkpoint"]["load_contents"] == ["model", "optimizer", "extra"]
    return differences


def download_resume_checkpoint(config, base):
    from huggingface_hub import hf_hub_download

    receipt = base.read(Path(config["step80_eval_root"]) / "checkpoint_receipt.json")
    destination = Path(config["model_storage"]) / "source_model"
    progress = Path(config["control_root"]) / "resume_download.json"

    def download(name):
        item = receipt["files"][name]
        filename = "global_step_80/" + name
        local = destination / filename
        if not local.exists() or local.stat().st_size != item["size"] or base.digest(local) != item["sha256"]:
            local = Path(hf_hub_download(receipt["repo_id"], filename, revision=receipt["remote_commit"], local_dir=destination))
        assert local.stat().st_size == item["size"] and base.digest(local) == item["sha256"]
        return name

    complete = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, name) for name in receipt["files"]]
        for future in as_completed(futures):
            complete.append(future.result())
            base.write(progress, {"verified_files": len(complete), "total_files": len(receipt["files"]),
                                  "completed": sorted(complete), "updated_at": now()})
    assert len(complete) == 33 and "data.pt" in complete


def resume_training(config, base):
    import yaml

    control, training = Path(config["control_root"]), Path(config["training_root"])
    original = base.read(config["original_training_plan"])
    download_resume_checkpoint(config, base)
    for name, checksum in base.read(original["model_manifest"])["files_sha256"].items():
        assert base.digest(Path(original["model_path"]) / name) == checksum
    for name, checksum in config["verified_training_files"].items():
        assert base.digest(name) == checksum
    scratch = Path(config["training_scratch"])
    for name in ("tmp", "ray", "triton", "checkpoints"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(training / "checkpoints/wandb_id.txt", scratch / "checkpoints/wandb_id.txt")
    env = train_environment(config, original)
    command = resume_command(config, original)
    preview = subprocess.run(command + ["--cfg", "job", "--resolve"], cwd=original["repo_root"], env=env,
                             text=True, capture_output=True, check=True)
    (control / "resume_config.yaml").write_text(preview.stdout)
    differences = verify_resume_configuration(yaml.safe_load((training / "resolved_config.yaml").read_text()),
                                               yaml.safe_load(preview.stdout))
    base.write(control / "config_differences.json", differences)
    with ExitStack() as locks:
        for name in config["holder_locks"]:
            lock = locks.enter_context(Path(name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            time.sleep(30)
        state = base.read(training / "status.json")
        state.update(state="resuming", last_completed_step=80, last_completed_before_interruption=94,
                     resume_from_step=80, recovery_started_at=now(), supervisor_pid=os.getpid(),
                     supervisor_hostname=os.uname().nodename, checkpoint_directory=str(scratch / "checkpoints"),
                     recovery_controller_pid=os.getpid(), recovery_controller_hostname=os.uname().nodename,
                     recovery_attempt=config.get("attempt", 1), updated_at=now())
        state.pop("exit_code", None)
        state.pop("finished_at", None)
        base.write(training / "status.json", state)
        archive_env = {**env, "PYTHON_BIN": sys.executable, "MAXRL_ARCHIVE_UPLOAD_LATEST": "1",
                       "MAXRL_ARCHIVE_VERIFY_HASHES": "1", "MAXRL_ARCHIVE_DEFER_FINAL_UNTIL_EXIT": "1",
                       "MAXRL_ARCHIVE_RECEIPT_DIR": str(training / "hf_checkpoint_archive/receipts"),
                       "MAXRL_ARCHIVE_POLL_SECONDS": "60", "MAXRL_HF_UPLOAD_WORKERS": "4",
                       "MAXRL_TRAINING_EXIT_STATUS_FILE": str(control / "training_exit_status"),
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        assert not (control / "training_exit_status").exists()
        archive_command = ["flock", "-n", str(training / "hf_checkpoint_archive/monitor.lock"), "bash",
                           str(Path(original["repo_root"]) / "qwen3_experiments/archive_checkpoints_to_hf.sh"),
                           str(scratch / "checkpoints"), original["checkpoint_hf_prefix"], str(os.getpid()),
                           str(training / "train.log"), "100"]
        with (training / "hf_checkpoint_archive/upload.log").open("ab", buffering=0) as log:
            archiver = subprocess.Popen(archive_command, env=archive_env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state["archive_monitor_pid"] = archiver.pid
        slurm_command = slurm_training_command(config, command)
        with (training / "train.log").open("a", buffering=1) as log:
            log.write(f"\nRecovery from step 80 at {now()} after both requested evaluations\n")
            child = subprocess.Popen(slurm_command, cwd=original["repo_root"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            state["launcher_pid"] = child.pid
            base.write(training / "status.json", state)
            for line in child.stdout:
                log.write(line)
                clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
                matched = re.search(r"(?<![/\w])step:\s*(\d+)\b", clean)
                if matched and int(matched[1]) > state["last_completed_step"]:
                    state.update(state="training", last_completed_step=int(matched[1]), updated_at=now())
                    base.write(training / "status.json", state)
                elif "Setting global step to 80" in clean:
                    state.update(state="loading_checkpoint", checkpoint_restore_started=True, updated_at=now())
                    base.write(training / "status.json", state)
            code = child.wait()
        good = code == 0 and state["last_completed_step"] == 100 and (scratch / "checkpoints/global_step_100").is_dir()
        state.update(state="complete" if good else "failed", exit_code=code, finished_at=now(), updated_at=now())
        base.write(training / "status.json", state)
        temporary = control / "training_exit_status.tmp"
        temporary.write_text(str(code if code else (0 if good else 1)) + "\n")
        temporary.replace(control / "training_exit_status")
        assert good, f"Resumed training failed with exit {code}"
        assert archiver.wait() == 0, "Training completed but checkpoint archival failed"


def pipeline(config):
    control = Path(config["control_root"])
    control.mkdir(parents=True, exist_ok=True)
    base = module("recovery_pipeline_core", Path(config["step80_eval_root"]) / "provenance/eval_polaris_step80.py")
    assert os.uname().nodename == config["node"]
    with (control / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "hostname": os.uname().nodename, "job_id": config["job_id"],
                 "started_at": now(), "order": ["l0_step80_evaluation", "training_difficulty", "resume_l0_training"],
                 "attempt": config.get("attempt", 1), "completed_evaluations": []}

        def update(**values):
            state.update(values, updated_at=now())
            base.write(control / "status.json", state)

        active_root = None
        try:
            for name, root_name, launcher_name, expected in (
                ("l0_step80_evaluation", "step80_eval_root", "run_l0_step80_eval.sh", 7276),
                ("training_difficulty", "difficulty_root", "run_training_difficulty_eval.sh", 3200),
            ):
                root = Path(config[root_name])
                if completed_evaluation(root, base, expected):
                    state["completed_evaluations"].append(name)
                    update(state="verified_completed_evaluation", phase=name)
                    print(f"Verified completed {name}; preserving its results", flush=True)
                    continue
                active_root = root
                update(state="evaluating", phase=name)
                queue = {"pid": os.getpid(), "hostname": os.uname().nodename, "job_id": config["job_id"],
                         "gpus": 8, "state": "running", "total_responses": expected, "updated_at": time.time()}
                base.write(root / "queue_status.json", queue)
                while True:
                    update(state="evaluating", phase=name)
                    command = ["srun", f"--jobid={config['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
                               "--cpus-per-task=96", "--gres=gpu:8", "--kill-on-bad-exit=1", f"--job-name={name}",
                               "bash", str(root / "provenance" / launcher_name), sys.executable, str(root)]
                    with (root / "launch.log").open("a", buffering=1) as log:
                        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            update(launcher_pid=child.pid)
                            time.sleep(15)
                    if child.returncode == 75:
                        update(state="waiting_for_all_eight_gpus")
                        time.sleep(30)
                        continue
                    assert child.returncode == 0, f"{name} failed with exit {child.returncode}"
                    break
                audit = base.read(root / "report/audit.json")
                assert audit["complete"] and audit["responses_verified"] == expected
                queue.update(state="complete", completed_responses=expected, updated_at=time.time())
                base.write(root / "queue_status.json", queue)
                state["completed_evaluations"].append(name)
                active_root = None
            update(state="resuming_training", phase="resume_l0_training")
            resume_training(config, base)
            update(state="complete", training_final_step=100)
        except BaseException as exc:
            update(state="failed", error=str(exc))
            if active_root is not None:
                queue.update(state="failed", error=str(exc), updated_at=time.time())
                base.write(active_root / "queue_status.json", queue)
            raise


def launch(config):
    control = Path(config["control_root"])
    assert os.uname().nodename == config["node"]
    control.mkdir(parents=True, exist_ok=True)
    with (control / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (control / "launch.json").exists(), "Pipeline already launched"
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        env.pop("PYTHONHOME", None)
        with (control / "pipeline.log").open("ab", buffering=0) as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "pipeline", "--config",
                                      str(control / "config.json")], stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
        receipt = {"pid": child.pid, "hostname": os.uname().nodename, "launched_at": now(), "job_id": config["job_id"]}
        (control / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-step80", "step80", "pipeline", "launch"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.command == "step80":
        evaluate_step80(args.output_root.resolve())
    else:
        config = json.loads(args.config.read_text())
        {"prepare-step80": prepare_step80, "pipeline": pipeline, "launch": launch}[args.command](config)


if __name__ == "__main__":
    main()
