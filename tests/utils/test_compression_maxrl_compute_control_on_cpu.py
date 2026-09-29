"""MaxRL configuration and the complete preceding Polaris evaluation handoff."""

import copy
import fcntl
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_l0_compute_control as control
from qwen3_experiments import evaluation_model_cache_cleanup as cleanup


def polaris_predecessor(tmp_path):
    root = tmp_path / "polaris"
    previous_er = tmp_path / "er"
    control.write(previous_er / "plan.json", {"job_id": "146103"})
    specs = [{"key": key, "repo": f"owner/{key}-step_100", "revision": str(i) * 40, "step": 100}
             for i, key in enumerate(("polaris_l0_step100", "polaris_er_step100", "polaris_maxrl_step100"), 1)]
    previous = {"job_id": "146103", "node": "compute", "models": specs,
                "nine_root": str(root / "nine"), "minerva_root": str(root / "minerva"), "frozen_files": {},
                "predecessor_root": str(previous_er), "predecessor_plan_sha256": control.digest(previous_er / "plan.json")}
    control.write(root / "plan.json", previous)
    control.write(root / "launch.json", {"plan_sha256": control.digest(root / "plan.json")})
    control.write(root / "queue_status.json", {"state": "complete", "nine_responses": 7276, "minerva_points": 15})
    nine = root / "nine"
    control.write(nine / "status.json", {"state": "complete"})
    control.write(nine / "report/per_sample.json", [])
    control.write(nine / "report/metrics.json", [])
    control.write(nine / "report/audit.json", {
        "complete": True, "questions": 1819, "responses_verified": 7276, "budget_points": 54,
        "all_budgets_regraded": True, "grader_errors": {},
        "per_sample_sha256": control.digest(nine / "report/per_sample.json"),
        "metrics_sha256": control.digest(nine / "report/metrics.json"),
    })
    metrics = []
    for spec in specs:
        folder = root / "minerva" / spec["key"]
        control.write(folder / "status.json", {"state": "complete"})
        manifest = {"fingerprint": "m" + spec["key"], "models": {spec["key"]: spec}}
        control.write(folder / "execution_manifest.json", manifest)
        for budget in [8192, 16384, 32768, 49152, 65536]:
            directory = folder / "results" / spec["key"] / f"budget_{budget}"
            control.write(directory / "rollouts.json", ["verified rollouts"])
            control.write(directory / "prompts.json", ["verified prompts"])
            control.write(directory / "summary.json", {
                "state": "complete", "num_prompts": 272, "ledger_audit": "passed", "seed": 0,
                "identity": {"manifest": manifest["fingerprint"], "model": spec["key"], "budget": budget},
                "artifacts": {name: {"file": f"{name}.json", "size": (directory / f"{name}.json").stat().st_size,
                                     "sha256": control.digest(directory / f"{name}.json")}
                              for name in ("rollouts", "prompts")},
            })
            metrics.append({"model_key": spec["key"], "budget_tokens": budget, "questions": 272, "seed": 0})
    control.write(root / "minerva/report/metrics.json", metrics)
    control.write(root / "minerva/report/audit.json", {
        "complete": True, "points": 15, "questions_per_point": 272, "all_rollout_ledgers_verified": True,
        "metrics_sha256": control.digest(root / "minerva/report/metrics.json"),
    })
    plan = {"job_id": "146103", "node": "compute", "predecessor_kind": "polaris_evaluation",
            "predecessor_evaluation_root": str(root), "predecessor_plan_sha256": control.digest(root / "plan.json")}
    return plan, root


def test_maxrl_waits_for_nine_and_all_fifteen_minerva_points(tmp_path):
    plan, root = polaris_predecessor(tmp_path)
    assert control.predecessor_ready(plan)
    for state in ("waiting_for_compression_er_evaluations", "running_or_waiting_for_gpus", "retrying"):
        control.write(root / "queue_status.json", {"state": state})
        assert not control.predecessor_ready(plan)


@pytest.mark.parametrize("damage", ["queue_failed", "partial_nine", "changed_nine", "partial_minerva",
                                    "changed_metrics", "unfinished_model", "wrong_checkpoint", "wrong_point",
                                    "changed_rollout", "changed_plan", "missing_artifacts"])
def test_incomplete_or_changed_polaris_results_cannot_release_maxrl(tmp_path, damage):
    plan, root = polaris_predecessor(tmp_path)
    model = root / "minerva/polaris_maxrl_step100"
    point = model / "results/polaris_maxrl_step100/budget_65536"
    if damage == "queue_failed":
        control.write(root / "queue_status.json", {"state": "failed"})
    elif damage == "partial_nine":
        control.write(root / "nine/status.json", {"state": "running"})
    elif damage == "changed_nine":
        control.write(root / "nine/report/metrics.json", ["changed"])
    elif damage == "partial_minerva":
        value = control.read(root / "queue_status.json")
        control.write(root / "queue_status.json", {**value, "minerva_points": 14})
    elif damage == "changed_metrics":
        control.write(root / "minerva/report/metrics.json", [])
    elif damage == "unfinished_model":
        control.write(model / "status.json", {"state": "running"})
    elif damage == "wrong_checkpoint":
        value = control.read(model / "execution_manifest.json")
        value["models"]["polaris_maxrl_step100"]["revision"] = "9" * 40
        control.write(model / "execution_manifest.json", value)
    elif damage == "wrong_point":
        value = control.read(point / "summary.json")
        value["identity"]["budget"] = 8192
        control.write(point / "summary.json", value)
    elif damage == "changed_rollout":
        control.write(point / "rollouts.json", ["changed"])
    elif damage == "missing_artifacts":
        value = control.read(point / "summary.json")
        control.write(point / "summary.json", {**value, "artifacts": {}})
    else:
        control.write(root / "plan.json", {"changed": True})
    with pytest.raises((ValueError, RuntimeError)):
        control.predecessor_ready(plan)


def test_only_maxrl_estimator_eos_gate_and_label_change():
    before = {"algorithm": {"adv_estimator": control.RB_ESTIMATOR, "cost_offset_tokens": 0},
              "reward_model": {"reward_kwargs": {"check_eos": True, "score_after_thinking": True}},
              "trainer": {"experiment_name": "l0"}, "data": {"train_batch_size": 32}}
    after = copy.deepcopy(before)
    after["algorithm"]["adv_estimator"] = "maxrl"
    after["reward_model"]["reward_kwargs"]["check_eos"] = False
    after["trainer"]["experiment_name"] = "maxrl"
    assert {item["path"] for item in control.compare_training_recipe(before, after, "maxrl")} == {
        "algorithm.adv_estimator", "reward_model.reward_kwargs.check_eos", "trainer.experiment_name"}
    after["data"]["train_batch_size"] = 16
    with pytest.raises(ValueError, match="train_batch_size"):
        control.compare_training_recipe(before, after, "maxrl")


def test_standalone_maxrl_launcher_keeps_all_rollouts_and_requested_grading():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(["bash", str(repo / control.MAXRL_LAUNCHER)], cwd=repo,
                            env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "PYTHON_BIN": "/not-installed"},
                            text=True, capture_output=True, check=True)
    for option in ("algorithm.adv_estimator=maxrl", "reward_model.reward_kwargs.check_eos=false",
                   "reward_model.reward_kwargs.score_after_thinking=True", "actor_rollout_ref.rollout.force_eos=False",
                   "actor_rollout_ref.rollout.ignore_eos=False", "trainer.rollout_dataset.enabled=true",
                   "trainer.rollout_dataset.private=false", "actor_rollout_ref.rollout.n=16",
                   "data.train_batch_size=32", "trainer.total_training_steps=100"):
        assert option in result.stdout
    assert "algorithm.adv_estimator=fixed_n_rb_offset_cost_aware_marginrl" not in result.stdout
    assert "MaxRL: binary after-thinking correctness reward" in result.stdout


def test_maxrl_training_uses_the_requested_allocation_and_launcher():
    plan = {"job_id": "146103", "node": "compute", "runtime": "/frozen snapshot", "adv_estimator": "maxrl",
            "launcher": control.MAXRL_LAUNCHER}
    command = control.training_command(plan)
    assert "--jobid=146103" in command and "--gres=gpu:8" in command
    assert "--job-name=compression-maxrl" in command
    assert command[-1] == "/frozen snapshot/" + control.MAXRL_LAUNCHER


def test_queued_training_plan_cannot_change_after_launch(tmp_path, monkeypatch):
    plan = {"output_root": str(tmp_path), "runtime": str(tmp_path / "runtime"), "input_hashes": {}}
    control.write(tmp_path / "plan.json", plan)
    control.write(tmp_path / "launch.json", {"plan_sha256": control.digest(tmp_path / "plan.json")})
    monkeypatch.setattr(control, "verify_runtime", lambda _: None)
    control.verify_training_inputs(plan)
    control.write(tmp_path / "plan.json", {**plan, "adv_estimator": "grpo"})
    with pytest.raises(ValueError, match="plan changed"):
        control.verify_training_inputs(plan)


def cleanup_plan(tmp_path):
    plan, root = polaris_predecessor(tmp_path)
    previous = control.read(root / "plan.json")
    scratch = tmp_path / "scratch"
    base = tmp_path / "base_model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"keep training weights")
    previous.update(scratch=str(scratch), base_model=str(base))
    for spec in previous["models"]:
        folder = scratch / "models" / spec["key"]
        for subdir in ("model", "source_model"):
            (folder / subdir).mkdir(parents=True)
            (folder / subdir / "weights.pt").write_bytes(b"cached weights")
        control.write(root / "prepared_models" / f"{spec['key']}.json", spec)
    control.write(root / "plan.json", previous)
    control.write(root / "launch.json", {"plan_sha256": control.digest(root / "plan.json")})
    plan.update(output_root=str(tmp_path / "training"), model_path=str(base),
                cleanup_predecessor_models_before_training=True,
                predecessor_plan_sha256=control.digest(root / "plan.json"),
                input_hashes={str(base / "model.safetensors"): control.digest(base / "model.safetensors")})
    control.write(Path(plan["output_root"]) / "plan.json", plan)
    return plan, root, scratch, base


def test_cleanup_removes_only_completed_model_caches_and_is_idempotent(tmp_path):
    plan, root, scratch, base = cleanup_plan(tmp_path)
    before = control.digest(root / "minerva/report/metrics.json")
    receipt = control.cleanup_before_training(plan)
    assert receipt["state"] == "complete"
    assert all(model["state"] == "deleted" for model in receipt["models"])
    assert not list((scratch / "models").iterdir())
    assert (base / "model.safetensors").read_bytes() == b"keep training weights"
    assert control.digest(root / "minerva/report/metrics.json") == before
    assert control.predecessor_ready(plan)
    assert control.cleanup_before_training(plan) == receipt


def test_handoff_accepts_caches_already_deleted_after_each_model(tmp_path):
    plan, root, scratch, base = cleanup_plan(tmp_path)
    for folder in (scratch / "models").iterdir():
        cleanup.shutil.rmtree(folder)
    receipt = control.cleanup_before_training(plan)
    assert receipt["state"] == "complete"
    assert all(model["state"] == "already_absent" for model in receipt["models"])
    assert control.predecessor_ready(plan)
    assert (base / "model.safetensors").read_bytes() == b"keep training weights"


def test_unfinished_evaluations_cannot_trigger_model_cleanup(tmp_path):
    plan, root, scratch, _ = cleanup_plan(tmp_path)
    control.write(root / "queue_status.json", {"state": "running_or_waiting_for_gpus"})
    with pytest.raises(RuntimeError, match="must finish"):
        control.cleanup_before_training(plan)
    assert len(list((scratch / "models").iterdir())) == 3


@pytest.mark.parametrize("lock_name", ["queue.lock", "run.lock"])
def test_cleanup_waits_for_evaluation_processes_to_release_their_locks(tmp_path, lock_name):
    plan, root, scratch, _ = cleanup_plan(tmp_path)
    with (root / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            control.cleanup_before_training(plan)
    assert len(list((scratch / "models").iterdir())) == 3


@pytest.mark.parametrize("damage", ["training_input", "symlink", "wrong_checkpoint"])
def test_cleanup_preflights_every_model_before_deleting_any_cache(tmp_path, damage):
    plan, root, scratch, base = cleanup_plan(tmp_path)
    last = scratch / "models/polaris_maxrl_step100"
    if damage == "training_input":
        plan["model_path"] = str(last / "model")
    elif damage == "symlink":
        (last / "model/base_alias").symlink_to(base, target_is_directory=True)
    else:
        p = root / "prepared_models/polaris_maxrl_step100.json"
        control.write(p, {**control.read(p), "revision": "9" * 40})
    with pytest.raises(ValueError):
        control.cleanup_before_training(plan)
    assert all((folder / "model/weights.pt").exists() for folder in (scratch / "models").iterdir())
    assert (base / "model.safetensors").exists()


def test_partial_cleanup_can_resume_without_removing_results(tmp_path, monkeypatch):
    plan, root, scratch, _ = cleanup_plan(tmp_path)
    real_delete = cleanup.shutil.rmtree
    calls = []

    def interrupted(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated cleanup failure")
        real_delete(path)

    monkeypatch.setattr(cleanup.shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="cleanup failure"):
        control.cleanup_before_training(plan)
    assert control.read(Path(plan["output_root"]) / "evaluation_model_cache_cleanup.json")["state"] == "cleaning"
    monkeypatch.setattr(cleanup.shutil, "rmtree", real_delete)
    assert control.cleanup_before_training(plan)["state"] == "complete"
    assert not list((scratch / "models").iterdir())
    assert control.predecessor_ready(plan)


def test_cleanup_failure_prevents_training_process_start(tmp_path, monkeypatch):
    plan, root, _, _ = cleanup_plan(tmp_path)
    plan.update(runtime=str(tmp_path), evaluate_after_training=False, adv_estimator="maxrl",
                hf_repo_prefix="owner/maxrl", holder_locks=[], predecessor_status=str(root / "queue_status.json"))
    monkeypatch.setattr(control, "require_compute", lambda _: None)
    monkeypatch.setattr(control, "verify_training_inputs", lambda _: None)
    monkeypatch.setattr(control, "gpu_idle", lambda: True)
    monkeypatch.setattr(control, "spawn", lambda *args: SimpleNamespace(pid=1, poll=lambda: None))

    def fail_cleanup(_):
        raise OSError("cleanup must succeed first")

    monkeypatch.setattr(control, "cleanup_before_training", fail_cleanup)
    monkeypatch.setattr(control.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Training started before cleanup"))
    with pytest.raises(OSError, match="cleanup must succeed first"):
        control.supervise(plan)
    assert not (Path(plan["output_root"]) / "training_started.json").exists()
