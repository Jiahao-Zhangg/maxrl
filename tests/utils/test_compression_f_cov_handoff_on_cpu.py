"""f_cov must wait for successful MaxRL training, archival, and safe cleanup."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import compression_l0_compute_control as control


def predecessor(tmp_path):
    root = tmp_path / "maxrl"
    previous = {"job_id": "146103", "node": "compute", "total_steps": 100, "rows_per_step": 512,
                "checkpoint_steps": list(range(10, 101, 10)), "hf_repo_prefix": "owner/maxrl",
                "rollout_hf_repo": "owner/maxrl-rollouts", "rollout_dir": str(root / "rollouts")}
    control.write(root / "plan.json", previous)
    control.write(root / "status.json", {"state": "complete", "exit_code": 0, "last_completed_step": 100})
    control.write(root / "training_exit.json", {"exit_code": 0})
    control.write(root / "hf_checkpoint_archive/status.json",
                  {"state": "complete", "archived_steps": previous["checkpoint_steps"]})
    for step in previous["checkpoint_steps"]:
        control.write(root / f"hf_checkpoint_archive/receipts/global_step_{step}.json", {
            "state": "archived_and_deleted", "checkpoint": f"global_step_{step}",
            "repo_id": f"owner/maxrl-step_{step}", "remote_commit": "a" * 40,
        })
    control.write(root / "rollout_upload.json", {
        "state": "verified", "repo_id": previous["rollout_hf_repo"],
        "num_steps": 100, "num_rollouts": 51200, "remote_commit": "b" * 40,
    })
    plan = {"job_id": "146103", "node": "compute", "predecessor_kind": "training",
            "predecessor_training_root": str(root), "predecessor_plan_sha256": control.digest(root / "plan.json"),
            "predecessor_status": str(root / "status.json"), "predecessor_rollout_data_root": previous["rollout_dir"],
            "cleanup_predecessor_rollouts_before_training": True, "output_root": str(tmp_path / "f_cov"),
            "handoff_local_receipt_dir": str(tmp_path / "local/handoff")}
    return plan, root


def test_wait_for_training_even_when_intermediate_uploads_exist(tmp_path):
    plan, root = predecessor(tmp_path)
    assert control.predecessor_ready(plan)
    control.write(root / "status.json", {"state": "training", "last_completed_step": 99})
    assert not control.predecessor_ready(plan)
    with pytest.raises(RuntimeError, match="must finish"):
        control.cleanup_before_training(plan)


@pytest.mark.parametrize("file,changes", [
    ("status.json", {"state": "failed"}),
    ("status.json", {"last_completed_step": 99}),
    ("training_exit.json", {"exit_code": 1}),
    ("plan.json", {"node": "other"}),
    ("hf_checkpoint_archive/status.json", {"state": "monitoring"}),
    ("hf_checkpoint_archive/status.json", {"archived_steps": [100]}),
    ("hf_checkpoint_archive/receipts/global_step_100.json", {"state": "uploading"}),
    ("hf_checkpoint_archive/receipts/global_step_10.json", {"repo_id": "other/model"}),
    ("hf_checkpoint_archive/receipts/global_step_100.json", {"remote_commit": ""}),
    ("rollout_upload.json", {"num_steps": 99}),
    ("rollout_upload.json", {"num_rollouts": 50688}),
    ("rollout_upload.json", {"repo_id": "other/rollouts"}),
    ("rollout_upload.json", {"remote_commit": ""}),
])
def test_incomplete_or_wrong_uploads_cannot_release_f_cov(tmp_path, file, changes):
    plan, root = predecessor(tmp_path)
    control.write(root / file, {**control.read(root / file), **changes})
    with pytest.raises((ValueError, RuntimeError)):
        control.predecessor_ready(plan)


def test_cleanup_checks_public_commit_and_retains_other_inputs(tmp_path, monkeypatch):
    import huggingface_hub

    plan, root = predecessor(tmp_path)
    source = root / "rollouts"
    steps, siblings = {}, []
    for step in range(1, 101):
        relative = f"data/step_{step:06d}.jsonl.gz"
        control.write(source / relative, {"closed_step": step})
        steps[str(step)] = {"file": relative, "num_rollouts": 512}
        siblings.append(SimpleNamespace(rfilename=relative, size=(source / relative).stat().st_size,
                                        lfs=SimpleNamespace(sha256=control.digest(source / relative))))
    control.write(source / "rollout_manifest.json", {"steps": steps, "num_steps": 100, "num_rollouts": 51200})
    protected = tmp_path / "146102" / "model.json"
    control.write(protected, {"keep": True})
    calls = []

    def repo_info(repo, **kwargs):
        calls.append((repo, kwargs))
        return SimpleNamespace(private=False, sha="b" * 40, siblings=siblings)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(repo_info=repo_info))
    result = control.cleanup_before_training(plan)
    assert result["state"] == "uploaded_verified_and_deleted" and len(result["files"]) == 100
    assert calls == [("owner/maxrl-rollouts", {"repo_type": "dataset", "revision": "b" * 40, "files_metadata": True})]
    assert not list((source / "data").iterdir())
    assert (source / "rollout_manifest.json").exists() and control.read(protected) == {"keep": True}
    assert control.cleanup_before_training(plan) == result
    assert len(calls) == 1
    assert control.predecessor_ready(plan)


def test_changed_rollout_location_is_never_cleaned(tmp_path):
    plan, root = predecessor(tmp_path)
    elsewhere = tmp_path / "146102"
    elsewhere.mkdir()
    (root / "rollouts").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="moved"):
        control.cleanup_before_training(plan)


def test_full_batch_f_cov_changes_only_requested_recipe_fields():
    before = {"algorithm": {"adv_estimator": control.RB_ESTIMATOR, "cost_offset_tokens": 0, "f_cov_num_prompts": 256},
              "reward_model": {"reward_kwargs": {"check_eos": True, "score_after_thinking": True}},
              "data": {"train_batch_size": 32}, "trainer": {"experiment_name": "l0"}}
    after = copy.deepcopy(before)
    after["algorithm"].update(adv_estimator="f_cov", f_cov_num_prompts=32)
    after["reward_model"]["reward_kwargs"]["check_eos"] = False
    assert {item["path"] for item in control.compare_training_recipe(before, after, "f_cov")} == {
        "algorithm.adv_estimator", "algorithm.f_cov_num_prompts", "reward_model.reward_kwargs.check_eos"}
    after["algorithm"]["f_cov_num_prompts"] = 256
    with pytest.raises(ValueError, match="full-batch"):
        control.compare_training_recipe(before, after, "f_cov")
    command = control.training_command({"job_id": "146103", "node": "compute", "runtime": "/frozen",
                                        "adv_estimator": "f_cov", "launcher": control.FCOV_LAUNCHER})
    assert "--job-name=compression-f-cov" in command and "--gres=gpu:8" in command
    assert command[-1] == "/frozen/" + control.FCOV_LAUNCHER


def test_cleanup_failure_cannot_start_f_cov(tmp_path, monkeypatch):
    plan, _ = predecessor(tmp_path)
    plan.update(runtime=str(tmp_path), evaluate_after_training=False, adv_estimator="f_cov",
                hf_repo_prefix="owner/f-cov", holder_locks=[])
    Path(plan["output_root"]).mkdir()
    monkeypatch.setattr(control, "require_compute", lambda _: None)
    monkeypatch.setattr(control, "verify_training_inputs", lambda _: None)
    monkeypatch.setattr(control, "gpu_idle", lambda: True)
    monkeypatch.setattr(control, "spawn", lambda *args: SimpleNamespace(pid=1, poll=lambda: None))

    def fail_cleanup(_):
        raise OSError("cleanup failed")

    monkeypatch.setattr(control, "cleanup_before_training", fail_cleanup)
    monkeypatch.setattr(control.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Training started before cleanup"))
    with pytest.raises(OSError, match="cleanup failed"):
        control.supervise(plan)
    assert not (Path(plan["output_root"]) / "training_started.json").exists()
