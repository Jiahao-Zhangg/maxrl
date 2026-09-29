import errno
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import grpo_coding_release as release


def test_checkpoint_requires_completion_marker_and_all_rank_states(tmp_path):
    plan = {"checkpoint_dir": str(tmp_path)}
    checkpoint = tmp_path / "global_step_10"
    for relative in release.required_checkpoint_files():
        target = checkpoint / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data")
    marker = tmp_path / "latest_checkpointed_iteration.txt"
    assert not release.checkpoint_complete(plan, 10)
    marker.write_text("9")
    assert not release.checkpoint_complete(plan, 10)
    marker.write_text("10")
    assert release.checkpoint_complete(plan, 10)
    (checkpoint / "actor/optim_world_size_8_rank_7.pt").unlink()
    assert not release.checkpoint_complete(plan, 10)


def test_control_records_survive_home_disk_full(tmp_path, monkeypatch):
    scratch, home = tmp_path / "scratch", tmp_path / "home"
    scratch.mkdir()
    home.mkdir()
    plan = {"scratch": str(scratch), "output_root": str(home)}
    original = release.write

    def disk_full(path, value):
        if path.is_relative_to(home):
            raise OSError(errno.ENOSPC, "Disk full")
        original(path, value)

    monkeypatch.setattr(release, "write", disk_full)
    assert release.persist(plan, "status.json", {"state": "training"})
    assert release.state(plan, "status.json")["state"] == "training"
    assert json.loads((scratch / "control_mirrors/status.json").read_text())["state"] == "training"


def test_successor_waits_for_four_verified_evaluations_and_rollout_archives(tmp_path):
    root, scratch = tmp_path / "predecessor", tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    predecessor = {"output_root": str(root), "scratch": str(scratch), "job_id": "123", "node": "compute"}
    release.write(root / "plan.json", predecessor)
    sha = release.digest(root / "plan.json")
    successor = {"job_id": "123", "node": "compute", "predecessor": {"root": str(root), "plan_sha256": sha}}
    assert not release.predecessor_complete(successor)
    release.persist(predecessor, "training_exit.json", {"exit_code": 0})
    release.persist(predecessor, "queue_status.json", {"state": "complete"})
    assert not release.predecessor_complete(successor)
    release.persist(predecessor, "rollout_status.json", {"state": "complete"})
    for name, count in release.evaluation.COUNTS.items():
        assert not release.predecessor_complete(successor)
        release.persist(predecessor, f"evaluation/{name}/audit.json", {
            "complete": True, "questions": count, "plan_sha256": sha,
        })
    assert release.predecessor_complete(successor)
    first = next(iter(release.evaluation.COUNTS))
    release.persist(predecessor, f"evaluation/{first}/audit.json", {
        "complete": True, "questions": release.evaluation.COUNTS[first], "plan_sha256": "wrong-run",
    })
    assert not release.predecessor_complete(successor)
    release.write(root / "plan.json", {**predecessor, "node": "other-compute"})
    with pytest.raises(ValueError, match="identity changed"):
        release.predecessor_complete(successor)


def test_model_cleanup_refuses_changed_files_and_can_resume_partial_deletion(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model.safetensors"
    weight.write_bytes(b"verified weights")
    hashes = {weight.name: release.digest(weight), "already-deleted": "irrelevant"}
    weight.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="changed; retaining"):
        release.delete_verified_model_tree(model, hashes)
    assert weight.exists()
    weight.write_bytes(b"verified weights")
    release.delete_verified_model_tree(model, hashes)
    assert not model.exists()
    release.delete_verified_model_tree(model, hashes)


def test_model_cleanup_refuses_symlinks(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    other = tmp_path / "other-run.bin"
    other.write_bytes(b"in use")
    (model / "weights").symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        release.delete_verified_model_tree(model, {"weights": release.digest(other)})
    assert other.read_bytes() == b"in use"


def continuation_plans(tmp_path):
    predecessor = {
        "output_root": str(tmp_path / "first"), "scratch": str(tmp_path / "first_scratch"),
        "hf_repo_prefix": "owner/grpo-first", "total_steps": 100,
        "dataset_sha256": "same-data", "dataset_rows": 3200, "base_model_hashes": {"model": "same-model"},
        "grading": {"name": "livecodebench", "check_eos": False},
        "training": {"batch_size": 32, "n": 8, "max_num_seqs": 16, "temperature": 1.0,
                     "top_p": 1.0, "top_k": -1, "shuffle": True, "seed": 42, "epochs": 1},
    }
    release.write(Path(predecessor["output_root"]) / "plan.json", predecessor)
    plan = deepcopy(predecessor)
    plan.update(output_root=str(tmp_path / "second"), scratch=str(tmp_path / "second_scratch"),
                hf_repo_prefix="owner/grpo-second", predecessor={
                    "root": predecessor["output_root"],
                    "plan_sha256": release.digest(Path(predecessor["output_root"]) / "plan.json"),
                })
    plan["training"]["algorithm"] = "grpo"
    return plan, predecessor


def test_continuation_runs_only_epoch_two_and_retains_training_settings(tmp_path):
    plan, predecessor = continuation_plans(tmp_path)
    release.configure_continuation(plan, predecessor)
    assert list(release.training_steps(plan)) == list(range(101, 201))
    assert plan["checkpoint_steps"] == list(range(110, 201, 10))
    assert plan["training"]["epochs"] == 2
    assert plan["training"]["epochs_this_stage"] == 1
    assert plan["continuation"]["restore_contents"] == ["model", "optimizer", "extra", "dataloader"]
    assert plan["continuation"]["checkpoint_repo"] == "owner/grpo-first-step_100"
    assert list(release.training_steps({"total_steps": 100})) == list(range(1, 101))


@pytest.mark.parametrize("changed", ["dataset", "n", "algorithm", "incomplete_epoch"])
def test_continuation_rejects_incompatible_training(tmp_path, changed):
    plan, predecessor = continuation_plans(tmp_path)
    if changed == "dataset":
        plan["dataset_sha256"] = "different-data"
    elif changed == "n":
        plan["training"]["n"] = 16
    elif changed == "algorithm":
        plan["training"]["algorithm"] = "maxrl"
    else:
        predecessor["total_steps"] = 90
    with pytest.raises(ValueError):
        release.configure_continuation(plan, predecessor)


def test_continuation_restores_full_checkpoint_into_own_scratch_then_prefers_own_recovery(tmp_path, monkeypatch):
    plan, predecessor = continuation_plans(tmp_path)
    release.configure_continuation(plan, predecessor)
    parent_receipt = {"repo_id": plan["continuation"]["checkpoint_repo"], "remote_commit": "pinned-commit"}
    own_steps = []

    def receipt(source, step):
        if source["output_root"] == predecessor["output_root"] and step == 100:
            return parent_receipt
        return {"verified": True} if step in own_steps else None

    restored = []

    def restore(source, step, **kwargs):
        restored.append((source["output_root"], step, kwargs))
        return Path(plan["scratch"]) / "resume_source" / f"global_step_{step}"

    monkeypatch.setattr(release, "archive_receipt", receipt)
    monkeypatch.setattr(release, "restore_checkpoint", restore)
    step, path = release.training_resume(plan)
    assert step == 100 and str(path).startswith(plan["scratch"])
    assert restored == [(predecessor["output_root"], 100, {"destination_scratch": plan["scratch"]})]
    assert release.state(plan, "continuation_checkpoint.json")["receipt"] == parent_receipt
    own_steps.extend([110, 120])
    assert release.training_resume(plan)[0] == 120
    assert restored[-1] == (plan["output_root"], 120, {})


def test_continuation_never_falls_back_to_base_weights_when_checkpoint_is_missing(tmp_path, monkeypatch):
    plan, predecessor = continuation_plans(tmp_path)
    release.configure_continuation(plan, predecessor)
    monkeypatch.setattr(release, "archive_receipt", lambda *_: None)
    with pytest.raises(ValueError, match="No verified full predecessor"):
        release.training_resume(plan)
    release.write(Path(predecessor["output_root"]) / "plan.json", {**predecessor, "total_steps": 200})
    with pytest.raises(ValueError, match="identity changed"):
        release.training_resume(plan)


@pytest.mark.parametrize("stage_complete", [False, True])
def test_rollout_monitor_counts_only_continuation_steps(tmp_path, monkeypatch, stage_complete):
    import huggingface_hub

    plan, predecessor = continuation_plans(tmp_path)
    release.configure_continuation(plan, predecessor)
    plan.update(expected_rollouts_per_step=256, experiment_name="epoch-two")
    steps = range(101, 201) if stage_complete else range(1, 101)
    for step in steps:
        release.persist(plan, f"rollout_receipts/{step}.json", {"state": "archived_and_deleted"})
    release.persist(plan, "training_exit.json", {"exit_code": 0})
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(upload_file=lambda **_: None))
    monkeypatch.setattr(release, "ensure_public_repository", lambda *_: None)

    def waiting(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(release.time, "sleep", waiting)
    if stage_complete:
        release.monitor_rollouts(plan)
    else:
        with pytest.raises(KeyboardInterrupt):
            release.monitor_rollouts(plan)
    state = release.state(plan, "rollout_status.json")
    assert state["state"] == ("complete" if stage_complete else "monitoring")
    assert state["archived_steps"] == (list(range(101, 201)) if stage_complete else [])


def test_continuation_queue_evaluates_step_200_and_completes(tmp_path, monkeypatch):
    plan, predecessor = continuation_plans(tmp_path)
    release.configure_continuation(plan, predecessor)
    plan["plan_sha256"] = "epoch-two-plan"
    release.persist(plan, "training_exit.json", {"exit_code": 0})
    release.persist(plan, "rollout_status.json", {"state": "complete"})
    checked_steps, evaluated = [], []

    def receipt(_, step):
        checked_steps.append(step)
        assert step == 200
        return {"verified": True}

    def evaluate(_, action, dataset):
        assert action == "evaluate"
        evaluated.append(dataset)
        release.persist(plan, f"evaluation/{dataset}/audit.json", {
            "complete": True, "plan_sha256": "epoch-two-plan",
        })
        release.persist(plan, f"evaluation/{dataset}/metrics.json", {"questions": release.evaluation.COUNTS[dataset]})

    monkeypatch.setattr(release, "archive_receipt", receipt)
    monkeypatch.setattr(release, "wait_child", evaluate)
    release.queue(plan)
    assert checked_steps == [200]
    assert evaluated == list(release.evaluation.COUNTS)
    assert release.state(plan, "queue_status.json")["state"] == "complete"


@pytest.mark.parametrize("workers", [0, 4])
def test_v1_epoch_boundary_resume_matches_uninterrupted_second_epoch(tmp_path, monkeypatch, workers):
    import torch
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler

    from verl.trainer.ppo.v1 import trainer_base

    dataset = [{"index": i, "raw_prompt": str(i)} for i in range(3200)]

    def loader():
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(42))
        return StatefulDataLoader(dataset, batch_size=32, num_workers=workers, drop_last=True, sampler=sampler)

    def epoch(trainer):
        batches = [trainer_base.PPOTrainer._fetch_one_gen_batch(trainer) for _ in range(100)]
        return torch.cat([batch["index"] for batch in batches]).tolist()

    monkeypatch.setattr(trainer_base.tu, "get_tensordict", lambda data: data)
    original = SimpleNamespace(train_dataloader=loader(), train_dataloader_it=None)
    first_epoch = epoch(original)
    checkpoint = tmp_path / "global_step_100"
    checkpoint.mkdir()
    torch.save(original.train_dataloader.state_dict(), checkpoint / "data.pt")
    expected = epoch(original)
    actor_loads = []
    resumed = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"resume_mode": "resume_path", "resume_from_path": str(checkpoint),
                                             "del_local_ckpt_after_load": False}}),
        actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda **kwargs: actor_loads.append(kwargs)),
        use_critic=False, trainer_mode="sync", train_dataloader=loader(), train_dataloader_it=None,
    )
    trainer_base.PPOTrainer._load_checkpoint(resumed)
    actual = epoch(resumed)
    assert resumed.global_steps == 100
    assert actor_loads == [{"local_path": str(checkpoint / "actor"), "del_local_after_load": False}]
    assert actual == expected
    assert actual != first_epoch
    assert sorted(actual) == list(range(3200))
