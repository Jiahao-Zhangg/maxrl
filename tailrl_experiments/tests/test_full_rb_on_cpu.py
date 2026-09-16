"""Check the full four-GPU configuration and checkpoint-gated launch on CPU."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tailrl_experiments import queue_full_rb as queued
from tailrl_experiments.prepare_text_maze_full import grid_digest
from tailrl_experiments.run_text_maze_rb_full import (
    CHECKPOINT_ORDER,
    NAME,
    final_checkpoint_saved,
    identity,
    is_complete,
    mark_complete,
    overrides,
    run_name,
)


class FakeSlurm:
    def __init__(self, config):
        self.config = config
        self.state = "RUNNING"
        self.active = set()
        self.busy = False
        self.cancelled = []

    def job(self, job_id):
        return {
            "JobId": job_id,
            "JobState": self.state,
            "JobName": self.config["holder_name"],
            "Account": self.config["account"],
            "Partition": self.config["partition"],
            "Command": self.config["holder_command"],
            "UserId": "tester(1)",
            "ReqTRES": "cpu=288,gres/gpu=4",
            "NodeList": "gh087",
        }

    def steps(self, job_id):
        return set(self.active)

    def idle(self, job):
        return not self.busy and not self.active

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.state = "COMPLETING"


@pytest.fixture
def queue(tmp_path):
    control = tmp_path / "queue"
    control.mkdir()
    ready = tmp_path / "CPU_READY.json"
    ready.write_text("{}")
    source = tmp_path / "er_state.json"
    source.write_text(json.dumps({"cancel_source_holder": False}))
    runner = tmp_path / "runner.sh"
    runner.touch()
    config = {
        "job_id": "3114675",
        "world_size": 4,
        "owner": "tester",
        "account": "account",
        "partition": "ghx4",
        "holder_name": "gpu-holder-48h",
        "holder_command": "/hold.sh",
        "runner": str(runner),
        "python": __import__("sys").executable,
        "cpu_ready": str(ready),
        "prerequisite_state": str(source),
        "prerequisite_repo": "owner/checkpoint-step_60",
        "holder_lock_dir": str(control),
        "repo_root": str(tmp_path),
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_order": list(CHECKPOINT_ORDER),
    }
    path = control / "config.json"
    path.write_text(json.dumps(config))
    return queued.Queue(path, slurm=FakeSlurm(config))


def finish_er(queue):
    Path(queue.config["prerequisite_state"]).write_text(
        json.dumps(
            {
                "cancel_source_holder": False,
                "source_training_stopped": True,
                "source_holder_retained": True,
                "receipt": {"checkpoint": {"step": 60, "repo": queue.config["prerequisite_repo"]}},
            }
        )
    )


def finish_arm(queue):
    queue.run_dir.mkdir(parents=True, exist_ok=True)
    (queue.run_dir / "COMPLETE.json").write_text(json.dumps({"step": 5001, **identity(queue.state["checkpoint_step"])}))


def test_gpu_idle_before_er_archive_does_not_launch(queue):
    queue.launch = Mock()
    assert queue.tick()
    queue.launch.assert_not_called()
    assert queue.state["phase"] == "WAITING_ER_STEP60"


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_training_stopped", False),
        ("source_holder_retained", False),
        ("cancel_source_holder", True),
    ],
)
def test_all_er_handoff_conditions_are_required(queue, key, value):
    finish_er(queue)
    path = Path(queue.config["prerequisite_state"])
    source = json.loads(path.read_text())
    source[key] = value
    path.write_text(json.dumps(source))
    assert not queue.ready()


def test_er_archive_with_busy_gpu_does_not_launch(queue):
    finish_er(queue)
    queue.slurm.busy = True
    queue.launch = Mock()
    assert queue.tick()
    queue.launch.assert_not_called()
    assert queue.state["phase"] == "WAITING_FOUR_IDLE_GPUS"


def test_launch_is_checkpoint_gated_and_persisted_before_srun(queue, monkeypatch):
    finish_er(queue)
    calls = []

    def spawn(command, **kwargs):
        assert json.loads(queue.path.read_text())["launch_requested"]
        calls.append(command)
        return SimpleNamespace(pid=123, poll=lambda: None)

    monkeypatch.setattr(queued.subprocess, "Popen", spawn)
    monkeypatch.setattr(queued, "process_start", lambda pid: "1234")
    assert queue.tick()
    assert queue.tick()
    assert len(calls) == 1
    assert "--gpus-per-task=4" in calls[0]
    assert "--network=no_vni" in calls[0]
    assert "--jobid=3114675" in calls[0]
    assert calls[0][-1] == "3000"
    restarted = queued.Queue(queue.plan_path, slurm=queue.slurm)
    restarted.launch = Mock()
    assert restarted.tick()
    restarted.launch.assert_not_called()
    queue.holder_lock.close()


def test_failed_attempt_keeps_retrying_without_cancelling_holder(queue):
    finish_er(queue)
    queue.state["launch_requested"] = True
    queue.child = SimpleNamespace(poll=lambda: 1)
    assert queue.tick()
    assert not queue.state["launch_requested"]
    assert queue.state["phase"] == "RETRYING"
    assert queue.slurm.state == "RUNNING"


def test_queue_stop_does_not_stop_existing_training(queue):
    queue.path.with_name("STOP").touch()
    assert not queue.tick()
    assert queue.slurm.state == "RUNNING"


def test_holder_timeout_records_need_to_continue_without_false_completion(queue):
    queue.slurm.state = "TIMEOUT"
    assert not queue.tick()
    assert queue.state["phase"] == "ALLOCATION_ENDED"


def test_arms_run_in_requested_order_with_one_active_at_a_time(queue, monkeypatch):
    finish_er(queue)
    commands = []
    child = SimpleNamespace(pid=123, poll=lambda: None)

    def spawn(command, **kwargs):
        commands.append(command)
        return child

    monkeypatch.setattr(queued.subprocess, "Popen", spawn)
    monkeypatch.setattr(queued, "process_start", lambda pid: "1234")
    for index, checkpoint in enumerate(CHECKPOINT_ORDER):
        assert queue.state["checkpoint_step"] == checkpoint
        assert queue.tick()
        assert len(commands) == index + 1
        assert commands[-1][-1] == str(checkpoint)
        assert f"--job-name=tailrl-rb-{checkpoint}" in commands[-1]
        finish_arm(queue)
        assert queue.tick()  # Completion marker alone cannot overlap the previous process.
        assert queue.state["checkpoint_step"] == checkpoint
        child.poll = lambda: 0
        queue.slurm.active = {"3114675.9"}
        assert queue.tick()  # Slurm must also finish cleaning the step.
        assert queue.state["phase"] == "WAITING_ARM_SHUTDOWN"
        queue.slurm.active.clear()
        assert queue.tick() == (index < len(CHECKPOINT_ORDER) - 1)
        assert queue.state["completed_checkpoints"] == list(CHECKPOINT_ORDER[: index + 1])
        assert queue.holder_lock is None
        child.poll = lambda: None
    assert queue.state["phase"] == "COMPLETE"
    assert queue.slurm.state == "RUNNING"
    assert not queue.tick()
    assert [int(command[-1]) for command in commands] == [3000, 2450, 3250, 3350, 3400, 3450, 3550]


def test_restart_after_completion_continues_next_arm_with_separate_progress(queue):
    finish_er(queue)
    finish_arm(queue)
    (queue.run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 5001, "metrics": {"training/global_step": 5001}}) + "\n"
    )
    queue.state["completed_step"] = 5001
    assert queue.tick()  # Recover completion even if the controller lost its launch flag.
    restarted = queued.Queue(queue.plan_path, slurm=queue.slurm)
    assert restarted.state["checkpoint_step"] == 2450
    assert restarted.state["completed_checkpoints"] == [3000]
    assert restarted.state["completed_step"] == 0
    assert restarted.progress() == 0
    assert restarted.run_dir.name == run_name(2450)
    restarted.state["launch_requested"] = True
    restarted.child = SimpleNamespace(poll=lambda: 1)
    assert restarted.tick()
    assert restarted.state["phase"] == "RETRYING"
    assert restarted.state["checkpoint_step"] == 2450
    assert restarted.state["completed_checkpoints"] == [3000]


@pytest.mark.parametrize("field,value", [("step", 250), ("checkpoint_step", 2450), ("seed", 1), ("smoke", True)])
def test_wrong_or_incomplete_completion_record_cannot_advance(queue, field, value):
    finish_arm(queue)
    path = queue.run_dir / "COMPLETE.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="does not match"):
        queue.tick()
    assert queue.state["checkpoint_step"] == 3000
    assert queue.state["completed_checkpoints"] == []


def test_expired_deadline_prevents_next_arm_even_before_slurm_updates(queue):
    finish_er(queue)
    finish_arm(queue)
    assert queue.tick()
    queue.config["allocation_end_utc"] = "2000-01-01T00:00:00+00:00"
    queue.launch = Mock()
    assert not queue.tick()
    queue.launch.assert_not_called()
    assert queue.state["phase"] == "ALLOCATION_ENDED"
    assert queue.state["checkpoint_step"] == 2450
    assert queue.state["completed_checkpoints"] == [3000]
    assert queue.slurm.state == "RUNNING"


def test_holder_timeout_during_attempt_does_not_enter_retry_loop(queue):
    queue.state["launch_requested"] = True
    queue.slurm.state = "TIMEOUT"
    queue.run_dir.mkdir(parents=True)
    (queue.run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 900, "metrics": {"training/global_step": 900}}) + "\n"
    )
    assert not queue.tick()
    assert queue.state["phase"] == "ALLOCATION_ENDED"
    assert queue.state["completed_step"] == 900
    assert queue.state["completed_checkpoints"] == []


def prepare_last_arm_for_release(queue):
    queue.order = [3000, 2450, 3250]
    queue.config.update(checkpoint_order=queue.order, cancel_holder_after_completion=True)
    queue.plan_path.write_text(json.dumps(queue.config))
    queue.state.update(
        config_sha256=hashlib.sha256(json.dumps(queue.config, sort_keys=True).encode()).hexdigest(),
        completed_checkpoints=[3000, 2450],
        checkpoint_step=3250,
        launch_requested=True,
    )
    finish_arm(queue)
    export = queue.run_dir / "checkpoints/global_step_5001/actor/huggingface/config.json"
    export.parent.mkdir(parents=True)
    export.write_text("{}")
    (queue.run_dir / "checkpoints/latest_checkpointed_iteration.txt").write_text("5001")


def test_cancel_waits_for_final_save_and_step_cleanup_then_confirms_release(queue):
    prepare_last_arm_for_release(queue)
    queue.slurm.active.add("3114675.9")
    assert queue.tick()
    assert queue.state["phase"] == "WAITING_ARM_SHUTDOWN"
    assert queue.slurm.cancelled == []
    queue.slurm.active.clear()
    assert queue.tick()
    assert queue.state["phase"] == "READY_TO_RELEASE_HOLDER"
    assert queue.state["completed_checkpoints"] == [3000, 2450, 3250]
    assert queue.slurm.cancelled == []
    assert queue.tick()
    assert queue.state["phase"] == "WAITING_HOLDER_RELEASE"
    assert queue.slurm.cancelled == ["3114675"]
    assert queue.tick()  # COMPLETING is not yet a released allocation.
    assert queue.slurm.cancelled == ["3114675"]
    queue.slurm.state = "CANCELLED"
    assert not queue.tick()
    assert queue.state["phase"] == "COMPLETE_HOLDER_RELEASED"


@pytest.mark.parametrize("missing", ["marker", "final_pointer", "hf_export"])
def test_final_checkpoint_is_rechecked_before_holder_cancel(queue, missing):
    prepare_last_arm_for_release(queue)
    final_dir = queue.run_dir
    assert queue.tick()
    if missing == "marker":
        (final_dir / "COMPLETE.json").unlink()
    elif missing == "final_pointer":
        (final_dir / "checkpoints/latest_checkpointed_iteration.txt").write_text("5000")
    else:
        (final_dir / "checkpoints/global_step_5001/actor/huggingface/config.json").unlink()
    with pytest.raises(RuntimeError):
        queue.tick()
    assert queue.slurm.cancelled == []


@pytest.mark.parametrize("busy", ["slurm_step", "gpu_process", "holder_lock"])
def test_holder_cancel_waits_if_other_work_occupies_holder(queue, busy):
    prepare_last_arm_for_release(queue)
    assert queue.tick()
    lock_path = Path(queue.config["holder_lock_dir"]) / "gpu_holder_3114675.launch.lock"
    with lock_path.open("a") as lock:
        if busy == "slurm_step":
            queue.slurm.active.add("3114675.99")
        elif busy == "gpu_process":
            queue.slurm.busy = True
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert queue.tick()
        assert queue.slurm.cancelled == []
        assert queue.state["phase"] == "WAITING_HOLDER_IDLE"


def test_holder_identity_is_checked_before_cancel(queue):
    prepare_last_arm_for_release(queue)
    assert queue.tick()
    real_job = queue.slurm.job
    queue.slurm.job = lambda job_id: {**real_job(job_id), "JobName": "another-experiment"}
    with pytest.raises(queued.SafetyError, match="authorized holder"):
        queue.tick()
    assert queue.slurm.cancelled == []


def test_cancel_retries_after_rpc_failure_and_controller_restart(queue):
    prepare_last_arm_for_release(queue)
    assert queue.tick()
    cancel = queue.slurm.cancel

    def fail_cancel(job_id):
        assert json.loads(queue.path.read_text())["holder_cancel_requested"]
        raise RuntimeError("Slurm RPC temporarily unavailable")

    queue.slurm.cancel = fail_cancel
    with pytest.raises(RuntimeError, match="RPC temporarily"):
        queue.tick()
    assert queue.slurm.state == "RUNNING"
    queue.slurm.cancel = cancel
    restarted = queued.Queue(queue.plan_path, slurm=queue.slurm)
    assert restarted.tick()
    assert queue.slurm.cancelled == ["3114675"]
    queue.slurm.state = "CANCELLED"
    assert not restarted.tick()
    assert restarted.state["phase"] == "COMPLETE_HOLDER_RELEASED"


def test_cancellation_enabled_does_not_cancel_unfinished_training(queue):
    queue.config["cancel_holder_after_completion"] = True
    queue.state["launch_requested"] = True
    queue.child = SimpleNamespace(poll=lambda: None)
    assert queue.tick()
    assert queue.state["phase"] == "GPU_VALIDATION_OR_STARTUP"
    assert queue.slurm.cancelled == []


def test_full_update_is_required_for_training_status(queue):
    queue.state["launch_requested"] = True
    queue.child = SimpleNamespace(poll=lambda: None)
    metrics = Path(queue.config["output_dir"]) / NAME / "metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"step": 0, "metrics": {"val/accuracy": 0.1}}) + "\n")
    assert queue.tick()
    assert queue.state["phase"] == "GPU_VALIDATION_OR_STARTUP"
    with metrics.open("a") as stream:
        stream.write(json.dumps({"step": 1, "metrics": {"training/global_step": 1}}) + "\n")
    assert queue.tick()
    assert queue.state["phase"] == "TRAINING"
    assert queue.state["completed_step"] == 1


def test_full_configuration_has_one_update_for_256_prompts_and_no_pilot_limits(tmp_path):
    values = dict(value.lstrip("+").split("=", 1) for value in overrides(tmp_path, tmp_path, tmp_path))
    assert values["data.train_batch_size"] == "256"
    assert values["actor_rollout_ref.rollout.n"] == "16"
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == "256"
    assert values["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "1024"
    assert 256 * 16 == 1024 * 4
    assert values["trainer.total_training_steps"] == "5001"
    assert values["trainer.save_freq"] == "250"
    assert values["trainer.test_freq"] == "1000"
    assert values["actor_rollout_ref.rollout.val_kwargs.n"] == "64"
    assert "ckpt-3000" in values["actor_rollout_ref.model.path"]
    assert values["data.seed"] == "0"
    assert values["data.train_files"].endswith("full_train.parquet")
    assert "actor_rollout_ref.model.attn_implementation" not in values
    assert "actor_rollout_ref.model.enable_gradient_checkpointing" not in values
    assert "actor_rollout_ref.actor.use_torch_compile" not in values


def test_every_sft_arm_keeps_the_same_full_training_configuration(tmp_path):
    common = None
    for checkpoint in CHECKPOINT_ORDER:
        run_dir = tmp_path / run_name(checkpoint)
        values = dict(
            value.lstrip("+").split("=", 1)
            for value in overrides(tmp_path, tmp_path, run_dir, checkpoint_step=checkpoint)
        )
        assert values.pop("actor_rollout_ref.model.path") == str(tmp_path / f"checkpoints/ckpt-{checkpoint}")
        assert values.pop("trainer.experiment_name") == run_dir.name
        assert values.pop("trainer.default_local_dir") == str(run_dir / "checkpoints")
        assert values.pop("trainer.validation_data_dir") == str(run_dir / "validation")
        if common is None:
            common = values
        else:
            assert values == common


def test_final_save_recovers_completion_after_driver_exit(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    export = checkpoints / "global_step_5001/actor/huggingface/config.json"
    export.parent.mkdir(parents=True)
    export.write_text("{}")
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("5001")
    assert not is_complete(tmp_path, 2450)
    assert final_checkpoint_saved(tmp_path)
    mark_complete(tmp_path, 2450)
    assert is_complete(tmp_path, 2450)


def test_partial_save_cannot_be_marked_complete(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    latest = checkpoints / "latest_checkpointed_iteration.txt"
    latest.write_text("5000")
    assert not final_checkpoint_saved(tmp_path)
    with pytest.raises(RuntimeError, match="before saving"):
        mark_complete(tmp_path, 3000)
    latest.write_text("5001")
    with pytest.raises(RuntimeError, match="export is missing"):
        final_checkpoint_saved(tmp_path)
    assert not (tmp_path / "COMPLETE.json").exists()


def test_grid_identity_does_not_depend_on_target_path():
    assert grid_digest("<bos> GRID_START START PATH GOAL GRID_END PATH_START RIGHT RIGHT DONE") == grid_digest(
        "GRID_START START PATH GOAL GRID_END PATH_START DOWN RIGHT UP RIGHT DONE"
    )


def test_four_rank_weighting_matches_single_gpu_token_mean_gradient():
    import torch

    coefficient = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4)
    masks = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float64)
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    global_loss = (parameter.square() * coefficient * masks).sum() / masks.sum()
    expected = torch.autograd.grad(global_loss, parameter)[0]
    gradients = []
    for rank in range(4):
        weight = parameter.detach().clone().requires_grad_(True)
        local_mean = (weight.square() * coefficient[rank] * masks[rank]).sum() / masks[rank].sum()
        loss = local_mean * 4 * masks[rank].sum() / masks.sum()
        gradients.append(torch.autograd.grad(loss, weight)[0])
    torch.testing.assert_close(torch.stack(gradients).mean(), expected)


def test_global_token_count_survives_rb_dispatch_to_four_shards():
    import numpy as np
    import torch

    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    batch = DataProto.from_dict(
        tensors={"response_mask": mask, "token_level_rewards": torch.tensor([[1.0, 0.0, 0.0]] * 4)},
        non_tensors={
            "uid": np.array(["a", "a", "b", "b"]),
            "trajectory_cost": np.array([1.0, 2.0, 3.0, 2.0]),
            "generated_action_length": np.array([1.0, 2.0, 3.0, 2.0]),
            "shortest_distance": np.ones(4),
        },
    )
    result = compute_advantage(batch, "fixed_n_rb_cost_aware_marginrl", num_repeat=2)
    assert result.meta_info["rb_global_response_tokens"] == 8
    assert all(part.meta_info["rb_global_response_tokens"] == 8 for part in result.chunk(4))


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("native", ["native"]),
        ("collective", ["native", "collective"]),
        ("smoke", ["native", "collective", "smoke"]),
        ("", ["native", "collective", "smoke", "full"]),
    ],
)
def test_node_stops_before_training_if_any_gpu_validation_stage_fails(tmp_path, failure, expected):
    tools = tmp_path / "bin"
    tools.mkdir()
    profile = tmp_path / "conda.sh"
    profile.write_text('conda() { return 0; }\nexport PATH="${TEST_BIN}:${PATH}"\n')
    scripts = {
        "nvidia-smi": "#!/bin/bash\nexit 0\n",
        "timeout": '#!/bin/bash\nshift 2\nexec "$@"\n',
        "python": """#!/bin/bash
if [[ "$1" == "-" ]]; then exec "$TEST_PYTHON" "$@"; fi
case "$*" in
  *check_text_maze_gpu.py*--collective*) stage=collective ;;
  *check_text_maze_gpu.py*) stage=native ;;
  *--smoke*) stage=smoke ;;
  *) stage=full ;;
esac
echo "$stage" >> "$TEST_EVENTS"
echo "$*" >> "$TEST_COMMANDS"
[[ "$stage" != "$TEST_FAIL" ]]
""",
    }
    for name, content in scripts.items():
        path = tools / name
        path.write_text(content)
        path.chmod(0o700)
    ready = tmp_path / "CPU_READY.json"
    ready.write_text("{}")
    plan = tmp_path / "config.json"
    plan.write_text(
        json.dumps(
            {
                "experiment": str(tmp_path),
                "state_dir": str(tmp_path),
                "output_dir": str(tmp_path),
                "cpu_ready": str(ready),
            }
        )
    )
    events = tmp_path / "events.txt"
    env = dict(
        os.environ,
        SLURM_JOB_ID="11",
        CONDA_PROFILE=str(profile),
        TEST_BIN=str(tools),
        TEST_PYTHON=sys.executable,
        TEST_EVENTS=str(events),
        TEST_COMMANDS=str(tmp_path / "commands.txt"),
        TEST_FAIL=failure,
        TAILRL_RAY_TMPDIR=str(tmp_path / "ray"),
    )
    runner = Path(__file__).resolve().parents[1] / "run_full_rb_on_slurm.sh"
    result = subprocess.run(["/usr/bin/bash", str(runner), "11", str(plan), "3000"], env=env, capture_output=True)
    assert events.read_text().splitlines() == expected, result.stderr.decode()
    assert (result.returncode == 0) == (failure == "")
    assert (tmp_path / "gpu_validation_passed").exists() == (failure == "")
    if not failure:
        result = subprocess.run(["/usr/bin/bash", str(runner), "11", str(plan), "2450"], env=env, capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
        assert events.read_text().splitlines() == expected + ["full"]
        assert "--ckpt-step 2450" in (tmp_path / "commands.txt").read_text().splitlines()[-1]
