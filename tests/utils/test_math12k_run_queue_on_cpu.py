import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments import queue_math12k_after_run as queue_module


@pytest.fixture
def queued(tmp_path, monkeypatch):
    checkpoints = tmp_path / "predecessor/checkpoints"
    logs = checkpoints / "logs"
    logs.mkdir(parents=True)
    (logs / "training.exit_status").write_text("0\n")
    (logs / "checkpoint_upload.log").write_text("Checkpoint archival is complete\n")
    dataset = tmp_path / "predecessor/dataset"
    (dataset / "data").mkdir(parents=True)
    for name in ["README.md", "rollout_manifest.json", "data/step_000001.jsonl.gz", "data/step_000002.jsonl.gz"]:
        (dataset / name).write_bytes(b"verified rollout data")
    rollout_state = tmp_path / "predecessor/rollout_state.json"
    rollout_state.write_text(json.dumps({
        "status": "complete", "uploaded_steps": [1, 2],
        "identity": {"repo_id": "owner/previous-rollouts", "dataset_dir": str(dataset), "final_step": 2, "expected_rows": 4},
    }))
    plan = {
        "predecessor": {"pid": 12345, "start_time": "67890", "checkpoint_dir": str(checkpoints),
                        "checkpoint_hf_prefix": "owner/previous", "checkpoint_steps": [2],
                        "num_gpus": 2, "rows_per_step": 4, "final_step": 2, "rollout_upload_state": str(rollout_state)},
        "gpu_indices": [0, 1, 2, 3], "output_root": str(tmp_path / "next-run"),
        "python_bin": "/environment/bin/python", "repo_root": str(tmp_path), "launcher": "/repo/run.sh",
        "ray_dir": str(tmp_path / "ray"), "tmp_dir": str(tmp_path / "tmp"), "data_dir": "/prepared/data",
        "model_path": "Qwen/Qwen3-1.7B-Base", "cost_offset_tokens": 256, "total_steps": 150, "save_freq": 50,
        "experiment_name": "f_cov_test", "checkpoint_hf_prefix": "owner/f-cov", "rollout_hf_repo": "owner/f-cov-rollouts",
        "wandb_run_id": "next-run-id", "wandb_entity": "entity", "min_output_free_gib": 0,
    }
    monkeypatch.setattr(queue_module, "process_matches", lambda *args: False)
    monkeypatch.setattr(queue_module, "gpu_blockers", lambda indices: [])
    return queue_module.RunQueue(plan, tmp_path / "queue/state.json", api=None, inherited={"PATH": "/bin"})


def test_live_predecessor_is_never_interrupted_or_replaced(queued, monkeypatch):
    monkeypatch.setattr(queue_module, "process_matches", lambda *args: True)
    monkeypatch.setattr(queue_module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launched too early"))
    assert queued.tick()
    assert queued.state["status"] == "waiting_for_predecessor"
    assert not Path(queued.plan["output_root"]).exists()


@pytest.mark.parametrize("gate,status", [
    ("failure", "predecessor_failed"),
    ("checkpoint", "waiting_for_checkpoint_uploads"),
    ("cleanup", "waiting_for_checkpoint_cleanup"),
    ("rollouts", "waiting_for_rollout_uploads"),
])
def test_waits_for_successful_archival_and_rollout_upload(queued, gate, status):
    predecessor = queued.plan["predecessor"]
    checkpoints = Path(predecessor["checkpoint_dir"])
    if gate == "failure":
        (checkpoints / "logs/training.exit_status").write_text("1")
    elif gate == "checkpoint":
        (checkpoints / "logs/checkpoint_upload.log").write_text("still uploading")
    elif gate == "cleanup":
        (checkpoints / "global_step_2").mkdir()
    else:
        Path(predecessor["rollout_upload_state"]).write_text(json.dumps({"status": "watching", "uploaded_steps": [1]}))
    queued.tick()
    assert queued.state["status"] == status
    assert queued.child is None


def test_waits_for_selected_gpus_without_stopping_other_processes(queued, monkeypatch):
    monkeypatch.setattr(queue_module, "gpu_blockers", lambda indices: ["GPU-0, another-pid"])
    assert queued.tick()
    assert queued.state["status"] == "waiting_for_gpus"
    assert queued.child is None


def test_gpu_blockers_only_include_requested_devices(monkeypatch):
    def query(args, **kwargs):
        if "--query-gpu=index,uuid" in args:
            return "0, first\n1, second\n4, other\n"
        return "other, 123\nsecond, 456\n"
    monkeypatch.setattr(queue_module.subprocess, "check_output", query)
    assert queue_module.gpu_blockers([0, 1]) == ["second, 456"]


def test_launches_once_after_two_idle_checks_and_upload_verification(queued, monkeypatch):
    calls = []
    verified = []
    monkeypatch.setattr(queue_module, "verify_predecessor_uploads", lambda *args: verified.append(True))
    monkeypatch.setattr(queued, "preflight", lambda: None)
    def launch(command, **kwargs):
        assert json.loads(queued.state_file.read_text())["phase"] == "launching"
        calls.append((command, kwargs))
        return SimpleNamespace(pid=23456)
    monkeypatch.setattr(queue_module.subprocess, "Popen", launch)
    assert queued.tick()
    assert queued.state["status"] == "confirming_idle"
    assert not queued.tick()
    assert not queued.tick()
    assert len(calls) == len(verified) == 1
    command, kwargs = calls[0]
    assert command[-1] == "trainer.resume_mode=disable"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert kwargs["env"]["MAXRL_SAVE_ROLLOUT_DATASET"] == "1"
    assert kwargs["env"]["MAXRL_UPLOAD_CHECKPOINTS"] == "1"
    assert kwargs["env"]["MAXRL_TOTAL_TRAINING_STEPS"] == "150"
    resumed = queue_module.RunQueue(queued.plan, queued.state_file, api=None)
    assert not resumed.tick()


def test_rechecks_gpus_after_network_preflight(queued, monkeypatch):
    queued.idle_confirmations = 1
    calls = iter([[], ["new-occupant"]])
    monkeypatch.setattr(queue_module, "gpu_blockers", lambda indices: next(calls))
    monkeypatch.setattr(queue_module, "verify_predecessor_uploads", lambda *args: None)
    monkeypatch.setattr(queued, "preflight", lambda: None)
    assert queued.tick()
    assert queued.state["status"] == "waiting_for_gpus"
    assert queued.child is None


def test_launch_environment_drops_previous_algorithm_and_wandb_identity(queued):
    env = queue_module.launch_environment(queued.plan, {
        "PATH": "/bin", "MAXRL_ADVANTAGE_ESTIMATOR": "old", "MAXRL_MODEL_PATH": "old-checkpoint",
        "MAXRL_COST_OFFSET_TOKENS": "999", "WANDB_RUN_ID": "old", "WANDB_API_KEY": "keep-credential",
        "RAY_ADDRESS": "old-cluster", "PYTHONPATH": "old-code",
    })
    assert "MAXRL_ADVANTAGE_ESTIMATOR" not in env
    assert env["MAXRL_MODEL_PATH"] == "Qwen/Qwen3-1.7B-Base"
    assert env["MAXRL_COST_OFFSET_TOKENS"] == "256"
    assert env["WANDB_RUN_ID"] == "next-run-id"
    assert env["WANDB_API_KEY"] == "keep-credential"
    assert env["RAY_ADDRESS"] == "local"
    assert "PYTHONPATH" not in env


def test_remote_verification_checks_shards_and_rollout_sizes(queued):
    predecessor = queued.plan["predecessor"]
    dataset = Path(json.loads(Path(predecessor["rollout_upload_state"]).read_text())["identity"]["dataset_dir"])
    class Hub:
        corrupt = False
        def repo_info(self, repo_type, **kwargs):
            if repo_type == "model":
                names = ["global_step_2/data.pt"] + [
                    f"global_step_2/actor/{kind}_world_size_2_rank_{rank}.pt"
                    for kind in ("model", "optim", "extra_state") for rank in range(2)
                ]
                return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name, size=10) for name in names])
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=p.relative_to(dataset).as_posix(), size=p.stat().st_size + int(self.corrupt))
                for p in dataset.rglob("*") if p.is_file()
            ])
    api = Hub()
    queue_module.verify_predecessor_uploads(queued.plan, api)
    api.corrupt = True
    with pytest.raises(RuntimeError, match="rollout upload is not verified"):
        queue_module.verify_predecessor_uploads(queued.plan, api)
