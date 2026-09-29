import errno
import json

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
