import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen3_experiments.upload_training_rollouts_to_hf import RolloutUploader, process_matches


class FakeHub:
    def __init__(self):
        self.files = {}
        self.upload_count = 0
        self.fail_upload = False
        self.wrong_size = False

    def create_repo(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"

    def upload_file(self, **kwargs):
        self.files[kwargs["path_in_repo"]] = Path(kwargs["path_or_fileobj"]).read_bytes()

    def upload_folder(self, **kwargs):
        self.upload_count += 1
        if self.fail_upload:
            raise RuntimeError("temporary upload failure")
        root = Path(kwargs["folder_path"])
        self.files.update({p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def repo_info(self, **kwargs):
        return SimpleNamespace(siblings=[
            SimpleNamespace(rfilename=name, size=len(value) + int(self.wrong_size))
            for name, value in self.files.items()
        ])


@pytest.fixture
def upload(tmp_path, monkeypatch):
    monkeypatch.setattr("verl.utils.rollout_dataset.time.sleep", lambda _: None)
    source = tmp_path / "source"
    source.mkdir()
    api = FakeHub()
    return RolloutUploader(
        source, tmp_path / "dataset", tmp_path / "state.json", "owner/rollouts", 2, 2,
        {"experiment_name": "test", "wandb_url": "https://example.org/run"}, api,
    )


def save_batch(upload, step, count=2):
    rows = [
        {"input": f"prompt {index}", "output": f"response {index}", "step": step,
         "score": float(index % 2), "accuracy": float(index % 2)}
        for index in range(count)
    ]
    (upload.source_dir / f"{step}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_waits_for_full_batch_then_uploads_all_rows(upload):
    save_batch(upload, 1, count=1)
    assert upload.tick(True) == "waiting_for_rollouts"
    assert upload.api.upload_count == 0
    assert not (upload.dataset_dir / "data").exists()

    rows = save_batch(upload, 1)
    assert upload.tick(True) == "watching"
    uploaded = [json.loads(line) for line in gzip.decompress(upload.api.files["data/step_000001.jsonl.gz"]).splitlines()]
    assert uploaded == [{**row, "rollout_index": index} for index, row in enumerate(rows)]
    assert (upload.source_dir / "1.jsonl").exists()
    assert upload.tick(True) == "watching"
    assert upload.api.upload_count == 1
    save_batch(upload, 2)
    assert upload.tick(True) == "complete"
    manifest = json.loads(upload.api.files["rollout_manifest.json"])
    assert manifest["num_steps"] == 2
    assert manifest["num_rollouts"] == 4


def test_does_not_upload_incomplete_json_write(upload):
    (upload.source_dir / "1.jsonl").write_text('{"input": "partial')
    assert upload.tick(True) == "waiting_for_rollouts"
    assert upload.api.upload_count == 0


@pytest.mark.parametrize("failure", ["fail_upload", "wrong_size"])
def test_failed_upload_or_verification_is_retried_and_keeps_source(upload, failure):
    save_batch(upload, 1)
    setattr(upload.api, failure, True)
    with pytest.raises(RuntimeError):
        upload.tick(True)
    assert upload.state["uploaded_steps"] == []
    assert (upload.source_dir / "1.jsonl").exists()
    setattr(upload.api, failure, False)
    assert upload.tick(True) == "watching"
    assert upload.state["uploaded_steps"] == [1]


def test_uploads_available_steps_when_trainer_exits_early(upload):
    save_batch(upload, 1)
    assert upload.tick(False) == "incomplete"
    assert upload.state["uploaded_steps"] == [1]
    assert "data/step_000001.jsonl.gz" in upload.api.files


def test_restart_does_not_duplicate_verified_shards(upload):
    save_batch(upload, 1)
    upload.tick(True)
    resumed = RolloutUploader(
        upload.source_dir, upload.dataset_dir, upload.state_file, upload.repo_id,
        upload.expected_rows, upload.final_step, upload.metadata, upload.api,
    )
    assert resumed.tick(True) == "watching"
    assert upload.api.upload_count == 1
    with pytest.raises(ValueError, match="different rollout collection"):
        RolloutUploader(upload.source_dir, upload.dataset_dir, upload.state_file, "other/run", 2, 2, {}, upload.api)


def test_process_identity_rejects_reused_pid():
    pid = os.getpid()
    start_time = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    assert process_matches(pid, start_time)
    assert not process_matches(pid, int(start_time) + 1)
