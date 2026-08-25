import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.utils.rollout_dataset import dump_rollout_step, upload_rollout_dataset_to_hf


class _FakeHfApi:
    def __init__(self):
        self.created = []
        self.settings = []
        self.uploads = []
        self.siblings = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def update_repo_settings(self, **kwargs):
        self.settings.append(kwargs)

    def upload_large_folder(self, **kwargs):
        self.uploads.append(kwargs)
        root = Path(kwargs["folder_path"])
        self.siblings = [
            SimpleNamespace(rfilename=path.relative_to(root).as_posix(), size=path.stat().st_size)
            for path in root.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(root).parts
        ]

    def repo_info(self, **kwargs):
        return SimpleNamespace(siblings=self.siblings)


class _WrongSizeHfApi(_FakeHfApi):
    def repo_info(self, **kwargs):
        siblings = [
            SimpleNamespace(rfilename=item.rfilename, size=item.size + 1)
            for item in self.siblings
        ]
        return SimpleNamespace(siblings=siblings)


def _read_compressed_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file]


def test_dump_rollout_step_writes_compressed_shard_and_manifest(tmp_path):
    output_path = dump_rollout_step(
        tmp_path,
        step=7,
        inputs=["prompt one", "prompt two"],
        outputs=["answer one", "answer two"],
        scores=[1.0, 0.0],
        extra_fields={"uid": ["group-a", "group-b"], "ignored": [1]},
    )

    assert output_path == tmp_path / "data" / "step_000007.jsonl.gz"
    assert _read_compressed_jsonl(output_path) == [
        {
            "step": 7,
            "rollout_index": 0,
            "input": "prompt one",
            "output": "answer one",
            "score": 1.0,
            "uid": "group-a",
        },
        {
            "step": 7,
            "rollout_index": 1,
            "input": "prompt two",
            "output": "answer two",
            "score": 0.0,
            "uid": "group-b",
        },
    ]
    manifest = json.loads((tmp_path / "rollout_manifest.json").read_text())
    assert manifest["num_steps"] == 1
    assert manifest["num_rollouts"] == 2
    assert manifest["steps"]["7"]["file"] == "data/step_000007.jsonl.gz"


def test_dump_rollout_step_replaces_replayed_step(tmp_path):
    dump_rollout_step(tmp_path, step=3, inputs=["old"], outputs=["old"], scores=[0.0])
    output_path = dump_rollout_step(
        tmp_path,
        step=3,
        inputs=["new-a", "new-b"],
        outputs=["a", "b"],
        scores=[1.0, 1.0],
    )

    assert [record["input"] for record in _read_compressed_jsonl(output_path)] == [
        "new-a",
        "new-b",
    ]
    manifest = json.loads((tmp_path / "rollout_manifest.json").read_text())
    assert manifest["num_steps"] == 1
    assert manifest["num_rollouts"] == 2


def test_upload_rollout_dataset_creates_dataset_repo_and_verifies_files(tmp_path):
    dump_rollout_step(tmp_path, step=1, inputs=["question"], outputs=["answer"], scores=[1.0])
    api = _FakeHfApi()

    url = upload_rollout_dataset_to_hf(
        tmp_path,
        repo_id="owner/rollouts",
        private=True,
        num_workers=2,
        metadata={"experiment_name": "experiment", "total_training_steps": 1},
        api=api,
        verify_attempts=1,
        verify_sleep_seconds=0,
    )

    assert url == "https://huggingface.co/datasets/owner/rollouts"
    assert api.created == [
        {
            "repo_id": "owner/rollouts",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploads[0]["repo_type"] == "dataset"
    assert api.uploads[0]["num_workers"] == 2
    assert (tmp_path / "README.md").is_file()
    manifest = json.loads((tmp_path / "rollout_manifest.json").read_text())
    assert manifest["run"]["experiment_name"] == "experiment"


def test_upload_rollout_dataset_requires_step_shards(tmp_path):
    with pytest.raises(ValueError, match="No rollout step shards"):
        upload_rollout_dataset_to_hf(
            tmp_path,
            repo_id="owner/empty",
            api=_FakeHfApi(),
            verify_attempts=1,
        )


def test_upload_rollout_dataset_rejects_remote_size_mismatch(tmp_path):
    dump_rollout_step(tmp_path, step=1, inputs=["question"], outputs=["answer"], scores=[1.0])

    with pytest.raises(RuntimeError, match="verification failed"):
        upload_rollout_dataset_to_hf(
            tmp_path,
            repo_id="owner/wrong-size",
            api=_WrongSizeHfApi(),
            verify_attempts=1,
            verify_sleep_seconds=0,
        )
