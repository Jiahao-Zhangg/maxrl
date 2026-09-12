"""Exercise the offset launcher and real uploader using local training/Hub stand-ins."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "qwen3_experiments/run_qwen3_1_7b_math12k_fixed_n_rb_offset_marginrl.sh"
STEPS = (50, 100, 150)


def write_script(path, source):
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
    path.chmod(0o755)


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    modules = tmp_path / "modules"
    hub = modules / "huggingface_hub"
    hub.mkdir(parents=True)
    (hub / "__init__.py").write_text(textwrap.dedent("""
        import os
        import time
        from pathlib import Path
        from types import SimpleNamespace

        if os.environ.get("TEST_CORRUPT_UPLOAD") == "1":
            time.sleep = lambda _: None

        class HfApi:
            def whoami(self):
                Path(os.environ["TEST_AUTH_CHECK"]).touch()
                if os.environ.get("TEST_AUTH_FAIL") == "1":
                    raise RuntimeError("test: missing HF credentials")
                return {"name": "zjhhhh"}

            def create_repo(self, repo_id, **kwargs):
                assert kwargs["private"] is False
                (Path(os.environ["TEST_REMOTE"]) / repo_id).mkdir(parents=True, exist_ok=True)

            def repo_info(self, repo_id, **kwargs):
                root = Path(os.environ["TEST_REMOTE"]) / repo_id
                if kwargs.get("files_metadata"):
                    step = repo_id.rsplit("-step_", 1)[1]
                    # Verification must run before any local checkpoint is deleted.
                    assert (Path(os.environ["TEST_LOCAL_ROOT"]) / f"global_step_{step}/data.pt").is_file()
                files = [SimpleNamespace(rfilename=p.relative_to(root).as_posix(), size=p.stat().st_size)
                         for p in root.rglob("*") if p.is_file()]
                return SimpleNamespace(private=False, siblings=files)
    """))
    (hub / "utils.py").write_text(textwrap.dedent("""
        def validate_repo_id(repo_id):
            assert repo_id.count("/") == 1
            assert len(repo_id.split("/")[1]) <= 96
    """))
    write_script(bin_dir / "hf", """
        import json
        import os
        import shutil
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        assert args[0] == "upload-large-folder"
        name = args[args.index("--include") + 1].split("/")[0]
        with Path(os.environ["TEST_UPLOAD_CALLS"]).open("a") as handle:
            handle.write(json.dumps({"repo": args[1], "checkpoint": name}) + "\\n")
        failure_marker = Path(os.environ["TEST_FAILURE_MARKER"])
        if os.environ.get("TEST_UPLOAD_FAIL") == "1":
            raise SystemExit(3)
        if os.environ.get("TEST_UPLOAD_FAIL_ONCE") == "1" and not failure_marker.exists():
            failure_marker.touch()
            raise SystemExit(3)
        source = Path(args[2]) / name
        destination = Path(os.environ["TEST_REMOTE"]) / args[1] / name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        if os.environ.get("TEST_CORRUPT_UPLOAD") == "1":
            (destination / "data.pt").write_bytes(b"wrong remote size")
    """)
    write_script(bin_dir / "python", """
        import json
        import os
        import sys
        import time
        from pathlib import Path

        args = sys.argv[1:]
        if "verl.trainer.main_ppo" in args:
            Path(os.environ["TEST_TRAIN_STARTED"]).touch()
            settings = dict(arg.split("=", 1) for arg in args[args.index("verl.trainer.main_ppo") + 1:])
            Path(os.environ["TEST_TRAIN_CONFIG"]).write_text(json.dumps(settings))
            assert settings["trainer.total_training_steps"] == "150"
            assert settings["trainer.save_freq"] == "50"
            root = Path(settings["trainer.default_local_dir"])
            assert root == Path(os.environ["TEST_LOCAL_ROOT"])
            for step in (50, 100, 150):
                checkpoint = root / f"global_step_{step}"
                actor = checkpoint / "actor"
                actor.mkdir(parents=True, exist_ok=True)
                (checkpoint / "data.pt").write_bytes(b"data state")
                for rank in range(4):
                    for kind in ("model", "optim", "extra_state"):
                        (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"rank state")
                (root / "latest_checkpointed_iteration.txt").write_text(str(step))
                print(f"step:{step}", flush=True)
                if os.environ.get("TEST_WAIT_UPLOAD") == "1":
                    # Require each upload/deletion before allowing the next save.
                    deadline = time.monotonic() + 10
                    while checkpoint.exists():
                        if time.monotonic() >= deadline:
                            raise SystemExit(f"checkpoint {step} was not uploaded during training")
                        time.sleep(0.02)
                    prefix = os.environ["MAXRL_CHECKPOINT_HF_REPO_PREFIX"]
                    remote = Path(os.environ["TEST_REMOTE"]) / f"{prefix}-step_{step}" / checkpoint.name
                    assert (remote / "data.pt").read_bytes() == b"data state"
            raise SystemExit(int(os.environ.get("TEST_TRAIN_EXIT", "0")))
        elif args == ["-"]:
            pass  # Bypass the GPU preflight only; run HF checks against the fake module.
        else:
            os.execv(sys.executable, [sys.executable, *args])
    """)
    data_root = tmp_path / "data"
    for relative in ("math12k/train.parquet", "aime25/test.parquet", "math500/test.parquet"):
        path = data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"prepared")
    checkpoint_root = tmp_path / "outputs/checkpoints/Qwen3_MaxRL_Experiments/offset-upload-test"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MAXRL_")}
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "PYTHONPATH": str(modules),
        "PYTHON_BIN": str(tmp_path / "must-not-use-ambient-python"),
        "MAXRL_SKIP_ENV_SETUP": "1",
        "MAXRL_DATA_DIR": str(data_root),
        "MAXRL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "MAXRL_EXPERIMENT_NAME": "offset-upload-test",
        "MAXRL_ARCHIVE_POLL_SECONDS": "1",
        "MAXRL_HF_UPLOAD_WORKERS": "1",
        "MAXRL_HF_UPLOAD_LOCK": str(tmp_path / "upload.lock"),
        "TEST_REMOTE": str(tmp_path / "remote"),
        "TEST_LOCAL_ROOT": str(checkpoint_root),
        "TEST_AUTH_CHECK": str(tmp_path / "auth_checked"),
        "TEST_TRAIN_STARTED": str(tmp_path / "train_started"),
        "TEST_TRAIN_CONFIG": str(tmp_path / "train_config.json"),
        "TEST_UPLOAD_CALLS": str(tmp_path / "upload_calls.jsonl"),
        "TEST_FAILURE_MARKER": str(tmp_path / "failed_once"),
        "TEST_WAIT_UPLOAD": "1",
    })

    def run(*arguments, **environment):
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments], env=env | environment, cwd=tmp_path,
            capture_output=True, text=True, timeout=30,
        )

    return run, tmp_path, checkpoint_root


@pytest.mark.parametrize("prefix", [None, "another-owner/custom-offset"])
def test_every_50_steps_uploads_verifies_then_deletes_while_training(launch, prefix):
    run, root, checkpoints = launch
    environment = {} if prefix is None else {"MAXRL_CHECKPOINT_HF_REPO_PREFIX": prefix}
    result = run(**environment)
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = prefix or "zjhhhh/offset-upload-test"
    for step in STEPS:
        remote = root / f"remote/{prefix}-step_{step}/global_step_{step}"
        assert (remote / "data.pt").read_bytes() == b"data state"
        assert len(list((remote / "actor").glob("*.pt"))) == 12
        assert not (checkpoints / f"global_step_{step}").exists()
    calls = [json.loads(line) for line in (root / "upload_calls.jsonl").read_text().splitlines()]
    assert [call["checkpoint"] for call in calls] == [f"global_step_{step}" for step in STEPS]
    log = (checkpoints / "logs/checkpoint_upload.log").read_text()
    assert log.count("Verified 13 files") == 3
    assert "Checkpoint archival is complete" in log
    assert (checkpoints / "logs/training.exit_status").read_text().strip() == "0"


def test_transient_upload_failure_retries_before_deletion(launch):
    run, root, checkpoints = launch
    result = run(TEST_UPLOAD_FAIL_ONCE="1")
    assert result.returncode == 0, result.stdout + result.stderr
    log = (checkpoints / "logs/checkpoint_upload.log").read_text()
    assert "Upload failed for global_step_50; retaining it for retry" in log
    assert log.count("Verified 13 files") == 3
    assert not list(checkpoints.glob("global_step_*"))
    assert len((root / "upload_calls.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("failure", ["TEST_CORRUPT_UPLOAD", "TEST_UPLOAD_FAIL"])
def test_failed_upload_or_verification_retains_local_checkpoints(launch, failure):
    run, _, checkpoints = launch
    result = run(**{failure: "1", "TEST_WAIT_UPLOAD": "0", "TEST_TRAIN_EXIT": "7"})
    assert result.returncode == 7, result.stdout + result.stderr
    for step in STEPS:
        assert (checkpoints / f"global_step_{step}/data.pt").is_file()
    log = (checkpoints / "logs/checkpoint_upload.log").read_text()
    expected = "Verification failed" if failure == "TEST_CORRUPT_UPLOAD" else "Upload failed"
    assert expected in log
    assert "Checkpoint archival is complete" not in log
    assert (checkpoints / "logs/training.exit_status").read_text().strip() == "7"


def test_upload_authentication_failure_prevents_training(launch):
    run, root, _ = launch
    result = run(TEST_AUTH_FAIL="1")
    assert result.returncode != 0
    assert "missing HF credentials" in result.stderr
    assert not (root / "train_started").exists()
    assert not (root / "remote").exists()


def test_uploads_can_be_disabled_without_hf_credentials(launch):
    run, root, checkpoints = launch
    result = run(MAXRL_UPLOAD_CHECKPOINTS="0", MAXRL_SAVE_ROLLOUT_DATASET="0", TEST_AUTH_FAIL="1", TEST_WAIT_UPLOAD="0")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "auth_checked").exists()
    assert not (root / "remote").exists()
    for step in STEPS:
        assert (checkpoints / f"global_step_{step}/data.pt").is_file()


@pytest.mark.parametrize("override", ["trainer.total_training_steps=7", "trainer.default_local_dir=elsewhere"])
def test_upload_supervisor_rejects_mismatched_step_or_checkpoint_path(launch, override):
    run, root, _ = launch
    result = run(override)
    assert result.returncode != 0
    assert "With checkpoint uploads, use" in result.stderr
    assert not (root / "train_started").exists()
