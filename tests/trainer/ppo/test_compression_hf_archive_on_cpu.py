"""Exercise the real compression launchers and archiver with local GPU/Hub stand-ins."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHERS = [
    "run_deepseek_1_5b_compression_er_cost_marginrl.sh",
    "run_deepseek_1_5b_compression_hard_clip_256.sh",
]


def write_script(path, source):
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(source))
    path.chmod(0o755)


@pytest.fixture(params=LAUNCHERS)
def launch(tmp_path, request):
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
                (Path(os.environ["TEST_REMOTE"]) / repo_id).mkdir(parents=True, exist_ok=True)

            def repo_info(self, repo_id, **kwargs):
                root = Path(os.environ["TEST_REMOTE"]) / repo_id
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
        import os
        import shutil
        import sys
        from pathlib import Path

        args = sys.argv[1:]
        assert args[0] == "upload-large-folder"
        name = args[args.index("--include") + 1].split("/")[0]
        source = Path(args[2]) / name
        destination = Path(os.environ["TEST_REMOTE"]) / args[1] / name
        shutil.copytree(source, destination, dirs_exist_ok=True)
        if os.environ.get("TEST_CORRUPT_UPLOAD") == "1":
            (destination / "data.pt").write_bytes(b"wrong remote size")
    """)
    # An explicit PYTHON_BIN must also be used by the archive subprocess.
    write_script(bin_dir / "python", "raise SystemExit('wrong Python interpreter')\n")
    write_script(bin_dir / "nvidia-smi", "pass\n")
    selected_python = bin_dir / "selected_python"
    write_script(selected_python, """
        import os
        import sys
        import time
        from pathlib import Path

        args = sys.argv[1:]
        if "verl.trainer.main_ppo" in args:
            Path(os.environ["TEST_TRAIN_STARTED"]).touch()
            root = Path(os.environ["OUTPUT_ROOT"]) / os.environ["RUN_NAME"] / "checkpoints"
            for step in (20, 100):
                checkpoint = root / f"global_step_{step}"
                actor = checkpoint / "actor"
                actor.mkdir(parents=True, exist_ok=True)
                (checkpoint / "data.pt").write_bytes(b"data state")
                for rank in range(4):
                    for kind in ("model", "optim", "extra_state"):
                        (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"rank state")
                (root / "latest_checkpointed_iteration.txt").write_text(str(step))
            if os.environ.get("TEST_WAIT_FOR_ARCHIVE") == "1":
                deadline = time.monotonic() + 10
                while (root / "global_step_20").exists():
                    if time.monotonic() >= deadline:
                        raise SystemExit("older checkpoint was not archived during training")
                    time.sleep(0.02)
                assert (root / "global_step_100/data.pt").is_file()
                (root / "kept_latest_during_training").touch()
            print("step:" + os.environ.get("TEST_LOG_STEP", "100"), flush=True)
            raise SystemExit(int(os.environ.get("TEST_TRAIN_EXIT", "0")))
        elif args[0].endswith("/compression.py"):
            data = Path(os.environ["DATA_DIR"]) / "train.parquet"
            data.parent.mkdir(parents=True, exist_ok=True)
            data.write_bytes(b"prepared")
        elif args == ["-"]:
            pass  # GPU/version preflight; no accelerator access in these tests.
        else:
            os.execv(sys.executable, [sys.executable, *args])
    """)
    env = dict(os.environ)
    env.update({
        "PATH": str(bin_dir) + os.pathsep + env["PATH"],
        "PYTHONPATH": str(modules),
        "PYTHON_BIN": str(selected_python),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "RUN_NAME": "compression-test",
        "DATA_DIR": str(tmp_path / "data"),
        "RAY_TMPDIR": str(tmp_path / "ray"),
        "DRY_RUN": "0",
        "PREPARE_ONLY": "0",
        "RESUME": "0",
        "USE_WANDB": "0",
        "GPU_IDS": "0,1,2,3",
        "MAXRL_ARCHIVE_POLL_SECONDS": "1",
        "MAXRL_HF_UPLOAD_WORKERS": "1",
        "MAXRL_HF_UPLOAD_LOCK": str(tmp_path / "upload.lock"),
        "MAXRL_ARCHIVE_MIN_FREE_GIB": "0",
        "TEST_REMOTE": str(tmp_path / "remote"),
        "TEST_AUTH_CHECK": str(tmp_path / "auth_checked"),
        "TEST_TRAIN_STARTED": str(tmp_path / "train_started"),
    })
    # Exercise the defaults, independently of the developer's shell exports.
    for key in (
        "ARCHIVE_CHECKPOINTS", "HF_REPO_PREFIX", "MAXRL_TRAINING_EXIT_STATUS_FILE",
        "MAXRL_SAVE_ROLLOUT_DATASET", "MAXRL_ROLLOUT_DATASET_DIR", "MAXRL_ROLLOUT_DATASET_HF_REPO",
    ):
        env.pop(key, None)

    def run(**overrides):
        return subprocess.run(
            ["bash", str(REPO_ROOT / "qwen3_experiments" / request.param)],
            env=env | overrides, cwd=tmp_path, text=True, capture_output=True, timeout=30,
        )

    return run, tmp_path


def test_default_archive_uploads_and_verifies_all_checkpoints(launch):
    run, root = launch
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint_root = root / "outputs/compression-test/checkpoints"
    for step in (20, 100):
        remote = root / f"remote/zjhhhh/compression-test-step_{step}/global_step_{step}"
        assert (remote / "data.pt").read_bytes() == b"data state"
        assert len(list((remote / "actor").glob("*.pt"))) == 12
        assert not (checkpoint_root / f"global_step_{step}").exists()
    log = (root / "outputs/compression-test/logs/checkpoint_archiver.log").read_text()
    assert log.count("Verified 13 files") == 2
    assert "Checkpoint archival is complete" in log
    # The remaining step marker must not allow an accidental fresh training run.
    resumed = run(RESUME="1")
    assert resumed.returncode != 0
    assert "restore global_step_100 from HF" in resumed.stderr


def test_failed_training_keeps_newest_even_with_final_step_in_log(launch):
    run, root = launch
    result = run(TEST_TRAIN_EXIT="7")
    assert result.returncode == 7, result.stdout + result.stderr
    assert (root / "outputs/compression-test/checkpoints/global_step_100/data.pt").is_file()
    assert not (root / "remote/zjhhhh/compression-test-step_100").exists()
    assert (root / "remote/zjhhhh/compression-test-step_20/global_step_20/data.pt").is_file()
    # A resume must replace the previous exit code before starting its watcher.
    resumed = run(RESUME="1")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (root / "remote/zjhhhh/compression-test-step_100/global_step_100/data.pt").is_file()


def test_live_archival_keeps_newest_and_honors_custom_repo(launch):
    run, root = launch
    result = run(TEST_WAIT_FOR_ARCHIVE="1", HF_REPO_PREFIX="another-owner/custom-experiment")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "outputs/compression-test/checkpoints/kept_latest_during_training").exists()
    for step in (20, 100):
        assert (root / f"remote/another-owner/custom-experiment-step_{step}/global_step_{step}/data.pt").is_file()
        assert not (root / f"outputs/compression-test/checkpoints/global_step_{step}").exists()


def test_zero_exit_without_final_step_does_not_archive_newest(launch):
    run, root = launch
    result = run(TEST_LOG_STEP="80")
    assert result.returncode == 1, result.stdout + result.stderr
    assert (root / "outputs/compression-test/checkpoints/global_step_100/data.pt").is_file()


def test_verification_failure_retains_uploaded_checkpoint(launch):
    run, root = launch
    result = run(TEST_TRAIN_EXIT="7", TEST_CORRUPT_UPLOAD="1")
    assert result.returncode == 7, result.stdout + result.stderr
    for step in (20, 100):
        assert (root / f"outputs/compression-test/checkpoints/global_step_{step}/data.pt").is_file()
    log = (root / "outputs/compression-test/logs/checkpoint_archiver.log").read_text()
    assert "Verification failed" in log


def test_all_hf_uploads_can_be_disabled(launch):
    run, root = launch
    result = run(ARCHIVE_CHECKPOINTS="0", MAXRL_SAVE_ROLLOUT_DATASET="0", TEST_AUTH_FAIL="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "auth_checked").exists()
    assert not (root / "remote").exists()
    for step in (20, 100):
        assert (root / f"outputs/compression-test/checkpoints/global_step_{step}/data.pt").is_file()


@pytest.mark.parametrize("mode", ["DRY_RUN", "PREPARE_ONLY"])
def test_preview_and_preparation_do_not_require_hf_login(launch, mode):
    run, root = launch
    result = run(**{mode: "1", "TEST_AUTH_FAIL": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / "auth_checked").exists()
    assert not (root / "train_started").exists()
    assert not (root / "outputs").exists()


@pytest.mark.parametrize("archive_checkpoints", ["0", "1"])
def test_missing_hf_credentials_fails_before_training(launch, archive_checkpoints):
    run, root = launch
    result = run(TEST_AUTH_FAIL="1", ARCHIVE_CHECKPOINTS=archive_checkpoints)
    assert result.returncode != 0
    assert "missing HF credentials" in result.stderr
    assert not (root / "train_started").exists()
    assert not (root / "data/train.parquet").exists()
