"""Ensure resuming a interrupted experiment cannot silently change its training recipe."""

import copy
import importlib.util
import multiprocessing
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "qwen3_experiments/l0_recovery_pipeline.py"
SPEC = importlib.util.spec_from_file_location("l0_recovery_test", SCRIPT)
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def configurations():
    before = {
        "trainer": {"resume_mode": "disable", "resume_from_path": None,
                    "default_local_dir": "/shared/checkpoints", "total_training_steps": 100},
        "ray_init": {"ray_dir": "/tmp/old/ray"},
        "algorithm": {"cost_offset_tokens": 0},
        "actor_rollout_ref": {"actor": {"checkpoint": {"load_contents": ["model", "optimizer", "extra"]}}},
    }
    after = copy.deepcopy(before)
    after["trainer"].update(resume_mode="resume_path", resume_from_path="/tmp/source/global_step_80",
                            default_local_dir="/tmp/new/checkpoints")
    after["ray_init"]["ray_dir"] = "/tmp/new/ray"
    return before, after


def test_permits_storage_and_full_state_resume_only():
    before, after = configurations()
    assert len(RECOVERY.verify_resume_configuration(before, after)) == 4


def test_rejects_algorithm_changes():
    before, after = configurations()
    after["algorithm"]["cost_offset_tokens"] = 1000
    with pytest.raises(AssertionError, match="Unexpected training change"):
        RECOVERY.verify_resume_configuration(before, after)


def test_rejects_losing_optimizer_and_random_state():
    before, after = configurations()
    before["actor_rollout_ref"]["actor"]["checkpoint"]["load_contents"] = ["model"]
    after["actor_rollout_ref"]["actor"]["checkpoint"]["load_contents"] = ["model"]
    with pytest.raises(AssertionError):
        RECOVERY.verify_resume_configuration(before, after)


def test_frozen_helper_is_importable_by_spawned_graders(tmp_path, monkeypatch):
    shutil.copy2(SCRIPT.with_name("eval_l0_final.py"), tmp_path / "l0_step80_helpers.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    helper = RECOVERY.module("l0_step80_helpers", tmp_path / "l0_step80_helpers.py")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(helper.answer_suffix, "<think>work</think>42").result(timeout=30) == ("42", "eligible")


def test_slurm_injected_rocm_variables_removed_before_training(tmp_path):
    launcher = tmp_path / "srun"
    launcher.write_text("#!/bin/sh\nwhile [ \"${1#--}\" != \"$1\" ]; do shift; done\n"
                        "export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
                        "exec \"$@\"\n")
    launcher.chmod(0o700)
    script = ("import os; assert 'ROCR_VISIBLE_DEVICES' not in os.environ; "
              "assert 'HIP_VISIBLE_DEVICES' not in os.environ; "
              "assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'")
    command = RECOVERY.slurm_training_command({"job_id": "146103"}, [sys.executable, "-c", script])
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                   "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
