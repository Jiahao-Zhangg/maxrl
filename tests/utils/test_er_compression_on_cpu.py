"""Reference ER settings, full rollout capture, and the audited evaluation handoff."""

import gzip
import json
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from qwen3_experiments import er_compression_compute as control
from qwen3_experiments import er_compression_eos as eos
from qwen3_experiments import er_compression_rollouts as rollouts


def test_launcher_matches_reference_rloo_hyperparameters():
    repo = Path(__file__).resolve().parents[2]
    preview = subprocess.check_output(["bash", str(repo / control.LAUNCHER)], cwd=repo,
                                      env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1"}, text=True)
    command = shlex.split(preview)
    expected = {"--pretrain": "Qwen/Qwen3-1.7B", "--advantage_estimator": "rloo", "--rollout_batch_size": "32",
                "--n_samples_per_prompt": "8", "--train_batch_size": "128", "--generate_max_len": "32768",
                "--prompt_max_len": "1536", "--actor_learning_rate": "1e-6", "--lr_warmup_ratio": "0",
                "--temperature": "1.0", "--top_p": "1.0", "--seed": "79", "--init_kl_coef": "0.0",
                "--zero_stage": "3", "--vllm_num_engines": "8", "--vllm_gpu_memory_utilization": "0.5",
                "--save_steps": "20", "--max_samples": "3200"}
    assert all(command[command.index(key) + 1] == value for key, value in expected.items())
    assert all(flag in command for flag in ("--colocate_all_models", "--vllm_enable_sleep", "--deepspeed_enable_sleep"))
    assert "compression_thinking.jsonl" in preview


def completed_comparison(tmp_path):
    root = tmp_path / "minerva"
    previous = {"models": {f"model_{i}": {} for i in range(5)}, "budgets": [8192, 16384, 32768, 49152, 65536]}
    control.write(root / "plan.json", previous)
    plan = {"predecessor_root": str(root), "predecessor_plan_sha256": control.digest(root / "plan.json")}
    for name in ("status.json", "queue_status.json"):
        control.write(root / name, {"state": "complete", "points": 25})
    control.write(root / "report/metrics.json", {"results": "all"})
    control.write(root / "report/audit.json", {"complete": True, "points": 25, "questions_per_point": 272,
                                              "all_rollout_ledgers_verified": True,
                                              "metrics_sha256": control.digest(root / "report/metrics.json")})
    control.write(root / "execution_manifest.json", {"fingerprint": "fixed"})
    for key in previous["models"]:
        for budget in previous["budgets"]:
            control.write(root / "results" / key / f"budget_{budget}/summary.json", {
                "state": "complete", "identity": {"manifest": "fixed", "model": key, "budget": budget}, "artifacts": {},
            })
    return root, plan


def test_training_waits_for_all_five_models_and_every_budget(tmp_path):
    root, plan = completed_comparison(tmp_path)
    assert control.predecessor_ready(plan)
    control.write(root / "queue_status.json", {"state": "running", "completed_points": 24})
    assert not control.predecessor_ready(plan)
    control.write(root / "queue_status.json", {"state": "failed"})
    with pytest.raises(RuntimeError, match="has not been launched"):
        control.predecessor_ready(plan)


@pytest.mark.parametrize("damage", ["audit", "point", "report", "identity"])
def test_partial_or_changed_evaluation_cannot_unlock_er(tmp_path, damage):
    root, plan = completed_comparison(tmp_path)
    if damage == "audit":
        path = root / "report/audit.json"
        value = control.read(path)
        value["points"] = 24
    elif damage == "point":
        path = root / "results/model_4/budget_65536/summary.json"
        value = control.read(path)
        value["state"] = "failed"
    elif damage == "report":
        path, value = root / "report/metrics.json", {"changed": True}
    else:
        path, value = root / "plan.json", {"another": "experiment"}
    control.write(path, value)
    with pytest.raises(ValueError):
        control.predecessor_ready(plan)


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, prompt, **kwargs):
        return [1, 99, 10 if prompt == "p0" else 11]  # Like Qwen3, the prompt itself contains EOS.

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def saved_rollouts(tmp_path, monkeypatch):
    monkeypatch.setenv("ER_SHARED_ROOT", str(tmp_path))
    maker = SimpleNamespace(strategy=SimpleNamespace(args=SimpleNamespace(n_samples_per_prompt=8,
                            rollout_batch_size=2, generate_max_len=32768)), tokenizer=Tokenizer())
    count = 16
    prompts = [f"p{i // 8}" for i in range(count)]
    labels = [json.dumps({"row_id": i // 8, "dataset_name": control.DATASET_KEY, "extracted": "42"}) for i in range(count)]
    outputs = [SimpleNamespace(prompt_token_ids=maker.tokenizer.encode(p), outputs=[SimpleNamespace(
        token_ids=[20, 99] if i % 2 else [21, 22, 23], finish_reason="stop" if i % 2 else "length", stop_reason=None)])
        for i, p in enumerate(prompts)]
    tagged = rollouts.capture_generated_rollouts(maker, outputs, prompts, labels)
    queries = [maker.tokenizer.decode(output.prompt_token_ids + output.outputs[0].token_ids) for output in outputs]
    payload = {"query": queries, "prompts": prompts, "labels": tagged}
    metrics = {"rewards": [0.95] * count, control.DATASET_KEY + "_accuracy": [1.] * count,
               control.DATASET_KEY + "_response_length": [5.] * count}
    rollouts.record_rewards(payload, metrics)
    with gzip.open(tmp_path / "rollout_dataset/data/step_000001.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    return maker, payload, metrics, records


def audit_plan(**overrides):
    return {"total_steps": 1, "rows_per_step": 16, "num_questions": 2,
            "eos_token_id": 99, "pad_token_id": 0, **overrides}


def test_capture_preserves_raw_tokens_then_forces_training_eos(tmp_path, monkeypatch):
    maker, payload, metrics, records = saved_rollouts(tmp_path, monkeypatch)
    assert len(records) == 16
    assert records[0]["response_token_ids"] == [21, 22, 23] and records[0]["generated_eos"] is False
    assert records[1]["response_token_ids"] == [20, 99] and records[1]["generated_eos"] is True
    assert records[0]["training_response_token_ids"] == [21, 22, 99] and records[0]["force_eos_applied"] is True
    assert records[1]["training_response_token_ids"] == [20, 99] and records[1]["force_eos_applied"] is False
    assert all(row["reward_query"] == query for row, query in zip(records, payload["query"]))
    assert records[0]["score"] == .95  # Reference reward may accept a generation without EOS.
    assert len(set(payload["labels"][:8])) == len(set(payload["labels"][8:])) == 1
    rollouts.record_rewards(payload, metrics)  # HTTP retry must not duplicate records.
    assert control.read(tmp_path / "rollout_dataset/rollout_manifest.json")["num_rollouts"] == 16
    assert control.audit_local_rollouts(tmp_path, audit_plan()) == 16


@pytest.mark.parametrize("raw,expected", [([4, 5, 6], [4, 5, 99]), ([4, 99], [4, 99]),
                                        ([4, 0, 0], [4, 99, 0]), ([4, 99, 0], [4, 99, 0]), ([4], [99])])
def test_reference_force_eos_replaces_within_existing_sequence(raw, expected):
    before = list(raw)
    assert eos.force_eos_token_ids([1, 99, 7], raw, 99, 0) == expected
    assert raw == before
    assert len(raw) == len(expected)


def test_length_pool_uses_raw_response_eos_despite_eos_in_qwen_prompt(tmp_path, monkeypatch):
    maker, payload, _, records = saved_rollouts(tmp_path, monkeypatch)
    converted = eos.reference_reward_payload(payload, 8, maker.tokenizer)["query"]
    for start in (0, 8):
        group = records[start:start + 8]
        expected_raw = [maker.tokenizer.decode(row["prompt_token_ids"] + row["response_token_ids"]) for row in group]
        for item in converted[start:start + 8]:
            assert item["aux_info"]["all_responses"] == expected_raw
            assert item["aux_info"]["all_responses_have_eos"] == [False, True] * 4
            assert item["aux_info"]["response_has_eos"] is True
    assert converted[0]["response"] != converted[0]["aux_info"]["all_responses"][0]
    assert all("99" in raw for raw in converted[0]["aux_info"]["all_responses"])


def test_rollout_audit_rejects_incorrect_force_eos(tmp_path, monkeypatch):
    _, _, _, records = saved_rollouts(tmp_path, monkeypatch)
    records[0]["training_response_token_ids"] = [21, 22, 23, 99]  # Appending differs from the reference.
    path = tmp_path / "rollout_dataset/data/step_000001.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="reference EOS policy"):
        control.audit_local_rollouts(tmp_path, audit_plan())


def test_raw_generation_cannot_be_overwritten_and_rewards_cannot_be_reordered(tmp_path, monkeypatch):
    maker, payload, metrics, _ = saved_rollouts(tmp_path, monkeypatch)
    payload["labels"][0], payload["labels"][8] = payload["labels"][8], payload["labels"][0]
    with pytest.raises(ValueError, match="ordering"):
        rollouts.record_rewards(payload, metrics)
    maker._er_rollout_step = 0
    with pytest.raises(FileExistsError):
        rollouts.capture_generated_rollouts(maker, [None] * 16, [None] * 16, [None] * 16)


def test_rollout_audit_requires_all_rows(tmp_path, monkeypatch):
    saved_rollouts(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="every ER rollout step"):
        control.audit_local_rollouts(tmp_path, audit_plan(total_steps=2, num_questions=4))
    with pytest.raises(ValueError, match="every compression row"):
        control.audit_local_rollouts(tmp_path, audit_plan(num_questions=3))


@pytest.mark.parametrize("started", ["training_started.json", "raw_rollouts", "rollout_dataset"])
def test_eos_refresh_cannot_replace_started_training(tmp_path, monkeypatch, started):
    monkeypatch.setattr(control, "verify_plan", lambda root: {"job_id": "146103"})
    control.write(tmp_path / "launch.json", {"pid": 17})
    control.write(tmp_path / "status.json", {"pid": 17, "state": "waiting_for_all_evaluations"})
    assert control.check_waiting_refresh(tmp_path) == {"job_id": "146103"}
    (tmp_path / started).touch()
    with pytest.raises(ValueError, match="execution already began"):
        control.check_waiting_refresh(tmp_path)


def test_eos_refresh_rejects_active_or_other_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "verify_plan", lambda root: {})
    control.write(tmp_path / "launch.json", {"pid": 17})
    for value in ({"pid": 17, "state": "training_or_archiving"},
                  {"pid": 18, "state": "waiting_for_all_evaluations"}):
        control.write(tmp_path / "status.json", value)
        with pytest.raises(ValueError, match="identified waiting"):
            control.check_waiting_refresh(tmp_path)
