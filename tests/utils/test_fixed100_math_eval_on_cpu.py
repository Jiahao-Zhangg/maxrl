"""Portable replay must preserve the paper protocol and reject incomplete pools."""

import asyncio
import hashlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_experiments import fixed100_math_replay as mc
from qwen3_experiments.fixed100_math_analysis import (
    load_pools,
    replay_group,
    run_replays,
    stop_once,
    summarize,
    validate_pool,
)
from qwen3_experiments.fixed100_math_eval import (
    CAPS,
    DATASETS,
    SAMPLING,
    digest,
    generate_async,
    grade,
    load_run,
    read_gzip,
    response_identity,
    sample_seed,
    validate_questions,
    verify_response,
    write,
    write_gzip,
)


def questions(per_dataset=2):
    return [
        {
            "unique_id": f"{dataset}_{i:04d}",
            "position": d * 100 + i,
            "dataset": dataset,
            "ground_truth": "2",
            "prompt": [],
            "prompt_token_ids": [1, 2],
        }
        for d, dataset in enumerate(DATASETS)
        for i in range(per_dataset)
    ]


def pool(n):
    lengths = np.full((n, 100), 1000, dtype=np.int32)
    scores = np.zeros((n, 100), dtype=np.int8)
    scores[:, 0] = 1
    capped = np.repeat(scores[:, :, None], 6, axis=2)
    return {"label": "Toy", "lengths": lengths, "scores": scores, "capped": capped}


def test_frozen_boundary_and_deduplication_examples():
    assert mc.validate()["passed"]


def test_seed_uses_original_position_and_matches_historical_formula():
    seed = int.from_bytes(hashlib.blake2b(b"math-eval-v1:42:137:99", digest_size=8).digest(), "big") % (2**31 - 1)
    assert sample_seed(137, 99) == seed
    assert len({sample_seed(p, a) for p in [135, 136, 137] for a in range(100)}) == 300
    before = questions()
    after = [q for q in before if q["position"] != 0]
    assert sample_seed(after[0]["position"], 3) == sample_seed(before[1]["position"], 3)


def test_resume_rejects_wrong_seed_model_plan_or_truncated_record(tmp_path):
    plan = {"model": {"repo": "owner/model", "revision": "a" * 40}, "plan_sha256": "frozen"}
    q = questions()[0]
    record = {
        **response_identity(plan, q, 0),
        "token_ids": [1, 2],
        "output_tokens": 2,
        "response": "</think>2",
        "finish_reason": "stop",
    }
    path = tmp_path / "response.json.gz"
    write_gzip(path, record)
    verify_response(read_gzip(path), plan, q, 0)
    for key, value in [
        ("seed", -1),
        ("model_revision", "wrong"),
        ("plan_sha256", "wrong"),
        ("output_tokens", 3),
        ("finish_reason", "abort"),
        ("token_ids", []),
    ]:
        changed = {**record, key: value}
        with pytest.raises(ValueError):
            verify_response(changed, plan, q, 0)


def test_duplicate_question_positions_are_rejected():
    qs = questions()
    qs[1]["position"] = qs[0]["position"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_questions(qs)


def test_incomplete_or_wrong_prefix_pool_is_rejected():
    qs = questions()
    values = pool(len(qs))
    validate_pool(values, qs)
    values["scores"] = values["scores"][:, :99]
    with pytest.raises(ValueError, match="exactly 100"):
        validate_pool(values, qs)
    values = pool(len(qs))
    values["capped"][0, 0, -1] = 0
    with pytest.raises(AssertionError):
        validate_pool(values, qs)


def test_stop_replay_matches_frozen_rule_with_paired_seeds():
    values = pool(3)
    values["scores"][:, 2:5] = 1
    targets = np.array([4000, 8000, 16000, 32000, 64000]) * 3
    for trial in range(4):

        def rngs(trial=trial):
            return (np.random.default_rng([42, 2, 0, trial, 0]), np.random.default_rng([42, 2, 0, trial, 1]))

        actual, missing = stop_once(values["lengths"], values["scores"], targets, *rngs())
        expected, _ = mc.cross_exit_once(values["lengths"], values["scores"], targets, *rngs())
        assert missing is None
        np.testing.assert_array_equal(actual, expected)


def test_stop_reports_exhaustion_instead_of_assuming_failure():
    lengths = np.ones((2, 100), dtype=np.int32)
    scores = np.zeros_like(lengths)
    scores[1, :] = 1
    value, missing = stop_once(lengths, scores, [1000], np.random.default_rng(1), np.random.default_rng(2))
    assert value is None and missing == 0


def test_shared_exclusions_apply_to_every_model_and_first_correct_cohort():
    qs = questions()
    a, b = pool(len(qs)), pool(len(qs))
    # Each even question alone covers 64k, but exhausts in shared stop after
    # its cheap companion succeeds. This requires the shared coverage check.
    a["lengths"][::2] = 700
    a["scores"][::2] = 0
    a["capped"][::2] = 0
    a["lengths"][1::2] = 1
    a["scores"][1::2] = 1
    a["capped"][1::2] = 1
    b["scores"][:] = 1
    b["capped"][:] = 1
    # Continue pools must cover N*64k too; short solved answers alone cannot
    # support continue, so their later (unused by stop) answers remain long.
    a["lengths"][1::2, 1:] = 2000
    selected, groups, results, reasons = run_replays(qs, {"a": a, "b": b}, workers=1)
    assert selected == list(range(1, len(qs), 2))
    assert {r["reason"] for r in reasons} == {"shared_stop_64k_pool_exhaustion"}
    budgets, capped, first, _, common_ids = summarize(qs, {"a": a, "b": b}, selected, groups, results)
    assert len(common_ids) == 6
    assert all(r["questions"] == 6 for r in budgets + capped if r["aggregation"] in ("macro", "weighted"))
    assert all(r["common_solved_questions"] == 6 for r in first if r["aggregation"] in ("macro", "weighted"))


def test_replay_identity_does_not_depend_on_model_label():
    qs = questions(1)
    values = pool(len(qs))
    task = ("one", 3, [qs[3]], values["lengths"][3:4], values["scores"][3:4])
    _, _, one = replay_group(task)
    _, _, two = replay_group(("two", *task[1:]))
    for field in ["individual", "stop_after_success", "continue_after_success", "hits", "exact"]:
        np.testing.assert_array_equal(one[field], two[field])
    assert one["exact"][0] == 50500


def test_grading_refuses_to_drop_a_missing_generation(tmp_path):
    qs = questions(1)
    write(tmp_path / "questions.json", qs)
    plan = {
        "sampling": SAMPLING,
        "samples": 100,
        "seed": 42,
        "files_sha256": {"questions.json": digest(tmp_path / "questions.json")},
    }
    write(tmp_path / "plan.json", plan)
    source = tmp_path / "runtime/qwen3_experiments/fixed100_legacy_math_scorer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# not reached\n")
    (tmp_path / "grader_site").mkdir()
    with pytest.raises(ValueError, match="Missing response"):
        grade(SimpleNamespace(run_dir=tmp_path, grader_site=None, workers=1))
    (tmp_path / "questions.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen input changed"):
        load_run(tmp_path)


def test_generation_shards_are_disjoint_and_resume_without_resampling(tmp_path, monkeypatch):
    calls = []

    class FakeEngine:
        @classmethod
        def from_engine_args(cls, args):
            assert args.enforce_eager is False
            return cls()

        async def generate(self, prompt, params, request_id):
            assert prompt["prompt_token_ids"] == [1, 2]
            assert params.max_tokens == 32768 and params.temperature == 0.6
            calls.append((request_id, params.seed))
            sample = SimpleNamespace(text="</think>2", token_ids=[1, 2], finish_reason="stop")
            yield SimpleNamespace(finished=True, outputs=[sample])

        def shutdown(self):
            pass

    modules = {
        "vllm": {"SamplingParams": SimpleNamespace},
        "vllm.engine.arg_utils": {"AsyncEngineArgs": SimpleNamespace},
        "vllm.sampling_params": {"RequestOutputKind": SimpleNamespace(FINAL_ONLY="final")},
        "vllm.v1.engine.async_llm": {"AsyncLLM": FakeEngine},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    plan = {
        "model": {"repo": "owner/model", "revision": "a" * 40},
        "plan_sha256": "frozen",
        "shards": 2,
        "engine": {"enforce_eager": False, "max_num_seqs": 4},
    }
    qs = questions(1)[:1]
    asyncio.run(generate_async(tmp_path, plan, qs, 0, None))
    assert len(calls) == 50
    asyncio.run(generate_async(tmp_path, plan, qs, 1, None))
    assert len(calls) == len(set(calls)) == 100
    paths = sorted((tmp_path / "responses").glob("*.json.gz"))
    assert len(paths) == 100
    original_hashes = [digest(p) for p in paths]
    asyncio.run(generate_async(tmp_path, plan, qs, 0, None))
    assert len(calls) == 100
    assert [digest(p) for p in paths] == original_hashes


def test_reference_pools_align_original_positions_and_check_prompt_identity(tmp_path):
    qs = questions(1)
    values = pool(len(qs))
    write(tmp_path / "questions.json", qs)
    plan = {
        "sampling": SAMPLING,
        "samples": 100,
        "seed": 42,
        "model": {"key": "new", "label": "New"},
        "files_sha256": {"questions.json": digest(tmp_path / "questions.json")},
    }
    write(tmp_path / "plan.json", plan)
    np.savez_compressed(
        tmp_path / "samples.npz",
        lengths=values["lengths"],
        scores=values["scores"],
        capped_scores=values["capped"],
        caps=CAPS,
        original_positions=[q["position"] for q in qs],
    )
    write(
        tmp_path / "grading_receipt.json",
        {
            "complete": True,
            "plan_sha256": digest(tmp_path / "plan.json"),
            "samples_sha256": digest(tmp_path / "samples.npz"),
        },
    )
    reference = tmp_path / "reference"
    # Reference files may use a different question order; identity, not array
    # offsets, must determine the matched comparison.
    order = list(reversed(range(len(qs))))
    other = [qs[i] for i in order]
    write(reference / "manifests/questions.json", other)
    write(reference / "manifests/model_index.json", {"old": {"label": "Old"}})
    (reference / "arrays").mkdir()
    values["lengths"][:, 0] = np.arange(len(qs)) + 500
    np.savez_compressed(
        reference / "manifests/old_fixed100.npz",
        lengths=values["lengths"][order],
        scores=values["scores"][order],
        original_positions=[q["position"] for q in other],
    )
    np.savez_compressed(
        reference / "arrays/capped_fixed100_scores.npz",
        caps=CAPS,
        old_scores=values["capped"][order],
        old_lengths=values["lengths"][order],
        original_positions=[q["position"] for q in other],
    )
    args = SimpleNamespace(run_dir=tmp_path, reference_dir=reference, with_reference=False)
    _, pools, _ = load_pools(args)
    np.testing.assert_array_equal(pools["old"]["lengths"], values["lengths"])
    assert list(pools) == ["old", "new"]
    other[0] = {**other[0], "prompt_token_ids": [999]}
    write(reference / "manifests/questions.json", other)
    with pytest.raises(ValueError, match="same prompts"):
        load_pools(args)


def test_empty_common_solved_benchmark_preserves_weighted_cost_but_not_macro():
    qs = questions(1)
    values = pool(len(qs))
    values["scores"][0] = 0
    values["capped"][0] = 0
    results = {}
    for d in range(6):
        results["model", d] = {
            "individual": np.zeros((100, 5)),
            "stop_after_success": np.zeros((100, 5)),
            "continue_after_success": np.zeros((100, 5)),
            "hits": np.full((1, 100), 1000 + d),
            "exact": np.array([1000 + d]),
        }
    _, _, first, _, common_ids = summarize(qs, {"model": values}, list(range(6)), [[d] for d in range(6)], results)
    assert len(common_ids) == 5
    weighted = next(r for r in first if r["aggregation"] == "weighted")
    macro = next(r for r in first if r["aggregation"] == "macro")
    assert weighted["mean_tokens_mc100"] == pytest.approx(1003)
    assert macro["mean_tokens_mc100"] is None
