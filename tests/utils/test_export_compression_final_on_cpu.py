import copy
from types import SimpleNamespace

import pytest

from qwen3_experiments import export_compression_final as export


def merge_fixture():
    spec = {"source": {"repo": "owner/model-step_100", "revision": "a" * 40}}
    files = {f"actor/model_world_size_8_rank_{rank}.pt": {"size": rank + 1, "sha256": str(rank) * 64}
             for rank in range(8)}
    receipt = {**spec["source"], "source_files": {"global_step_100/" + k: v for k, v in files.items()},
               "merged_files": {k: "c" * 64 for k in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")}}
    return spec, {"files": files}, receipt


def test_rejects_model_from_different_checkpoint():
    spec, archive, receipt = merge_fixture()
    receipt["revision"] = "b" * 40
    with pytest.raises(ValueError, match="another checkpoint"):
        export.verify_merge_receipt(spec, archive, receipt)


@pytest.mark.parametrize("change", ["missing_rank", "wrong_hash", "missing_weights", "path_traversal"])
def test_rejects_incomplete_or_mismatched_model_provenance(change):
    spec, archive, receipt = merge_fixture()
    export.verify_merge_receipt(spec, archive, receipt)
    receipt = copy.deepcopy(receipt)
    first = next(iter(receipt["source_files"]))
    if change == "missing_rank":
        receipt["source_files"].pop(first)
    elif change == "wrong_hash":
        receipt["source_files"][first]["sha256"] = "bad"
    elif change == "missing_weights":
        receipt["merged_files"].pop("model.safetensors")
    else:
        receipt["merged_files"]["../unrelated.json"] = "bad"
    with pytest.raises(ValueError):
        export.verify_merge_receipt(spec, archive, receipt)


def test_cleanup_preserves_unverified_or_unrelated_model(tmp_path):
    scratch = tmp_path / "scratch"
    work = scratch / "work/model"
    work.mkdir(parents=True)
    model = work / "model.safetensors"
    model.write_text("weights")
    with pytest.raises(ValueError):
        export.cleanup(work, scratch, {"state": "uploading", "public": True})
    assert model.exists()
    verified = {"state": "published_verified", "public": True}
    with pytest.raises(ValueError):
        export.cleanup(work, tmp_path, verified)
    assert model.exists()
    export.cleanup(work, scratch, verified)
    assert not work.exists()


def test_public_verification_rejects_wrong_remote_weight(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    info = SimpleNamespace(private=False, sha="a" * 40, siblings=[
        SimpleNamespace(rfilename="model.safetensors", size=7, lfs=SimpleNamespace(sha256="bad"))])
    api = SimpleNamespace(model_info=lambda *a, **kw: info)
    with pytest.raises(ValueError, match="checksum"):
        export.verify_public(api, "owner/model-final", "a" * 40, tmp_path,
                             {"model.safetensors": export.digest(tmp_path / "model.safetensors")})


def test_training_must_be_complete_before_export(tmp_path):
    export.write(tmp_path / "plan.json", {"model": "qwen3"})
    export.write(tmp_path / "status.json", {"state": "training", "last_completed_step": 100})
    spec = {"training_root": str(tmp_path), "training_plan_sha256": export.digest(tmp_path / "plan.json")}
    with pytest.raises(ValueError, match="finish successfully"):
        export.checkpoint_identity(spec)


def test_tied_er_embeddings_do_not_require_duplicate_lm_head_storage():
    embedding = {"shape": [100, 16], "dtype": "BF16"}
    reference = {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding}
    actual = {"model.embed_tokens.weight": embedding}
    export.validate_inventory(actual, reference, tied_embeddings=True)
    with pytest.raises(ValueError, match="architecture"):
        export.validate_inventory(actual, reference, tied_embeddings=False)


def test_tied_embedding_exception_does_not_hide_other_missing_weights():
    reference = {"model.embed_tokens.weight": {"shape": [100, 16], "dtype": "BF16"},
                 "model.norm.weight": {"shape": [16], "dtype": "BF16"}}
    with pytest.raises(ValueError, match="architecture"):
        export.validate_inventory({"model.embed_tokens.weight": reference["model.embed_tokens.weight"]},
                                  reference, tied_embeddings=True)
