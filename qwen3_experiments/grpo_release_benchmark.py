"""Prepare and run a veRL 0.9.1 GRPO step against a recorded earlier batch."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import runpy
import socket
import sys
from pathlib import Path

from qwen3_experiments.grpo_graph_benchmark import write_json


def _first_loader_batch(dataset, config, collate_fn=None):
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.trainer.ppo.utils import create_rl_sampler

    loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=create_rl_sampler(config, dataset),
    )
    return next(iter(loader))


def replay_source_indices(indices, config):
    """Lay out a replay subset so the actual shuffled loader yields `indices`."""
    assert len(indices) == config.train_batch_size
    assert len(set(indices)) == len(indices)
    permutation = _first_loader_batch(range(len(indices)), config).tolist()
    assert sorted(permutation) == list(range(len(indices)))
    positions = [None] * len(indices)
    for position, source_index in zip(permutation, indices, strict=True):
        positions[position] = source_index
    return positions, permutation


def prepare(plan, variant):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    runtime = Path(plan["runtime"])
    folder = Path(plan["scratch"]) / variant["name"]
    folder.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(Path(plan["dataset"]).read_bytes()).hexdigest() == plan["dataset_sha256"]
    with initialize_config_dir(str(runtime / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(config, False)
    recipe = OmegaConf.load(runtime / "qwen3_experiments/grpo_verl091_benchmark.yaml")
    config = OmegaConf.merge(config, recipe)
    paths = {
        "data.val_files": plan["validation_dataset"],
        "actor_rollout_ref.model.path": plan["model"],
        "reward.custom_reward_function.path": str(runtime / "qwen3_experiments/lcb_verl_reward.py"),
        "reward.custom_reward_function.reward_kwargs.grading_plan": plan["grading_plan"],
        "trainer.rollout_data_dir": str(folder / "rollouts"),
        "trainer.default_local_dir": str(folder / "checkpoints"),
        "ray_kwargs.ray_init._temp_dir": plan["ray_dir"],
        "ray_kwargs.ray_init.runtime_env.env_vars": variant["environment"],
        "actor_rollout_ref.rollout.agent.custom_async_server.path":
            str(runtime / "qwen3_experiments/grpo_release_server.py"),
        "actor_rollout_ref.rollout.agent.custom_async_server.name": "BenchmarkvLLMHttpServer",
    }
    for key, value in paths.items():
        OmegaConf.update(config, key, value, force_add=True)
    tokenizer = AutoTokenizer.from_pretrained(plan["model"])
    indices = plan["first_batch_indices"]
    assert len(indices) == config.data.train_batch_size == 32
    assert len(set(indices)) == 32
    source = pq.read_table(plan["dataset"])
    assert len(source) == plan["dataset_rows"]
    rows = source.take(pa.array(indices)).to_pylist()

    # StatefulDataLoader can advance its sampler differently across torchdata
    # releases. Replaying a 32-row subset makes membership independent of that
    # behavior; invert the actual loader permutation to preserve prompt order.
    positions, permutation = replay_source_indices(indices, config.data)
    batch_path = folder / "benchmark_batch.parquet"
    pq.write_table(source.take(pa.array(positions)), batch_path)
    OmegaConf.update(config, "data.train_files", str(batch_path))
    dataset = RLHFDataset([str(batch_path)], tokenizer, config.data)
    actual_batch = _first_loader_batch(dataset, config.data, collate_fn=collate_fn)
    actual_prompts = [list(messages) for messages in actual_batch["raw_prompt"]]
    assert actual_prompts == [row["prompt"] for row in rows]

    # Keep original prompt text as well as token IDs to detect template drift.
    prompt_tokens = [tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                                   enable_thinking=True, return_dict=False) for row in rows]
    write_json(folder / "batch_receipt.json", {
        "indices": indices,
        "prompt_token_sha256": hashlib.sha256(json.dumps(prompt_tokens).encode()).hexdigest(),
        "prompt_lengths": [len(ids) for ids in prompt_tokens],
        "prompts": [row["prompt"] for row in rows],
        "torch": torch.__version__,
        "dataset_rows": len(dataset),
        "source_dataset_rows": len(source),
        "subset_source_indices": positions,
        "actual_loader_permutation": permutation,
        "actual_loader_checked": True,
        "benchmark_batch_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
    })
    OmegaConf.save(config, folder / "resolved_config.yaml", resolve=True)
    print(f"Prepared {len(indices)} matching prompts; config: {folder / 'resolved_config.yaml'}", flush=True)


def train_entry(plan, variant):
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    folder = Path(plan["scratch"]) / variant["name"]
    packages = ("verl", "vllm", "torch", "transformers", "flash-attn", "ray", "tensordict", "transferqueue")
    versions = {p: importlib.metadata.version(p) for p in packages}
    assert versions["verl"] == "0.9.1"
    assert versions["vllm"] == "0.24.0"
    write_json(folder / "process.json", {
        "pid": os.getpid(), "node": socket.gethostname(), "python": sys.executable,
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"), "packages": versions,
        "mrv2_requested": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
    })
    sys.argv = ["verl.trainer.main_ppo", "--config-path", str(folder), "--config-name", "resolved_config"]
    runpy.run_module("verl.trainer.main_ppo", run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-variant")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if socket.gethostname() != plan["node"]:
        raise RuntimeError("Run this benchmark on its assigned compute node")
    if args.prepare:
        for variant in plan["variants"]:
            prepare(plan, variant)
    elif args.train_variant:
        variant = next(v for v in plan["variants"] if v["name"] == args.train_variant)
        train_entry(plan, variant)
    else:
        from qwen3_experiments.grpo_graph_benchmark import main as run_controller

        run_controller()


if __name__ == "__main__":
    main()
