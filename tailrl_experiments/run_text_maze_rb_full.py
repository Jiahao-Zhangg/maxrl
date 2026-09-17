"""Run one full RB x released SFT checkpoint x seed/sample-count arm on four GPUs."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

SFT_STEPS = (2450, 3000, 3250, 3350, 3400, 3450, 3550)
CHECKPOINT_ORDER = (3000, *(step for step in SFT_STEPS if step != 3000))


def run_name(checkpoint_step, seed=0, n_rollouts=16):
    if checkpoint_step not in SFT_STEPS:
        raise ValueError(f"Unknown released SFT checkpoint: {checkpoint_step}")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if n_rollouts not in (16, 32):
        raise ValueError("This full-run launcher supports n_rollouts=16 or 32")
    return f"full_rb_ck{checkpoint_step}_seed{seed}_bs256_n{n_rollouts}_4gpu"


NAME = run_name(3000)


def identity(checkpoint_step, smoke=False, seed=0, n_rollouts=16):
    run_name(checkpoint_step, seed, n_rollouts)
    return {"checkpoint_step": checkpoint_step, "seed": seed, "n_rollouts": n_rollouts, "world_size": 4, "smoke": smoke}


def matches_identity(record, expected):
    # Older launch/completion records predate this option and always used N=16.
    return all(record.get(key, 16 if key == "n_rollouts" else None) == value for key, value in expected.items())


def is_complete(run_dir, checkpoint_step, smoke=False, seed=0, n_rollouts=16):
    path = run_dir / "COMPLETE.json"
    if not path.exists():
        return False
    expected = {**identity(checkpoint_step, smoke, seed, n_rollouts), "step": 2 if smoke else 5001}
    record = json.loads(path.read_text())
    if not matches_identity(record, expected):
        raise RuntimeError(f"Completion record does not match this RB arm: {path}")
    return True


def final_checkpoint_saved(run_dir, smoke=False):
    expected = 2 if smoke else 5001
    latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    actual = int(latest.read_text().strip()) if latest.exists() else None
    if actual is None or actual < expected:
        return False
    if actual != expected:
        raise RuntimeError(f"Expected final checkpoint {expected}; found {actual}")
    if not (run_dir / f"checkpoints/global_step_{expected}/actor/huggingface/config.json").exists():
        raise RuntimeError("Final Hugging Face export is missing")
    return True


def mark_complete(run_dir, checkpoint_step, smoke=False, seed=0, n_rollouts=16):
    if not final_checkpoint_saved(run_dir, smoke):
        raise RuntimeError("Training returned before saving its final checkpoint")
    marker = run_dir / "COMPLETE.json.tmp"
    marker.write_text(json.dumps({"step": 2 if smoke else 5001, **identity(checkpoint_step, smoke, seed, n_rollouts)}) + "\n")
    marker.replace(run_dir / "COMPLETE.json")


def overrides(experiment, state, run_dir, smoke=False, checkpoint_step=3000, seed=0, n_rollouts=16):
    run_name(checkpoint_step, seed, n_rollouts)
    batch, steps, val_n = (16, 2, 4) if smoke else (256, 5001, 64)
    train = run_dir / "smoke_train.parquet" if smoke else state / "data/full_train.parquet"
    evaluation = run_dir / "smoke_eval.parquet" if smoke else state / "data/full_eval.parquet"
    return [
        "algorithm.adv_estimator=fixed_n_rb_cost_aware_marginrl",
        "algorithm.use_kl_in_reward=False",
        "algorithm.kl_ctrl.kl_coef=0.0",
        "algorithm.reward_transform=raw",
        "algorithm.pass_k=8",
        f"+data.seed={seed}",
        f"data.train_files={train}",
        f"data.val_files={evaluation}",
        f"data.train_batch_size={batch}",
        "data.max_prompt_length=320",
        "data.max_response_length=180",
        "data.apply_chat_template=False",
        f"actor_rollout_ref.model.path={state / f'checkpoints/ckpt-{checkpoint_step}'}",
        "actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.dtype=float16",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        # Keep the existing memory cap. N=32 accumulates two microbatches per
        # rank, still making one globally token-weighted update per rollout.
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={batch * 16 // 4}",
        "+actor_rollout_ref.actor.rb_global_token_mean=True",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
        "actor_rollout_ref.rollout.name=hf",
        "+actor_rollout_ref.rollout.micro_batch_size=8000",
        "actor_rollout_ref.rollout.dtype=float16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8192",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.7",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8192",
        f"actor_rollout_ref.rollout.n={n_rollouts}",
        f"actor_rollout_ref.rollout.val_kwargs.n={val_n}",
        "+actor_rollout_ref.rollout.extra_eos_token_ids=[7]",
        "actor_rollout_ref.rollout.val_kwargs.gen_batch_size=128",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "reward_model.reward_manager=prime",
        "+reward_model.reward_kwargs.num_processes=16",
        "+reward_model.reward_kwargs.chunksize=64",
        f"custom_reward_function.path={experiment / 'src/maze_binary_goal_cost_reward.py'}",
        "custom_reward_function.name=compute_score",
        "trainer.project_name=tailrl-maze-per-context-rb",
        f"trainer.experiment_name={run_dir.name}",
        "trainer.logger=[console]",
        "trainer.val_before_train=True",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        f"trainer.save_freq={2 if smoke else 250}",
        f"trainer.test_freq={2 if smoke else 1000}",
        f"trainer.total_training_steps={steps}",
        "trainer.total_epochs=100",
        "trainer.max_actor_ckpt_to_keep=3",
        f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.validation_data_dir={run_dir / 'validation'}",
        "trainer.resume_mode=auto",
        "ray_init.num_cpus=64",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", type=int, choices=SFT_STEPS, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-rollouts", "--n_rollouts", type=int, choices=(16, 32), default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiment, state = args.experiment.resolve(), args.state_dir.resolve()
    run_dir = args.output_dir.resolve() / (run_name(args.ckpt_step, args.seed, args.n_rollouts) + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    arm_identity = identity(args.ckpt_step, args.smoke, args.seed, args.n_rollouts)
    launch_path = run_dir / "launch.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        if not matches_identity(previous, arm_identity):
            raise RuntimeError(f"Existing output directory belongs to a different arm: {run_dir}")
    for key in ("RAY_ADDRESS", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "TRANSFORMERS_CACHE"):
        os.environ.pop(key, None)
    os.environ.update(
        {
            "PYTHONPATH": str(experiment),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": str(args.seed),
            "TAILRL_RB_SEED": str(args.seed),
            "TAILRL_RB_METRICS_FILE": str(run_dir / "metrics.jsonl"),
            "HF_HOME": str(state.parent / "cache/hf"),
            "WANDB_MODE": "disabled",
            "RAY_USAGE_STATS_ENABLED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    sys.path.insert(0, str(experiment))
    os.chdir(experiment)
    import hydra
    from omegaconf import OmegaConf

    if args.smoke:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pq.read_table(state / "data/full_eval.parquet").slice(0, 16), run_dir / "smoke_eval.parquet")
        first = next(pq.ParquetFile(state / "data/full_train.parquet").iter_batches(batch_size=128))
        pq.write_table(pa.Table.from_batches([first]), run_dir / "smoke_train.parquet")
    resolved_overrides = overrides(experiment, state, run_dir, args.smoke, args.ckpt_step, args.seed, args.n_rollouts)
    with hydra.initialize_config_dir(config_dir=str(experiment / "verl/trainer/config"), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=resolved_overrides)
    # A node-local Ray directory is supplied before importing or initializing Ray.
    config.ray_init.ray_dir = os.environ.get("TAILRL_RAY_TMPDIR", "/tmp/tailrl_rb_pending")
    OmegaConf.save(config, run_dir / "config.yaml", resolve=True)
    launch_path.write_text(
        json.dumps(
            {
                "python": sys.executable,
                "overrides": resolved_overrides,
                **arm_identity,
            },
            indent=2,
        )
        + "\n"
    )
    if args.dry_run:
        print(f"Resolved complete configuration: {run_dir / 'config.yaml'}")
        return
    if is_complete(run_dir, args.ckpt_step, args.smoke, args.seed, args.n_rollouts):
        print(f"Already complete: {run_dir}")
        return
    if final_checkpoint_saved(run_dir, args.smoke):
        # A driver may exit after the final save but before writing COMPLETE.json.
        # Recover that completed run without performing an extra optimizer update.
        mark_complete(run_dir, args.ckpt_step, args.smoke, args.seed, args.n_rollouts)
        print(f"Recovered final checkpoint completion: {run_dir}")
        return
    import ray
    import torch

    from verl.trainer.main_ppo import run_ppo

    if torch.cuda.device_count() != 4:
        raise RuntimeError("This arm requires exactly four allocated GPUs")
    ray_root = Path(os.environ.get("TAILRL_RAY_TMPDIR", "/tmp/tailrl_rb"))
    ray_root.mkdir(parents=True, exist_ok=True)
    ray_dir = tempfile.mkdtemp(prefix="attempt_", dir=ray_root)
    try:
        ray.init(
            address="local",
            num_cpus=64,
            num_gpus=4,
            include_dashboard=False,
            object_store_memory=2 * 1024**3,
            _temp_dir=ray_dir,
        )
        run_ppo(config)
        mark_complete(run_dir, args.ckpt_step, args.smoke, args.seed, args.n_rollouts)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
