"""Run one isolated TailRL text-maze arm with the existing MaxRL RB estimator."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=[4, 5])
    parser.add_argument("--ckpt-step", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-rollouts", type=int, default=16)
    parser.add_argument("--val-n", type=int, default=128)
    parser.add_argument("--name", default="pilot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.experiment = args.experiment.resolve()
    args.state_dir = args.state_dir.resolve()
    name = f"{args.name}_rb_ck{args.ckpt_step}_seed{args.seed}"
    run_dir = args.state_dir / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).strip()
    for key in ("RAY_ADDRESS", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "TRANSFORMERS_CACHE"):
        os.environ.pop(key, None)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "PYTHONPATH": str(args.experiment),
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": str(args.seed),
        "TAILRL_RB_SEED": str(args.seed),
        "TAILRL_RB_METRICS_FILE": str(run_dir / "metrics.jsonl"),
        "HF_HOME": str(args.state_dir / "cache/hf"),
        "TMPDIR": str(args.state_dir / "tmp"),
        "WANDB_MODE": "disabled", "RAY_USAGE_STATS_ENABLED": "0",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    sys.path.insert(0, str(args.experiment))
    os.chdir(args.experiment)
    import hydra
    from omegaconf import OmegaConf

    batch_size = 4 if args.smoke else args.batch_size
    val_file = args.state_dir / "data/pilot_eval.parquet"
    if args.smoke:
        import pandas as pd
        val_file = run_dir / "smoke_eval.parquet"
        pd.read_parquet(args.state_dir / "data/pilot_eval.parquet").iloc[:4].to_parquet(val_file)
    steps = 2 if args.smoke else args.steps
    overrides = [
        "algorithm.adv_estimator=fixed_n_rb_cost_aware_marginrl",
        "algorithm.use_kl_in_reward=False", "algorithm.kl_ctrl.kl_coef=0.0",
        "algorithm.reward_transform=raw", f"+data.seed={args.seed}",
        f"data.train_files={args.state_dir / 'data/pilot_train.parquet'}", f"data.val_files={val_file}",
        f"data.train_batch_size={batch_size}", "data.val_batch_size=16",
        "+data.dataloader_num_workers=2", "data.validation_shuffle=False",
        "data.max_prompt_length=320", "data.max_response_length=180", "data.apply_chat_template=False",
        f"actor_rollout_ref.model.path={args.state_dir / 'checkpoints' / f'ckpt-{args.ckpt_step}'}",
        "actor_rollout_ref.model.attn_implementation=sdpa",
        "actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "actor_rollout_ref.actor.optim.lr=1e-4", "actor_rollout_ref.actor.dtype=float16",
        "actor_rollout_ref.actor.use_kl_loss=False", "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={batch_size}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]",
        "actor_rollout_ref.rollout.name=hf", "actor_rollout_ref.rollout.dtype=float16",
        "+actor_rollout_ref.rollout.micro_batch_size=1024",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.temperature=1.0", "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.top_k=-1", f"actor_rollout_ref.rollout.n={args.n_rollouts}",
        "+actor_rollout_ref.rollout.extra_eos_token_ids=[7]",
        f"actor_rollout_ref.rollout.val_kwargs.n={4 if args.smoke else args.val_n}",
        "actor_rollout_ref.rollout.val_kwargs.gen_batch_size=64",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        "actor_rollout_ref.rollout.val_kwargs.temperature=1.0",
        "reward_model.reward_manager=prime", "+reward_model.reward_kwargs.num_processes=2",
        "+reward_model.reward_kwargs.chunksize=64",
        f"custom_reward_function.path={args.experiment / 'src/maze_binary_goal_cost_reward.py'}",
        "custom_reward_function.name=compute_score",
        "trainer.project_name=tailrl-maze-per-context-rb", f"trainer.experiment_name={name}",
        "trainer.logger=[console]", "trainer.val_before_train=True",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1",
        f"trainer.save_freq={steps}", f"trainer.test_freq={steps}",
        f"trainer.total_training_steps={steps}", "trainer.total_epochs=100",
        "trainer.max_actor_ckpt_to_keep=2", f"trainer.default_local_dir={run_dir / 'checkpoints'}",
        f"trainer.validation_data_dir={run_dir / 'validation'}",
    ]
    with hydra.initialize_config_dir(config_dir=str(args.experiment / "verl/trainer/config"), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    OmegaConf.save(config, run_dir / "config.yaml", resolve=True)
    (run_dir / "launch.json").write_text(json.dumps({
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu_uuid": gpu_uuid, "python": sys.executable,
        "overrides": overrides,
    }, indent=2) + "\n")
    if args.dry_run:
        print(f"Resolved config: {run_dir / 'config.yaml'}")
        return
    if (run_dir / "COMPLETE.json").exists():
        print(f"Already complete: {run_dir}")
        return
    import ray

    from verl.trainer.main_ppo import run_ppo

    # A new Ray instance owns only this process's one visible GPU. Do not
    # attach to or stop any existing cluster on this shared training host.
    ray_dir = tempfile.mkdtemp(prefix=f"rb{args.gpu}_", dir=str(args.state_dir.parent))
    try:
        ray.init(address="local", num_cpus=8, num_gpus=1, include_dashboard=False,
                 object_store_memory=512 * 1024**2, _temp_dir=ray_dir)
        run_ppo(config)
        latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        actual_step = int(latest.read_text().strip()) if latest.exists() else None
        if actual_step != steps:
            raise RuntimeError(f"Expected completed step {steps}, found {actual_step}")
        (run_dir / "COMPLETE.json").write_text(json.dumps({"step": actual_step, "gpu": args.gpu}) + "\n")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
