"""Install a reviewable adapter into a dedicated, pinned TailRL checkout."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

TAILRL_REVISION = "5682c6ac03387355e017ce966693266bb148fa10"
FUNCTION = "compute_fixed_n_rb_cost_aware_marginrl_outcome_advantage"


def replace_once(text, old, new):
    if new in text:
        return text
    if text.count(old) != 1:
        raise ValueError(f"Expected one adapter insertion point: {old[:100]!r}")
    return text.replace(old, new, 1)


def install(checkout, maxrl_root):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != TAILRL_REVISION:
        raise ValueError(f"Expected TailRL {TAILRL_REVISION}, found {revision}")
    experiment = checkout / "experiments/text_maze"
    source_file = maxrl_root / "verl/trainer/ppo/core_algos.py"
    source = source_file.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    function = ast.get_source_segment(source, node)
    core_path = experiment / "verl/trainer/ppo/core_algos.py"
    core = core_path.read_text()
    core = replace_once(core, '    TAILRL = "tailrl"\n',
                        '    TAILRL = "tailrl"\n    FIXED_N_RB_COST_AWARE_MARGINRL = "fixed_n_rb_cost_aware_marginrl"\n')
    marker = "\n\n# BEGIN MAXRL TEXT-MAZE RB PORT\n"
    if marker in core:
        core = core.split(marker)[0]
    core += (marker + "# Exact function body copied from MaxRL; see rb_port_manifest.json.\n"
             + "@register_adv_est(AdvantageEstimator.FIXED_N_RB_COST_AWARE_MARGINRL)\n" + function + "\n")
    core_path.write_text(core)

    trainer_path = experiment / "verl/trainer/ppo/ray_trainer.py"
    trainer = trainer_path.read_text()
    trainer = replace_once(
        trainer,
        "    else:\n        # handle all other adv estimator type other than GAE, GRPO and SFT",
        "    elif adv_estimator == AdvantageEstimator.FIXED_N_RB_COST_AWARE_MARGINRL:\n"
        "        from verl.trainer.ppo.maze_rb_adapter import apply_maze_rb_advantage\n"
        "        return apply_maze_rb_advantage(data, num_repeat)\n\n"
        "    else:\n        # handle all other adv estimator type other than GAE, GRPO and SFT",
    )
    trainer = replace_once(trainer, "            AdvantageEstimator.TAILRL,\n",
                           "            AdvantageEstimator.TAILRL,\n            AdvantageEstimator.FIXED_N_RB_COST_AWARE_MARGINRL,\n")
    trainer = replace_once(trainer, "                        try:\n                            adv_index =",
                           '                        metrics.update(batch.meta_info.get("maze_rb_metrics", {}))\n\n'
                           "                        try:\n                            adv_index =")
    trainer_path.write_text(trainer)

    tracking_path = experiment / "verl/utils/tracking.py"
    tracking = tracking_path.read_text()
    tracking = replace_once(tracking, "    def log(self, data, step, backend=None):\n",
                            "    def log(self, data, step, backend=None):\n"
                            "        from verl.trainer.ppo.maze_rb_adapter import append_metrics\n"
                            "        append_metrics(data, step)\n")
    tracking_path.write_text(tracking)

    worker_path = experiment / "verl/workers/fsdp_workers.py"
    worker = worker_path.read_text()
    worker = replace_once(worker,
        "    def init_model(self):\n        from verl.workers.actor import DataParallelPPOActor",
        "    def init_model(self):\n"
        "        from transformers import set_seed\n"
        "        set_seed(int(os.environ.get('TAILRL_RB_SEED', '0')))\n"
        "        from verl.workers.actor import DataParallelPPOActor")
    worker_path.write_text(worker)

    main_path = experiment / "verl/trainer/main_ppo.py"
    main_text = main_path.read_text()
    main_text = replace_once(main_text,
        "    def run(self, config):\n        # Print the initial configuration.",
        "    def run(self, config):\n"
        "        from transformers import set_seed\n"
        "        set_seed(int(config.data.get('seed', 0)))\n"
        "        # Print the initial configuration.")
    main_path.write_text(main_text)

    actor_path = experiment / "verl/workers/actor/dp_actor.py"
    actor = actor_path.read_text()
    actor = replace_once(actor,
        "if is_cuda_available:\n    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input",
        "if is_cuda_available:\n"
        "    try:\n"
        "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
        "    except ImportError:\n"
        "        index_first_axis = pad_input = rearrange = unpad_input = None")
    actor = replace_once(actor,
        '        self.use_remove_padding = self.config.get("use_remove_padding", False)\n',
        '        self.use_remove_padding = self.config.get("use_remove_padding", False)\n'
        '        if self.use_remove_padding and is_cuda_available and unpad_input is None:\n'
        '            raise ImportError("flash_attn is required when use_remove_padding=True")\n')
    actor_path.write_text(actor)

    local_dir = Path(__file__).resolve().parent
    shutil.copy2(local_dir / "maze_rb_adapter.py", experiment / "verl/trainer/ppo/maze_rb_adapter.py")
    shutil.copy2(local_dir / "maze_binary_goal_cost_reward.py", experiment / "src/maze_binary_goal_cost_reward.py")
    manifest = {
        "tailrl_revision": revision,
        "maxrl_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=maxrl_root, text=True).strip(),
        "source_function": FUNCTION,
        "source_function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        "reward": "original binary goal reach with DONE required",
        "cost": "max(generated action-prefix length, BFS shortest distance)",
    }
    (experiment / "rb_port_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--maxrl-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    install(args.checkout.resolve(), args.maxrl_root.resolve())
