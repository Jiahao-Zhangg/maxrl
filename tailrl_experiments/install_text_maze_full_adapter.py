"""Install RB and preserve global token-mean scaling across the four GPU shards."""

import argparse
import json
from pathlib import Path

from install_text_maze_adapter import install, replace_once


def install_full(checkout, maxrl_root):
    install(checkout, maxrl_root)
    experiment = checkout / "experiments/text_maze"
    adapter = experiment / "verl/trainer/ppo/maze_rb_adapter.py"
    source = adapter.read_text()
    source = replace_once(
        source,
        '    data.batch["advantages"] = advantages\n',
        '    data.meta_info["rb_global_response_tokens"] = int(data.batch["response_mask"].sum().item())\n'
        '    data.batch["advantages"] = advantages\n',
    )
    adapter.write_text(source)
    actor = experiment / "verl/workers/actor/dp_actor.py"
    source = actor.read_text()
    source = replace_once(
        source,
        "        self.actor_module.train()\n\n"
        '        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error\n',
        "        self.actor_module.train()\n\n"
        "        rb_global_tokens = None\n"
        '        if self.config.get("rb_global_token_mean", False):\n'
        '            rb_global_tokens = data.meta_info["rb_global_response_tokens"]\n'
        '            assert rb_global_tokens > 0 and self.config.loss_agg_mode == "token-mean"\n'
        "            assert data.batch.batch_size[0] == self.config.ppo_mini_batch_size\n"
        '        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error\n',
    )
    source = replace_once(
        source,
        "                            loss = policy_loss / self.gradient_accumulation\n",
        "                            loss = policy_loss / self.gradient_accumulation\n"
        "                        if rb_global_tokens is not None:\n"
        "                            # FSDP averages rank gradients. Weight each local token sum\n"
        "                            # so their average equals the original full-batch token mean.\n"
        "                            loss = policy_loss * world_size * response_mask.sum() / rb_global_tokens\n",
    )
    actor.write_text(source)
    manifest_path = experiment / "rb_port_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["full_config_adapter"] = {
        "global_token_mean": "rank loss multiplied by world_size * local_tokens / global_tokens",
        "estimator_function_unchanged": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--maxrl-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    install_full(args.checkout.resolve(), args.maxrl_root.resolve())
