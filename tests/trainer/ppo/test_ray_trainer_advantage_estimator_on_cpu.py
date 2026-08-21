# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role


def test_efficient_reasoning_cost_estimator_does_not_require_critic(monkeypatch):
    monkeypatch.setattr(RayPPOTrainer, "_validate_config", lambda self: None)
    monkeypatch.setattr(RayPPOTrainer, "_create_dataloader", lambda self, *args: None)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {"lora_rank": 0},
            },
            "algorithm": {
                "adv_estimator": (
                    AdvantageEstimator.FIXED_N_RB_EFFICIENT_REASONING_COST_MARGINRL
                ),
                "use_kl_in_reward": False,
            },
        }
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=None,
        role_worker_mapping={Role.ActorRollout: object},
        resource_pool_manager=None,
    )

    assert trainer.use_critic is False
