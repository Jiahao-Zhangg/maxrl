# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    append_shortest_rollout_record,
    build_shortest_rollout_record,
)


class _TokenIdTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(str(token_id) for token_id in token_ids)


def _make_training_batch() -> DataProto:
    prompts = torch.tensor(
        [
            [0, 11, 12],
            [0, 21, 22],
            [0, 31, 32],
        ]
    )
    responses = torch.tensor(
        [
            [101, 102, 103, 0],
            [201, 0, 0, 0],
            [301, 0, 0, 0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    attention_mask = torch.cat(
        (
            torch.tensor([[0, 1, 1], [0, 1, 1], [0, 1, 1]]),
            response_mask,
        ),
        dim=1,
    )
    token_level_scores = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "token_level_scores": token_level_scores,
            },
            batch_size=[3],
        ),
        non_tensor_batch={"uid": np.array(["prompt-a", "prompt-b", "prompt-c"])},
    )


def test_build_shortest_rollout_record_selects_first_global_minimum():
    record = build_shortest_rollout_record(
        data=_make_training_batch(),
        tokenizer=_TokenIdTokenizer(),
        global_step=17,
    )

    assert record == {
        "global_step": 17,
        "batch_position": 1,
        "prompt_uid": "prompt-b",
        "prompt": "21 22",
        "response": "201",
        "response_tokens": 1,
        "reward": 0.0,
        "correct": False,
    }


def test_append_shortest_rollout_record_writes_jsonl(tmp_path):
    filename = tmp_path / "debug" / "shortest_rollouts.jsonl"
    first_record = {"global_step": 1, "prompt": "first"}
    second_record = {"global_step": 2, "prompt": "second"}

    append_shortest_rollout_record(str(filename), first_record)
    append_shortest_rollout_record(str(filename), second_record)

    records = [json.loads(line) for line in filename.read_text().splitlines()]
    assert records == [first_record, second_record]
