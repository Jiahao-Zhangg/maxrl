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

from verl.utils.reward_score.thinking_efficiency import (
    analyze_thinking_efficiency,
    gate_correctness_reward,
)


class CharacterTokenizer:
    """Small reversible tokenizer for boundary-exclusive span tests."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


def analyze(response: str):
    tokenizer = CharacterTokenizer()
    return analyze_thinking_efficiency(tokenizer.encode(response), tokenizer)


def test_counts_strictly_inside_both_marker_pairs():
    stats = analyze("<think>abc</think>xy\\boxed{7}")

    assert stats.thinking_tokens == 3
    assert stats.post_think_pre_box_tokens == 2
    assert stats.has_think_open
    assert stats.has_think_close
    assert stats.has_box_after_think
    assert not stats.thinking_span_censored


def test_post_think_reward_gate_is_strict_at_512_tokens():
    passes = analyze(f"<think>x</think>{'a' * 511}\\boxed{{1}}")
    fails = analyze(f"<think>x</think>{'a' * 512}\\boxed{{1}}")

    assert gate_correctness_reward(1.0, passes, 512) == 1.0
    assert gate_correctness_reward(0.0, passes, 512) == 0.0
    assert gate_correctness_reward(1.0, fails, 512) == 0.0


def test_missing_close_is_charged_through_response_end_and_cannot_reward():
    stats = analyze("<think>unfinished reasoning")

    assert stats.thinking_tokens == len("unfinished reasoning")
    assert stats.post_think_pre_box_tokens is None
    assert stats.thinking_span_censored
    assert gate_correctness_reward(1.0, stats, 512) == 0.0


def test_missing_open_charges_observed_prefix_and_cannot_reward():
    stats = analyze("reasoning</think>short\\boxed{1}")

    assert stats.thinking_tokens == len("reasoning")
    assert stats.post_think_pre_box_tokens == len("short")
    assert stats.thinking_span_censored
    assert gate_correctness_reward(1.0, stats, 512) == 0.0


def test_only_first_box_after_think_is_used():
    stats = analyze("<think>x</think>abc\\boxed{wrong}later\\boxed{right}")

    assert stats.post_think_pre_box_tokens == 3
