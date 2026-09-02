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
"""Token-span utilities for thinking-cost and concise-answer rewards."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ThinkingEfficiencyStats:
    """Marker locations and boundary-exclusive token counts for one response."""

    thinking_tokens: int
    post_think_pre_box_tokens: int | None
    has_think_open: bool
    has_think_close: bool
    has_box_after_think: bool
    thinking_span_censored: bool


def _find_subsequence(
    token_ids: Sequence[int],
    pattern: Sequence[int],
    start: int = 0,
) -> int | None:
    if not pattern:
        raise ValueError("marker token pattern must be nonempty")
    last_start = len(token_ids) - len(pattern)
    for position in range(start, last_start + 1):
        if list(token_ids[position : position + len(pattern)]) == list(pattern):
            return position
    return None


def analyze_thinking_efficiency(
    token_ids: Sequence[int],
    tokenizer,
    think_open: str = "<think>",
    think_close: str = "</think>",
    boxed_prefix: str = "\\boxed{",
) -> ThinkingEfficiencyStats:
    """Measure thinking and post-thinking spans with marker tokens excluded.

    A missing closing marker makes the thinking span right-censored: cost is
    measured from just after ``<think>`` through the observed response end. If
    the opening marker is also missing, the full observed response is charged.
    This prevents truncated or malformed responses from receiving minimum cost.
    """

    response_ids = [int(token_id) for token_id in token_ids]
    open_ids = tokenizer.encode(think_open, add_special_tokens=False)
    close_ids = tokenizer.encode(think_close, add_special_tokens=False)
    if not open_ids or not close_ids:
        raise ValueError("thinking markers must tokenize to nonempty sequences")

    open_position = _find_subsequence(response_ids, open_ids)
    thinking_start = 0 if open_position is None else open_position + len(open_ids)
    close_position = _find_subsequence(response_ids, close_ids, start=thinking_start)

    has_open = open_position is not None
    has_close = close_position is not None
    thinking_end = len(response_ids) if close_position is None else close_position
    thinking_tokens = max(thinking_end - thinking_start, 0)

    post_think_pre_box_tokens = None
    if close_position is not None:
        post_think_ids = response_ids[close_position + len(close_ids) :]
        post_think_text = tokenizer.decode(
            post_think_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        box_character_position = post_think_text.find(boxed_prefix)
        if box_character_position >= 0:
            pre_box_text = post_think_text[:box_character_position]
            post_think_pre_box_tokens = len(
                tokenizer.encode(pre_box_text, add_special_tokens=False)
            )

    return ThinkingEfficiencyStats(
        thinking_tokens=thinking_tokens,
        post_think_pre_box_tokens=post_think_pre_box_tokens,
        has_think_open=has_open,
        has_think_close=has_close,
        has_box_after_think=post_think_pre_box_tokens is not None,
        thinking_span_censored=not (has_open and has_close),
    )


def gate_correctness_reward(
    correctness: float,
    stats: ThinkingEfficiencyStats,
    post_think_pre_box_token_limit: int,
) -> float:
    """Keep correctness iff both markers exist and the open span is short."""

    if isinstance(post_think_pre_box_token_limit, bool):
        raise ValueError("post-thinking token limit must be a positive integer")
    token_limit = int(post_think_pre_box_token_limit)
    if token_limit <= 0 or token_limit != post_think_pre_box_token_limit:
        raise ValueError("post-thinking token limit must be a positive integer")

    passes_length_gate = (
        stats.has_think_open
        and stats.has_think_close
        and stats.has_box_after_think
        and stats.post_think_pre_box_tokens is not None
        and stats.post_think_pre_box_tokens < token_limit
    )
    return float(correctness) if passes_length_gate else 0.0
