# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fold DSV4 compressor state block tables the way V1 metadata does."""

import math

import torch

from vllm_ascend.models.deepseek_v4.compressor_state_window import (
    fold_compressor_state_window,
)

# Same MTP decode that faulted the nightly compressor kernel: 27697 computed
# tokens and 2 scheduled tokens (one of them speculative).
_LONG_MTP_START = 27697
_LONG_MTP_QUERY = 2
_MAX_QUERY_TOKENS = 2048


def _compress_count(start: int, query: int, compress_ratio: int) -> int:
    return (start + query + compress_ratio - 1) // compress_ratio - start // compress_ratio


def _fold(block_table, start_pos, sliding_window, block_size, compress_ratio, max_query_tokens=_MAX_QUERY_TOKENS):
    return fold_compressor_state_window(
        block_table,
        start_pos,
        sliding_window=sliding_window,
        block_size=block_size,
        compress_ratio=compress_ratio,
        max_query_tokens=max_query_tokens,
    )


def test_long_mtp_decode_uses_window_physical_block():
    cases = (
        # c4 overlap window, A3/A5 state page.
        {"sliding_window": 8, "block_size": 8, "compress_ratio": 4},
        # c128, compressed / A5 BF16 state page.
        {"sliding_window": 128, "block_size": 16, "compress_ratio": 128},
        # c128, non-compressed A3 state page.
        {"sliding_window": 128, "block_size": 32, "compress_ratio": 128},
    )
    for case in cases:
        block_size = case["block_size"]
        absolute_column = _LONG_MTP_START // block_size
        block_table = torch.arange(absolute_column + 64, dtype=torch.int32).unsqueeze(0) + 3
        start_pos = torch.tensor([_LONG_MTP_START], dtype=torch.int32)
        original_table = block_table.clone()
        original_start = start_pos.clone()

        folded_table, folded_start = _fold(block_table, start_pos, **case)

        folded_pos = int(folded_start[0])
        folded_column = folded_pos // block_size
        assert folded_table.shape[1] < block_table.shape[1]
        assert 0 <= folded_column < folded_table.shape[1]
        assert int(folded_table[0, folded_column]) == int(block_table[0, absolute_column])
        assert int(folded_table[0, folded_column]) != int(block_table[0, 0])
        assert folded_pos % block_size == _LONG_MTP_START % block_size
        assert _compress_count(folded_pos, _LONG_MTP_QUERY, case["compress_ratio"]) == _compress_count(
            _LONG_MTP_START, _LONG_MTP_QUERY, case["compress_ratio"]
        )
        align = math.lcm(block_size, case["compress_ratio"])
        assert folded_pos < case["sliding_window"] + align
        # Overlap reads almost one extra ratio before start_pos. That column
        # has to exist in the compact table.
        read_start = max((folded_pos // case["compress_ratio"]) * case["compress_ratio"] - case["compress_ratio"], 0)
        assert read_start // block_size < folded_table.shape[1]
        assert torch.equal(block_table, original_table)
        assert torch.equal(start_pos, original_start)


def test_short_sequence_on_a_wide_table_keeps_the_prefix_block():
    block_table = torch.arange(4000, dtype=torch.int32).unsqueeze(0) + 5
    start_pos = torch.tensor([5], dtype=torch.int32)
    folded_table, folded_start = _fold(
        block_table,
        start_pos,
        sliding_window=8,
        block_size=8,
        compress_ratio=4,
    )
    assert int(folded_start[0]) == 5
    assert int(folded_table[0, 0]) == int(block_table[0, 0])
    assert folded_table.shape[1] < block_table.shape[1]


def test_long_mtp_decode_folds_even_when_query_budget_covers_the_table():
    block_size = 8
    absolute_column = _LONG_MTP_START // block_size
    block_table = torch.arange(absolute_column + 8, dtype=torch.int32).unsqueeze(0) + 3
    start_pos = torch.tensor([_LONG_MTP_START], dtype=torch.int32)
    folded_table, folded_start = _fold(
        block_table,
        start_pos,
        sliding_window=8,
        block_size=block_size,
        compress_ratio=4,
        max_query_tokens=10**7,
    )
    folded_column = int(folded_start[0]) // block_size
    assert folded_column < folded_table.shape[1]
    assert int(folded_table[0, folded_column]) == int(block_table[0, absolute_column])
    assert int(folded_start[0]) != _LONG_MTP_START


def test_already_windowed_table_is_unchanged():
    block_table = torch.tensor([[7, 8, 9]], dtype=torch.int32)
    start_pos = torch.tensor([3], dtype=torch.int32)
    folded_table, folded_start = _fold(
        block_table,
        start_pos,
        sliding_window=8,
        block_size=8,
        compress_ratio=4,
        max_query_tokens=16,
    )
    assert folded_table is block_table
    assert folded_start is start_pos


def test_requests_fold_independently():
    block_size = 8
    width = 4000
    starts = torch.tensor([4, _LONG_MTP_START], dtype=torch.int32)
    block_table = torch.arange(width, dtype=torch.int32).unsqueeze(0).expand(2, -1).contiguous() + 1
    folded_table, folded_start = _fold(
        block_table,
        starts,
        sliding_window=8,
        block_size=block_size,
        compress_ratio=4,
    )
    assert int(folded_start[0]) == 4
    assert int(folded_table[0, 0]) == int(block_table[0, 0])
    long_column = int(folded_start[1]) // block_size
    absolute_column = _LONG_MTP_START // block_size
    assert int(folded_table[1, long_column]) == int(block_table[1, absolute_column])
