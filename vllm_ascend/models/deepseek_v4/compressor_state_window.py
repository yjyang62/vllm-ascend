# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Window-local block table for the DSV4 compressor state cache."""

import math

import torch

# CONTINUOUS cache mode indexes ``curSeqIdx / block_size`` and the overlap
# path also reads one compress window before ``start_pos``. Keep one extra
# column so the last in-block token of a long prefill still lands in range.
_STATE_WINDOW_COLUMN_MARGIN = 1


def compressor_state_window_num_columns(
    sliding_window: int,
    block_size: int,
    compress_ratio: int,
    max_query_tokens: int,
) -> int:
    """Columns a folded state block table needs for one request."""
    align = math.lcm(int(block_size), int(compress_ratio))
    span = int(sliding_window) + align + int(max_query_tokens)
    return (span + int(block_size) - 1) // int(block_size) + _STATE_WINDOW_COLUMN_MARGIN


def fold_compressor_state_window(
    block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    sliding_window: int,
    block_size: int,
    compress_ratio: int,
    max_query_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather sliding-window physical blocks and rebase ``start_pos``.

    ``torch.ops._C_ascend.compressor`` with ``cache_mode=1`` (CONTINUOUS)
    reads ``state_block_table[batch, start_pos / block_size]`` and does not
    check the column. Worker tables keep absolute columns, so a long MTP
    decode indexes a full-length block id. Copy the live window into a
    compact table and subtract a block-aligned origin.

    The origin is a multiple of both ``block_size`` and ``compress_ratio``,
    so the in-block offset and the compressed-token count match the absolute
    addresses. A table that is already no wider than the window is returned
    unchanged, which is what a V1 metadata row looks like after the same fold.
    ``start_pos`` and ``block_table`` are not written.
    """
    if block_table.ndim != 2 or start_pos.ndim != 1 or start_pos.numel() == 0 or block_size <= 0 or compress_ratio <= 0:
        return block_table, start_pos

    full_cols = int(block_table.shape[1])
    # A row this short cannot hold an absolute column past the window, which
    # is the V1 metadata shape. Wider rows are the full-length V2 table.
    window_cols = compressor_state_window_num_columns(
        sliding_window,
        block_size,
        compress_ratio,
        max_query_tokens=0,
    )
    if full_cols <= window_cols:
        return block_table, start_pos

    compact_cols = compressor_state_window_num_columns(
        sliding_window,
        block_size,
        compress_ratio,
        max_query_tokens,
    )
    n_cols = min(compact_cols, full_cols)
    num_reqs = int(start_pos.shape[0])
    table = block_table[:num_reqs]
    align = math.lcm(int(block_size), int(compress_ratio))
    min_pos = (start_pos - int(sliding_window)).clamp(min=0)
    origin = (min_pos // align) * align
    folded_start = start_pos - origin
    src_cols = (origin.to(device=table.device).unsqueeze(1) // int(block_size)).to(torch.long) + torch.arange(
        n_cols,
        device=table.device,
        dtype=torch.long,
    )
    valid = src_cols < full_cols
    gathered = torch.gather(table, 1, src_cols.clamp(max=full_cols - 1))
    folded_table = gathered.masked_fill(~valid, 0)
    return folded_table, folded_start
