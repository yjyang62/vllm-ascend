# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Centralized DSA attention-KV execution plan (SparseFlashMla vs sharedkv)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.ops.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """DSA attention-KV execution plan (not indexer KV).

    Owns SparseFlashMla vs quant/sharedkv choices so callers do not scatter
    ``attn_kv_dtype == torch.bfloat16`` checks.
    """

    attn_kv_dtype: torch.dtype | None
    uses_sparse_flash_mla: bool
    layout_kv: str
    compressor_slot_mapping_format: int
    requires_block_offset_slots: bool
    pack_kv_head_dim_extra: bool
    sparse_attn_op: Callable[..., Any]
    sparse_attn_metadata_op: Callable[..., Any]
    sparse_attn_base_kwargs: dict[str, Any]
    sparse_attn_metadata_extra_kwargs: dict[str, Any]
    # A2/A3 sharedkv and SparseFlashMla need device=; A5 FP8 quant metadata does not.
    include_metadata_device: bool

    def metadata_kwargs(self, device) -> dict[str, Any]:
        kwargs = dict(self.sparse_attn_metadata_extra_kwargs)
        if self.include_metadata_device:
            kwargs["device"] = str(device)
        return kwargs


def uses_bf16_sparse_flash_mla(kv_cache_dtype: torch.dtype | None) -> bool:
    return kv_cache_dtype == torch.bfloat16


def build_sparse_flash_mla_plan(kv_cache_dtype: torch.dtype | None = None) -> DsaAttnKvPlan:
    """BF16 SparseFlashMla plan shared by A3 and A5."""
    return DsaAttnKvPlan(
        attn_kv_dtype=kv_cache_dtype,
        uses_sparse_flash_mla=True,
        layout_kv="PA_BBND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
        requires_block_offset_slots=True,
        pack_kv_head_dim_extra=False,
        sparse_attn_op=sparse_flash_mla,
        sparse_attn_metadata_op=sparse_flash_mla_metadata,
        sparse_attn_base_kwargs={},
        sparse_attn_metadata_extra_kwargs={},
        include_metadata_device=True,
    )


def build_base_dsa_attn_kv_plan(kv_cache_dtype: torch.dtype | None = None) -> DsaAttnKvPlan:
    """A2/A3 DSA plan.

    A3 BF16 uses SparseFlashMla (same op path as A5 BF16). A2 and non-BF16 A3
    stay on sharedkv + PA_ND + block/offset slots.
    """
    if get_ascend_device_type() == AscendDeviceType.A3 and uses_bf16_sparse_flash_mla(kv_cache_dtype):
        return build_sparse_flash_mla_plan(kv_cache_dtype)
    return DsaAttnKvPlan(
        attn_kv_dtype=kv_cache_dtype,
        uses_sparse_flash_mla=False,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
        requires_block_offset_slots=True,
        pack_kv_head_dim_extra=False,
        sparse_attn_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={},
        sparse_attn_metadata_extra_kwargs={},
        include_metadata_device=True,
    )


def build_a5_dsa_attn_kv_plan(kv_cache_dtype: torch.dtype | None = None) -> DsaAttnKvPlan:
    """A5: explicit BF16 → SparseFlashMla; otherwise KV-quant sharedkv (default)."""
    if uses_bf16_sparse_flash_mla(kv_cache_dtype):
        return build_sparse_flash_mla_plan(kv_cache_dtype)
    return DsaAttnKvPlan(
        attn_kv_dtype=kv_cache_dtype,
        uses_sparse_flash_mla=False,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
        requires_block_offset_slots=False,
        pack_kv_head_dim_extra=True,
        sparse_attn_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        # Match main A5: quant metadata takes kv_quant_mode only (no device=).
        sparse_attn_metadata_extra_kwargs={"kv_quant_mode": 1},
        include_metadata_device=False,
    )
