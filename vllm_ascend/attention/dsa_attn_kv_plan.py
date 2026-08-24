# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.dsa_kv_mode import uses_explicit_bf16_kv
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """The attention-KV plan only; indexer KV remains independently FP8."""

    uses_sparse_flash_mla: bool
    uses_kv_compress_epilog: bool
    layout_kv: str
    compressor_slot_mapping_format: int
    requires_block_offset_slots: bool
    sparse_attn_op: Callable[..., Any]
    sparse_attn_metadata_op: Callable[..., Any]
    sparse_attn_base_kwargs: dict[str, Any]
    sparse_attn_metadata_kwargs: dict[str, Any]
    include_metadata_device: bool
    applies_sparse_attn_runtime_kwargs: bool

    def get_dsa_sparse_attn_metadata_op(self):
        return self.sparse_attn_metadata_op

    def get_dsa_sparse_attn_metadata_kwargs(self, device) -> dict[str, Any]:
        kwargs = dict(self.sparse_attn_metadata_kwargs)
        if self.include_metadata_device:
            kwargs["device"] = str(device)
        return kwargs

    def get_dsa_sparse_attn_op(self):
        return self.sparse_attn_op

    def get_dsa_sparse_attn_base_kwargs(self) -> dict[str, Any]:
        return dict(self.sparse_attn_base_kwargs)

    def add_dsa_sparse_attn_extra_kwargs(self, extra_kwargs: dict[str, Any], **kwargs_to_add) -> None:
        if self.applies_sparse_attn_runtime_kwargs:
            extra_kwargs.update(kwargs_to_add)

    def get_dsa_compressor_slot_mapping_format(self) -> int:
        return self.compressor_slot_mapping_format

    def format_dsa_slot_mapping(self, slot_mapping: torch.Tensor, block_size: int) -> torch.Tensor:
        if not self.requires_block_offset_slots:
            return slot_mapping
        valid = slot_mapping >= 0
        invalid = torch.full_like(slot_mapping, -1)
        block_idx = torch.where(valid, torch.div(slot_mapping, block_size, rounding_mode="floor"), invalid)
        offset = torch.where(valid, slot_mapping % block_size, invalid)
        return torch.stack([block_idx, offset], dim=-1).to(torch.int32)

    def dsa_kv_compress_scatter(self, cache: torch.Tensor, x: torch.Tensor | None, slot_mapping: torch.Tensor) -> None:
        if x is None:
            return
        if self.uses_sparse_flash_mla:
            if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
                raise ValueError(f"BF16 DSA slot_mapping must be [num_tokens, 2], got {tuple(slot_mapping.shape)}.")
            # Keep fixed [T, 2] shape under ACLGraph. SparseFlashMla's
            # scatter path receives padded [-1, -1] rows directly; do not
            # introduce a data-dependent Nonzero/gather operation here.
            indices = slot_mapping.to(torch.int64).contiguous()
            updates = x.reshape((slot_mapping.shape[0],) + tuple(cache.shape[2:])).contiguous()
            torch_npu.npu_scatter_nd_update_(cache, indices, updates)
            return
        if not self.uses_kv_compress_epilog:
            torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)
            return
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=x.view(-1, x.shape[-1]),
            slot_mapping=slot_mapping,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )


def get_dsa_attn_kv_plan(vllm_config=None) -> DsaAttnKvPlan:
    """Return the explicit A5 BF16 or upstream-compatible FP8 DSA plan."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=False,
            uses_kv_compress_epilog=False,
            layout_kv="PA_ND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv,
            sparse_attn_metadata_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )

    use_bf16 = uses_explicit_bf16_kv(vllm_config)
    if use_bf16:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=True,
            uses_kv_compress_epilog=False,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=sparse_flash_mla,
            sparse_attn_metadata_op=sparse_flash_mla_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )
    return DsaAttnKvPlan(
        uses_sparse_flash_mla=False,
        uses_kv_compress_epilog=True,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
        requires_block_offset_slots=False,
        sparse_attn_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        sparse_attn_metadata_kwargs={"kv_quant_mode": 1},
        include_metadata_device=False,
        applies_sparse_attn_runtime_kwargs=False,
    )
