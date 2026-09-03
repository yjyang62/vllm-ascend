# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})
_AUTO_KV_CACHE_DTYPES = frozenset({"auto", "none"})
_FP8_KV_CACHE_DTYPES = frozenset({"fp8", "fp8_e4m3", "fp8_e4m3fn", "float8_e4m3fn", "fp8_ds_mla"})
_NO_QUANT_METHODS = frozenset({"", "none"})
# DSV4 A5 FP8 KV is a custom MXFP8 packed page (rope + nope + e8m0 + pad),
# not vLLM's generic ``--kv-cache-dtype fp8`` tensor. Keep the engine string
# as ``auto`` so Worker / _init_kv_cache_quant do not install generic FP8 KV.
_CUSTOM_MXFP8_CACHE_DTYPE = "auto"


def _normalize_dtype_name(dtype) -> str:
    return str(dtype).lower().removeprefix("torch.")


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str, quant_method: str | None = None) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5 an explicit cache dtype always wins. ``auto`` selects BF16 for an
    unquantized BF16 checkpoint and keeps FP8-quantized checkpoints on the
    existing MXFP8 KV path. Explicit ``fp8`` must stay on that same custom
    path: writing ``fp8`` into ``CacheConfig`` turns on vLLM's generic FP8
    KV machinery and corrupts the packed 640-wide pages.
    """
    if get_ascend_device_type() != AscendDeviceType.A5:
        return model_dtype
    normalized_cache_dtype = _normalize_dtype_name(cache_dtype)
    if normalized_cache_dtype in _BF16_KV_CACHE_DTYPES:
        return "bfloat16"
    if normalized_cache_dtype in _FP8_KV_CACHE_DTYPES:
        return _CUSTOM_MXFP8_CACHE_DTYPE
    normalized_model_dtype = _normalize_dtype_name(model_dtype)
    normalized_quant_method = "" if quant_method is None else str(quant_method).lower()
    is_unquantized_bf16 = (
        normalized_model_dtype in _BF16_KV_CACHE_DTYPES and normalized_quant_method in _NO_QUANT_METHODS
    )
    if normalized_cache_dtype not in _AUTO_KV_CACHE_DTYPES:
        return _CUSTOM_MXFP8_CACHE_DTYPE
    if is_unquantized_bf16:
        return "bfloat16"
    return "auto"


def is_a5_bf16_kv_enabled(vllm_config) -> bool:
    """Return whether BF16 SparseFlashMla KV is enabled on A5.

    Callers must pass the engine ``vllm_config``. Do not look it up from the
    process-global current config: that context is only set during
    ``load_model()`` and a missing lookup would silently pick the FP8 plan.
    """
    if get_ascend_device_type() != AscendDeviceType.A5:
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return False
    return _normalize_dtype_name(cache_config.cache_dtype) in _BF16_KV_CACHE_DTYPES


def get_dsv4_attn_kv_dtype(vllm_config) -> torch.dtype:
    """Return the attention KV dtype while preserving non-A5 behavior."""
    return (
        torch.bfloat16
        if get_ascend_device_type() != AscendDeviceType.A5 or is_a5_bf16_kv_enabled(vllm_config)
        else torch.float8_e4m3fn
    )


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

    def prepare_sparse_attn_query(self, q: torch.Tensor) -> torch.Tensor:
        # npu_kv_quant_sparse_attn_sharedkv rejects non-contiguous Q. BF16
        # wq_b/unflatten/rope often leaves a strided [T, N, D] view; the BF16
        # SparseFlashMla path tolerates that, the FP8 kernel decodes 乱码.
        if self.uses_kv_compress_epilog:
            return q.contiguous()
        return q

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
            if slot_mapping.ndim != 1:
                raise ValueError(f"BF16 DSA slot_mapping must be [num_tokens], got {tuple(slot_mapping.shape)}.")
            # ACLGraph cannot capture host syncs (torch.any/item) or
            # data-dependent Nonzero/gather. Flatten the paged dimensions and
            # keep a static [T, 1] index tensor so auxiliary-stream capture
            # follows the same one-dimensional slot convention as A5 FP8.
            flat_cache = cache.view((-1,) + tuple(cache.shape[2:]))
            indices = slot_mapping.to(torch.int64).clamp(min=0).view(-1, 1).contiguous()
            updates = x.reshape((slot_mapping.shape[0],) + tuple(flat_cache.shape[1:])).contiguous()
            torch_npu.npu_scatter_nd_update_(flat_cache, indices, updates)
            return
        if not self.uses_kv_compress_epilog:
            torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)
            return
        # Unquantized BF16 wkv/rope often leaves a non-contiguous [T, 1, D]
        # view. kv_compress_epilog reads a dense [T, D] BF16 row; a strided
        # view makes it pack garbage MXFP8 pages and decode as 乱码.
        # The cache argument must stay a view of the live page: reshape()
        # copies a padded/strided cache and the write is discarded.
        packed_x = x.reshape(-1, x.shape[-1]).contiguous()
        packed_slots = slot_mapping.reshape(-1).contiguous()
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=packed_x,
            slot_mapping=packed_slots,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )


def get_dsa_attn_kv_plan(vllm_config) -> DsaAttnKvPlan:
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

    use_bf16 = is_a5_bf16_kv_enabled(vllm_config)
    if use_bf16:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=True,
            uses_kv_compress_epilog=False,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
            requires_block_offset_slots=False,
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
