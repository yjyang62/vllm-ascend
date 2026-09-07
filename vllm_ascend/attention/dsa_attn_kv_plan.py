# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def _supports_dsv4_compressed_cache() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5 the launch request has to stay readable afterwards, because it is the
    only thing that separates an explicit bfloat16 KV request from ``auto``.
    ``auto`` and the model dtype resolve identically everywhere downstream, so
    collapsing every non-bfloat16 request to ``auto`` preserves the upstream
    values while keeping the mode recoverable.
    """
    if not _supports_dsv4_compressed_cache():
        return model_dtype
    return "bfloat16" if str(cache_dtype).lower() in _BF16_KV_CACHE_DTYPES else "auto"


def is_a5_bf16_kv_enabled(vllm_config) -> bool:
    """Return whether BF16 SparseFlashMla KV is enabled on A5.

    Callers must pass the engine ``vllm_config``. Do not look it up from the
    process-global current config: that context is only set during
    ``load_model()`` and a missing lookup would silently pick the FP8 plan.
    """
    if not _supports_dsv4_compressed_cache():
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return False
    return str(cache_config.cache_dtype).lower() in _BF16_KV_CACHE_DTYPES


def get_dsv4_attn_kv_dtype(vllm_config) -> torch.dtype:
    """Return the attention KV dtype while preserving non-A5 behavior."""
    return (
        torch.bfloat16
        if not _supports_dsv4_compressed_cache() or is_a5_bf16_kv_enabled(vllm_config)
        else torch.float8_e4m3fn
    )


def get_dsv4_indexer_kv_dtype(vllm_config) -> torch.dtype:
    """Return the indexer K cache dtype.

    A5 BF16 SparseFlashMla stores indexer KV in float16. A5 ``auto`` stays
    FP8, and non-A5 keeps the int8 lightning-indexer cache.
    """
    if not _supports_dsv4_compressed_cache():
        return torch.int8
    return torch.float16 if is_a5_bf16_kv_enabled(vllm_config) else torch.float8_e4m3fn


def dsa_indexer_uses_quant(vllm_config) -> bool:
    """Return whether indexer KV is quantized (int8/FP8) rather than FP16."""
    return not is_a5_bf16_kv_enabled(vllm_config)


DSA_INDEXER_CMP_RATIO = 4
# Unquant lightning_indexer has no cmp_ratio. rightDownCausal (mode 3) treats
# query_len as original tokens and key_len as compressed slots, so prefill
# validS2Len becomes (S/4 - S) and TopK collapses. defaultMask (mode 0) scores
# every stored compressed key; SparseFlashMla still applies cmp_ratio causal.
DSA_INDEXER_FP16_SPARSE_MODE = 0


def get_dsv4_indexer_key_seq_lens(seq_lens: torch.Tensor, cmp_ratio: int = DSA_INDEXER_CMP_RATIO) -> torch.Tensor:
    """Return compressed indexer K lengths for unquantized lightning_indexer.

    Quant lightning indexer consumes original sequence lengths plus
    ``cmp_ratio``. The unquantized operator has no compression attribute, so
    callers must pass the number of compressed keys actually stored in cache.
    """
    return torch.div(seq_lens, cmp_ratio, rounding_mode="floor")


def fill_dsv4_indexer_key_seq_lens(out: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Write compressed indexer K lengths into a persistent ACLGraph buffer."""
    n = seq_lens.shape[0]
    out[:n].copy_(get_dsv4_indexer_key_seq_lens(seq_lens))
    return out[:n]


def dsa_fp16_indexer_key_seq_lens(metadata) -> torch.Tensor:
    """Return compressed indexer K lengths, preferring the graph-stable buffer."""
    key_seq_lens = getattr(metadata, "indexer_key_seq_lens", None)
    if key_seq_lens is not None:
        return key_seq_lens
    return get_dsv4_indexer_key_seq_lens(metadata.seq_lens)


def _unquant_lightning_indexer(**kwargs):
    """Prefer torch_npu on A5; fall back to the custom op for tests/A3 stubs."""
    op = getattr(torch_npu, "npu_lightning_indexer", None)
    if callable(op):
        return op(**kwargs)
    return torch.ops._C_ascend.npu_lightning_indexer(**kwargs)


def select_dsa_indexer_fp16_topk(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    index_topk: int,
) -> torch.Tensor:
    """Select indexer TopK from FP16 query/key caches.

    ``actual_seq_lengths_key`` must already be compressed (seq_len // 4).
    Do not divide inside this call: ACLGraph would otherwise capture a fresh
    tensor instead of the persistent metadata buffer.
    """
    if query.dtype != key_cache.dtype:
        query = query.to(dtype=key_cache.dtype)
    topk_idxs, _ = _unquant_lightning_indexer(
        query=query,
        key=key_cache,
        weights=weights.to(dtype=key_cache.dtype),
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=index_topk,
        sparse_mode=DSA_INDEXER_FP16_SPARSE_MODE,
    )
    return topk_idxs


DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """Attention-KV plan. Indexer KV follows the A5 BF16 switch as FP16."""

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
            if slot_mapping.ndim != 1:
                raise ValueError(f"BF16 DSA slot_mapping must be [num_tokens], got {tuple(slot_mapping.shape)}.")
            # Flatten the paged cache and keep a static [T, 1] index tensor so
            # ACLGraph capture matches the A5 FP8 / SFA one-dimensional slot
            # convention. Do not clamp PAD_SLOT_ID (-1) to 0: that overwrites
            # a live physical slot. The scatter kernel skips negative indices.
            flat_cache = cache.flatten(end_dim=1)
            indices = slot_mapping.view(-1, 1)
            updates = x.reshape((slot_mapping.shape[0],) + tuple(flat_cache.shape[1:]))
            torch_npu.npu_scatter_nd_update_(flat_cache, indices, updates)
            return
        if not self.uses_kv_compress_epilog:
            torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, slot_mapping, x)
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


def get_dsa_attn_kv_plan(vllm_config) -> DsaAttnKvPlan:
    """Return the explicit A5 BF16 or upstream-compatible FP8 DSA plan."""
    if not _supports_dsv4_compressed_cache():
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
