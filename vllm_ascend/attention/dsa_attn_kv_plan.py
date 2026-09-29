# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})
_AUTO_KV_DTYPE = "auto"
_A5_DSV4_DEFAULT_KV_DTYPE = "fp8"


def _supports_dsv4_compressed_cache() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5, ``auto`` must become ``fp8``. Downstream ``auto`` means the model
    dtype, which is BF16, while the compressed-cache plan treats a non-BF16
    request as the FP8 ``kv_compress_epilog`` path. Explicit ``bfloat16`` stays
    BF16 so SparseFlashMla remains selectable. Any other explicit dtype is
    left unchanged.
    """
    if not _supports_dsv4_compressed_cache():
        return model_dtype
    normalized = str(cache_dtype).lower()
    if normalized in _BF16_KV_CACHE_DTYPES:
        return "bfloat16"
    if normalized == _AUTO_KV_DTYPE:
        return _A5_DSV4_DEFAULT_KV_DTYPE
    return str(cache_dtype)


def _is_exact_deepseek_v4(model_config) -> bool:
    for candidate in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if getattr(candidate, "model_type", None) == "deepseek_v4":
            return True
    return False


def apply_a5_deepseek_v4_default_kv_dtypes(vllm_config) -> None:
    """Pin DeepSeek-V4 ``auto`` KV dtypes to FP8 on A5 before cache allocation.

    The main cache and the indexer cache are independent. An explicit main
    BF16 request does not stop an indexer ``auto`` from becoming FP8.
    """
    if not _supports_dsv4_compressed_cache():
        return
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not _is_exact_deepseek_v4(model_config):
        return

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and str(getattr(cache_config, "cache_dtype", "")).lower() == _AUTO_KV_DTYPE:
        model_dtype = getattr(model_config, "dtype", "bfloat16")
        cache_config.cache_dtype = resolve_dsv4_cache_dtype(_AUTO_KV_DTYPE, str(model_dtype))
        logger.info_once(
            "DeepSeek-V4 on Ascend A5 resolves kv cache dtype auto to %s "
            "so the compressed KV plan matches the allocated cache.",
            cache_config.cache_dtype,
        )

    attention_config = getattr(vllm_config, "attention_config", None)
    if (
        attention_config is not None
        and str(getattr(attention_config, "indexer_kv_dtype", "")).lower() == _AUTO_KV_DTYPE
    ):
        attention_config.indexer_kv_dtype = _A5_DSV4_DEFAULT_KV_DTYPE
        logger.info_once(
            "DeepSeek-V4 on Ascend A5 resolves indexer kv dtype auto to %s.",
            attention_config.indexer_kv_dtype,
        )


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
