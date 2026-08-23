# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 attention-KV mode selection for Ascend A5.

Resolution order (see ``resolve_dsv4_use_bf16_kv``):

1. Explicit ``--kv-cache-dtype`` (anything other than ``auto``).
2. ``auto`` → upstream-compatible FP8 KV.
"""

from __future__ import annotations

DSV4_EXPLICIT_BF16_KV_KEY = "dsv4_use_bf16_sparse_flash_mla"

_FP8_KV_CACHE_DTYPES = frozenset({"fp8", "float8", "float8_e4m3fn", "float8_e5m2"})
_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def _is_dsv4_model(model_config) -> bool:
    hf_config = getattr(model_config, "hf_text_config", None)
    return hf_config is not None and hasattr(hf_config, "index_topk")


def resolve_dsv4_use_bf16_kv(vllm_config) -> bool:
    """Return whether DSV4 should use BF16 SparseFlashMla KV on Ascend A5."""
    if not _is_dsv4_model(vllm_config.model_config):
        return False

    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in (None, "auto"):
        normalized = str(cache_dtype).lower()
        if normalized in _BF16_KV_CACHE_DTYPES:
            return True
        if normalized in _FP8_KV_CACHE_DTYPES:
            return False

    return False


def record_dsv4_kv_mode(vllm_config, additional_config: dict) -> None:
    """Persist the resolved KV mode before platform rewrites ``cache_dtype``."""
    if not _is_dsv4_model(vllm_config.model_config):
        return
    additional_config[DSV4_EXPLICIT_BF16_KV_KEY] = resolve_dsv4_use_bf16_kv(vllm_config)


def uses_explicit_bf16_kv(vllm_config=None) -> bool:
    """Return whether the launch selected BF16 SparseFlashMla KV."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except AssertionError:
            # Module inspection and direct unit tests can reach device helpers
            # without a current vLLM config. Fall back to the upstream FP8
            # behavior there.
            return False
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if DSV4_EXPLICIT_BF16_KV_KEY in additional_config:
        return bool(additional_config[DSV4_EXPLICIT_BF16_KV_KEY])
    return resolve_dsv4_use_bf16_kv(vllm_config)
