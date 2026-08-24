# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 attention-KV mode selection for Ascend A5.

Only an explicit bfloat16 ``--kv-cache-dtype`` selects the BF16 SparseFlashMla
path. ``auto`` and every FP8 spelling keep the upstream-compatible FP8 KV.
"""

from __future__ import annotations

# Internal snapshot written by ``record_dsv4_kv_mode``; not a user-facing option.
DSV4_EXPLICIT_BF16_KV_KEY = "_dsv4_use_bf16_sparse_flash_mla"

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def resolve_dsv4_use_bf16_kv(vllm_config) -> bool:
    """Return whether DSV4 should use BF16 SparseFlashMla KV on Ascend A5."""
    return str(vllm_config.cache_config.cache_dtype).lower() in _BF16_KV_CACHE_DTYPES


def record_dsv4_kv_mode(vllm_config, additional_config: dict) -> None:
    """Persist the resolved KV mode before platform rewrites ``cache_dtype``."""
    additional_config[DSV4_EXPLICIT_BF16_KV_KEY] = resolve_dsv4_use_bf16_kv(vllm_config)


def uses_explicit_bf16_kv(vllm_config=None) -> bool:
    """Return whether the launch recorded BF16 SparseFlashMla KV.

    Only the recorded snapshot is trusted. Re-reading ``cache_dtype`` here
    cannot tell an explicit bfloat16 request apart from ``auto``, because the
    platform rewrites it to the model dtype right after recording. A missing
    snapshot therefore means the upstream FP8 KV path.
    """
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except AssertionError:
            # Module inspection and direct unit tests can reach device helpers
            # without a current vLLM config.
            return False
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get(DSV4_EXPLICIT_BF16_KV_KEY, False))
