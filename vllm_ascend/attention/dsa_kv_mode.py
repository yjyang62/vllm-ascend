# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 attention-KV mode selection for Ascend A5.

Only an explicit bfloat16 ``--kv-cache-dtype`` selects the BF16 SparseFlashMla
path. ``auto`` and every FP8 spelling keep the upstream-compatible FP8 KV.
"""

from __future__ import annotations

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5 the launch request has to stay readable afterwards, because it is the
    only thing that separates an explicit bfloat16 KV request from ``auto``.
    ``auto`` and the model dtype resolve identically everywhere downstream, so
    collapsing every non-bfloat16 request to ``auto`` preserves the upstream
    values while keeping the mode recoverable.
    """
    if get_ascend_device_type() != AscendDeviceType.A5:
        return model_dtype
    return "bfloat16" if str(cache_dtype).lower() in _BF16_KV_CACHE_DTYPES else "auto"


def uses_explicit_bf16_kv(vllm_config=None) -> bool:
    """Return whether the launch asked for BF16 SparseFlashMla KV on A5."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        return False
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except AssertionError:
            # Module inspection and direct unit tests can reach device helpers
            # without a current vLLM config.
            return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return False
    return str(cache_config.cache_dtype).lower() in _BF16_KV_CACHE_DTYPES
