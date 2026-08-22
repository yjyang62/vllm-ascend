# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit DeepSeek-V4 attention-KV mode selection."""

DSV4_EXPLICIT_BF16_KV_KEY = "dsv4_use_bf16_sparse_flash_mla"


def uses_explicit_bf16_kv(vllm_config=None) -> bool:
    """Return whether the original CLI request explicitly selected BF16 KV."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get(DSV4_EXPLICIT_BF16_KV_KEY, False))
