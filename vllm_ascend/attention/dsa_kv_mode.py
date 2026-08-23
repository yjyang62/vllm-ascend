# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit DeepSeek-V4 attention-KV mode selection.

BF16 KV is opt-in through ``--additional-config`` only. Deriving it from
``--kv-cache-dtype`` is not possible without mutating shared config state:
the platform rewrites ``cache_config.cache_dtype`` to the model dtype for
``index_topk`` models, so the original request is no longer recoverable
later in startup. Requiring an explicit key keeps every other launch, FP8
and ``auto`` alike, on exactly the upstream code path.
"""

DSV4_EXPLICIT_BF16_KV_KEY = "dsv4_use_bf16_sparse_flash_mla"


def uses_explicit_bf16_kv(vllm_config=None) -> bool:
    """Return whether the launch explicitly opted into BF16 SparseFlashMla KV."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        try:
            vllm_config = get_current_vllm_config()
        except AssertionError:
            # Module inspection and direct unit tests can reach device helpers
            # without a current vLLM config. Fall back to the upstream FP8
            # behavior there; BF16 stays opt-in inside a configured engine.
            return False
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get(DSV4_EXPLICIT_BF16_KV_KEY, False))
