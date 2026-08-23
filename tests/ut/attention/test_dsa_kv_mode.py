# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend.attention.dsa_kv_mode import (
    DSV4_EXPLICIT_BF16_KV_KEY,
    record_dsv4_kv_mode,
    resolve_dsv4_use_bf16_kv,
    uses_explicit_bf16_kv,
)


def _config(cache_dtype: str = "auto"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=512), dtype=torch.bfloat16),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        additional_config={},
    )


def test_auto_defaults_to_fp8_and_explicit_dtype_overrides():
    assert not resolve_dsv4_use_bf16_kv(_config())
    assert not resolve_dsv4_use_bf16_kv(_config("fp8"))
    assert resolve_dsv4_use_bf16_kv(_config("bfloat16"))


def test_recorded_mode_survives_cache_dtype_normalization():
    config = _config()
    record_dsv4_kv_mode(config, config.additional_config)
    assert config.additional_config[DSV4_EXPLICIT_BF16_KV_KEY] is False
    config.cache_config.cache_dtype = "bfloat16"
    assert not uses_explicit_bf16_kv(config)
