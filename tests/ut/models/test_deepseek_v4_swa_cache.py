# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.core.kv_cache_interface import AscendCompressorStateSpec
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.utils import AscendDeviceType


def _swa_cache(*, window_size: int = 128, block_size: int = 128) -> AscendDeepseekV4SWACache:
    cache = AscendDeepseekV4SWACache.__new__(AscendDeepseekV4SWACache)
    cache.head_dim = 512
    cache.window_size = window_size
    cache.dtype = torch.bfloat16
    cache.block_size = block_size
    cache.cache_config = SimpleNamespace(cache_dtype="auto")
    return cache


class TestDeepseekV4SWACacheAdmission:
    def test_swa_cache_spec_is_compressor_state_spec(self):
        cache = _swa_cache()
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"))

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.model.get_ascend_device_type",
                return_value=AscendDeviceType.A3,
            ),
            patch(
                "vllm_ascend.models.deepseek_v4.model.uses_explicit_bf16_kv",
                return_value=False,
            ),
        ):
            spec = cache.get_kv_cache_spec(vllm_config)

        assert isinstance(spec, AscendCompressorStateSpec)
        assert spec.sliding_window == 128
        assert spec.block_size == 128
        assert spec.model_version == "deepseek_v4"

    def test_32k_request_is_charged_sliding_window_not_in_flight(self):
        from vllm.utils.math_utils import cdiv

        cache = _swa_cache()
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="auto"),
            model_config=SimpleNamespace(max_model_len=36864),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            max_in_flight_tokens=8192,
        )

        with (
            patch(
                "vllm_ascend.models.deepseek_v4.model.get_ascend_device_type",
                return_value=AscendDeviceType.A3,
            ),
            patch(
                "vllm_ascend.models.deepseek_v4.model.uses_explicit_bf16_kv",
                return_value=False,
            ),
        ):
            spec = cache.get_kv_cache_spec(vllm_config)

        expected_blocks = cdiv(cache.window_size, cache.block_size) + 1
        swa_formula_blocks = cdiv(min(cache.window_size - 1 + 8192, 36864), cache.block_size) + 1
        linear_32k_blocks = cdiv(32768, cache.block_size)

        assert spec.max_admission_blocks_per_request(8192, 36864) == expected_blocks
        assert spec.max_admission_blocks_per_request(8192, 32768) == expected_blocks
        assert spec.max_memory_usage_bytes(vllm_config) == expected_blocks * spec.page_size_bytes
        assert swa_formula_blocks > expected_blocks
        assert linear_32k_blocks > expected_blocks
