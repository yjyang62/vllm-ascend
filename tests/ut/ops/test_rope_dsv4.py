from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ops import rope_dsv4
from vllm_ascend.ops.rope_dsv4 import (
    ComplexExpRotaryEmbedding,
    clear_global_dsa_rope_cache,
    get_cos_and_sin_dsa,
    restore_global_dsa_rope_cache,
)


def _reset_rope_state():
    rope_dsv4._ROPE_STATE.static_cache.clear()
    rope_dsv4._ROPE_STATE.runtime_buffer.clear()
    rope_dsv4._ROPE_STATE.layer_info.clear()
    rope_dsv4._ROPE_STATE.registry_summary.clear()
    rope_dsv4._ROPE_STATE.cache_configs.clear()


def test_clear_and_restore_global_dsa_rope_cache():
    _reset_rope_state()
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_batched_tokens = 4

    with patch("vllm_ascend.ops.rope_dsv4.current_platform") as mock_platform:
        mock_platform.device_type = "cpu"
        ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername="model.layers.0.attn",
            head_size=4,
            rotary_dim=4,
            max_position_embeddings=8,
            base=10000,
            scaling_factor=1.0,
        )

    assert rope_dsv4._ROPE_STATE.static_cache
    assert rope_dsv4._ROPE_STATE.runtime_buffer

    assert clear_global_dsa_rope_cache()
    assert not rope_dsv4._ROPE_STATE.static_cache
    assert not rope_dsv4._ROPE_STATE.runtime_buffer

    assert restore_global_dsa_rope_cache()
    assert rope_dsv4._ROPE_STATE.static_cache
    assert rope_dsv4._ROPE_STATE.runtime_buffer

    _reset_rope_state()


def test_get_cos_and_sin_dsa_lazily_restores_cleared_cache():
    _reset_rope_state()
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_batched_tokens = 4

    with patch("vllm_ascend.ops.rope_dsv4.current_platform") as mock_platform:
        mock_platform.device_type = "cpu"
        ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername="model.layers.0.attn",
            head_size=4,
            rotary_dim=4,
            max_position_embeddings=8,
            base=10000,
            scaling_factor=1.0,
        )

    clear_global_dsa_rope_cache()
    cos, sin = get_cos_and_sin_dsa(torch.tensor([0, 1]), use_cache=True)

    assert cos["model.layers.0.attn"].shape == (2, 1, 1, 4)
    assert sin["model.layers.0.attn"].shape == (2, 1, 1, 4)

    _reset_rope_state()
