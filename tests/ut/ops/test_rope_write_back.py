# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Contract tests for vllm_ascend.ops.rope_dsv4 components.

``ComplexExpRotaryEmbedding.forward`` writes its rotated result back into
the input storage (``y.copy_``). ``DeepseekV4DSparkModel._project_shared_kv``
relies on this in-place behavior (it discards the rope return value), so a
contract test locks it: a future out-of-place refactor fails loudly here
instead of silently disabling draft RoPE.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.ops.rope_dsv4 import (
    ComplexExpRotaryEmbedding,
    dsa_rope_cache_len,
    get_cos_and_sin_dsa,
)


def test_dsv4_rope_writes_back_inplace():
    # DeepseekV4DSparkModel._project_shared_kv discards the return value of
    # _apply_dsv4_rope; RoPE only takes effect because forward writes the
    # rotated values back into the input storage (``y.copy_``). Lock that
    # contract so a future out-of-place refactor fails loudly instead of
    # silently disabling draft RoPE.
    with (
        patch("vllm_ascend.ops.rope_dsv4.current_platform") as fake_platform,
        # create=True: the torch_npu mock installed by tests/ut/conftest.py on
        # CPU-only CI has no npu_rotary_mul attribute.
        patch(
            "torch_npu.npu_rotary_mul",
            lambda x, cos, sin, rotary_mode=None: x * 2 + 1,
            create=True,
        ),
    ):
        fake_platform.device_type = "cpu"
        vllm_config = SimpleNamespace(
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        )
        rotary = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername="ut.dspark_rope",
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=64,
            base=10000,
            scaling_factor=1.0,
        )
        x = torch.randn(4, 1, 8)
        snapshot = x.clone()
        out = rotary(x, torch.ones(1), torch.zeros(1))

    assert not torch.equal(x, snapshot)
    assert out is x


def test_dsa_rope_cache_len_covers_max_model_len():
    assert dsa_rope_cache_len(32, 2.0, 1000) == 1000
    assert dsa_rope_cache_len(4096, 40, 1000) == 4096 * 40
    assert dsa_rope_cache_len(32, 1.5, 0) == 48


def test_get_cos_and_sin_dsa_gathers_positions_out_to_max_model_len():
    max_model_len = 1000
    with patch("vllm_ascend.ops.rope_dsv4.current_platform") as fake_platform:
        fake_platform.device_type = "cpu"
        vllm_config = SimpleNamespace(
            speculative_config=None,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
            model_config=SimpleNamespace(max_model_len=max_model_len),
        )
        rotary = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername="ut.dsv4_rope.max_model_len",
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=32,
            base=10000,
            scaling_factor=2.0,
            rope_groups=["default"],
        )

    assert rotary.full_rope_cos.shape[0] >= max_model_len
    assert rotary.full_rope_sin.shape[0] == rotary.full_rope_cos.shape[0]
    # Yarn still sees the original sequence length; only the position axis grew.
    assert rotary.full_rope_cos.shape[-1] == 8
    position = torch.tensor([max_model_len - 1])
    cos, sin = get_cos_and_sin_dsa(position, layer_names="ut.dsv4_rope.max_model_len")
    gathered_cos = cos["ut.dsv4_rope.max_model_len"]
    gathered_sin = sin["ut.dsv4_rope.max_model_len"]
    torch.testing.assert_close(gathered_cos, rotary.full_rope_cos[position])
    torch.testing.assert_close(gathered_sin, rotary.full_rope_sin[position])
