from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_ascend.models.deepseek_v4 import DeepseekV4Attention
from vllm_ascend.utils import AscendDeviceType


def _new_uninitialized_attention() -> DeepseekV4Attention:
    """DeepseekV4Attention.__new__ without running __init__, so
    `super().__init__()` inside the real __init__ (called explicitly below)
    resolves correctly -- it requires `self` to actually be a
    DeepseekV4Attention instance."""
    return DeepseekV4Attention.__new__(DeepseekV4Attention)


def _make_config(num_attention_heads: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=128,
        num_attention_heads=num_attention_heads,
        q_lora_rank=16,
        o_lora_rank=16,
        head_dim=64,
        qk_rope_head_dim=32,
        o_groups=1,
        sliding_window=128,
        rms_norm_eps=1e-6,
    )


def _init_attention(*, num_attention_heads: int, tp_size: int, ascend_device_type, use_kv_bf16: bool):
    """Drive DeepseekV4Attention.__init__ far enough to reach (or pass) the
    N1==64 head-count check, without constructing the rest of the (heavy)
    attention module. A bare nn.Module() stands in for `self`."""
    fake_self = _new_uninitialized_attention()
    config = _make_config(num_attention_heads)
    with (
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_world_size", return_value=tp_size),
        mock.patch("vllm_ascend.models.deepseek_v4.get_ascend_device_type", return_value=ascend_device_type),
        mock.patch("vllm_ascend.models.deepseek_v4.dsv4_use_kv_bf16", return_value=use_kv_bf16),
        mock.patch("vllm_ascend.models.deepseek_v4.extract_dsv4_layer_index", return_value=0),
    ):
        DeepseekV4Attention.__init__(
            fake_self,
            vllm_config=mock.MagicMock(),
            config=config,
            prefix="model.layers.0.self_attn",
        )


def test_a5_bf16_rejects_tp_size_that_does_not_yield_64_local_heads():
    """Regression: TP=2 (32 local heads) silently produced garbled output and
    TP=8 (8 local heads) crashed the AICore kernel, both because
    npu_sparse_flash_mla hard-requires exactly 64 local query heads
    (csrc/attention/sparse_flash_mla/README.md: "N1仅支持64"). Fail fast with
    a clear error instead of corrupting output or crashing hardware."""
    for bad_tp_size in (2, 8):
        with pytest.raises(ValueError, match="N1 constraint"):
            _init_attention(
                num_attention_heads=64,
                tp_size=bad_tp_size,
                ascend_device_type=AscendDeviceType.A5,
                use_kv_bf16=True,
            )


def test_a5_bf16_accepts_tp_size_that_yields_64_local_heads():
    # Should not raise, and should reach past the check (n_local_heads set).
    fake_self = _new_uninitialized_attention()
    config = _make_config(num_attention_heads=64)
    with (
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_world_size", return_value=1),
        mock.patch("vllm_ascend.models.deepseek_v4.get_ascend_device_type", return_value=AscendDeviceType.A5),
        mock.patch("vllm_ascend.models.deepseek_v4.dsv4_use_kv_bf16", return_value=True),
        mock.patch("vllm_ascend.models.deepseek_v4.extract_dsv4_layer_index", return_value=0),
        mock.patch("vllm_ascend.models.deepseek_v4.enable_dsa_cp", return_value=False),
        # Stop init right after our check (before the heavy Linear/quant
        # setup, which needs a real VllmConfig) by making the next attribute
        # access blow up in a way we can distinguish from our ValueError.
        mock.patch(
            "vllm_ascend.models.deepseek_v4.ReplicatedLinear",
            side_effect=RuntimeError("reached past the head-count check"),
        ),
        pytest.raises(RuntimeError, match="reached past the head-count check"),
    ):
        DeepseekV4Attention.__init__(
            fake_self,
            vllm_config=mock.MagicMock(),
            config=config,
            prefix="model.layers.0.self_attn",
        )
    assert fake_self.n_local_heads == 64


@pytest.mark.parametrize(
    "ascend_device_type,use_kv_bf16",
    [
        (AscendDeviceType.A2, True),  # BF16 flag only takes effect on A5
        (AscendDeviceType.A5, False),  # FP8 path (default) has no N1==64 constraint here
    ],
)
def test_check_only_applies_to_a5_bf16_path(ascend_device_type, use_kv_bf16):
    """Non-A5, or A5 without VLLM_ASCEND_DSV4_KV_BF16, must not be affected by
    this new check (dsv4_use_kv_bf16() itself already returns False for
    non-A5 regardless of the env var, but assert the attention init doesn't
    add its own device/flag-independent restriction)."""
    fake_self = _new_uninitialized_attention()
    config = _make_config(num_attention_heads=64)
    with (
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_world_size", return_value=2),
        mock.patch("vllm_ascend.models.deepseek_v4.get_ascend_device_type", return_value=ascend_device_type),
        mock.patch("vllm_ascend.models.deepseek_v4.dsv4_use_kv_bf16", return_value=use_kv_bf16),
        mock.patch("vllm_ascend.models.deepseek_v4.extract_dsv4_layer_index", return_value=0),
        mock.patch("vllm_ascend.models.deepseek_v4.enable_dsa_cp", return_value=False),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.ReplicatedLinear",
            side_effect=RuntimeError("reached past the head-count check"),
        ),
        pytest.raises(RuntimeError, match="reached past the head-count check"),
    ):
        DeepseekV4Attention.__init__(
            fake_self,
            vllm_config=mock.MagicMock(),
            config=config,
            prefix="model.layers.0.self_attn",
        )
    assert fake_self.n_local_heads == 32
