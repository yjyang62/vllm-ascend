from unittest import mock

import pytest

from vllm_ascend.attention.dsa_v1 import AscendDSAImpl


def _make_impl(*, multistream_overlap_config: bool, use_kv_bf16: bool) -> AscendDSAImpl:
    ascend_config = mock.MagicMock()
    ascend_config.multistream_dsv4_dsa_overlap = multistream_overlap_config

    with (
        mock.patch("vllm_ascend.attention.dsa_v1.get_ascend_config", return_value=ascend_config),
        mock.patch("vllm_ascend.attention.dsa_v1.dsv4_use_kv_bf16", return_value=use_kv_bf16),
        mock.patch("vllm_ascend.attention.dsa_v1.get_current_vllm_config", return_value=mock.MagicMock()),
    ):
        return AscendDSAImpl(
            n_heads=8,
            scale=1.0,
            n_local_heads=8,
            q_lora_rank=16,
            o_lora_rank=16,
            head_dim=64,
            rope_head_dim=32,
            nope_head_dim=32,
            n_groups=1,
            n_local_groups=1,
            window_size=128,
            compress_ratio=0,
            wq_a=mock.MagicMock(),
            wq_b=mock.MagicMock(),
            wkv=mock.MagicMock(),
            q_norm=mock.MagicMock(),
            q_norm_without_weight=mock.MagicMock(),
            kv_norm=mock.MagicMock(),
            wo_a=mock.MagicMock(),
            wo_b=mock.MagicMock(),
            eps=1e-6,
            attn_sink=mock.MagicMock(),
        )


def test_bf16_kv_forces_multistream_overlap_off():
    """Regression: VLLM_ASCEND_DSV4_BF16_DEBUG=1 (which forces a device sync
    after every layer via .item()) reliably "fixed" otherwise
    garbled/truncated BF16 output, indicating the multistream_dsv4_dsa_overlap
    optimization's aux-stream/main-stream synchronization does not (yet)
    cover the BF16 KV scatter path correctly. Disable it automatically on
    that path until verified, as a safe (perf-only) interim workaround."""
    impl = _make_impl(multistream_overlap_config=True, use_kv_bf16=True)
    assert impl.multistream_dsv4_dsa_overlap is False


@pytest.mark.parametrize(
    "multistream_overlap_config,use_kv_bf16",
    [
        (True, False),  # FP8 path (or non-A5): overlap stays as configured
        (False, True),  # Already disabled: no-op, stays disabled
        (False, False),
    ],
)
def test_non_bf16_or_already_disabled_is_unaffected(multistream_overlap_config, use_kv_bf16):
    impl = _make_impl(multistream_overlap_config=multistream_overlap_config, use_kv_bf16=use_kv_bf16)
    assert impl.multistream_dsv4_dsa_overlap == multistream_overlap_config
