from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import (
    A5_DSA_SPARSE_FLASH_MLA_METADATA_OP,
    A5_DSA_SPARSE_FLASH_MLA_OP,
    A5DeviceAdaptor,
    AscendDeviceType,
    BaseDeviceAdaptor,
)


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]


def test_base_dsa_sparse_attention_keeps_sharedkv_ops():
    with (
        mock.patch.object(torch.ops._C_ascend, "npu_sparse_attn_sharedkv_metadata", create=True) as metadata_op,
        mock.patch.object(torch.ops._C_ascend, "npu_sparse_attn_sharedkv", create=True) as attn_op,
    ):
        assert BaseDeviceAdaptor.get_dsa_sparse_attn_metadata_op() is metadata_op
        assert BaseDeviceAdaptor.get_dsa_sparse_attn_op() is attn_op


def test_a5_dsa_sparse_attention_uses_sparse_flash_mla_ops():
    vllm_config = SimpleNamespace(quant_config=None)
    q = torch.randn(2, 4, 512, dtype=torch.bfloat16)
    kv = torch.randn(8, 128, 512, dtype=torch.bfloat16)
    seq_lens = torch.tensor([18, 16], dtype=torch.int32)
    expected = torch.empty_like(q)

    with (
        mock.patch("vllm_ascend.device.device_op.get_ascend_device_type", return_value=AscendDeviceType.A5),
        mock.patch.object(
            torch.ops.cann_ops_transformer, A5_DSA_SPARSE_FLASH_MLA_METADATA_OP, create=True
        ) as metadata_op,
        mock.patch.object(torch.ops.cann_ops_transformer, A5_DSA_SPARSE_FLASH_MLA_OP, create=True) as attn_op,
    ):
        metadata_op.return_value = torch.zeros(1024, dtype=torch.int32)
        attn_op.return_value = (expected, torch.empty([], dtype=torch.float32))

        metadata = A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op(vllm_config)(
            num_heads_q=4,
            num_heads_kv=1,
            head_dim=512,
            seqused_kv=seq_lens,
            max_seqlen_q=1,
            max_seqlen_kv=18,
            batch_size=2,
            cmp_topk=512,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=True,
        )
        output = A5DeviceAdaptor.get_dsa_sparse_attn_op(vllm_config)(
            q,
            ori_kv=kv,
            cmp_kv=kv,
            cmp_sparse_indices=torch.zeros(2, 512, dtype=torch.int32),
            ori_block_table=torch.zeros(2, 1, dtype=torch.int32),
            cmp_block_table=torch.zeros(2, 1, dtype=torch.int32),
            seqused_kv=seq_lens,
            sinks=torch.zeros(4, dtype=torch.float32),
            metadata=metadata,
            softmax_scale=0.1,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )[0]

    assert output is expected
    assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs("npu:0", vllm_config) == {}
    assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs(vllm_config) == {}
    metadata_kwargs = metadata_op.call_args.kwargs
    assert metadata_kwargs["layout_kv"] == "PA_BBND"
    torch.testing.assert_close(metadata_kwargs["seqused_cmp_kv"], torch.tensor([4, 4], dtype=torch.int32))
    torch.testing.assert_close(metadata_kwargs["cmp_residual_kv"], torch.tensor([2, 0], dtype=torch.int32))
    assert metadata_kwargs["max_seqlen_cmp_kv"] == 4
    attn_kwargs = attn_op.call_args.kwargs
    assert attn_kwargs["layout_kv"] == "PA_BBND"
    assert attn_kwargs["ori_kv"].shape == (8, 128, 1, 512)
    assert attn_kwargs["cmp_kv"].shape == (8, 128, 1, 512)
    assert attn_kwargs["cmp_sparse_indices"].shape == (2, 1, 512)
    assert "kv_quant_mode" not in attn_kwargs
    torch.testing.assert_close(attn_kwargs["seqused_ori_kv"], seq_lens)
    torch.testing.assert_close(attn_kwargs["seqused_cmp_kv"], torch.tensor([4, 4], dtype=torch.int32))


def test_a5_quantized_dsa_sparse_attention_keeps_kv_quant_ops():
    vllm_config = SimpleNamespace(quant_config=object())

    with (
        mock.patch("vllm_ascend.device.device_op.get_ascend_device_type", return_value=AscendDeviceType.A5),
        mock.patch.object(
            torch.ops._C_ascend, "npu_kv_quant_sparse_attn_sharedkv_metadata", create=True
        ) as metadata_op,
        mock.patch.object(torch.ops._C_ascend, "npu_kv_quant_sparse_attn_sharedkv", create=True) as attn_op,
    ):
        assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op(vllm_config) is metadata_op
        assert A5DeviceAdaptor.get_dsa_sparse_attn_op(vllm_config) is attn_op
        assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs("npu:0", vllm_config) == {"kv_quant_mode": 1}
        assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs(vllm_config) == {
            "kv_quant_mode": 1,
            "tile_size": 64,
            "rope_head_dim": 64,
        }


def test_a5_non_quant_dsa_sparse_attention_reports_missing_sparse_flash_mla_op():
    vllm_config = SimpleNamespace(quant_config=None)
    with mock.patch("vllm_ascend.device.device_op.get_ascend_device_type", return_value=AscendDeviceType.A5):
        if hasattr(torch.ops.cann_ops_transformer, A5_DSA_SPARSE_FLASH_MLA_OP) or hasattr(
            torch.ops._C_ascend, A5_DSA_SPARSE_FLASH_MLA_OP
        ):
            pytest.skip(f"{A5_DSA_SPARSE_FLASH_MLA_OP} is registered in this environment")

        with pytest.raises(RuntimeError, match="--ops=sparse_flash_mla,sparse_flash_mla_metadata"):
            A5DeviceAdaptor.get_dsa_sparse_attn_op(vllm_config)(torch.empty(1))


def test_a5_dsa_kv_compress_scatter_non_quant_filters_invalid_slots():
    cache = torch.zeros((4, 1, 3), dtype=torch.float32)
    x = torch.tensor(
        [
            [[1.0, 1.1, 1.2]],
            [[2.0, 2.1, 2.2]],
            [[3.0, 3.1, 3.2]],
            [[4.0, 4.1, 4.2]],
        ],
        dtype=torch.float32,
    )
    slot_mapping = torch.tensor([2, -1, 9, 0], dtype=torch.int64)

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping, quantized=False)

    torch.testing.assert_close(cache[2], x[0])
    torch.testing.assert_close(cache[0], x[3])
    torch.testing.assert_close(cache[1], torch.zeros_like(cache[1]))
    torch.testing.assert_close(cache[3], torch.zeros_like(cache[3]))


def test_a5_dsa_kv_compress_scatter_non_quant_handles_mapping_length_mismatch():
    cache = torch.zeros((3, 1, 2), dtype=torch.float32)
    x = torch.tensor(
        [
            [[10.0, 11.0]],
            [[20.0, 21.0]],
            [[30.0, 31.0]],
        ],
        dtype=torch.float32,
    )
    slot_mapping = torch.tensor([1, 2, -1, 0], dtype=torch.int64)

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping, quantized=False)

    torch.testing.assert_close(cache[1], x[0])
    torch.testing.assert_close(cache[2], x[1])
    torch.testing.assert_close(cache[0], torch.zeros_like(cache[0]))


def test_a5_dsa_kv_compress_scatter_non_quant_accepts_2d_source_tensor():
    cache = torch.zeros((3, 1, 2), dtype=torch.float32)
    x = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    slot_mapping = torch.tensor([2, 0, -1], dtype=torch.int64)

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping, quantized=False)

    torch.testing.assert_close(cache[2, 0], x[0])
    torch.testing.assert_close(cache[0, 0], x[1])
    torch.testing.assert_close(cache[1], torch.zeros_like(cache[1]))


def test_a5_dsa_kv_compress_scatter_non_quant_aligns_source_with_valid_slots():
    cache = torch.zeros((4, 1, 2), dtype=torch.float32)
    x = torch.tensor(
        [
            [11.0, 12.0],
            [21.0, 22.0],
        ],
        dtype=torch.float32,
    )
    # Interleaved invalid slots: source rows should map to valid slots [3, 1].
    slot_mapping = torch.tensor([-1, 3, -1, 1], dtype=torch.int64)

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping, quantized=False)

    torch.testing.assert_close(cache[3, 0], x[0])
    torch.testing.assert_close(cache[1, 0], x[1])
    torch.testing.assert_close(cache[0], torch.zeros_like(cache[0]))
    torch.testing.assert_close(cache[2], torch.zeros_like(cache[2]))
