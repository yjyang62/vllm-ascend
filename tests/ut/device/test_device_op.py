from unittest import mock

import pytest
import torch

from vllm_ascend.device import device_op
from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def test_reshape_and_cache_makes_scatter_inputs_contiguous():
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn(2, 3, 4).transpose(0, 1)
    slot_mapping = torch.arange(8, dtype=torch.int32)[::2]
    key_cache = object()
    value_cache = object()

    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert not slot_mapping.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_scatter:
        BaseDeviceAdaptor.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    mock_scatter.assert_called_once()
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["key"] is not key
    assert call_kwargs["value"] is not value
    assert call_kwargs["slot_mapping"] is not slot_mapping
    assert call_kwargs["key"].is_contiguous()
    assert call_kwargs["value"].is_contiguous()
    assert call_kwargs["slot_mapping"].is_contiguous()
    torch.testing.assert_close(call_kwargs["key"], key)
    torch.testing.assert_close(call_kwargs["value"], value)
    torch.testing.assert_close(call_kwargs["slot_mapping"], slot_mapping)
    assert call_kwargs["key_cache"] is key_cache
    assert call_kwargs["value_cache"] is value_cache
    assert call_kwargs["cache_mode"] == "Norm"


def test_kv_cache_load_makes_seq_lens_contiguous():
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather:
        BaseDeviceAdaptor.kv_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts,
            key,
            value,
        )

    mock_gather.assert_called_once()
    call_args = mock_gather.call_args.args
    assert call_args[0] is cache_kv_c
    assert call_args[1] is cache_k_pe
    assert call_args[2] is block_table
    assert call_args[3] is not context_seq_len_npu
    assert call_args[3].is_contiguous()
    torch.testing.assert_close(call_args[3], context_seq_len_npu)
    assert mock_gather.call_args.kwargs["seq_offset"] is seq_starts
    assert mock_gather.call_args.kwargs["key"] is key
    assert mock_gather.call_args.kwargs["value"] is value


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


def test_a5_dsa_bf16_sparse_flash_mla_metadata_adapter(monkeypatch):
    expected = object()
    seqused_kv = torch.tensor([7, 9], dtype=torch.int32)
    max_seqlen_kv = torch.tensor(9, dtype=torch.int32)

    monkeypatch.setattr(device_op, "dsv4_use_kv_bf16", lambda: True)
    with mock.patch.object(
        device_op.torch.ops._C_ascend,
        "npu_sparse_flash_mla_metadata",
        return_value=expected,
        create=True,
    ) as mock_metadata:
        metadata_op = A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op()
        result = metadata_op(
            device="npu:0",
            seqused_kv=seqused_kv,
            max_seqlen_kv=max_seqlen_kv,
            cmp_ratio=4,
        )

    assert result is expected
    call_kwargs = mock_metadata.call_args.kwargs
    assert call_kwargs["seqused_ori_kv"] is seqused_kv
    assert call_kwargs["max_seqlen_ori_kv"] is max_seqlen_kv
    torch.testing.assert_close(call_kwargs["seqused_cmp_kv"], torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(call_kwargs["cmp_residual_kv"], torch.tensor([3, 1], dtype=torch.int32))
    assert call_kwargs["max_seqlen_cmp_kv"].item() == 2
    assert "seqused_kv" not in call_kwargs
    assert "max_seqlen_kv" not in call_kwargs


def test_a5_dsa_bf16_selectors_and_block_offset_scatter(monkeypatch):
    monkeypatch.setattr(device_op, "dsv4_use_kv_bf16", lambda: True)

    assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs(torch.device("cpu")) == {"device": "cpu"}
    assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs() == {}
    assert A5DeviceAdaptor.get_dsa_kv_layout() == "PA_BBND"
    assert (
        A5DeviceAdaptor.get_dsa_compressor_slot_mapping_format()
        == device_op.DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
    )

    flat_slot_mapping = torch.tensor([1, 6], dtype=torch.int64)
    formatted = A5DeviceAdaptor.format_dsa_slot_mapping(flat_slot_mapping, block_size=4)
    torch.testing.assert_close(formatted, torch.tensor([[0, 1], [1, 2]], dtype=torch.int64))

    cache = torch.zeros((2, 4, 1, 3), dtype=torch.bfloat16)
    x = torch.arange(9, dtype=torch.float32).reshape(3, 1, 3).to(torch.bfloat16)
    slot_mapping = torch.tensor([[0, 1], [1, 2], [-1, -1]], dtype=torch.int64)

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, slot_mapping)

    torch.testing.assert_close(cache[0, 1], x[0])
    torch.testing.assert_close(cache[1, 2], x[1])
    torch.testing.assert_close(cache[0, 0], x[2])
