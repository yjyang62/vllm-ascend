from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    _dsa_o_proj_matmul,
    _dsa_o_proj_weight_for_batch_matmul,
    _has_weight_scale,
    _is_w8a8_dynamic,
)
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import AscendDeviceType


def test_is_w8a8_dynamic_detects_method_without_weight_scale():
    quant_method = AscendW8A8DynamicLinearMethod.__new__(AscendW8A8DynamicLinearMethod)
    linear = SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_method))

    assert not _has_weight_scale(linear)
    assert _is_w8a8_dynamic(linear)

    linear.weight_scale = object()

    assert _has_weight_scale(linear)
    assert _is_w8a8_dynamic(linear)


def test_dsa_o_proj_weight_for_batch_matmul_views_2d_weight_by_group():
    weight = torch.arange(24).reshape(6, 4)

    grouped = _dsa_o_proj_weight_for_batch_matmul(weight, n_local_groups=3)

    assert grouped.shape == (3, 2, 4)
    assert grouped.data_ptr() == weight.data_ptr()
    torch.testing.assert_close(grouped[2, 1], weight[5])


def test_dsa_o_proj_weight_for_batch_matmul_keeps_3d_weight():
    weight = torch.empty(3, 2, 4)

    assert _dsa_o_proj_weight_for_batch_matmul(weight, n_local_groups=3) is weight


def test_dsa_o_proj_weight_for_batch_matmul_rejects_unexpected_rank():
    with pytest.raises(ValueError, match="must be 2D or 3D"):
        _dsa_o_proj_weight_for_batch_matmul(torch.empty(4), n_local_groups=2)


def test_dsa_o_proj_matmul_matches_grouped_einsum():
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    grouped_weight = weight.view(3, 2, 4)

    output = _dsa_o_proj_matmul(o_proj_input, weight, n_local_groups=3)
    expected = torch.einsum("tgd,grd->tgr", o_proj_input, grouped_weight)

    torch.testing.assert_close(output, expected)


def test_a5_bf16_o_proj_does_not_access_weight_scale():
    impl = SimpleNamespace(
        n_local_groups=3,
        wo_a=SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4)),
        wo_b=lambda x: x,
    )
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = torch.empty(2, 6)

    with (
        mock.patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        mock.patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.olora_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_quant_batchmatmul") as quant_batch_matmul,
        mock.patch("vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_batchmatmul") as batch_matmul,
    ):
        result = AscendDSAImpl._forward_o_proj(impl, o_proj_input, output)

    quant_batch_matmul.assert_not_called()
    batch_matmul.assert_not_called()
    expected = _dsa_o_proj_matmul(o_proj_input, impl.wo_a.weight, impl.n_local_groups).reshape(2, 6)
    torch.testing.assert_close(result, expected)


def test_a3_o_proj_keeps_npu_transpose_batchmatmul():
    impl = SimpleNamespace(
        n_local_groups=3,
        wo_a=SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4)),
        wo_b=lambda x: x,
    )
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = torch.empty(2, 6)
    expected = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    with (
        mock.patch(
            "vllm_ascend.attention.dsa_v1.get_ascend_device_type",
            return_value=AscendDeviceType.A3,
        ),
        mock.patch("vllm_ascend.attention.dsa_v1.oproj_tp_enable", return_value=False),
        mock.patch("vllm_ascend.attention.dsa_v1.olora_tp_enable", return_value=False),
        mock.patch(
            "vllm_ascend.attention.dsa_v1.torch_npu.npu_transpose_batchmatmul",
            return_value=expected,
        ) as batch_matmul,
        mock.patch(
            "vllm_ascend.attention.dsa_v1._dsa_o_proj_matmul",
            side_effect=AssertionError("A3 must not use BF16 torch.matmul o_proj"),
        ),
    ):
        result = AscendDSAImpl._forward_o_proj(impl, o_proj_input, output)

    batch_matmul.assert_called_once()
    torch.testing.assert_close(result, expected)
