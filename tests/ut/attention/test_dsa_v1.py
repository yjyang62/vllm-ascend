from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import _dsa_o_proj_weight_for_batch_matmul, _has_weight_scale, _is_w8a8_dynamic
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod


def test_w8a8_dynamic_requires_loaded_weight_scale():
    quant_method = AscendW8A8DynamicLinearMethod.__new__(AscendW8A8DynamicLinearMethod)
    linear = SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_method))

    assert not _has_weight_scale(linear)
    assert not _is_w8a8_dynamic(linear)

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
