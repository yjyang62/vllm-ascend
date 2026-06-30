from types import SimpleNamespace

from vllm_ascend.attention.dsa_v1 import _has_weight_scale, _is_w8a8_dynamic
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod


def test_w8a8_dynamic_requires_loaded_weight_scale():
    quant_method = AscendW8A8DynamicLinearMethod.__new__(AscendW8A8DynamicLinearMethod)
    linear = SimpleNamespace(quant_method=SimpleNamespace(quant_method=quant_method))

    assert not _has_weight_scale(linear)
    assert not _is_w8a8_dynamic(linear)

    linear.weight_scale = object()

    assert _has_weight_scale(linear)
    assert _is_w8a8_dynamic(linear)
