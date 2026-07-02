from types import SimpleNamespace
from unittest import mock

from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod


def _w8a8_quant_method_wrapper(apply=None):
    quant_method = AscendW8A8DynamicLinearMethod.__new__(AscendW8A8DynamicLinearMethod)
    return SimpleNamespace(quant_method=quant_method, apply=apply or mock.Mock(return_value="fallback"))


def test_cv_linear_w8a8_detection_requires_loaded_weight_scale():
    linear = SimpleNamespace(
        quant_method=_w8a8_quant_method_wrapper(),
        custom_op=None,
        gather_output=False,
        weight=object(),
    )

    wrapper = CVLinearWrapper(linear)

    assert not wrapper._is_w8a8_dynamic
    assert wrapper.quantize("x") == ("x", None)
    assert wrapper.matmul("x") == "fallback"
    linear.quant_method.apply.assert_called_once_with(linear, "x", None)


def test_cv_linear_w8a8_detection_accepts_loaded_weight_scale():
    linear = SimpleNamespace(
        quant_method=_w8a8_quant_method_wrapper(),
        custom_op=None,
        gather_output=False,
        weight=object(),
        weight_scale=object(),
    )

    wrapper = CVLinearWrapper(linear)

    assert wrapper._is_w8a8_dynamic
