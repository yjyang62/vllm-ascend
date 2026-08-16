# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.quantization import fp8_config as fp8_config_module
from vllm_ascend.quantization.fp8_config import (
    AscendFp8Config,
    checkpoint_experts_look_unquantized,
    should_use_unquantized_dsv4_moe,
)


@pytest.mark.parametrize(
    ("expert_dtype", "expected"),
    [
        ("bf16", True),
        ("bfloat16", True),
        ("fp16", True),
        ("fp4", False),
        ("fp8", False),
        (None, False),
    ],
)
def test_should_use_unquantized_dsv4_moe_from_expert_dtype(expert_dtype, expected):
    hf_config = SimpleNamespace(expert_dtype=expert_dtype, model_type="deepseek_v4")
    assert should_use_unquantized_dsv4_moe(hf_config, model=None) is expected


def test_should_use_unquantized_when_fp4_but_checkpoint_has_no_expert_scales(monkeypatch):
    monkeypatch.setattr(
        fp8_config_module,
        "checkpoint_experts_look_unquantized",
        lambda model, revision=None: True,
    )
    hf_config = SimpleNamespace(expert_dtype="fp4", model_type="deepseek_v4")
    assert should_use_unquantized_dsv4_moe(hf_config, model="/weights/DeepSeek-V4-HF-BF16") is True


def test_checkpoint_experts_look_unquantized_detects_scale_keys(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map": {'
        '"model.layers.0.mlp.experts.0.gate_proj.weight": "a.safetensors",'
        '"model.layers.0.mlp.experts.0.gate_proj.weight_scale": "a.safetensors"'
        "}}"
    )
    monkeypatch_model = str(tmp_path)
    # Point get_model_file at the temp index via Path.exists layout.
    from vllm_ascend.quantization import utils as quant_utils

    assert checkpoint_experts_look_unquantized(monkeypatch_model) is False

    index.write_text(
        '{"weight_map": {'
        '"model.layers.0.mlp.experts.0.gate_proj.weight": "a.safetensors",'
        '"model.layers.0.mlp.experts.0.up_proj.weight": "a.safetensors",'
        '"model.layers.0.mlp.experts.0.down_proj.weight": "a.safetensors"'
        "}}"
    )
    assert checkpoint_experts_look_unquantized(monkeypatch_model) is True
    assert quant_utils.get_model_file(monkeypatch_model, "model.safetensors.index.json") == index


def test_ascend_fp8_moe_uses_unquantized_for_bf16_experts(monkeypatch):
    config = AscendFp8Config(ignore=[], quant_format=None, config={})
    moe_config = MagicMock()
    layer = MagicMock(spec=fp8_config_module.RoutedExperts)
    layer.moe_config = moe_config

    unquant_method = object()
    monkeypatch.setattr(config, "_should_use_unquantized_moe", lambda: True)

    import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module

    fake_cls = MagicMock(return_value=unquant_method)
    monkeypatch.setattr(routed_experts_module, "AscendUnquantizedFusedMoEMethod", fake_cls)

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
    assert method is unquant_method
    fake_cls.assert_called_once_with(moe_config, tid2eid=None)
    assert layer.ascend_quant_method == "unquantized"


def test_ascend_fp8_moe_keeps_w4a8_for_fp4_experts(monkeypatch):
    config = AscendFp8Config(ignore=[], quant_format=None, config={})
    moe_config = MagicMock()
    layer = MagicMock(spec=fp8_config_module.RoutedExperts)
    layer.moe_config = moe_config

    monkeypatch.setattr(config, "_should_use_unquantized_moe", lambda: False)

    scheme = object()
    fused_method = object()
    monkeypatch.setattr(fp8_config_module, "create_scheme_for_layer", lambda *a, **k: scheme)

    import vllm_ascend.quantization.method_adapters as adapters

    monkeypatch.setattr(adapters, "AscendFusedMoEMethod", MagicMock(return_value=fused_method))

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
    assert method is fused_method
    adapters.AscendFusedMoEMethod.assert_called_once_with(scheme, moe_config, tid2eid=None)
    assert layer.ascend_quant_method == "fp8"
