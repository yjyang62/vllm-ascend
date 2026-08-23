# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_ascend.attention.dsa_attn_kv_plan import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
    get_dsa_attn_kv_plan,
)
from vllm_ascend.attention.dsa_kv_mode import (
    DSV4_EXPLICIT_BF16_KV_KEY,
    record_dsv4_kv_mode,
    resolve_dsv4_use_bf16_kv,
    uses_explicit_bf16_kv,
)
from vllm_ascend.attention.dsa_v1 import _dsa_layout_kv, _dsa_o_proj_matmul, _dsa_swa_only_cmp_ratio
from vllm_ascend.attention.sparse_flash_mla import (
    sparse_flash_mla,
    sparse_flash_mla_metadata,
)
from vllm_ascend.ops.linear import _requires_a5_bf16_wo_a_layout
from vllm_ascend.utils import AscendDeviceType


def _dsv4_config(
    *,
    cache_dtype="auto",
    model_dtype=torch.bfloat16,
    quantization=None,
    quant_config=None,
    recorded=None,
):
    hf_text_config = SimpleNamespace(index_topk=512)
    additional_config = {}
    if recorded is not None:
        additional_config[DSV4_EXPLICIT_BF16_KV_KEY] = recorded
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=hf_text_config,
            dtype=model_dtype,
            quantization=quantization,
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        quant_config=quant_config,
        additional_config=additional_config,
    )


def test_auto_bf16_checkpoint_defaults_to_fp8_kv():
    config = _dsv4_config(cache_dtype="auto", model_dtype=torch.bfloat16)
    assert not resolve_dsv4_use_bf16_kv(config)


def test_auto_fp8_quantized_checkpoint_defaults_to_fp8_kv():
    quant_config = SimpleNamespace(get_name=lambda: "deepseek_v4_fp8")
    config = _dsv4_config(
        cache_dtype="auto",
        model_dtype=torch.bfloat16,
        quantization="deepseek_v4_fp8",
        quant_config=quant_config,
    )
    assert not resolve_dsv4_use_bf16_kv(config)


def test_explicit_kv_cache_dtype_overrides_auto():
    bf16_model = _dsv4_config(cache_dtype="auto", model_dtype=torch.bfloat16)
    assert not resolve_dsv4_use_bf16_kv(bf16_model)

    fp8_override = _dsv4_config(cache_dtype="fp8", model_dtype=torch.bfloat16)
    assert not resolve_dsv4_use_bf16_kv(fp8_override)

    bf16_override = _dsv4_config(
        cache_dtype="bfloat16",
        model_dtype=torch.bfloat16,
        quantization="deepseek_v4_fp8",
        quant_config=SimpleNamespace(get_name=lambda: "deepseek_v4_fp8"),
    )
    assert resolve_dsv4_use_bf16_kv(bf16_override)


def test_record_dsv4_kv_mode_persists_before_platform_rewrite():
    config = _dsv4_config(cache_dtype="auto", model_dtype=torch.bfloat16)
    record_dsv4_kv_mode(config, config.additional_config)
    assert config.additional_config[DSV4_EXPLICIT_BF16_KV_KEY] is False

    config.cache_config.cache_dtype = "bfloat16"
    assert not uses_explicit_bf16_kv(config)


def test_recorded_mode_is_used_after_platform_rewrite():
    config = _dsv4_config(cache_dtype="fp8", model_dtype=torch.bfloat16)
    record_dsv4_kv_mode(config, config.additional_config)
    assert config.additional_config[DSV4_EXPLICIT_BF16_KV_KEY] is False

    config.cache_config.cache_dtype = "bfloat16"
    assert not uses_explicit_bf16_kv(config)


def test_mode_without_current_vllm_config_defaults_to_fp8():
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        side_effect=AssertionError("Current vLLM config is not set"),
    ):
        assert not uses_explicit_bf16_kv()


def test_a5_fp8_selectors_remain_identical_to_main():
    flat_slots = torch.tensor([5, -1], dtype=torch.int32)
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_dsv4_config(recorded=False))
        assert plan.get_dsa_sparse_attn_op() is torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}
        assert plan.get_dsa_sparse_attn_base_kwargs() == {
            "kv_quant_mode": 1,
            "tile_size": 64,
            "rope_head_dim": 64,
        }
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert plan.format_dsa_slot_mapping(flat_slots, 128) is flat_slots


def test_a5_bf16_selectors_use_sparse_flash_mla():
    flat_slots = torch.tensor([5, -1], dtype=torch.int32)
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_dsv4_config(recorded=True))
        assert plan.get_dsa_sparse_attn_op() is sparse_flash_mla
        assert plan.get_dsa_sparse_attn_metadata_op() is sparse_flash_mla_metadata
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"device": "npu:0"}
        assert plan.get_dsa_sparse_attn_base_kwargs() == {}
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        torch.testing.assert_close(
            plan.format_dsa_slot_mapping(flat_slots, 128),
            torch.tensor([[0, 5], [-1, -1]], dtype=torch.int32),
        )


def test_a5_bf16_wo_a_layout_is_independent_of_kv_dtype():
    with mock.patch("vllm_ascend.ops.linear.get_ascend_device_type", return_value=AscendDeviceType.A5):
        assert _requires_a5_bf16_wo_a_layout("model.layers.0.wo_a", None, torch.bfloat16)
        assert not _requires_a5_bf16_wo_a_layout("model.layers.0.wo_a", None, torch.float8_e4m3fn)
        assert not _requires_a5_bf16_wo_a_layout("model.layers.0.wo_a", object(), torch.bfloat16)


def test_sparse_flash_mla_adapter_enforces_bf16_paged_layout():
    metadata_op = mock.Mock(return_value=torch.empty(0))
    attention_op = mock.Mock(return_value=torch.empty(0))
    with mock.patch(
        "vllm_ascend.attention.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        sparse_flash_mla_metadata(layout_kv="PA_ND")
        sparse_flash_mla(torch.empty(0), layout_kv="PA_ND")

    assert metadata_op.call_args.kwargs["layout_kv"] == "PA_BBND"
    assert attention_op.call_args.kwargs["layout_kv"] == "PA_BBND"


def test_layout_and_cmp_ratio_match_main_outside_bf16():
    with mock.patch("vllm_ascend.attention.dsa_v1.uses_explicit_bf16_kv", return_value=False):
        assert _dsa_layout_kv() == "PA_ND"
        for compress_ratio in (0, 1, 4, 128):
            assert _dsa_swa_only_cmp_ratio(compress_ratio) == max(compress_ratio, 1)


def test_layout_and_cmp_ratio_switch_for_bf16():
    with mock.patch("vllm_ascend.attention.dsa_v1.uses_explicit_bf16_kv", return_value=True):
        assert _dsa_layout_kv() == "PA_BBND"
        assert _dsa_swa_only_cmp_ratio(0) == 0
        assert _dsa_swa_only_cmp_ratio(1) == 0
        assert _dsa_swa_only_cmp_ratio(4) == 4

    config = _dsv4_config(recorded=True)
    assert _dsa_layout_kv(config) == "PA_BBND"
    assert _dsa_swa_only_cmp_ratio(1, config) == 0


def test_a5_bf16_o_proj_matmul_matches_grouped_einsum():
    o_proj_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    output = _dsa_o_proj_matmul(o_proj_input, weight, num_groups=3)
    expected = torch.einsum("tgd,grd->tgr", o_proj_input, weight.view(3, 2, 4))
    torch.testing.assert_close(output, expected)
