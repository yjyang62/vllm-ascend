# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_ascend.attention.dsa_kv_mode import DSV4_EXPLICIT_BF16_KV_KEY, uses_explicit_bf16_kv
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.device_op import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
    A5DeviceAdaptor,
)


def _config(explicit_bf16: bool):
    return SimpleNamespace(additional_config={DSV4_EXPLICIT_BF16_KV_KEY: explicit_bf16})


def test_explicit_bf16_mode_is_opt_in():
    assert uses_explicit_bf16_kv(_config(True))
    assert not uses_explicit_bf16_kv(_config(False))
    assert not uses_explicit_bf16_kv(SimpleNamespace(additional_config={}))


def test_mode_without_current_vllm_config_defaults_to_fp8():
    with mock.patch(
        "vllm.config.get_current_vllm_config",
        side_effect=AssertionError("Current vLLM config is not set"),
    ):
        assert not uses_explicit_bf16_kv()


def test_a5_fp8_selectors_remain_identical_to_main():
    flat_slots = torch.tensor([5, -1], dtype=torch.int32)
    with mock.patch("vllm_ascend.device.device_op.uses_explicit_bf16_kv", return_value=False):
        assert (
            A5DeviceAdaptor.get_dsa_sparse_attn_op()
            is torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv
        )
        assert (
            A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op()
            is torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
        )
        assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}
        assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs() == {
            "kv_quant_mode": 1,
            "tile_size": 64,
            "rope_head_dim": 64,
        }
        assert A5DeviceAdaptor.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert A5DeviceAdaptor.get_dsa_layout_kv() == "PA_ND"
        assert not A5DeviceAdaptor.dsa_requires_block_offset_slots()
        assert A5DeviceAdaptor.format_dsa_slot_mapping(flat_slots, 128) is flat_slots


def test_a5_explicit_bf16_selectors_use_sparse_flash_mla():
    flat_slots = torch.tensor([5, -1], dtype=torch.int32)
    with mock.patch("vllm_ascend.device.device_op.uses_explicit_bf16_kv", return_value=True):
        assert A5DeviceAdaptor.get_dsa_sparse_attn_op() is sparse_flash_mla
        assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op() is sparse_flash_mla_metadata
        assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"device": "npu:0"}
        assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs() == {}
        assert (
            A5DeviceAdaptor.get_dsa_compressor_slot_mapping_format()
            == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        )
        assert A5DeviceAdaptor.get_dsa_layout_kv() == "PA_BBND"
        assert A5DeviceAdaptor.dsa_requires_block_offset_slots()
        torch.testing.assert_close(
            A5DeviceAdaptor.format_dsa_slot_mapping(flat_slots, 128),
            torch.tensor([[0, 5], [-1, -1]], dtype=torch.int32),
        )
