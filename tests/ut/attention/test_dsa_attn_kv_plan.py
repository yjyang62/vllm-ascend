# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import mock

import torch

from vllm_ascend.attention.dsa_attn_kv_plan import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
    get_dsa_attn_kv_plan,
)
from vllm_ascend.attention.dsa_kv_mode import DSV4_EXPLICIT_BF16_KV_KEY
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla
from vllm_ascend.utils import AscendDeviceType


def _config(use_bf16: bool):
    return SimpleNamespace(additional_config={DSV4_EXPLICIT_BF16_KV_KEY: use_bf16})


def test_a5_fp8_plan_uses_flat_shared_kv():
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}


def test_a5_bf16_plan_uses_sparse_flash_mla():
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_sparse_attn_op() is sparse_flash_mla
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        torch.testing.assert_close(
            plan.format_dsa_slot_mapping(torch.tensor([5, -1], dtype=torch.int32), 128),
            torch.tensor([[0, 5], [-1, -1]], dtype=torch.int32),
        )


def test_non_a5_plan_preserves_shared_kv_runtime_kwargs():
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A3):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        kwargs = {}
        plan.add_dsa_sparse_attn_extra_kwargs(kwargs, cu_seqlens_ori_kv=torch.tensor([0, 1]))
        assert "cu_seqlens_ori_kv" in kwargs


def test_scatter_skips_none_updates():
    with mock.patch("vllm_ascend.attention.dsa_attn_kv_plan.get_ascend_device_type", return_value=AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        cache = torch.zeros(2, 1, 4)
        with mock.patch.object(torch.ops._C_ascend, "kv_compress_epilog") as epilog:
            plan.dsa_kv_compress_scatter(cache, None, torch.tensor([0], dtype=torch.int32))
            epilog.assert_not_called()
