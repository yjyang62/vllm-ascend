# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention import dsa_attn_kv_plan
from vllm_ascend.models.deepseek_v4.compressor import (
    AscendCompressorMetadata,
    AscendCompressorStateCache,
    Compressor,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage


class TestCompressorMetadata:
    def test_compute_metadata_waits_for_precomputed_output(self):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        outputs = (torch.ones(1), torch.zeros(1), torch.zeros(1, dtype=torch.int32))
        metadata = SimpleNamespace(
            compressor_metadata=outputs,
            compressor_metadata_group_id=11,
        )

        with patch("vllm_ascend.models.deepseek_v4.compressor.wait_for_device_metadata") as wait:
            assert compressor._compute_metadata(metadata) is outputs

        wait.assert_called_once_with(DeviceMetadataStage.COMPRESSOR, 11)

    def test_compute_metadata_flattens_rotary_inputs(self):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.compress_ratio = 4
        compressor.vllm_config = SimpleNamespace()
        full_cos = torch.arange(8).view(2, 1, 1, 4)
        full_sin = -full_cos
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        start_pos = torch.tensor([1, 3], dtype=torch.int32)
        block_table = torch.tensor([[0], [1]], dtype=torch.int32)
        metadata = SimpleNamespace(
            cache_group_key="model.layers.0.self_attn.attn",
            full_compress_cos=full_cos,
            full_compress_sin=full_sin,
            query_start_loc=query_start_loc,
            start_pos=start_pos,
            block_table=block_table,
            storage_block_size=128,
            num_compressed_tokens=3,
            num_actual_reqs=2,
        )
        result_cos = torch.ones((3, 4))
        result_sin = torch.zeros((3, 4))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        plan = SimpleNamespace(get_dsa_compressor_slot_mapping_format=MagicMock(return_value=7))

        with (
            patch.object(
                dsa_attn_kv_plan,
                "get_dsa_attn_kv_plan",
                return_value=plan,
            ),
            patch(
                "vllm_ascend.attention.dsa_v1.get_dsa_attn_kv_plan",
                return_value=plan,
            ),
            patch(
                "vllm_ascend.attention.dsa_v1.get_forward_context",
                return_value=SimpleNamespace(additional_kwargs={}),
            ),
            patch.object(
                torch.ops._C_ascend,
                "compressor_metadata",
                create=True,
                return_value=(result_cos, result_sin, slot_mapping),
            ) as metadata_op,
        ):
            actual = compressor._compute_metadata(metadata)

        assert actual[0] is result_cos
        assert actual[1] is result_sin
        assert actual[2] is slot_mapping
        plan.get_dsa_compressor_slot_mapping_format.assert_called_once_with()
        args = metadata_op.call_args.args
        assert torch.equal(args[0], full_cos.view(2, 4))
        assert torch.equal(args[1], full_sin.view(2, 4))
        assert args[2] is query_start_loc
        assert args[3] is start_pos
        assert args[4] is block_table
        assert args[5:] == (128, 7, 4, 3, 2)


class TestCompressorForward:
    @pytest.mark.parametrize(
        ("compress_ratio", "overlap", "expected_coff"),
        [(4, True, 2), (128, False, 1)],
    )
    def test_routes_cache_and_state_metadata(
        self,
        compress_ratio: int,
        overlap: bool,
        expected_coff: int,
    ):
        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.overlap = overlap
        compressor.compress_ratio = compress_ratio
        compressor.rope_head_dim = 2
        compressor.norm_eps = 1e-6
        compressor.ape = torch.ones((compress_ratio, 4))
        compressor.wkv = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.wgate = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.norm = SimpleNamespace(weight=torch.ones(4))
        compressor.state_cache = SimpleNamespace(
            sliding_window=8 if compress_ratio == 4 else 128,
            block_size=8 if compress_ratio == 4 else 16,
        )
        compressor.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        )
        cache_req_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            start_pos=torch.tensor([1], dtype=torch.int32),
        )
        state_req_metadata = SimpleNamespace(block_table=torch.tensor([[3]], dtype=torch.int32))
        metadata = AscendCompressorMetadata(
            cache=SimpleNamespace(req_metadata=cache_req_metadata),
            state=SimpleNamespace(req_metadata=state_req_metadata),
        )
        hidden_states = torch.ones((2, 4))
        state_cache = torch.ones((1, 2, 1, 4))
        compress_cos = torch.ones((1, 1, 2))
        compress_sin = torch.zeros((1, 1, 2))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        compressed_kv = torch.ones((1, 1, 4))
        compute_metadata = MagicMock(return_value=(compress_cos, compress_sin, slot_mapping))
        compressor._compute_metadata = compute_metadata

        with patch.object(
            torch.ops._C_ascend,
            "compressor",
            create=True,
            return_value=compressed_kv,
        ) as compressor_op:
            actual_kv, actual_slot_mapping = compressor(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=metadata,
            )

        assert actual_kv is compressed_kv
        assert actual_slot_mapping is slot_mapping
        compute_metadata.assert_called_once_with(cache_req_metadata)
        call = compressor_op.call_args
        assert call.args[0] is hidden_states
        assert torch.equal(call.args[3], state_cache.squeeze(-2))
        assert call.kwargs["state_block_table"] is state_req_metadata.block_table
        assert call.kwargs["cu_seqlens"] is cache_req_metadata.query_start_loc
        assert call.kwargs["start_pos"] is cache_req_metadata.start_pos
        assert call.kwargs["cmp_ratio"] == compress_ratio
        assert call.kwargs["coff"] == expected_coff

    def test_long_mtp_decode_folds_state_block_table_into_window(self):
        """A >20k MTP decode must hand the kernel window-local physical blocks.

        V1 metadata already rebases ``start_pos`` and the state block ids by
        ``sliding_window``. V2 keeps absolute columns, and CONTINUOUS mode
        indexes ``start_pos / block_size`` with no bounds check.
        """
        start_pos_value = 27697
        query_tokens = 2
        compress_ratio = 4
        block_size = 8
        sliding_window = 8
        absolute_column = start_pos_value // block_size
        block_table = torch.arange(absolute_column + 8, dtype=torch.int32).view(1, -1) + 10
        stale_prefix_id = int(block_table[0, 0])
        live_block_id = int(block_table[0, absolute_column])
        start_pos = torch.tensor([start_pos_value], dtype=torch.int32)
        original_start = start_pos.clone()
        original_table = block_table.clone()

        compressor = Compressor.__new__(Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.overlap = True
        compressor.compress_ratio = compress_ratio
        compressor.rope_head_dim = 2
        compressor.norm_eps = 1e-6
        compressor.ape = torch.ones((compress_ratio, 4))
        compressor.wkv = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.wgate = SimpleNamespace(weight=torch.ones((4, 4)))
        compressor.norm = SimpleNamespace(weight=torch.ones(4))
        compressor.state_cache = SimpleNamespace(sliding_window=sliding_window, block_size=block_size)
        compressor.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=4),
        )
        cache_req_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, query_tokens], dtype=torch.int32),
            start_pos=start_pos,
        )
        state_req_metadata = SimpleNamespace(block_table=block_table)
        metadata = AscendCompressorMetadata(
            cache=SimpleNamespace(req_metadata=cache_req_metadata),
            state=SimpleNamespace(req_metadata=state_req_metadata),
        )
        compress_cos = torch.ones((1, 1, 2))
        compress_sin = torch.zeros((1, 1, 2))
        slot_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        compressor._compute_metadata = MagicMock(return_value=(compress_cos, compress_sin, slot_mapping))

        with patch.object(
            torch.ops._C_ascend,
            "compressor",
            create=True,
            return_value=torch.ones((1, 1, 4)),
        ) as compressor_op:
            compressor(
                hidden_states=torch.ones((query_tokens, 4)),
                state_cache=torch.ones((1, block_size, 1, 4)),
                metadata=metadata,
            )

        passed_table = compressor_op.call_args.kwargs["state_block_table"]
        passed_start = compressor_op.call_args.kwargs["start_pos"]
        folded_start = int(passed_start[0])
        folded_column = folded_start // block_size
        assert passed_table.shape[1] < block_table.shape[1]
        assert folded_column < passed_table.shape[1]
        assert int(passed_table[0, folded_column]) == live_block_id
        assert int(passed_table[0, folded_column]) != stale_prefix_id
        assert folded_start % block_size == start_pos_value % block_size
        assert passed_start is not start_pos
        assert torch.equal(start_pos, original_start)
        assert torch.equal(block_table, original_table)
        compressor._compute_metadata.assert_called_once_with(cache_req_metadata)


class TestCompressorStateCache:
    @pytest.mark.parametrize(
        ("state_dim", "compress_ratio", "padding_index"),
        [(2 * 256, 4, 0), (2 * 1024, 4, 1), (2 * 512, 128, 1)],
    )
    def test_cache_spec_selects_expected_page_padding(
        self,
        state_dim: int,
        compress_ratio: int,
        padding_index: int,
    ):
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        cache = AscendCompressorStateCache.__new__(AscendCompressorStateCache)
        cache.state_dim = state_dim
        cache.compress_ratio = compress_ratio
        cache.block_size = 8
        cache.dtype = torch.float32
        cache.sliding_window = 64
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        )

        spec = cache.get_kv_cache_spec(vllm_config)

        assert spec.block_size == 8
        assert spec.head_size == state_dim
        assert spec.sliding_window == 64
        assert spec.page_size_padded == DSV4_BLOCK_SIZES[128][1][padding_index]
