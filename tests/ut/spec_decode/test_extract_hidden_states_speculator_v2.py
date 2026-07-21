from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.extract_hidden_states.speculator import (
    AscendExtractHiddenStatesSpeculator,
    _update_valid_hidden_state_slots,
)


def _make_speculator() -> AscendExtractHiddenStatesSpeculator:
    speculator = AscendExtractHiddenStatesSpeculator.__new__(AscendExtractHiddenStatesSpeculator)
    speculator.device = torch.device("cpu")
    speculator.runner = MagicMock()
    speculator.runner._sync_metadata_across_dp.return_value = (
        4,
        None,
        CUDAGraphMode.NONE,
    )
    speculator.dp_rank = 0
    speculator.attn_layer_names = ["cache_only_layers.0"]
    speculator.kv_cache_gid = 1
    speculator.block_tables = MagicMock()
    speculator.block_tables.input_block_tables = [
        torch.zeros(2, 4, dtype=torch.int32),
        torch.ones(2, 4, dtype=torch.int32),
    ]
    speculator.hidden_states = torch.zeros(8, 2, 8)
    speculator.slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)
    speculator._dispatch_and_sync = MagicMock(return_value=(CUDAGraphMode.NONE, 3, None))
    speculator._run_cache_only_model = MagicMock()
    return speculator


def _make_input_batch():
    input_batch = MagicMock()
    input_batch.num_reqs = 2
    input_batch.num_tokens = 3
    input_batch.num_tokens_after_padding = 4
    input_batch.idx_mapping = torch.tensor([1, 3], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor(
        [0, 1, 3],
        dtype=torch.int32,
    )
    input_batch.query_start_loc_np = np.array([0, 1, 3], dtype=np.int32)
    input_batch.seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    input_batch.seq_lens_np = np.array([5, 7], dtype=np.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.tensor(
        [5, 7],
        dtype=torch.int32,
    )
    input_batch.num_scheduled_tokens = np.array([1, 2], dtype=np.int32)
    input_batch.positions = torch.tensor([4, 5, 6, 0], dtype=torch.int64)
    input_batch.attn_state = MagicMock()
    return input_batch


def test_propose_runs_native_v2_cache_only_path():
    speculator = _make_speculator()
    input_batch = _make_input_batch()
    last_sampled = torch.tensor(
        [[10], [13], [20], [23]],
        dtype=torch.int64,
    )
    next_prefill_tokens = torch.tensor(
        [7, 8, 9, 29],
        dtype=torch.int32,
    )
    aux_hidden_states = [
        torch.randn(4, 8),
        torch.randn(4, 8),
    ]
    slot_mapping = torch.tensor([0, 1, 2, -1], dtype=torch.int32)

    output = speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={"cache_only_layers.0": slot_mapping},
        last_hidden_states=torch.randn(4, 8),
        aux_hidden_states=aux_hidden_states,
        num_sampled=torch.tensor([1, 0], dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill_tokens,
        temperature=torch.ones(4),
        seeds=torch.zeros(4, dtype=torch.int64),
    )

    assert torch.equal(
        output,
        torch.tensor([[13], [29]], dtype=torch.int64),
    )
    expected_hidden_states = torch.stack(
        [hidden[:3] for hidden in aux_hidden_states],
        dim=1,
    )
    assert torch.equal(
        speculator.hidden_states[:3],
        expected_hidden_states,
    )

    kwargs = speculator._run_cache_only_model.call_args.kwargs
    common_metadata = kwargs["common_attn_metadata"]
    assert common_metadata.num_actual_tokens == 3
    assert common_metadata.max_query_len == 2
    assert common_metadata.max_seq_len == 7
    assert common_metadata.block_table_tensor.data_ptr() == (speculator.block_tables.input_block_tables[1].data_ptr())
    assert "cache_only_layers.0" in kwargs["slot_mapping"]


def test_propose_requires_aux_hidden_states():
    speculator = _make_speculator()

    with pytest.raises(ValueError, match="aux_hidden_states"):
        speculator.propose(
            input_batch=_make_input_batch(),
            attn_metadata={},
            slot_mappings={"cache_only_layers.0": torch.zeros(4)},
            last_hidden_states=torch.zeros(4, 8),
            aux_hidden_states=None,
            num_sampled=torch.ones(2, dtype=torch.int32),
            num_rejected=torch.zeros(2, dtype=torch.int32),
            last_sampled=torch.zeros(4, 1, dtype=torch.int64),
            next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
            temperature=torch.ones(4),
            seeds=torch.zeros(4, dtype=torch.int64),
        )


def test_dummy_run_uses_native_speculator_method():
    speculator = _make_speculator()
    speculator._dummy_run = MagicMock()
    input_batch = _make_input_batch()

    output = speculator.propose(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.zeros(4, 8),
        aux_hidden_states=None,
        num_sampled=torch.ones(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        last_sampled=torch.zeros(4, 1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(4, dtype=torch.int32),
        temperature=torch.ones(4),
        seeds=torch.zeros(4, dtype=torch.int64),
        dummy_run=True,
        is_profile=True,
    )

    speculator._dummy_run.assert_called_once()
    assert output.shape == (2, 1)


def test_capture_forwards_slot_mappings_to_native_dummy_run():
    speculator = _make_speculator()
    speculator._dummy_run = MagicMock()
    batch_desc = MagicMock()
    batch_desc.num_tokens = 4
    attention_state = MagicMock()
    attention_state.slot_mappings = {
        "cache_only_layers.0": torch.tensor(
            [0, 1, 2, -1],
            dtype=torch.int32,
        )
    }

    speculator.capture({batch_desc: attention_state})

    kwargs = speculator._dummy_run.call_args.kwargs
    assert kwargs["num_tokens"] == 4
    assert kwargs["slot_mappings"] is attention_state.slot_mappings


def test_padding_slots_do_not_write_hidden_state_cache():
    impl = MagicMock()
    impl.kv_cache_torch_dtype = torch.float32
    kv_cache = torch.zeros(2, 2, 1, 1, dtype=torch.float32)
    to_cache = torch.tensor([1.0, 2.0, 9.0]).reshape(3, 1, 1)
    slot_mapping = torch.tensor([0, 3, -1], dtype=torch.int64)

    _update_valid_hidden_state_slots(
        impl,
        MagicMock(),
        to_cache,
        kv_cache,
        slot_mapping,
    )

    assert torch.equal(
        kv_cache.flatten(),
        torch.tensor([1.0, 0.0, 0.0, 2.0]),
    )


def test_dummy_without_slot_mapping_skips_cache_update():
    speculator = _make_speculator()

    speculator._dummy_run(
        num_tokens=4,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
    )

    kwargs = speculator._run_cache_only_model.call_args.kwargs
    assert kwargs["slot_mapping"] == {}
