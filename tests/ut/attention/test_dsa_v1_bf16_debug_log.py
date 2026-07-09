import logging

import pytest
import torch

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.dsa_v1 import (
    _dsv4_bf16_debug_accumulate_output_stats,
    _dsv4_bf16_debug_flush_deferred,
    _dsv4_bf16_debug_layer_sort_key,
    _dsv4_bf16_debug_log_output_stats,
)


@pytest.fixture(autouse=True)
def _clear_deferred_debug_state():
    dsa_v1._DSV4_BF16_DEBUG_DEFERRED_ACCUM.clear()
    dsa_v1._DSV4_BF16_DEBUG_DEFERRED_SEEN.clear()
    yield
    dsa_v1._DSV4_BF16_DEBUG_DEFERRED_ACCUM.clear()
    dsa_v1._DSV4_BF16_DEBUG_DEFERRED_SEEN.clear()


def test_logs_finite_stats(caplog):
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with caplog.at_level(logging.INFO, logger="vllm"):
        _dsv4_bf16_debug_log_output_stats("final_o_proj", "model.layers.3.self_attn", 4, output)

    messages = [r.message for r in caplog.records]
    assert any("[DSV4_BF16_DEBUG]" in m for m in messages)
    logged = next(m for m in messages if "[DSV4_BF16_DEBUG]" in m)
    assert "stage=final_o_proj" in logged
    assert "layer=model.layers.3.self_attn" in logged
    assert "compress_ratio=4" in logged
    assert "has_nan=False" in logged
    assert "has_inf=False" in logged
    assert "mean=2.5" in logged


def test_logs_nan_and_inf():
    output = torch.tensor([[float("nan"), 1.0], [float("inf"), -1.0]])
    # Should not raise even though part of the tensor is non-finite.
    _dsv4_bf16_debug_log_output_stats("raw_attn_decode", "model.layers.0.self_attn", 1, output)


def test_logs_all_non_finite_without_crashing():
    output = torch.tensor([float("nan"), float("inf"), float("-inf")])
    # finite subset is empty -- must not divide-by-zero / crash on empty .mean().
    _dsv4_bf16_debug_log_output_stats("raw_attn_prefill", "model.layers.0.self_attn", 128, output)


def test_deferred_layer_sort_key_orders_numerically_not_lexically():
    keys = [
        "final_o_proj:model.layers.10.self_attn",
        "final_o_proj:model.layers.2.self_attn",
        "final_o_proj:model.layers.1.self_attn",
    ]
    assert sorted(keys, key=_dsv4_bf16_debug_layer_sort_key) == [
        "final_o_proj:model.layers.1.self_attn",
        "final_o_proj:model.layers.2.self_attn",
        "final_o_proj:model.layers.10.self_attn",
    ]


def test_deferred_accumulate_does_not_log_until_a_new_step_starts(caplog):
    """The whole point of the deferred variant: no per-layer .item()/log at
    accumulation time (that would mask the race the same way the immediate
    variant does). Only flush -- and only then call .item() -- once a
    (stage, layer) key repeats, signaling the previous step is complete."""
    with caplog.at_level(logging.INFO, logger="vllm"):
        for layer in range(3):
            _dsv4_bf16_debug_accumulate_output_stats(
                "final_o_proj", f"model.layers.{layer}.self_attn", 4, torch.tensor([1.0, 2.0])
            )
        assert not any("[DSV4_BF16_DEBUG_DEFERRED]" in r.message for r in caplog.records)

        # Layer 0 repeats -> a new step started -> flush the previous one now.
        _dsv4_bf16_debug_accumulate_output_stats(
            "final_o_proj", "model.layers.0.self_attn", 4, torch.tensor([1.0, 2.0])
        )

    messages = [r.message for r in caplog.records if "[DSV4_BF16_DEBUG_DEFERRED]" in r.message]
    assert len(messages) == 3
    # Flushed in ascending layer order.
    assert [f"model.layers.{i}.self_attn" in messages[i] for i in range(3)] == [True, True, True]
    for m in messages:
        assert "bad=False" in m


def test_deferred_flush_reports_nan_and_clears_state():
    _dsv4_bf16_debug_accumulate_output_stats(
        "raw_attn_decode", "model.layers.5.self_attn", 128, torch.tensor([float("nan"), 1.0])
    )
    _dsv4_bf16_debug_flush_deferred()

    assert dsa_v1._DSV4_BF16_DEBUG_DEFERRED_ACCUM == {}
    assert set() == dsa_v1._DSV4_BF16_DEBUG_DEFERRED_SEEN

    # Safe (no-op) when there is nothing pending, e.g. the atexit handler
    # firing after a normal flush already happened.
    _dsv4_bf16_debug_flush_deferred()
