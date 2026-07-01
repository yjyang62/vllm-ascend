import logging

import torch

from vllm_ascend.attention.dsa_v1 import _dsv4_bf16_debug_log_output_stats


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
