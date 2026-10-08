# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4


def _run_forward(model, inputs_embeds):
    from vllm.forward_context import ForwardContext, override_forward_context

    layer = MagicMock()
    layer.layer_idx = 0
    layer.side_effect = lambda _positions, hidden_states, *_args, **_kwargs: (hidden_states, None)
    model.layers = [layer]
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
    )
    with override_forward_context(forward_context):
        deepseek_v4.DeepseekV4Model.forward(
            model,
            torch.zeros(inputs_embeds.shape[0], dtype=torch.int32),
            positions=torch.arange(inputs_embeds.shape[0]),
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )
    return layer


def _model(monkeypatch, *, hc_mult: int, buffer: torch.Tensor | None):
    monkeypatch.setattr(
        deepseek_v4,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    hidden_size = 3
    return SimpleNamespace(
        hc_mult=hc_mult,
        use_sequence_parallel_moe=False,
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=set(),
        _needs_mtp_hidden_states=True,
        _mtp_hidden_buffer=buffer,
        hc_head=lambda hidden_states, *_args: hidden_states.flatten(1)[:, :hidden_size],
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda hidden_states: hidden_states,
    )


def test_forward_copies_pre_hc_residual_into_existing_buffer(monkeypatch):
    hc_mult = 4
    hidden_size = 3
    buffer = torch.full((4, hc_mult * hidden_size), -7.0)
    model = _model(monkeypatch, hc_mult=hc_mult, buffer=buffer)
    first = torch.arange(2 * hidden_size, dtype=torch.float32).view(2, hidden_size)
    second = torch.arange(hidden_size, dtype=torch.float32).view(1, hidden_size) + 20

    _run_forward(model, first)

    expected_first = first.unsqueeze(1).repeat(1, hc_mult, 1).flatten(1)
    torch.testing.assert_close(buffer[:2], expected_first)
    torch.testing.assert_close(buffer[2:], torch.full_like(buffer[2:], -7.0))
    assert model._mtp_hidden_buffer is buffer

    _run_forward(model, second)

    expected_second = second.unsqueeze(1).repeat(1, hc_mult, 1).flatten(1)
    torch.testing.assert_close(buffer[:1], expected_second)
    torch.testing.assert_close(buffer[1:2], expected_first[1:2])
    assert model._mtp_hidden_buffer is buffer


def test_forward_does_not_allocate_mtp_hidden_buffer(monkeypatch):
    model = _model(monkeypatch, hc_mult=4, buffer=None)
    inputs = torch.arange(6, dtype=torch.float32).view(2, 3)

    _run_forward(model, inputs)

    assert model._mtp_hidden_buffer is None
