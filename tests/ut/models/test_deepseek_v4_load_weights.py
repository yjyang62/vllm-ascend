from unittest import mock

import torch

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM


class _FakeConfig:
    n_routed_experts = 4
    n_shared_experts = 1
    num_attention_heads = 4


class _FakeModelForCausalLM:
    """Minimal duck-typed stand-in for AscendDeepseekV4ForCausalLM.

    Only exposes what ``load_weights`` touches, so the (heavy, NPU-only)
    real model does not need to be constructed for this test.
    """

    def __init__(self, params: dict[str, torch.nn.Parameter]):
        self.config = _FakeConfig()
        self.num_redundant_experts = 0
        self.model = object()  # only passed through to the mocked make_expert_params_mapping
        self._params = params

    def named_parameters(self):
        return self._params.items()


def _run_load_weights(params, weights):
    fake_self = _FakeModelForCausalLM(params)
    with (
        mock.patch(
            "vllm_ascend.models.deepseek_v4.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
            return_value=False,
        ),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.get_ascend_config",
            return_value=object(),  # no `mix_placement` attr -> getattr(..., False)
        ),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.FusedMoE.make_expert_params_mapping",
            return_value=[],
        ),
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_rank", return_value=0),
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_world_size", return_value=1),
        mock.patch("vllm_ascend.models.deepseek_v4.get_spec_layer_idx_from_weight_name", return_value=None),
        # PP size is 1 in this scenario: no layer is actually PP-missing.
        # This isolates the new "layer-count-truncated" guard from the
        # pre-existing pipeline-parallel one.
        mock.patch("vllm_ascend.models.deepseek_v4.is_pp_missing_parameter", return_value=False),
        mock.patch("vllm_ascend.models.deepseek_v4.enable_dsa_cp", return_value=False),
    ):
        return AscendDeepseekV4ForCausalLM.load_weights(fake_self, weights)


def test_load_weights_skips_attn_sink_for_layer_count_truncated_model():
    """Regression for KeyError: 'model.layers.5.self_attn.attn_sink'.

    A layer-count-truncated dev/debug run ("减层" testing: hf_config's
    num_hidden_layers reduced so only the first N decoder layers are
    instantiated) still iterates the full checkpoint, which includes
    attn_sink weights for layers beyond N. is_pp_missing_parameter does not
    cover this (it only tracks real pipeline-parallel sharding), so
    load_weights must fall back to a plain params_dict membership check
    instead of crashing.
    """
    # Model only has layer 0 (simulates num_hidden_layers truncated to 1).
    sink_param = torch.nn.Parameter(torch.zeros(4))
    params = {"model.layers.0.self_attn.attn_sink": sink_param}

    weights = [
        ("model.layers.0.self_attn.attn_sink", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        # Checkpoint still has layer 5 (beyond the truncated model); this
        # used to raise KeyError.
        ("model.layers.5.self_attn.attn_sink", torch.tensor([5.0, 6.0, 7.0, 8.0])),
    ]

    loaded_params = _run_load_weights(params, weights)

    assert loaded_params == {"model.layers.0.self_attn.attn_sink"}
    torch.testing.assert_close(sink_param.data, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_load_weights_skips_generic_weight_for_layer_count_truncated_model():
    """Same truncated-layer scenario, but for a plain (non-sink, non-expert,
    non-stacked) weight that only hits the generic fallback branch.

    Note: like the pre-existing GPTQ-bias skip a few lines above it, this
    fallback's ``continue`` only exits the (trivial, single-iteration)
    ``for j in range(num_chunks)`` loop, so the checkpoint weight name is
    still recorded in the returned ``loaded_params`` set even though nothing
    was actually copied -- this matches existing behavior for other
    intentionally-skipped weights in this loader. What matters here is that
    it does not raise, and that a real parameter for an existing layer is
    still loaded correctly.
    """
    real_param = torch.nn.Parameter(torch.zeros(2, 2))
    params = {"model.layers.0.self_attn.o_proj.weight": real_param}

    weights = [
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(2, 2)),
        ("model.layers.5.self_attn.o_proj.weight", torch.full((2, 2), 9.0)),
    ]

    loaded_params = _run_load_weights(params, weights)

    assert "model.layers.0.self_attn.o_proj.weight" in loaded_params
    torch.testing.assert_close(real_param.data, torch.ones(2, 2))
