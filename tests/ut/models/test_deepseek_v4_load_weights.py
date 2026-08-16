# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from unittest import mock

import torch

from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM, _expert_param_name_candidates


class _FakeConfig:
    n_routed_experts = 4
    n_shared_experts = 1
    num_attention_heads = 4


class _FakeModelForCausalLM:
    """Minimal duck-typed stand-in for AscendDeepseekV4ForCausalLM."""

    def __init__(self, params: dict[str, torch.nn.Parameter]):
        self.config = _FakeConfig()
        self.num_redundant_experts = 0
        self.model = object()
        self._params = params

    def named_parameters(self):
        return self._params.items()


def _run_load_weights(params, weights, expert_mapping=None):
    fake_self = _FakeModelForCausalLM(params)
    mapping = [] if expert_mapping is None else expert_mapping
    with (
        mock.patch(
            "vllm_ascend.models.deepseek_v4.rocm_aiter_ops.is_fusion_moe_shared_experts_enabled",
            return_value=False,
        ),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.get_ascend_config",
            return_value=object(),
        ),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.fused_moe_make_expert_params_mapping",
            return_value=mapping,
        ),
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_rank", return_value=0),
        mock.patch("vllm_ascend.models.deepseek_v4.get_tensor_model_parallel_world_size", return_value=1),
        mock.patch("vllm_ascend.models.deepseek_v4.get_spec_layer_idx_from_weight_name", return_value=None),
        mock.patch("vllm_ascend.models.deepseek_v4.is_pp_missing_parameter", return_value=False),
        mock.patch("vllm_ascend.models.deepseek_v4.enable_dsa_cp", return_value=False),
        mock.patch(
            "vllm_ascend.models.deepseek_v4.maybe_remap_kv_scale_name",
            side_effect=lambda name, params_dict: name,
        ),
    ):
        return AscendDeepseekV4ForCausalLM.load_weights(fake_self, weights)


def test_load_weights_skips_attn_sink_for_layer_count_truncated_model():
    sink_param = torch.nn.Parameter(torch.zeros(4))
    params = {"model.layers.0.self_attn.attn_sink": sink_param}
    weights = [
        ("model.layers.0.self_attn.attn_sink", torch.tensor([1.0, 2.0, 3.0, 4.0])),
        ("model.layers.5.self_attn.attn_sink", torch.tensor([5.0, 6.0, 7.0, 8.0])),
    ]
    loaded_params = _run_load_weights(params, weights)
    assert loaded_params == {"model.layers.0.self_attn.attn_sink"}
    torch.testing.assert_close(sink_param.data, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_load_weights_skips_generic_weight_for_layer_count_truncated_model():
    real_param = torch.nn.Parameter(torch.zeros(2, 2))
    params = {"model.layers.0.self_attn.o_proj.weight": real_param}
    weights = [
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(2, 2)),
        ("model.layers.5.self_attn.o_proj.weight", torch.full((2, 2), 9.0)),
    ]
    loaded_params = _run_load_weights(params, weights)
    assert "model.layers.0.self_attn.o_proj.weight" in loaded_params
    torch.testing.assert_close(real_param.data, torch.ones(2, 2))


def test_load_weights_skips_unmapped_per_expert_gate_proj():
    """Regression for KeyError: 'model.layers.0.mlp.experts.100.gate_proj.weight'.

    BF16 HF checkpoints ship per-expert gate/up/down keys. FusedMoE only
    registers fused w13/w2. When expert_params_mapping is empty/mismatched,
    load_weights must skip instead of crashing on params_dict[name].
    """
    params = {"model.layers.0.self_attn.o_proj.weight": torch.nn.Parameter(torch.zeros(2, 2))}
    weights = [
        ("model.layers.0.mlp.experts.100.gate_proj.weight", torch.ones(4, 8)),
        ("model.layers.0.self_attn.o_proj.weight", torch.ones(2, 2)),
    ]
    loaded_params = _run_load_weights(params, weights, expert_mapping=[])
    assert "model.layers.0.self_attn.o_proj.weight" in loaded_params


def test_load_weights_resolves_nested_routed_experts_param():
    """Mapping may emit flat ``experts.w13_`` while MoERunner nests under
    ``experts.routed_experts.w13_`` — both must resolve.
    """
    w13 = torch.nn.Parameter(torch.zeros(2, 8, 4))
    # Expert-aware weight_loader used by the fused MoE path.
    loaded = {}

    def weight_loader(param, weight, name, shard_id=None, expert_id=None, return_success=False):
        loaded["args"] = (name, shard_id, expert_id, tuple(weight.shape))
        # Copy into the w1 half for expert 0.
        param.data[expert_id, :4, :].copy_(weight)
        return True if return_success else None

    w13.weight_loader = weight_loader  # type: ignore[attr-defined]
    params = {"model.layers.0.mlp.experts.routed_experts.w13_weight": w13}
    mapping = [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
    ]
    weights = [("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(4, 4))]
    _run_load_weights(params, weights, expert_mapping=mapping)
    assert loaded["args"][0] == "model.layers.0.mlp.experts.routed_experts.w13_weight"
    assert loaded["args"][1] == "w1"
    assert loaded["args"][2] == 0
    torch.testing.assert_close(w13.data[0, :4, :], torch.ones(4, 4))


def test_expert_param_name_candidates_nested_and_flat():
    nested = "model.layers.0.mlp.experts.routed_experts.w13_weight"
    flat = "model.layers.0.mlp.experts.w13_weight"
    assert _expert_param_name_candidates(nested) == [nested, flat]
    assert _expert_param_name_candidates(flat) == [flat, nested]
