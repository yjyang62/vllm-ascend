from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.compilation.acl_graph import (
    GraphParams,
    clear_attention_workspaces_for_sleep,
    reset_graph_params_for_sleep,
)


def _make_graph_params(workspace):
    return GraphParams(
        events={4: [MagicMock()]},
        workspaces={4: workspace},
        handles={4: [MagicMock()]},
        attn_params={4: [("attn",)]},
        conv1d_params={4: [("conv",)]},
        conv1d_handles={4: [MagicMock()]},
        conv1d_events={4: [MagicMock()]},
    )


def test_clear_attention_workspaces_for_sleep_clears_all_graph_workspace_refs():
    graph_params = _make_graph_params(torch.randn(2, 2))
    draft_graph_params = _make_graph_params(torch.randn(2, 2))
    draft_prefill_graph_params = _make_graph_params(None)

    with (
        patch("vllm_ascend.compilation.acl_graph._graph_params", graph_params),
        patch("vllm_ascend.compilation.acl_graph._draft_graph_params", draft_graph_params),
        patch("vllm_ascend.compilation.acl_graph._draft_graph_prefill_params", draft_prefill_graph_params),
    ):
        num_cleared = clear_attention_workspaces_for_sleep()

    assert num_cleared == 2
    assert graph_params.workspaces[4] is None
    assert draft_graph_params.workspaces[4] is None
    assert draft_prefill_graph_params.workspaces[4] is None


def test_reset_graph_params_for_sleep_clears_workspace_and_runtime_states():
    graph_params = _make_graph_params(torch.randn(2, 2))
    draft_graph_params = _make_graph_params(torch.randn(2, 2))
    draft_prefill_graph_params = _make_graph_params(torch.randn(2, 2))
    wrapper = MagicMock()

    with (
        patch("vllm_ascend.compilation.acl_graph._graph_params", graph_params),
        patch("vllm_ascend.compilation.acl_graph._draft_graph_params", draft_graph_params),
        patch("vllm_ascend.compilation.acl_graph._draft_graph_prefill_params", draft_prefill_graph_params),
        patch("vllm_ascend.compilation.acl_graph._acl_graph_wrappers", {wrapper}),
    ):
        reset_graph_params_for_sleep()

    for params in (graph_params, draft_graph_params, draft_prefill_graph_params):
        assert params.workspaces[4] is None
        assert params.events[4] == []
        assert params.handles[4] == []
        assert params.attn_params[4] == []
        assert params.conv1d_params[4] == []
        assert params.conv1d_handles[4] == []
        assert params.conv1d_events[4] == []
    wrapper.reset_aclgraph_cache.assert_called_once()

