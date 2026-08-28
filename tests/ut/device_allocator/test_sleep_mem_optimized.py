#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from contextlib import nullcontext
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from vllm_ascend.device_allocator.sleep_mem_optimized import (
    AclGraphSleepWakeupManager,
    HcclSleepWakeupManager,
    SleepWakeupManager,
)


@dataclass
class DummyGraphParams:
    events: dict[int, list]
    workspaces: dict[int, object]
    extra_handles: dict[int, list]
    metadata: dict[int, tuple]


def test_acl_graph_reset_graph_params_clears_list_values_only():
    workspace = object()
    params = DummyGraphParams(
        events={1: ["event"]},
        workspaces={1: workspace},
        extra_handles={1: ["handle"]},
        metadata={1: ("keep",)},
    )

    AclGraphSleepWakeupManager.reset_graph_params(params)

    assert params.events == {1: []}
    assert params.extra_handles == {1: []}
    assert params.workspaces == {1: workspace}
    assert params.metadata == {1: ("keep",)}


def test_acl_graph_wakeup_waits_for_kv_cache_tag():
    model_runner = MagicMock()
    manager = AclGraphSleepWakeupManager(MagicMock(), lambda: model_runner)

    manager.wakeup(tags=["weights"])
    model_runner.capture_model.assert_not_called()

    manager.wakeup(tags=["kv_cache"])
    model_runner.capture_model.assert_called_once_with()


def test_sleep_wakeup_manager_skips_acl_sleep_when_aclgraph_disabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = False
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    manager.acl_graph.sleep = MagicMock()
    manager.hccl.sleep = MagicMock()
    with patch(
        "vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info",
        side_effect=[(10, 20), (12, 20)],
    ):
        manager.sleep()

    manager.acl_graph.sleep.assert_not_called()
    manager.hccl.sleep.assert_called_once_with()


def test_sleep_wakeup_manager_cleans_acl_before_hccl_when_aclgraph_enabled():
    model_runner = MagicMock()
    model_runner.use_aclgraph = True
    manager = SleepWakeupManager(MagicMock(), MagicMock(), lambda: model_runner)
    calls = []
    manager.acl_graph.sleep = MagicMock(side_effect=lambda: calls.append("acl"))
    manager.hccl.sleep = MagicMock(side_effect=lambda: calls.append("hccl"))

    mem_info = [(10, 20), (12, 20), (12, 20), (13, 20)]
    with patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.mem_get_info", side_effect=mem_info):
        manager.sleep()

    assert calls == ["acl", "hccl"]


def test_hccl_wakeup_restores_and_refreshes_moe_groups():
    manager = HcclSleepWakeupManager(MagicMock(), MagicMock())

    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.set_current_vllm_config", return_value=nullcontext()),
        patch.object(manager, "restore_hccl", return_value=2) as mock_restore,
        patch.object(manager, "refresh_moe_hccl_groups") as mock_refresh,
    ):
        manager.wakeup()

    mock_restore.assert_called_once_with()
    mock_refresh.assert_called_once_with()


def _make_vllm_config(model_type: str) -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config.model_type = model_type
    return vllm_config


def _make_hccl_group(
    *,
    group_name: str,
    world_size: int = 8,
    device_group: bool = True,
    destroy_result: bool = True,
    restore_result: bool = True,
) -> MagicMock:
    group = MagicMock()
    group.group_name = group_name
    group.world_size = world_size
    group.device_group = object() if device_group else None
    group.destroy_hccl.return_value = destroy_result
    group.restore_hccl.return_value = restore_result
    return group


def test_destroy_hccl_destroys_all_groups_for_non_deepseek_v4():
    manager = HcclSleepWakeupManager(_make_vllm_config("llama"), MagicMock())
    tp = _make_hccl_group(group_name="tp", world_size=8)
    dp = _make_hccl_group(group_name="dp", world_size=2)

    with patch.object(manager, "iter_alive_group_coordinators", return_value=[tp, dp]):
        num_destroyed = manager.destroy_hccl()

    tp.destroy_hccl.assert_called_once_with()
    dp.destroy_hccl.assert_called_once_with()
    assert num_destroyed == 2
    assert manager._preserved_hccl_group_ids == set()
    assert not manager._skip_hccl_cleanup_for_cycle


def test_destroy_hccl_preserves_deepseek_v4_multi_rank_tp():
    manager = HcclSleepWakeupManager(_make_vllm_config("deepseek_v4"), MagicMock())
    tp = _make_hccl_group(group_name="tp", world_size=8)
    dp = _make_hccl_group(group_name="dp", world_size=2)
    ep = _make_hccl_group(group_name="ep", world_size=16)

    with patch.object(manager, "iter_alive_group_coordinators", return_value=[tp, dp, ep]):
        num_destroyed = manager.destroy_hccl()

    tp.destroy_hccl.assert_not_called()
    dp.destroy_hccl.assert_called_once_with()
    ep.destroy_hccl.assert_called_once_with()
    assert num_destroyed == 2
    assert manager._preserved_hccl_group_ids == {id(tp)}
    assert not manager._skip_hccl_cleanup_for_cycle


def test_destroy_hccl_skips_all_when_deepseek_v4_tp_anchor_missing():
    manager = HcclSleepWakeupManager(_make_vllm_config("deepseek_v4"), MagicMock())
    tp = _make_hccl_group(group_name="tp", world_size=1)
    dp = _make_hccl_group(group_name="dp", world_size=2)

    with patch.object(manager, "iter_alive_group_coordinators", return_value=[tp, dp]):
        num_destroyed = manager.destroy_hccl()

    tp.destroy_hccl.assert_not_called()
    dp.destroy_hccl.assert_not_called()
    assert num_destroyed == 0
    assert manager._skip_hccl_cleanup_for_cycle


def test_destroy_hccl_skips_all_when_deepseek_v4_tp_device_group_is_none():
    manager = HcclSleepWakeupManager(_make_vllm_config("deepseek_v4"), MagicMock())
    tp = _make_hccl_group(group_name="tp", world_size=8, device_group=False)
    dp = _make_hccl_group(group_name="dp", world_size=2)

    with patch.object(manager, "iter_alive_group_coordinators", return_value=[tp, dp]):
        num_destroyed = manager.destroy_hccl()

    tp.destroy_hccl.assert_not_called()
    dp.destroy_hccl.assert_not_called()
    assert num_destroyed == 0
    assert manager._skip_hccl_cleanup_for_cycle


def test_restore_hccl_skips_preserved_tp_and_clears_ids():
    manager = HcclSleepWakeupManager(_make_vllm_config("deepseek_v4"), MagicMock())
    tp = _make_hccl_group(group_name="tp", world_size=8)
    dp = _make_hccl_group(group_name="dp", world_size=2)
    groups = [tp, dp]

    with patch.object(manager, "iter_alive_group_coordinators", return_value=groups):
        manager.destroy_hccl()
        num_restored = manager.restore_hccl()

    tp.restore_hccl.assert_not_called()
    dp.restore_hccl.assert_called_once_with()
    assert num_restored == 1
    assert manager._preserved_hccl_group_ids == set()


def test_restore_hccl_is_noop_after_skipped_cleanup():
    manager = HcclSleepWakeupManager(_make_vllm_config("deepseek_v4"), MagicMock())
    dp = _make_hccl_group(group_name="dp", world_size=2)

    with patch.object(manager, "iter_alive_group_coordinators", return_value=[dp]):
        manager.destroy_hccl()
        num_restored = manager.restore_hccl()

    dp.destroy_hccl.assert_not_called()
    dp.restore_hccl.assert_not_called()
    assert num_restored == 0
    assert not manager._skip_hccl_cleanup_for_cycle
