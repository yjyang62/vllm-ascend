from unittest.mock import MagicMock, patch

from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch, get_hccl_group_debug_info


def test_destroy_hccl_for_sleep_preserves_cpu_group():
    """验证 sleep 只销毁 HCCL device_group，不影响 Gloo cpu_group。"""
    group = object.__new__(GroupCoordinatorPatch)
    device_communicator = MagicMock()
    group.device_communicator = device_communicator
    device_group = MagicMock()
    group.device_group = device_group
    group.cpu_group = MagicMock()

    with patch('vllm_ascend.patch.worker.patch_distributed.torch.distributed.destroy_process_group') as mock_destroy:
        assert group.destroy_hccl_for_sleep()

    device_communicator.destroy.assert_called_once()
    mock_destroy.assert_called_once_with(device_group)
    assert group.device_communicator is None
    assert group.device_group is None
    assert group.cpu_group is not None


def test_restore_hccl_after_sleep_recreates_device_group_and_communicator():
    """验证 wakeup 会按原 rank 列表重建 HCCL group 和 communicator。"""
    group = object.__new__(GroupCoordinatorPatch)
    group.device_group = None
    group.cpu_group = MagicMock()
    group.group_name = 'tp'
    group.group_ranks = [[0, 1], [2, 3]]
    group.torch_distributed_backend = 'hccl'
    group.rank = 0
    group.world_size = 2
    group.use_device_communicator = True
    group.unique_name = 'tp:0'

    device_groups = [MagicMock(name='group_0'), MagicMock(name='group_1')]
    with patch('vllm_ascend.patch.worker.patch_distributed.create_hccl_pg_options') as mock_options, \
         patch('vllm_ascend.patch.worker.patch_distributed.torch.distributed.new_group',
               side_effect=device_groups) as mock_new_group, \
         patch('vllm_ascend.patch.worker.patch_distributed.torch.npu.current_device', return_value=0), \
         patch('vllm_ascend.patch.worker.patch_distributed.NPUCommunicator') as mock_communicator:
        assert group.restore_hccl_after_sleep()

    mock_options.assert_called_once_with('tp')
    assert mock_new_group.call_count == 2
    assert group.device_group is device_groups[0]
    mock_communicator.assert_called_once_with(
        cpu_group=group.cpu_group,
        device=0,
        device_group=device_groups[0],
        unique_name='tp:0',
    )


def test_get_hccl_group_debug_info_reports_device_group_address():
    """验证调试信息包含 HCCL group 和 communicator 地址指纹。"""
    group = MagicMock()
    group.unique_name = 'tp:0'
    group.group_name = 'tp'
    group.device_group = MagicMock(name='device_group')
    group.device_communicator = MagicMock(name='device_communicator')

    with patch('vllm_ascend.patch.worker.patch_distributed._iter_alive_group_coordinators',
               return_value=[group]):
        debug_info = get_hccl_group_debug_info()

    assert len(debug_info) == 1
    assert 'tp:0' in debug_info[0]
    assert f'device_group_id=0x{id(group.device_group):x}' in debug_info[0]
    assert f'device_communicator_id={hex(id(group.device_communicator))}' in debug_info[0]
