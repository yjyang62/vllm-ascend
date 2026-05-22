from unittest.mock import MagicMock, patch

from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch


def test_destroy_hccl_for_sleep_preserves_cpu_group():
    group = object.__new__(GroupCoordinatorPatch)
    device_communicator = MagicMock()
    device_group = MagicMock()
    hccl_key = MagicMock()
    group.backend = "hccl"
    group.device_communicator = device_communicator
    group.device_group = device_group
    group.cpu_group = MagicMock()
    group._acquired_hccl_keys = [hccl_key]
    group._unshared_hccl_groups = []
    group._hccl_destroyed_for_sleep = False

    with patch("vllm_ascend.patch.worker.patch_distributed._HCCL_PG_REGISTRY") as mock_registry:
        mock_registry.release.return_value = device_group
        assert group.destroy_hccl_for_sleep()

    device_communicator.destroy.assert_called_once()
    mock_registry.release.assert_called_once_with(hccl_key)
    assert group.device_communicator is None
    assert group.device_group is None
    assert group.cpu_group is not None
    assert group._hccl_destroyed_for_sleep


def test_restore_hccl_after_sleep_recreates_device_group_and_communicator():
    group = object.__new__(GroupCoordinatorPatch)
    group.backend = "hccl"
    group.device_group = None
    group.cpu_group = MagicMock()
    group.group_name = "tp"
    group.group_ranks = [[0, 1], [2, 3]]
    group.reuse_domain = "shared"
    group.rank = 0
    group.world_size = 2
    group.use_device_communicator = True
    group.unique_name = "tp:0"
    group._hccl_destroyed_for_sleep = True

    device_groups = [MagicMock(name="group_0"), MagicMock(name="group_1")]
    hccl_keys = [MagicMock(name="key_0"), MagicMock(name="key_1")]
    with patch("vllm_ascend.patch.worker.patch_distributed.create_hccl_pg_options") as mock_options, \
         patch("vllm_ascend.patch.worker.patch_distributed._acquire_hccl_group",
               side_effect=list(zip(device_groups, hccl_keys))) as mock_acquire, \
         patch("vllm_ascend.patch.worker.patch_distributed.torch.npu.current_device", return_value=0), \
         patch("vllm_ascend.patch.worker.patch_distributed.NPUCommunicator") as mock_communicator:
        assert group.restore_hccl_after_sleep()

    assert mock_options.call_count == 2
    assert mock_acquire.call_count == 2
    assert group.device_group is device_groups[0]
    assert group._acquired_hccl_keys == hccl_keys
    mock_communicator.assert_called_once_with(
        cpu_group=group.cpu_group,
        device=0,
        device_group=device_groups[0],
        unique_name="tp:0",
    )
    assert not group._hccl_destroyed_for_sleep
