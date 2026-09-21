# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.device_allocator.hccp_lease import HccpLease
from vllm_ascend.device_allocator.sleep_mem_optimized import HcclSleepWakeupManager


def test_staged_release_and_balanced_ownership():
    library = MagicMock()
    library.noanchor_hold.return_value = 0
    library.noanchor_release.return_value = 0
    with patch("ctypes.CDLL", return_value=library):
        lease = HccpLease()
    lease.acquire()
    with pytest.raises(RuntimeError):
        lease.acquire()
    assert lease.release()
    assert not lease.release()
    library.noanchor_hold.assert_called_once()
    library.noanchor_release.assert_called_once()


def test_release_error_keeps_ownership():
    library = MagicMock()
    library.noanchor_hold.return_value = 0
    library.noanchor_release.return_value = 3
    with patch("ctypes.CDLL", return_value=library):
        lease = HccpLease()
    lease.acquire()
    with pytest.raises(RuntimeError):
        lease.release()
    assert lease.held


def test_native_lease_acquires_before_destroying_groups():
    with patch("vllm_ascend.device_allocator.hccp_lease.HccpLease") as factory:
        manager = HcclSleepWakeupManager(MagicMock(), MagicMock(), experimental_hccp_lease=True)
    worker = MagicMock()
    worker._pp_send_work = []
    manager.worker = worker
    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.is_available", return_value=True),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.is_initialized", return_value=True),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size", return_value=2),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.synchronize"),
        patch.object(manager, "destroy_hccl", return_value=1) as destroy,
    ):
        manager.sleep()
    factory.return_value.acquire.assert_called_once()
    destroy.assert_called_once_with()


def test_acquire_failure_does_not_destroy_groups():
    with patch("vllm_ascend.device_allocator.hccp_lease.HccpLease") as factory:
        factory.return_value.acquire.side_effect = RuntimeError("native unavailable")
        manager = HcclSleepWakeupManager(MagicMock(), MagicMock(), experimental_hccp_lease=True)
    worker = MagicMock()
    worker._pp_send_work = []
    manager.worker = worker
    with (
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.is_available", return_value=True),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.is_initialized", return_value=True),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.distributed.get_world_size", return_value=2),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.torch.npu.synchronize"),
        patch.object(manager, "destroy_hccl") as destroy,
        pytest.raises(RuntimeError, match="native unavailable"),
    ):
        manager.sleep()
    destroy.assert_not_called()
