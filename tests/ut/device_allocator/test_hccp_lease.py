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


def test_native_lease_never_creates_anchor():
    with patch("vllm_ascend.device_allocator.hccp_lease.HccpLease") as factory:
        manager = HcclSleepWakeupManager(MagicMock(), MagicMock(), experimental_hccp_lease=True)
    with (
        patch("torch.distributed.get_world_size", return_value=2),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.init_model_parallel_group") as create,
        patch("torch.npu.synchronize"),
    ):
        assert manager._ensure_lifecycle_anchor()
        manager.release_lifecycle_anchor()
    create.assert_not_called()
    assert manager._lifecycle_anchor_group is None
    factory.return_value.acquire.assert_called_once()
    factory.return_value.release.assert_called_once()


def test_acquire_failure_does_not_silently_fallback_to_anchor():
    with patch("vllm_ascend.device_allocator.hccp_lease.HccpLease") as factory:
        factory.return_value.acquire.side_effect = RuntimeError("native unavailable")
        manager = HcclSleepWakeupManager(MagicMock(), MagicMock(), experimental_hccp_lease=True)
    with (
        patch("torch.distributed.get_world_size", return_value=2),
        patch("vllm_ascend.device_allocator.sleep_mem_optimized.init_model_parallel_group") as create,
        pytest.raises(RuntimeError),
    ):
        manager._ensure_lifecycle_anchor()
    create.assert_not_called()
