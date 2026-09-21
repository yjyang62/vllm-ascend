# SPDX-License-Identifier: Apache-2.0
"""Experimental process-local native service lease; creates no HCCL group."""

import ctypes


class HccpLease:
    def __init__(self):
        self.library = ctypes.CDLL(None)
        try:
            self.acquire_fn = self.library.noanchor_hold
            self.release_fn = self.library.noanchor_release
        except AttributeError as exc:
            raise RuntimeError("HCCP lease requires the tested native shim in LD_PRELOAD") from exc
        for fn in (self.acquire_fn, self.release_fn):
            fn.argtypes = []
            fn.restype = ctypes.c_int
        self.held = False

    def acquire(self):
        if self.held:
            raise RuntimeError("HCCP lease already held")
        rc = self.acquire_fn()
        if rc:
            raise RuntimeError(f"HCCP lease acquire failed: {rc}")
        self.held = True

    def release(self):
        if not self.held:
            return False
        rc = self.release_fn()
        if rc:
            raise RuntimeError(f"HCCP lease release failed: {rc}")
        self.held = False
        return True
