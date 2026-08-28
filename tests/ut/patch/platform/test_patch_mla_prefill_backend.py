# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

from vllm_ascend.patch.platform import patch_mla_prefill_backend


def test_mla_prefill_backend_patch_skips_when_upstream_module_missing():
    with patch.object(patch_mla_prefill_backend, "find_spec", return_value=None):
        assert patch_mla_prefill_backend._apply_mla_prefill_backend_patch() is False


def test_mla_prefill_backend_patch_module_imports_without_prefill_package():
    """Importing the platform patch must not require MLAPrefillBackend."""
    assert patch_mla_prefill_backend._MLA_PREFILL_BACKEND_MODULE.endswith("prefill.base")
