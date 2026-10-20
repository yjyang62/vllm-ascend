# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register Ascend runtime tensors before layerwise reload metadata is captured."""

from vllm_ascend.distributed.layerwise_runtime_tensors import (
    register_layerwise_runtime_tensors,
)


def _apply() -> None:
    try:
        from vllm.model_executor.model_loader.reload import meta as reload_meta
    except ImportError:
        return
    register_layerwise_runtime_tensors(reload_meta)


_apply()
