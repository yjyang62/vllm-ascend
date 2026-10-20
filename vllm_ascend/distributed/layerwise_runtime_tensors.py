# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend tensors that layerwise weight reload must not treat as checkpoints.

``expert_ids_per_ep_rank`` is the AllToAll routing table. ``ascend_conv1d_weight``
is the GDN/KDA kernel layout packed from ``conv1d.weight`` after loading.
Neither tensor is stored in the checkpoint. vLLM's layerwise reload finishes a
layer, copies values back, and drops ``loaded_weights`` only when the loaded
element count reaches ``get_layer_size``. Counting these tensors leaves every
MoE and convolution layer short, so reload keeps the incoming weights alive
until the update runs out of memory.
"""

from collections.abc import Mapping, Set
from typing import Any

EXPERT_IDS_PER_EP_RANK = "expert_ids_per_ep_rank"
ASCEND_CONV1D_WEIGHT = "ascend_conv1d_weight"

# Names added to both upstream skip sets. ``SKIP_LOAD_TENSORS`` is what
# ``get_layer_size`` excludes. ``SKIP_TENSORS`` is a separate set created from
# that one at import time, and it is what keeps a tensor on its original storage.
LAYERWISE_RUNTIME_TENSOR_NAMES: tuple[str, ...] = (
    EXPERT_IDS_PER_EP_RANK,
    ASCEND_CONV1D_WEIGHT,
)


def checkpoint_load_target(named_numel: Mapping[str, int], skip_load: Set[str]) -> int:
    """Return the element count layerwise reload waits for.

    This follows ``vllm.model_executor.model_loader.reload.utils.get_layer_size``:
    sum the direct parameters and buffers, skipping ``SKIP_LOAD_TENSORS``.
    """

    return sum(numel for name, numel in named_numel.items() if name not in skip_load)


def register_layerwise_runtime_tensors(reload_meta: Any) -> None:
    """Add Ascend runtime tensors to the upstream layerwise skip sets.

    Older vLLM builds only publish ``SKIP_TENSORS``. Current builds publish
    both, and adding to one does not update the other.
    """

    skip_load = getattr(reload_meta, "SKIP_LOAD_TENSORS", None)
    skip_tensors = getattr(reload_meta, "SKIP_TENSORS", None)
    for name in LAYERWISE_RUNTIME_TENSOR_NAMES:
        if isinstance(skip_load, set):
            skip_load.add(name)
        if isinstance(skip_tensors, set):
            skip_tensors.add(name)
