from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.models.extract_hidden_states import (
    CacheOnlyAttentionLayer,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
)

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _hidden_state_spec() -> HiddenStateCacheSpec:
    return HiddenStateCacheSpec(
        block_size=2,
        num_kv_heads=3,
        head_size=4,
        dtype=torch.float32,
        cache_dtype_str="auto",
    )


def test_get_kv_cache_spec_preserves_hidden_state_marker():
    source_spec = _hidden_state_spec()
    cache_only_layer = object.__new__(CacheOnlyAttentionLayer)

    with (
        patch(
            "vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config",
            return_value={"cache_only_layers.0": cache_only_layer},
        ),
        patch.object(
            CacheOnlyAttentionLayer,
            "get_kv_cache_spec",
            return_value=source_spec,
        ),
    ):
        specs = get_kv_cache_spec(MagicMock())

    assert isinstance(
        specs["cache_only_layers.0"],
        HiddenStateCacheSpec,
    )


def test_allocate_and_reshape_hidden_state_cache_as_single_tensor():
    spec = _hidden_state_spec()
    layer_name = "cache_only_layers.0"
    num_blocks = 5

    kv_cache_tensor = MagicMock()
    kv_cache_tensor.size = num_blocks * spec.page_size_bytes
    kv_cache_tensor.shared_by = [layer_name]

    group_spec = MagicMock()
    group_spec.kv_cache_spec = spec
    group_spec.layer_names = [layer_name]

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_tensors = [kv_cache_tensor]
    kv_cache_config.kv_cache_groups = [group_spec]

    vllm_config = MagicMock()
    vllm_config.kv_transfer_config = None
    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=vllm_config,
    ):
        raw_caches = _allocate_kv_cache(
            kv_cache_config,
            shared_layers={},
            device=torch.device("cpu"),
        )

    raw_cache = raw_caches[layer_name]
    assert isinstance(raw_cache, torch.Tensor)
    assert raw_cache.numel() == kv_cache_tensor.size

    backend = MagicMock()
    backend.get_kv_cache_shape.return_value = (
        num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
    )
    attn_group = MagicMock()
    attn_group.kv_cache_group_id = 0
    attn_group.kv_cache_spec = spec
    attn_group.layer_names = [layer_name]
    attn_group.backend = backend

    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=vllm_config,
    ):
        caches = _reshape_kv_cache_v2(
            [attn_group],
            raw_caches,
            cache_dtype="auto",
            kernel_block_sizes=[spec.block_size],
            shared_kv_cache_layers={},
        )

    cache = caches[layer_name]
    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (
        num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
    )


def test_hidden_and_attention_groups_share_one_raw_tensor():
    hidden_spec = _hidden_state_spec()
    attention_spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=4,
        dtype=torch.float32,
    )
    assert hidden_spec.page_size_bytes == attention_spec.page_size_bytes

    hidden_layer = "cache_only_layers.0"
    attention_layer = "model.layers.0.self_attn"
    num_blocks = 5
    kv_cache_tensor = MagicMock()
    kv_cache_tensor.size = num_blocks * hidden_spec.page_size_bytes
    kv_cache_tensor.shared_by = [attention_layer, hidden_layer]

    attention_group_spec = MagicMock()
    attention_group_spec.kv_cache_spec = attention_spec
    attention_group_spec.layer_names = [attention_layer]
    hidden_group_spec = MagicMock()
    hidden_group_spec.kv_cache_spec = hidden_spec
    hidden_group_spec.layer_names = [hidden_layer]

    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_tensors = [kv_cache_tensor]
    kv_cache_config.kv_cache_groups = [
        attention_group_spec,
        hidden_group_spec,
    ]
    vllm_config = MagicMock()
    vllm_config.kv_transfer_config = None

    with patch(
        "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
        return_value=vllm_config,
    ):
        raw_caches = _allocate_kv_cache(
            kv_cache_config,
            shared_layers={},
            device=torch.device("cpu"),
        )

    assert raw_caches[attention_layer] is raw_caches[hidden_layer]
    assert isinstance(raw_caches[hidden_layer], torch.Tensor)

    attention_backend = MagicMock()
    attention_backend.get_kv_cache_shape.return_value = (
        2,
        num_blocks,
        attention_spec.block_size,
        attention_spec.num_kv_heads,
        attention_spec.head_size,
    )
    hidden_backend = MagicMock()
    hidden_backend.get_kv_cache_shape.return_value = (
        num_blocks,
        hidden_spec.block_size,
        hidden_spec.num_kv_heads,
        hidden_spec.head_size,
    )

    attention_group = MagicMock()
    attention_group.kv_cache_group_id = 0
    attention_group.kv_cache_spec = attention_spec
    attention_group.layer_names = [attention_layer]
    attention_group.backend = attention_backend
    hidden_group = MagicMock()
    hidden_group.kv_cache_group_id = 1
    hidden_group.kv_cache_spec = hidden_spec
    hidden_group.layer_names = [hidden_layer]
    hidden_group.backend = hidden_backend

    with (
        patch(
            "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
            return_value=vllm_config,
        ),
        patch(
            "vllm_ascend.worker.v2.attn_utils.enable_fa_quant",
            return_value=False,
        ),
    ):
        caches = _reshape_kv_cache_v2(
            [attention_group, hidden_group],
            raw_caches,
            cache_dtype="auto",
            kernel_block_sizes=[
                attention_spec.block_size,
                hidden_spec.block_size,
            ],
            shared_kv_cache_layers={},
        )

    k_cache, v_cache = caches[attention_layer]
    assert k_cache.shape == (
        num_blocks,
        attention_spec.block_size,
        attention_spec.num_kv_heads,
        attention_spec.head_size,
    )
    assert v_cache.shape == k_cache.shape
    assert caches[hidden_layer].shape == (
        num_blocks,
        hidden_spec.block_size,
        hidden_spec.num_kv_heads,
        hidden_spec.head_size,
    )
