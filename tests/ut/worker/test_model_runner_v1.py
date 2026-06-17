import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    def test_initialize_kv_cache_tensors_pools_only_raw_allocation(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enable_sleep_mode = True
        runner.model_config = MagicMock()
        runner.model_config.hf_text_config.model_type = "test_model"
        runner.shared_kv_cache_layers = {}
        runner.compilation_config = MagicMock()
        runner.compilation_config.static_forward_context = {}
        runner.kv_caches = []

        events = []
        raw_tensors = {"attn": torch.empty(1)}
        kv_caches = {"attn": torch.empty(1)}
        runner._allocate_kv_cache_tensors = MagicMock(
            side_effect=lambda kv_config: events.append("allocate") or raw_tensors
        )
        runner._reshape_kv_cache_tensors = MagicMock(
            side_effect=lambda kv_config, raw: events.append("reshape") or kv_caches
        )

        context = MagicMock()
        context.__enter__.side_effect = lambda: events.append("enter_pool")
        context.__exit__.side_effect = lambda *args: events.append("exit_pool")
        allocator = MagicMock()
        allocator.use_memory_pool.return_value = context

        with (
            patch("vllm_ascend.worker.model_runner_v1.CaMemAllocator.get_instance", return_value=allocator),
            patch("vllm.v1.worker.utils.bind_kv_cache", side_effect=lambda *args: events.append("bind")),
        ):
            result = runner.initialize_kv_cache_tensors(MagicMock())

        self.assertIs(result, kv_caches)
        allocator.use_memory_pool.assert_called_once_with(tag="kv_cache")
        self.assertEqual(events, ["enter_pool", "allocate", "exit_pool", "reshape", "bind"])


if __name__ == "__main__":
    unittest.main()