from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops.rotary_embedding as rotary_embedding


class _DummyRotaryModule(torch.nn.Module):
    def __init__(self, with_split_cache: bool):
        super().__init__()
        self.max_position_embeddings = 8
        self.use_flashinfer = False
        self.cos_sin_cache = torch.randn(8, 16)
        if with_split_cache:
            self.cos_cached = torch.randn(8, 16)
            self.sin_cached = torch.randn(8, 16)
        self.cos = torch.randn(1)
        self.sin = torch.randn(1)

    def _compute_cos_sin_cache(self):
        return torch.randn(8, 16)


class _DummyModel(torch.nn.Module):
    def __init__(self, with_split_cache: bool):
        super().__init__()
        self.rotary = _DummyRotaryModule(with_split_cache=with_split_cache)


def _set_all_global_caches():
    rotary_embedding._cos_mla = torch.randn(1)
    rotary_embedding._sin_mla = torch.randn(1)
    rotary_embedding._cos_cache = torch.randn(1)
    rotary_embedding._sin_cache = torch.randn(1)
    rotary_embedding._cos_sin_cache = torch.randn(1)
    rotary_embedding._cos = torch.randn(1)
    rotary_embedding._sin = torch.randn(1)
    rotary_embedding._cos_slice = torch.randn(1)
    rotary_embedding._sin_slice = torch.randn(1)


def _clear_all_global_caches():
    rotary_embedding._cos_mla = None
    rotary_embedding._sin_mla = None
    rotary_embedding._cos_cache = None
    rotary_embedding._sin_cache = None
    rotary_embedding._cos_sin_cache = None
    rotary_embedding._cos = None
    rotary_embedding._sin = None
    rotary_embedding._cos_slice = None
    rotary_embedding._sin_slice = None


@pytest.fixture(autouse=True)
def _reset_rotary_globals():
    _clear_all_global_caches()
    yield
    _clear_all_global_caches()


def test_clear_global_cos_sin_runtime_cache_clears_globals_and_module_cache():
    model = _DummyModel(with_split_cache=True)
    _set_all_global_caches()

    cleared = rotary_embedding.clear_global_cos_sin_runtime_cache(model)

    assert cleared is True
    assert rotary_embedding._cos_mla is None
    assert rotary_embedding._sin_mla is None
    assert rotary_embedding._cos_cache is None
    assert rotary_embedding._sin_cache is None
    assert rotary_embedding._cos_sin_cache is None
    assert rotary_embedding._cos is None
    assert rotary_embedding._sin is None
    assert rotary_embedding._cos_slice is None
    assert rotary_embedding._sin_slice is None
    assert model.rotary.cos_sin_cache is None
    assert model.rotary.cos_cached is None
    assert model.rotary.sin_cached is None
    assert model.rotary.cos is None
    assert model.rotary.sin is None


def test_restore_global_cos_sin_cache_from_model_with_split_cache():
    model = _DummyModel(with_split_cache=True)
    _clear_all_global_caches()

    restored = rotary_embedding.restore_global_cos_sin_cache_from_model(model)

    assert restored is True
    assert rotary_embedding._cos_sin_cache is model.rotary.cos_sin_cache
    assert rotary_embedding._cos_cache is model.rotary.cos_cached
    assert rotary_embedding._sin_cache is model.rotary.sin_cached


def test_restore_global_cos_sin_cache_from_model_interleaved_fallback():
    model = _DummyModel(with_split_cache=False)
    _clear_all_global_caches()

    restored = rotary_embedding.restore_global_cos_sin_cache_from_model(model)

    assert restored is True
    assert rotary_embedding._cos_sin_cache is model.rotary.cos_sin_cache
    assert rotary_embedding._cos_cache is not None
    assert rotary_embedding._sin_cache is not None


def test_set_cos_and_sin_rebuilds_destroyed_module_cache(monkeypatch):
    model = _DummyModel(with_split_cache=False)
    rotary_embedding.clear_global_cos_sin_runtime_cache(model)
    monkeypatch.setattr(rotary_embedding, "has_rope", lambda _: False)
    monkeypatch.setattr(rotary_embedding, "is_vl_model", lambda _: False)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
    )

    rotary_embedding.set_cos_and_sin(cfg, 4, 1, torch.float16, torch.device("cpu"), model=model)

    assert model.rotary.cos_sin_cache is not None
    assert rotary_embedding._cos_sin_cache is model.rotary.cos_sin_cache

