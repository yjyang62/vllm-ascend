# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]
_IMPL_PATH = _ROOT / "vllm_ascend/distributed/layerwise_runtime_tensors.py"
_PATCH_PATH = _ROOT / "vllm_ascend/patch/worker/patch_layerwise_reload.py"


def _load_impl():
    spec = importlib.util.spec_from_file_location("layerwise_runtime_tensors_under_test", _IMPL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
ASCEND_CONV1D_WEIGHT = _impl.ASCEND_CONV1D_WEIGHT
EXPERT_IDS_PER_EP_RANK = _impl.EXPERT_IDS_PER_EP_RANK
LAYERWISE_RUNTIME_TENSOR_NAMES = _impl.LAYERWISE_RUNTIME_TENSOR_NAMES
checkpoint_load_target = _impl.checkpoint_load_target
register_layerwise_runtime_tensors = _impl.register_layerwise_runtime_tensors

# Counts measured while reloading Qwen3.5-35B-A3B. The checkpoint had already
# supplied these elements, and the layer target stayed larger by exactly the
# runtime tensor.
_MOE_CHECKPOINT_NUMEL = 201_326_592
_ROUTING_TABLE_NUMEL = 256
_CONV_CHECKPOINT_NUMEL = 8_192


def _reload_meta() -> SimpleNamespace:
    skip_load = {
        "_expert_map",
        "expert_mask",
        "e_score_correction_bias",
    }
    return SimpleNamespace(
        SKIP_LOAD_TENSORS=skip_load,
        SKIP_TENSORS=skip_load | {"bias"},
    )


def _moe_tensors() -> dict[str, int]:
    return {
        "experts": _MOE_CHECKPOINT_NUMEL,
        EXPERT_IDS_PER_EP_RANK: _ROUTING_TABLE_NUMEL,
    }


def _conv_tensors() -> dict[str, int]:
    return {
        "weight": _CONV_CHECKPOINT_NUMEL,
        ASCEND_CONV1D_WEIGHT: _CONV_CHECKPOINT_NUMEL,
    }


def test_runtime_names_join_both_skip_sets():
    meta = _reload_meta()
    register_layerwise_runtime_tensors(meta)

    for name in LAYERWISE_RUNTIME_TENSOR_NAMES:
        assert name in meta.SKIP_LOAD_TENSORS
        assert name in meta.SKIP_TENSORS
    assert "bias" in meta.SKIP_TENSORS
    assert "bias" not in meta.SKIP_LOAD_TENSORS
    assert meta.SKIP_LOAD_TENSORS is not meta.SKIP_TENSORS


def test_registration_is_idempotent():
    meta = _reload_meta()
    register_layerwise_runtime_tensors(meta)
    register_layerwise_runtime_tensors(meta)

    assert {name for name in LAYERWISE_RUNTIME_TENSOR_NAMES if name in meta.SKIP_LOAD_TENSORS} == set(
        LAYERWISE_RUNTIME_TENSOR_NAMES
    )


def test_older_reload_meta_with_only_skip_tensors():
    skip_tensors = {"_expert_map", "bias"}
    meta = SimpleNamespace(SKIP_TENSORS=skip_tensors)

    register_layerwise_runtime_tensors(meta)

    for name in LAYERWISE_RUNTIME_TENSOR_NAMES:
        assert name in skip_tensors


def test_missing_skip_sets_are_ignored():
    register_layerwise_runtime_tensors(SimpleNamespace())


def test_reported_shortfall_blocks_layer_completion():
    moe_target = checkpoint_load_target(_moe_tensors(), set())
    conv_target = checkpoint_load_target(_conv_tensors(), set())

    assert moe_target == _MOE_CHECKPOINT_NUMEL + _ROUTING_TABLE_NUMEL
    assert conv_target == _CONV_CHECKPOINT_NUMEL * 2
    assert moe_target > _MOE_CHECKPOINT_NUMEL
    assert conv_target > _CONV_CHECKPOINT_NUMEL


def test_excluding_only_the_routing_table_leaves_conv_pending():
    meta = _reload_meta()
    meta.SKIP_LOAD_TENSORS.add(EXPERT_IDS_PER_EP_RANK)
    meta.SKIP_TENSORS.add(EXPERT_IDS_PER_EP_RANK)

    assert checkpoint_load_target(_moe_tensors(), meta.SKIP_LOAD_TENSORS) == _MOE_CHECKPOINT_NUMEL
    assert checkpoint_load_target(_conv_tensors(), meta.SKIP_LOAD_TENSORS) == _CONV_CHECKPOINT_NUMEL * 2


def test_both_runtime_tensors_let_reload_release_inputs():
    meta = _reload_meta()
    register_layerwise_runtime_tensors(meta)

    assert checkpoint_load_target(_moe_tensors(), meta.SKIP_LOAD_TENSORS) == _MOE_CHECKPOINT_NUMEL
    assert checkpoint_load_target(_conv_tensors(), meta.SKIP_LOAD_TENSORS) == _CONV_CHECKPOINT_NUMEL
    for name in (EXPERT_IDS_PER_EP_RANK, ASCEND_CONV1D_WEIGHT):
        assert name in meta.SKIP_TENSORS


def test_registration_sites_use_the_canonical_names():
    fused_moe = (_ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py").read_text()
    gdn = (_ROOT / "vllm_ascend/ops/gdn.py").read_text()
    kda = (_ROOT / "vllm_ascend/ops/kimi_kda.py").read_text()

    assert "EXPERT_IDS_PER_EP_RANK" in fused_moe
    assert "ASCEND_CONV1D_WEIGHT" in gdn
    assert "ASCEND_CONV1D_WEIGHT" in kda


def test_worker_patch_updates_reload_meta_on_import():
    meta = types.ModuleType("vllm.model_executor.model_loader.reload.meta")
    meta.SKIP_LOAD_TENSORS = {"_expert_map"}
    meta.SKIP_TENSORS = {"_expert_map", "bias"}
    reload_mod = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_mod.meta = meta

    inserted = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType("vllm.model_executor.model_loader"),
        "vllm.model_executor.model_loader.reload": reload_mod,
        "vllm.model_executor.model_loader.reload.meta": meta,
        "vllm_ascend": types.ModuleType("vllm_ascend"),
        "vllm_ascend.distributed": types.ModuleType("vllm_ascend.distributed"),
        "vllm_ascend.distributed.layerwise_runtime_tensors": _impl,
    }
    for name, module in inserted.items():
        module.__package__ = name.rpartition(".")[0]
    previous = {name: sys.modules.get(name) for name in inserted}
    sys.modules.update(inserted)
    try:
        spec = importlib.util.spec_from_file_location("patch_layerwise_reload_under_test", _PATCH_PATH)
        assert spec is not None and spec.loader is not None
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    for name in LAYERWISE_RUNTIME_TENSOR_NAMES:
        assert name in meta.SKIP_LOAD_TENSORS
        assert name in meta.SKIP_TENSORS
    assert patch_module is not None
