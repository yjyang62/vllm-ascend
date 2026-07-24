# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

_SPARSE_FLASH_MLA_JIT_PRELOADED = False


def _sparse_flash_mla_vendor_hint() -> str:
    ascend_home = __import__("os").environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")
    return (
        "Install SparseFlashMla from https://gitcode.com/cann/ops-transformer, then run "
        f"./build_out/cann-ops-transformer-custom_linux-*.run and ensure "
        f"{ascend_home}/opp/vendors/custom_transformer (or custom) is on "
        "ASCEND_CUSTOM_OPP_PATH. Call bootstrap_custom_op_env(include_vendor_lib=True) "
        "before inference."
    )


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla torch ops from an installed cann_ops_transformer."""
    try:
        import cann_ops_transformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires cann_ops_transformer "
            "with sparse_flash_mla / sparse_flash_mla_metadata. Install from "
            "https://gitcode.com/cann/ops-transformer "
            "(e.g. /home/y00899261/ops-transformer)."
        ) from exc

    namespace = torch.ops.cann_ops_transformer
    try:
        attention_op = namespace.sparse_flash_mla
        metadata_op = namespace.sparse_flash_mla_metadata
    except AttributeError as exc:
        raise RuntimeError(
            "The installed cann_ops_transformer package does not provide "
            "sparse_flash_mla and sparse_flash_mla_metadata. Rebuild those "
            "ops from https://gitcode.com/cann/ops-transformer against the "
            "matching CANN toolkit."
        ) from exc
    return attention_op, metadata_op


def preload_sparse_flash_mla_jit_extensions(*, verbose: bool = False) -> None:
    """JIT-compile cann_ops_transformer C++ wrappers before the first forward pass."""
    global _SPARSE_FLASH_MLA_JIT_PRELOADED
    if _SPARSE_FLASH_MLA_JIT_PRELOADED:
        return

    from cann_ops_transformer.ops.sparse_flash_mla import sparse_flash_mla_op_builder

    sparse_flash_mla_op_builder.load(verbose=verbose)
    _get_sparse_flash_mla_ops()
    _SPARSE_FLASH_MLA_JIT_PRELOADED = True


def preload_sparse_flash_mla_for_worker(*, verbose: bool = False) -> None:
    """Register CANN vendor ops and prebuild SparseFlashMla torch extensions in workers."""
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    preload_sparse_flash_mla_jit_extensions(verbose=verbose)


def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    if cmp_ratio <= 1 or seqused_ori_kv is None:
        return

    if kwargs.get("seqused_cmp_kv") is None:
        kwargs["seqused_cmp_kv"] = seqused_ori_kv // cmp_ratio
    if kwargs.get("cmp_residual_kv") is None:
        kwargs["cmp_residual_kv"] = seqused_ori_kv % cmp_ratio
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def sparse_flash_mla_metadata(**kwargs):
    """Adapt the existing DSA metadata convention to SparseFlashMla."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _add_compressed_kv_lengths(kwargs)

    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt the existing DSA attention convention to SparseFlashMla."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _add_compressed_kv_lengths(kwargs)

    attention_op, _ = _get_sparse_flash_mla_ops()
    try:
        return attention_op(q, **kwargs)
    except RuntimeError as exc:
        if "aclnnSparseFlashMla" in str(exc) or "SparseFlashMla" in str(exc):
            raise RuntimeError(f"{exc}\n\n{_sparse_flash_mla_vendor_hint()}") from exc
        raise
