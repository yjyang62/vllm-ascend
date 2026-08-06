# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
import torch_npu


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla ops from torch_npu, else cann_ops_transformer."""
    attention_op = getattr(torch_npu, "npu_sparse_flash_mla", None)
    metadata_op = getattr(torch_npu, "npu_sparse_flash_mla_metadata", None)
    if attention_op is not None and metadata_op is not None:
        return attention_op, metadata_op

    try:
        import cann_ops_transformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires SparseFlashMla. "
            "Neither torch_npu.npu_sparse_flash_mla nor cann_ops_transformer "
            "is available. Install a torch_npu/op-plugin build that exposes "
            "npu_sparse_flash_mla*, or install cann_ops_transformer with an "
            "Ascend950 SparseFlashMla OPP (ASCEND_CUSTOM_OPP_PATH)."
        ) from exc

    namespace = torch.ops.cann_ops_transformer
    try:
        return namespace.sparse_flash_mla, namespace.sparse_flash_mla_metadata
    except AttributeError as exc:
        raise RuntimeError(
            "SparseFlashMla is not exposed by torch_npu, and the installed "
            "cann_ops_transformer package does not provide sparse_flash_mla / "
            "sparse_flash_mla_metadata."
        ) from exc


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
    return attention_op(q, **kwargs)
