# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs

_SPARSE_FLASH_MLA_LOGGED_STATES: set[tuple[int, str]] = set()


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load the SparseFlashMla torch operators shipped by ops-transformer."""
    try:
        import cann_ops_transformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires the "
            "cann_ops_transformer package that provides sparse_flash_mla and "
            "sparse_flash_mla_metadata."
        ) from exc

    namespace = torch.ops.cann_ops_transformer
    try:
        attention_op = namespace.sparse_flash_mla
        metadata_op = namespace.sparse_flash_mla_metadata
    except AttributeError as exc:
        raise RuntimeError(
            "The installed cann_ops_transformer package does not provide "
            "sparse_flash_mla and sparse_flash_mla_metadata. Install a version "
            "matched to the CANN toolkit that contains these operators."
        ) from exc
    return attention_op, metadata_op


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


def _sparse_flash_mla_q_heads(q: torch.Tensor, layout_q: str) -> int | None:
    if layout_q == "TND" and q.dim() == 3:
        return q.shape[1]
    if layout_q == "BSND" and q.dim() == 4:
        return q.shape[2]
    return None


def _is_npu_graph_capturing() -> bool:
    npu = getattr(torch, "npu", None)
    is_capturing = getattr(npu, "is_current_stream_capturing", None)
    return bool(is_capturing()) if callable(is_capturing) else False


def _log_sparse_flash_mla_output(q: Any, output: Any, layout_q: str) -> None:
    if not envs.VLLM_ASCEND_DSV4_SPARSE_MLA_OUTPUT_CHECK:
        return
    # Tensor reductions followed by item() synchronize NPU and cannot be
    # captured by ACLGraph. Warmup/eager calls still provide the diagnostic.
    if _is_npu_graph_capturing():
        return
    if not isinstance(q, torch.Tensor) or not isinstance(output, torch.Tensor):
        return
    num_heads = _sparse_flash_mla_q_heads(q, layout_q)
    if num_heads is None:
        return

    out = output.detach().float()
    finite_mask = torch.isfinite(out)
    total = out.numel()
    finite = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(out).sum().item())
    posinf_count = int(torch.isposinf(out).sum().item())
    neginf_count = int(torch.isneginf(out).sum().item())
    if finite > 0:
        finite_values = out[finite_mask]
        abs_max = float(finite_values.abs().max().item())
    else:
        abs_max = float("inf")

    status = "bad" if finite != total or abs_max > 1.0e6 else "ok"
    log_key = (num_heads, status)
    if log_key in _SPARSE_FLASH_MLA_LOGGED_STATES:
        return
    _SPARSE_FLASH_MLA_LOGGED_STATES.add(log_key)
    logger.warning(
        "sparse_flash_mla BF16 output check: N1=%s shape=%s status=%s "
        "finite=%s/%s nan=%s +inf=%s -inf=%s finite_abs_max=%.4e",
        num_heads,
        tuple(output.shape),
        status,
        finite,
        total,
        nan_count,
        posinf_count,
        neginf_count,
        abs_max,
    )


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
    result = attention_op(q, **kwargs)
    output = result[0] if isinstance(result, tuple) else result
    _log_sparse_flash_mla_output(q, output, kwargs.get("layout_q", "BSND"))
    return result
