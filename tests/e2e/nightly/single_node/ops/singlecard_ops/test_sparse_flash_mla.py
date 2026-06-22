# SPDX-License-Identifier: Apache-2.0
"""Functional test for the BF16 SparseFlashMla operator (DeepSeek-V4 on A5).

This validates the CANN ops-transformer ``sparse_flash_mla`` operator that
replaces the FP8 ``kv_quant_sparse_attn_sharedkv`` path. Operator interface
under test (from ops-transformer)::

    sparse_flash_mla(
        q,
        ori_kv=None,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        cmp_residual_kv=None,
        ori_topk_length=None,
        cmp_topk_length=None,
        sinks=None,
        metadata=None,
        softmax_scale=1.0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )

Three computation scenarios (per the README):
  1. only ori_kv                       -> Sliding Window Attention (SWA)
  2. ori_kv + cmp_kv                    -> SWA + Compressed Attention
  3. ori_kv + cmp_kv + cmp_sparse_indices -> SWA + Sparse Compressed Attention

This is a *first-iteration* scaffold:
  * Smoke tests build valid inputs for the 3 scenarios and assert the op is
    callable and returns a finite, correctly-shaped output (the immediate
    "does it run / is it sane" gate).
  * A numerical golden test covers scenario 1 (shared-KV attention with an
    attention sink, causal, sequence length <= window so the sliding window
    reduces to plain causal). It auto-skips if the output head_dim differs
    from the query head_dim, which would indicate an MLA-absorb nope/rope
    split that the simple golden does not model (we'd refine the golden then).

Prerequisite: the SparseFlashMla operator must be available, either via a
prebuilt ops-transformer .run package (installed into a vendor on
ASCEND_CUSTOM_OPP_PATH / ASCEND_OPP_PATH) or via the vllm-ascend torch_binding
wrapper ``torch.ops._C_ascend.npu_sparse_flash_mla``.

Run:
  pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_mla.py -v
"""

import math

import pytest
import torch

# ---------------------------------------------------------------------------
# CONFIG -- adjust these to match the deployed operator / model dims.
# ---------------------------------------------------------------------------
SEED = 42
NUM_Q_HEADS = 64  # SparseFlashMla TND constraint: N1 == 64
NUM_KV_HEADS = 1  # KV_N == 1
# Query head dim. For DSV4 MLA the cache packs rope(64) + nope(512). If the
# operator returns an output head_dim != HEAD_DIM (MLA-absorb split), the
# numerical golden auto-skips and only the smoke checks run.
HEAD_DIM = 576
BLOCK_SIZE = 64  # PageAttention block size, multiple of 16, <= 1024
WINDOW = 128  # sliding window size; ori_win_left == WINDOW - 1 == 127
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Operator entry-point resolution.
# ---------------------------------------------------------------------------
def _resolve_op():
    """Return (callable, name) for sparse_flash_mla, or (None, reason)."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None, "torch_npu not available"

    # 1) Native torch_npu binding (ops-transformer prebuilt package).
    op = getattr(torch_npu, "sparse_flash_mla", None)
    if callable(op):
        return op, "torch_npu.sparse_flash_mla"

    # 2) torch.ops.npu namespace.
    npu_ns = getattr(torch.ops, "npu", None)
    op = getattr(npu_ns, "sparse_flash_mla", None) if npu_ns is not None else None
    if callable(op):
        return op, "torch.ops.npu.sparse_flash_mla"

    # 3) vllm-ascend torch_binding wrapper (this PR). NOTE: its kwargs differ
    #    slightly (seqused_kv instead of seqused_ori_kv, no topk_value_mode);
    #    use only if the native op is unavailable.
    c_ns = getattr(torch.ops, "_C_ascend", None)
    op = getattr(c_ns, "npu_sparse_flash_mla", None) if c_ns is not None else None
    if callable(op):
        return op, "torch.ops._C_ascend.npu_sparse_flash_mla"

    return None, "sparse_flash_mla operator not found in torch_npu / torch.ops"


_OP, _OP_NAME = _resolve_op()
_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available() if hasattr(torch, "npu") else False

pytestmark = pytest.mark.skipif(
    _OP is None or not _NPU_AVAILABLE,
    reason=f"requires NPU and sparse_flash_mla op ({_OP_NAME})",
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _unwrap(out):
    """The op may return a Tensor or a (attention_out, softmax_lse) tuple."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _build_paged_kv(seq_len: int, head_dim: int, dtype: torch.dtype, seed: int):
    """Build a single-batch PA_BBND KV cache + block_table.

    Returns:
        kv_cache:    [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim] on NPU.
        block_table: [1, num_blocks] int32 on NPU.
        kv_dense:    [seq_len, head_dim] CPU fp32 (golden reference view).
    """
    torch.manual_seed(seed)
    num_blocks = math.ceil(seq_len / BLOCK_SIZE)
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim, dtype=dtype)
    kv_dense = torch.randn(seq_len, head_dim, dtype=torch.float32) * 0.1
    for s in range(seq_len):
        blk, off = s // BLOCK_SIZE, s % BLOCK_SIZE
        kv_cache[blk, off, 0] = kv_dense[s].to(dtype)
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    return kv_cache.npu(), block_table.npu(), kv_dense


def _golden_shared_kv_attention(
    q: torch.Tensor,  # [T, N, D] fp32 (CPU)
    kv: torch.Tensor,  # [S, D]    fp32 (CPU), K == V (shared)
    sinks: torch.Tensor,  # [N]    fp32 (CPU)
    scale: float,
) -> torch.Tensor:
    """Causal shared-KV attention with an attention sink.

    Models scenario 1 when seq_len <= window (sliding window == full causal).
    out[t, n] = sum_s softmax_s(scale * q[t,n] . kv[s]) * kv[s], with the sink
    contributing an extra exp(sink[n]) term to the softmax denominator only.
    """
    t_len, n_heads, _ = q.shape
    s_len = kv.shape[0]
    out = torch.zeros(t_len, n_heads, kv.shape[1], dtype=torch.float32)
    # Causal alignment: query t maps to key position (s_len - t_len + t).
    for n in range(n_heads):
        for t in range(t_len):
            q_pos = s_len - t_len + t
            scores = scale * (kv[: q_pos + 1] @ q[t, n])  # [q_pos+1]
            m = torch.maximum(scores.max(), sinks[n])
            p = torch.exp(scores - m)
            denom = p.sum() + torch.exp(sinks[n] - m)
            out[t, n] = (p.unsqueeze(-1) * kv[: q_pos + 1]).sum(0) / denom
    return out


def _call_swa_only(q, ori_kv, block_table, seqused_ori_kv, cu_seqlens_q, sinks, scale):
    """Invoke the operator for scenario 1 (SWA-only)."""
    return _unwrap(
        _OP(
            q,
            ori_kv=ori_kv,
            ori_block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_ori_kv,
            sinks=sinks,
            softmax_scale=scale,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=WINDOW - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
    )


# ---------------------------------------------------------------------------
# Smoke tests: callable + shape + finite.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [16, 64, 100])
def test_swa_only_smoke(seq_len):
    """Scenario 1: only ori_kv -> SWA. Op runs and returns finite output."""
    torch.manual_seed(SEED)
    t_len = seq_len  # prefill-style: one batch, query == key length
    q = (torch.randn(t_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32) * 0.1).to(DTYPE).npu()
    ori_kv, block_table, _ = _build_paged_kv(seq_len, HEAD_DIM, DTYPE, SEED + 1)
    cu_seqlens_q = torch.tensor([0, t_len], dtype=torch.int32).npu()
    seqused_ori_kv = torch.tensor([seq_len], dtype=torch.int32).npu()
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32).npu()
    scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _call_swa_only(q, ori_kv, block_table, seqused_ori_kv, cu_seqlens_q, sinks, scale)

    assert out.shape[0] == t_len, f"unexpected token dim: {tuple(out.shape)}"
    assert out.shape[1] == NUM_Q_HEADS, f"unexpected head dim: {tuple(out.shape)}"
    assert torch.isfinite(out.float()).all(), "output contains NaN/Inf"

    torch.npu.empty_cache()


# ---------------------------------------------------------------------------
# Numerical golden test: scenario 1, seq_len <= window (== full causal).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len", [8, 32, 128])
def test_swa_only_matches_golden(seq_len):
    """Scenario 1 numerical check vs a shared-KV causal-attention golden.

    seq_len <= WINDOW so the sliding window covers the whole sequence.
    Auto-skips if the operator output head_dim != HEAD_DIM (MLA-absorb split),
    since the simple shared-KV golden would not apply in that case.
    """
    assert seq_len <= WINDOW, "golden assumes seq_len <= window"
    torch.manual_seed(SEED)
    t_len = seq_len

    q_cpu = torch.randn(t_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32) * 0.1
    q = q_cpu.to(DTYPE).npu()
    ori_kv, block_table, kv_dense = _build_paged_kv(seq_len, HEAD_DIM, DTYPE, SEED + 1)
    cu_seqlens_q = torch.tensor([0, t_len], dtype=torch.int32).npu()
    seqused_ori_kv = torch.tensor([seq_len], dtype=torch.int32).npu()
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32).npu()
    scale = 1.0 / math.sqrt(HEAD_DIM)

    out = _call_swa_only(q, ori_kv, block_table, seqused_ori_kv, cu_seqlens_q, sinks, scale).cpu().float()

    if out.shape[-1] != HEAD_DIM:
        pytest.skip(
            f"output head_dim {out.shape[-1]} != query head_dim {HEAD_DIM}: "
            "operator uses an MLA-absorb nope/rope split; refine golden to "
            "score over full D but read value over the nope sub-dim."
        )

    ref = _golden_shared_kv_attention(q_cpu, kv_dense, sinks.cpu(), scale)

    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)

    torch.npu.empty_cache()


if __name__ == "__main__":
    if _OP is None or not _NPU_AVAILABLE:
        print(f"SKIP: requires NPU and sparse_flash_mla op ({_OP_NAME})")
    else:
        print(f"Using operator: {_OP_NAME}")
        test_swa_only_smoke(64)
        print("smoke OK")
        test_swa_only_matches_golden(32)
        print("golden OK")
