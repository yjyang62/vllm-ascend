# SPDX-License-Identifier: Apache-2.0
"""Standalone runnable check for the BF16 SparseFlashMla operator (DSV4 / A5).

NOT a pytest file. Run directly:
    python tests/e2e/nightly/single_node/ops/singlecard_ops/run_sparse_flash_mla.py

You import the operators yourself. ``sparse_flash_mla`` requires a ``metadata``
tensor (the aicpu core-split result), so import BOTH the op and its metadata
builder at the top of this file, e.g.:

    from cann_ops_transformer.ops.sparse_flash_mla import sparse_flash_mla
    from cann_ops_transformer.ops.sparse_flash_mla_metadata import sparse_flash_mla_metadata
    # (adjust module paths to your install)

The script then calls ``sparse_flash_mla_metadata(...)`` to build ``metadata``
and ``sparse_flash_mla(...)`` directly. The metadata builder's exact argument
names vary by version, so the script discovers them via ``inspect.signature``
and only passes matching kwargs (it prints the discovered signature).

Operator interface under test (ops-transformer)::

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

This first iteration covers scenario 1 (only ori_kv -> Sliding Window
Attention). It runs a smoke check (callable / shape / finite) and a numerical
comparison against a shared-KV causal-attention golden with an attention sink,
using seq_len <= window so the sliding window reduces to plain causal.
"""

import inspect
import math

import torch

# ---------------------------------------------------------------------------
# >>> IMPORT THE OPERATORS YOURSELF (uncomment / edit to match your install) <<<
# from cann_ops_transformer.ops.sparse_flash_mla import sparse_flash_mla
# from cann_ops_transformer.ops.sparse_flash_mla_metadata import sparse_flash_mla_metadata
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONFIG -- adjust to match the deployed operator / model dims.
# ---------------------------------------------------------------------------
SEED = 42
NUM_Q_HEADS = 64  # SparseFlashMla TND constraint: N1 == 64
NUM_KV_HEADS = 1  # KV_N == 1
# Query head dim. For DSV4 MLA the cache packs rope(64) + nope(512) = 576.
# If the operator output head_dim differs (MLA-absorb nope/rope split), the
# golden comparison is skipped and only the smoke check is reported.
HEAD_DIM = 576
BLOCK_SIZE = 64  # PageAttention block size, multiple of 16, <= 1024
WINDOW = 128  # sliding window size; ori_win_left == WINDOW - 1 == 127
DTYPE = torch.bfloat16
DEVICE = "npu"


def _unwrap(out):
    """Op may return a Tensor or a (attention_out, softmax_lse) tuple."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _build_paged_kv(seq_len, head_dim, dtype, seed):
    """Build a single-batch PA_BBND KV cache + block_table.

    Returns:
        kv_cache:    [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim] on device.
        block_table: [1, num_blocks] int32 on device.
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
    return kv_cache.to(DEVICE), block_table.to(DEVICE), kv_dense


def _golden_shared_kv_attention(q, kv, sinks, scale):
    """Causal shared-KV attention with an attention sink (CPU fp32).

    q:     [T, N, D]   kv: [S, D] (K == V)   sinks: [N]
    out[t, n] = sum_s softmax_s(scale * q[t,n] . kv[s]) * kv[s], where the sink
    adds an extra exp(sink[n]) term to the softmax denominator only.
    Query t is aligned to key position (S - T + t).
    """
    t_len, n_heads, _ = q.shape
    s_len = kv.shape[0]
    out = torch.zeros(t_len, n_heads, kv.shape[1], dtype=torch.float32)
    for n in range(n_heads):
        for t in range(t_len):
            q_pos = s_len - t_len + t
            scores = scale * (kv[: q_pos + 1] @ q[t, n])
            m = torch.maximum(scores.max(), sinks[n])
            p = torch.exp(scores - m)
            denom = p.sum() + torch.exp(sinks[n] - m)
            out[t, n] = (p.unsqueeze(-1) * kv[: q_pos + 1]).sum(0) / denom
    return out


def _build_metadata(t_len, seq_len, cu_seqlens_q, seqused_ori_kv):
    """Build the required `metadata` tensor via sparse_flash_mla_metadata.

    The metadata builder's argument names differ across versions, so we
    introspect its signature and only pass the kwargs it accepts (printing the
    discovered signature). If the call fails, paste the printed signature back
    and the kwargs can be adjusted.
    """
    meta_op = globals().get("sparse_flash_mla_metadata")
    if meta_op is None:
        raise RuntimeError(
            "`sparse_flash_mla_metadata` is not imported. The op requires a "
            "`metadata` tensor; import the metadata builder at the top of this file."
        )

    sig = inspect.signature(meta_op)
    print(f"[metadata] sparse_flash_mla_metadata{sig}")

    # Superset of candidate kwargs for scenario 1 (SWA-only). Names cover the
    # observed sparse_flash_mla / kv_quant_sparse_attn_sharedkv_metadata vocab.
    candidates = {
        "num_heads_q": NUM_Q_HEADS,
        "num_heads_kv": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_ori_kv": cu_seqlens_q,
        "cu_seqlens_cmp_kv": None,
        "seqused_q": None,
        "seqused_ori_kv": seqused_ori_kv,
        "seqused_cmp_kv": None,
        "seqused_kv": seqused_ori_kv,
        "batch_size": 1,
        "max_seqlen_q": t_len,
        "max_seqlen_kv": seq_len,
        "max_seqlen_ori_kv": seq_len,
        "ori_topk": 0,
        "cmp_topk": 0,
        "cmp_ratio": 1,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": WINDOW - 1,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "has_ori_kv": True,
        "has_cmp_kv": False,
        "topk_value_mode": 1,
    }

    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        kwargs = candidates
    else:
        kwargs = {k: v for k, v in candidates.items() if k in params}
    print(f"[metadata] passing kwargs: {sorted(kwargs)}")
    return meta_op(**kwargs)


def _call_swa_only(q, ori_kv, block_table, seqused_ori_kv, cu_seqlens_q, sinks, scale, metadata):
    """Invoke sparse_flash_mla for scenario 1 (SWA-only). Edit kwargs to match
    your operator binding if needed."""
    return _unwrap(
        sparse_flash_mla(  # noqa: F821  (imported by the user at the top)
            q,
            ori_kv=ori_kv,
            ori_block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=seqused_ori_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=scale,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=WINDOW - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
    )


def run_scenario_one(seq_len):
    """Scenario 1 (only ori_kv -> SWA): smoke + golden compare."""
    print(f"\n=== scenario 1 (SWA-only), seq_len={seq_len} ===")
    torch.manual_seed(SEED)
    t_len = seq_len  # prefill-style single batch: query length == key length

    q_cpu = torch.randn(t_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32) * 0.1
    q = q_cpu.to(DTYPE).to(DEVICE)
    ori_kv, block_table, kv_dense = _build_paged_kv(seq_len, HEAD_DIM, DTYPE, SEED + 1)
    cu_seqlens_q = torch.tensor([0, t_len], dtype=torch.int32).to(DEVICE)
    seqused_ori_kv = torch.tensor([seq_len], dtype=torch.int32).to(DEVICE)
    sinks = torch.zeros(NUM_Q_HEADS, dtype=torch.float32).to(DEVICE)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    metadata = _build_metadata(t_len, seq_len, cu_seqlens_q, seqused_ori_kv)
    out = _call_swa_only(q, ori_kv, block_table, seqused_ori_kv, cu_seqlens_q, sinks, scale, metadata)
    out_cpu = out.cpu().float()

    # --- smoke ---
    print(f"output shape={tuple(out.shape)} dtype={out.dtype}")
    finite = torch.isfinite(out_cpu).all().item()
    print(f"all finite : {finite}")
    assert out.shape[0] == t_len, f"token dim mismatch: {tuple(out.shape)}"
    assert out.shape[1] == NUM_Q_HEADS, f"head dim mismatch: {tuple(out.shape)}"
    assert finite, "output contains NaN/Inf"

    # --- golden compare (only when output head_dim == query head_dim) ---
    if out_cpu.shape[-1] != HEAD_DIM:
        print(
            f"[skip golden] output head_dim {out_cpu.shape[-1]} != q head_dim {HEAD_DIM}; "
            "operator likely uses MLA-absorb nope/rope split (score over full D, "
            "value over nope sub-dim). Tell me the real dims and I'll refine the golden."
        )
        return

    if seq_len > WINDOW:
        print(f"[skip golden] seq_len {seq_len} > window {WINDOW}; golden assumes full-causal.")
        return

    ref = _golden_shared_kv_attention(q_cpu, kv_dense, sinks.cpu(), scale)
    abs_err = (out_cpu - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    print(f"golden max abs err = {abs_err.max().item():.4e}")
    print(f"golden max rel err = {rel_err.max().item():.4e}")
    ok = torch.allclose(out_cpu, ref, rtol=4e-2, atol=4e-2)
    print(f"golden match (rtol=4e-2, atol=4e-2): {ok}")


def main():
    if "sparse_flash_mla" not in globals():
        raise RuntimeError(
            "`sparse_flash_mla` is not imported. Edit the import block at the top of this file, e.g. "
            "`from cann_ops_transformer.ops.sparse_flash_mla import sparse_flash_mla`."
        )
    if "sparse_flash_mla_metadata" not in globals():
        raise RuntimeError(
            "`sparse_flash_mla_metadata` is not imported, but sparse_flash_mla requires a `metadata` "
            "tensor. Add e.g. `from cann_ops_transformer.ops.sparse_flash_mla_metadata import "
            "sparse_flash_mla_metadata` to the import block."
        )
    print(f"device={DEVICE} dtype={DTYPE} N1={NUM_Q_HEADS} D={HEAD_DIM} block={BLOCK_SIZE} window={WINDOW}")
    for seq_len in (8, 32, 128):
        run_scenario_one(seq_len)
    print("\nDONE.")


if __name__ == "__main__":
    main()
