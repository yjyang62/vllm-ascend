# DeepSeek-V4-Flash 结构

Document-Kind: model-knowledge

## 总览

```text
token ids → VocabParallelEmbedding [B,S,H]
  → Decoder × 43（每层：mHC → RMSNorm → DSA → mHC → RMSNorm → DeepSeekMoE）
  → Final RMSNorm → LM Head [B,S,V]
  → 可选 MTP / DSpark
```

H = 4096，L = 43，V = 129280，Nₕ = 64，Dₕ = 512，R_q = R_o = 1024，G_o = 8。

## mHC

`DeepseekV2DecoderLayer.forward` 把 Attention 和 MoE 各自包在 `hc_pre` / `hc_post` 里：

1. `residual = hidden_states.clone()`
2. `hidden_states, post, comb = hc_pre(...)`（Sinkhorn，n_hc=4）
3. RMSNorm → 子模块
4. `hc_post(hidden_states, residual, post, comb)`

这不是逐元素 `x + f(x)`。权重：`hc_{attn,ffn}_{fn,base,scale}`。

## Attention 三态

`compress_ratio = get_dsv4_compress_ratio(config, layer_idx)`。

公共路径：

- Q：`wq_a` (H→1024) → RMSNorm → `wq_b` (1024→64×512)
- KV：`wkv` (H→512) → RMSNorm；Nₖᵥ = 1
- RoPE：只转 `qk_rope_head_dim=64`；压缩层用 `compress_rope_theta=160000`
- O：`wo_a` grouped LoRA → `wo_b`；加上 per-head `attn_sink`
- 所有层都有 SWA cache（window=128）

CSA（m=4）：

- `Compressor` overlap，`coff=2`
- `DeepseekV4Indexer`：Index Q 从 Q LoRA 再投，Index K 经 Hadamard 后写入独立 FP8 cache
- `npu_quant_lightning_indexer_v2`，`topk=512`，`cmp_ratio=4`
- 主注意力只在选中的压缩槽上计算

HCA（m=128）：

- 只有 Compressor，无 Indexer
- 在压缩长度 `⌈S/128⌉` 上稠密注意力

SWA（m=0）：

- 不创建 Compressor / Indexer

## MoE

- 256 routed + 1 shared，Top-6，`scoring_func=sqrtsoftplus`，`topk_method=noaux_tc`
- `routed_scaling_factor=1.5`，`swiglu_limit=10`
- `num_hash_layers=3`：前三层可用 hash gate（需要 `input_ids`）

## 缓存 dtype

`DsaAttnKvPlan` 只管 **attention KV**。Indexer KV 在 A5 上仍是 FP8。`--kv-cache-dtype bfloat16` 只改 attention KV。
