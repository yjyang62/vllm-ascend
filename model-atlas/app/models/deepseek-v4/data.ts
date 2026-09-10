import type { Node } from "../types";

export const MODEL = "https://github.com/yjyang62/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v4/model.py";
export const INDEXER = "https://github.com/yjyang62/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v4/indexer.py";
export const COMPRESSOR = "https://github.com/yjyang62/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v4/compressor.py";

export function layerShard(_layer: number) {
  return "official DeepSeek-V4-Flash safetensors";
}

function key(layer: number, suffix: string) {
  return `model.layers.${layer}.${suffix}`;
}

function sharedWeights(layer: number) {
  const shard = layerShard(layer);
  return {
    inputNorm: { key: key(layer, "input_layernorm.weight"), shape: "[4096]", dtype: "BF16" as const, shard, params: "4,096" },
    postNorm: { key: key(layer, "post_attention_layernorm.weight"), shape: "[4096]", dtype: "BF16" as const, shard, params: "4,096" },
    wqa: { key: key(layer, "self_attn.wq_a.weight"), shape: "[1024,4096]", dtype: "BF16" as const, shard, runtime: "wq_a · Q LoRA down", params: "4.19M" },
    qnorm: { key: key(layer, "self_attn.q_norm.weight"), shape: "[1024]", dtype: "BF16" as const, shard, params: "1,024" },
    wqb: { key: key(layer, "self_attn.wq_b.weight"), shape: "[32768,1024]", dtype: "BF16" as const, shard, runtime: "wq_b · 64×512", params: "33.55M" },
    wkv: { key: key(layer, "self_attn.wkv.weight"), shape: "[512,4096]", dtype: "BF16" as const, shard, params: "2.10M" },
    kvnorm: { key: key(layer, "self_attn.kv_norm.weight"), shape: "[512]", dtype: "BF16" as const, shard, params: "512" },
    woa: { key: key(layer, "self_attn.wo_a.weight"), shape: "[8192,4096]", dtype: "BF16" as const, shard, runtime: "wo_a · grouped LoRA", params: "33.55M" },
    wob: { key: key(layer, "self_attn.wo_b.weight"), shape: "[4096,8192]", dtype: "BF16" as const, shard, runtime: "wo_b · RowParallel", params: "33.55M" },
    sink: { key: key(layer, "self_attn.attn_sink"), shape: "[64]", dtype: "F32" as const, shard, params: "64" },
    hcAttn: { key: key(layer, "hc_attn_fn"), shape: "[24,16384]", dtype: "F32" as const, shard, note: "mHC attention mix · (2+n_hc)·n_hc × n_hc·H" },
    hcFfn: { key: key(layer, "hc_ffn_fn"), shape: "[24,16384]", dtype: "F32" as const, shard, note: "mHC FFN mix" },
    gate: { key: key(layer, "mlp.gate.weight"), shape: "[256,4096]", dtype: "F32" as const, shard, runtime: "FP32 router", params: "1.05M" },
    expert: { key: key(layer, "mlp.experts.*.{gate,up,down}_proj.weight"), shape: "256 × [2048,4096]×2 + [4096,2048]", dtype: "BF16" as const, shard, note: "routed expert FP4/FP8 in checkpoint" },
    shared: { key: key(layer, "mlp.shared_experts.{gate,up,down}_proj.weight"), shape: "gate/up [2048,4096] · down [4096,2048]", dtype: "BF16" as const, shard },
    compressor: { key: key(layer, "self_attn.compressor.{wkv,wgate,ape,norm}"), shape: "wkv/wgate [coff·512,4096] · ape [m,coff·512]", dtype: "BF16" as const, shard, note: "CSA coff=2 overlap；HCA coff=1" },
    indexer: { key: key(layer, "self_attn.indexer.{wq_b,weights_proj}"), shape: "wq_b [8192,1024] · weights_proj [64,4096]", dtype: "BF16" as const, shard, note: "仅 CSA；Index K cache 独立 FP8" },
  };
}

export function layerNodes(layer: number): Record<string, Node> {
  const w = sharedWeights(layer);
  return {
    norm: {
      id: "norm", tone: "norm", kicker: `L${layer} · PRE-ATTN RMSNORM`, title: "RMSNorm",
      summary: "mHC 预混合之后，对进入 Attention 的流做 RMSNorm。residual 由 mHC 旁路保留。",
      input: "hc_pre 输出", inputShape: "[B,S,4096]", output: "normalized hidden_states", outputShape: "[B,S,4096]",
      formula: "y = x/√(mean(x²)+ε)⊙γ", formulaNote: "标准 RMSNorm，ε=1e−6；与 MiniMax 的 Gemma (1+γ) 不同。",
      runtime: "RMSNorm · input_layernorm", source: "model.py · DeepseekV2DecoderLayer.forward · L730–734", sourceUrl: MODEL,
      code: `hidden_states, post, comb = self.hc_pre(...)\nhidden_states = self.input_layernorm(hidden_states)`,
      weights: [w.inputNorm],
    },
    attn: {
      id: "attn", tone: "attention", kicker: "DSA · Q LoRA + KV LATENT", title: "DeepSeek Sparse Attention",
      summary: "Q 走 LoRA（wq_a → RMSNorm → wq_b）；KV 投到 512 维 latent；RoPE 只转 qk_rope_head_dim=64。",
      input: "normalized hidden + positions", inputShape: "[B,S,4096] + [Nq]", output: "attention heads", outputShape: "[B,S,32768]",
      formula: "Q=Wq_b RMSNorm(X Wq_a); KV=RMSNorm(X Wkv)", formulaNote: "head_dim=512；Nₖᵥ=1。后续按 compress_ratio 走 SWA / CSA / HCA。",
      runtime: "DeepseekV4Attention · AscendDeepseekSparseAttention", source: "model.py · DeepseekV4Attention", sourceUrl: MODEL,
      code: `q = self.wq_b(self.q_norm(self.wq_a(x)))\nkv = self.kv_norm(self.wkv(x))`,
      weights: [w.wqa, w.qnorm, w.wqb, w.wkv, w.kvnorm, w.sink],
    },
    oproj: {
      id: "oproj", tone: "projection", kicker: "GROUPED O-LORA", title: "O Projection",
      summary: "o_groups=8 的 grouped LoRA：wo_a 再 wo_b 投回 H，并加入 per-head attention sink。",
      input: "heads", inputShape: "[B,S,32768]", output: "Yattn", outputShape: "[B,S,4096]",
      formula: "Yattn = RowParallel(wo_b(wo_a(O)))", formulaNote: "wo_a 保持 ND 以喂 npu_transpose_batchmatmul。",
      runtime: "ColumnParallelLinear wo_a + RowParallelLinear wo_b", source: "model.py · DeepseekV4Attention", sourceUrl: MODEL,
      code: `o = self.wo_a(heads)\nYattn, _ = self.wo_b(o)`,
      weights: [w.woa, w.wob, w.sink],
    },
    compressor: {
      id: "compressor", tone: "attention", kicker: "COMPRESS-m KV", title: "Compressor",
      summary: "把 token KV 压成 1/m 的压缩状态。CSA m=4 且 overlap（coff=2）；HCA m=128 无 overlap。",
      input: "hidden_states + compressor state", inputShape: "[B,S,4096] + state cache", output: "compressed KV", outputShape: "[B,⌈S/m⌉,512]",
      formula: "C = Compress_m(X; Wkv, Wgate, APE, RoPE_c)", formulaNote: "compress_rope_theta=160000。内核 torch.ops._C_ascend.compressor。",
      runtime: "Compressor.forward · npu compressor", source: "compressor.py · Compressor.forward · L193–224", sourceUrl: COMPRESSOR,
      code: `compressed_kv = torch.ops._C_ascend.compressor(\n    hidden_states, self.wkv.weight, self.wgate.weight, ...)`,
      weights: [w.compressor],
    },
    indexer: {
      id: "indexer", tone: "index", kicker: "LIGHTNING INDEXER · TOP-512", title: "Lightning Indexer",
      summary: "仅 CSA。从 Q LoRA 再投 Index Q，Hadamard 旋转后的 Index K 写入独立 FP8 cache，然后 Top-512 选压缩位置。",
      input: "qr + hidden_states + index K cache", inputShape: "[B,S,1024] + FP8 pages", output: "topk_indices", outputShape: "[B,S,512]",
      formula: "I = TopK_{512}(Lightning(Qidx, Kidx_fp8))", formulaNote: "Indexer KV 与 attention KV 独立；A5 上 attention KV 可改为 BF16，indexer 仍走 FP8。",
      runtime: "npu_quant_lightning_indexer_v2 · cmp_ratio=4", source: "indexer.py · AscendIndexerOps.select_topk", sourceUrl: INDEXER,
      code: `topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(\n    query=query, key=key_cache, topk=self.index_topk, cmp_ratio=4, ...)`,
      weights: [w.indexer],
    },
    moe: {
      id: "moe", tone: "moe", kicker: "TOP-6 / 256 + SHARED", title: "DeepSeekMoE",
      summary: "全部 43 层都是 MoE。前 3 层可用 hash gate；其余 sigmoid/sqrtsoftplus + noaux_tc，Top-6 加 1 个 shared expert。",
      input: "post-attn RMSNorm 输出", inputShape: "[B,S,4096]", output: "Ymoe", outputShape: "[B,S,4096]",
      formula: "Y = s_route·Σ_{e∈Top6} w_e E_e(Û) + E_shared(Û)", formulaNote: "routed_scaling_factor=1.5；SwiGLU clamp c=10。",
      runtime: "DeepseekV4MoE · FusedMoEFactory", source: "model.py · DeepseekV4MoE.forward", sourceUrl: MODEL,
      code: `router_logits = F.linear(hidden_states.float(), self.gate.weight)\nfused_moe_out = self.experts(hidden_states, router_logits=router_logits)`,
      weights: [w.gate, w.expert, w.shared],
    },
    mhc: {
      id: "mhc", tone: "output", kicker: "mHC · n_hc=4", title: "Manifold-Constrained Hyper-Connections",
      summary: "Attention 与 FFN 各包一层 hc_pre / hc_post。Sinkhorn 在 n_hc=4 条流上混合，再与 clone 的 residual 合并。",
      input: "Xₗ", inputShape: "[B,S,4096]", output: "mixed streams + residual", outputShape: "[B,S,4096]",
      formula: "Y = hc_post(f(hc_pre(X)), residual, post, comb)", formulaNote: "hc_mult=4，sinkhorn_iters=20。替代普通 x+f(x)。",
      runtime: "npu_hc_pre_v2 / npu_hc_post", source: "model.py · DeepseekV2DecoderLayer.forward · L730–740", sourceUrl: MODEL,
      code: `residual = hidden_states.clone()\nhidden_states, post, comb = self.hc_pre(...)\n# attn or moe\nhidden_states = self.hc_post(hidden_states, residual, post, comb)`,
      weights: [w.hcAttn, w.hcFfn],
    },
  };
}
