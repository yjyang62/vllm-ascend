import type { CodeSection, CodeSymbol, CodeDetail, IoBinding } from "../types";
import {
  CODE_URL, DECODER_FORWARD_URL, MOE_FORWARD_URL,
  COMPRESSOR_FORWARD_URL, INDEXER_TOPK_URL, KV_PLAN_COMMENT_URL, COMPRESS_RATIO_URL,
} from "./sources";

const MHC_SECTIONS: CodeSection[] = [
  {stage:"1 · FORWARD",title:"DecoderLayer：mHC 包住 Attention 与 MoE",location:"model.py · L730–742",url:DECODER_FORWARD_URL,code:`residual = hidden_states.clone()
hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(**attn_kwargs)
hidden_states = self.hc_post(hidden_states, residual, post, comb)
residual = hidden_states.clone()
hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states, input_ids)
hidden_states = self.hc_post(hidden_states, residual, post, comb)`},
];

const MHC_SYMBOLS: CodeSymbol[] = [
  {symbol:"hc_pre / hc_post",resolvesTo:"npu_hc_pre_v2 / npu_hc_post",meaning:"Manifold-Constrained Hyper-Connections；n_hc=4 条流，Sinkhorn 20 步。"},
  {symbol:"residual",resolvesTo:"clone(X)",meaning:"进入 hc_pre 前的完整 hidden；hc_post 再与计算分支混合。"},
];

const COMPRESS_SECTIONS: CodeSection[] = [
  {stage:"1 · RATIO",title:"按层读取 compress_ratios",location:"utils.py · get_dsv4_compress_ratio",url:COMPRESS_RATIO_URL,code:`def get_dsv4_compress_ratio(config, layer_idx):
    compress_ratios = getattr(config, "compress_ratios", None)
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0
    return compress_ratios[layer_idx]`},
  {stage:"2 · KERNEL",title:"Compressor.forward：写出压缩 KV",location:"compressor.py · L193–224",url:COMPRESSOR_FORWARD_URL,code:`compressed_kv = torch.ops._C_ascend.compressor(
    hidden_states, self.wkv.weight, self.wgate.weight,
    state_cache.squeeze(-2), self.ape, self.norm.weight, ...,
    cmp_ratio=self.compress_ratio,
    coff=2 if self.overlap else 1,
)`},
];

const INDEX_SECTIONS: CodeSection[] = [
  {stage:"1 · TOPK",title:"Lightning Indexer 选出 512 个压缩位置",location:"indexer.py · npu_quant_lightning_indexer_v2",url:INDEXER_TOPK_URL,code:`topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
    query=query, key=key_cache, weights=..., topk=self.index_topk,
    quant_mode=2, cmp_ratio=4, layout_q="TND", layout_k="PA_BBND", ...
)`},
  {stage:"2 · DTYPE",title:"Indexer KV 独立于 attention KV",location:"dsa_attn_kv_plan.py",url:KV_PLAN_COMMENT_URL,code:`class DsaAttnKvPlan:
    """The attention-KV plan only; indexer KV remains independently FP8."""`},
];

const MOE_SECTIONS: CodeSection[] = [
  {stage:"1 · ROUTE",title:"FP32 gate 产生 router_logits",location:"model.py · DeepseekV4MoE.forward",url:MOE_FORWARD_URL,code:`router_input = hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
router_logits = F.linear(router_input, self.gate.weight)
fused_moe_out = self.experts(hidden_states, router_logits=router_logits, input_ids=input_ids)`},
  {stage:"2 · MERGE",title:"routed 输出乘 scaling 再加 shared",location:"model.py · DeepseekV4MoE.forward",url:MOE_FORWARD_URL,code:`final_hidden_states = muls_add_triton(
    final_hidden_states, shared_output, self.routed_scaling_factor
)`},
];

const ATTN_SECTIONS: CodeSection[] = [
  {stage:"1 · MODULES",title:"CSA 同时持有 compressor 与 indexer；HCA 只有 compressor",location:"model.py · DeepseekV4Attention",url:`${CODE_URL}#L572-L594`,code:`if self.compress_ratio > 1:
    self.compressor = Compressor(..., self.compress_ratio, ...)
    if self.compress_ratio == 4:
        self.indexer = DeepseekV4Indexer(...)`},
];

export const CODE_BY_ID: Record<string, CodeDetail> = {};
for (const prefix of ["c", "h", "w"]) {
  CODE_BY_ID[`${prefix}-hcpre`] = {sections:MHC_SECTIONS,symbols:MHC_SYMBOLS};
  CODE_BY_ID[`${prefix}-hcpost`] = {sections:MHC_SECTIONS,symbols:MHC_SYMBOLS};
  CODE_BY_ID[`${prefix}-hcpre2`] = {sections:MHC_SECTIONS,symbols:MHC_SYMBOLS};
  CODE_BY_ID[`${prefix}-hcpost2`] = {sections:MHC_SECTIONS,symbols:MHC_SYMBOLS};
  CODE_BY_ID[`${prefix}-router`] = {sections:MOE_SECTIONS,symbols:[
    {symbol:"gate.weight",resolvesTo:"[E,H]",meaning:"FP32 router；E=256。"},
    {symbol:"input_ids",resolvesTo:"hash gate",meaning:"前 num_hash_layers=3 层可能用 token id 做 hash 路由。"},
  ]};
  CODE_BY_ID[`${prefix}-experts`] = {sections:MOE_SECTIONS,symbols:[
    {symbol:"K",resolvesTo:"6",meaning:"num_experts_per_tok。"},
    {symbol:"s_route",resolvesTo:"1.5",meaning:"routed_scaling_factor。"},
  ]};
  CODE_BY_ID[`${prefix}-shared`] = {sections:MOE_SECTIONS,symbols:[]};
  CODE_BY_ID[`${prefix}-sum`] = {sections:MOE_SECTIONS,symbols:[]};
}

CODE_BY_ID["c-compress"] = {sections:COMPRESS_SECTIONS,symbols:[
  {symbol:"m",resolvesTo:"4",meaning:"CSA 压缩比；overlap 时 coff=2。"},
]};
CODE_BY_ID["h-compress"] = {sections:COMPRESS_SECTIONS,symbols:[
  {symbol:"m",resolvesTo:"128",meaning:"HCA 压缩比；无 overlap。"},
]};
CODE_BY_ID["c-indexer"] = {sections:INDEX_SECTIONS,symbols:[
  {symbol:"K_idx",resolvesTo:"512",meaning:"index_topk。"},
  {symbol:"Kidx cache",resolvesTo:"FP8 e4m3",meaning:"与 attention KV dtype 独立。"},
]};
CODE_BY_ID["c-topk"] = {sections:INDEX_SECTIONS,symbols:[]};
CODE_BY_ID["c-idxcache"] = {sections:INDEX_SECTIONS,symbols:[]};
CODE_BY_ID["c-idxproj"] = {sections:INDEX_SECTIONS,symbols:[]};
for (const id of ["c-wqa","c-wqb","c-wkv","h-wqa","h-wqb","h-wkv","w-wqa","w-wqb","w-wkv","c-qk","h-qk","w-qk"]) {
  CODE_BY_ID[id] = {sections:ATTN_SECTIONS,symbols:[]};
}

export const INPUT_OVERRIDES: Record<string, IoBinding[]> = {
  "c-indexer":[{kind:"upstream",label:"Index Q + compressed Index K",shape:"[B,S,N_idx,D_idx] + FP8 pages",from:"Q LoRA 与 Indexer compressor"}],
  "c-topk":[{kind:"upstream",label:"Lightning scores",shape:"[B,S,T_cmp]",from:"Lightning Indexer"}],
};

export const NEXT_BY_ID: Record<string,string> = {
  "c-topk":"CSA sparse attention · 只在选中的压缩 KV 上做 softmax",
  "h-compress":"HCA dense attention · 在 1/128 压缩流上稠密计算",
  "w-swa":"滑动窗口 Attention + attention sink",
  "c-sum":"mHC FFN hc_post → 下一层",
};
