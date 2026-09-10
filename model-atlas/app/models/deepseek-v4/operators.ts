import type { Node, OpNode, OpKind } from "../types";
import { layerNodes } from "./data";
import { CODE_BY_ID } from "./evidence";
import { LATEX_BY_ID } from "./formulas";
import { ASCEND_COMMIT } from "./sources";

const cloneOp = (base: Node, values: Partial<OpNode> & { id: string; kind: OpKind; title: string }): OpNode => {
  const detail=CODE_BY_ID[values.id];
  return { ...base, ...values, latex:values.latex??LATEX_BY_ID[values.id], codeSections:values.codeSections??detail?.sections, codeSymbols:values.codeSymbols??detail?.symbols };
};
export const pinSource = (url: string) => url.replace("/blob/main/", `/blob/${ASCEND_COMMIT}/`);

function commonGraph(layer: number, prefix: "c"|"h"|"w"): Record<string, OpNode> {
  const n = layerNodes(layer);
  return {
    hcpre: cloneOp(n.mhc,{id:`${prefix}-hcpre`,kind:"add",title:"mHC Pre (Attention)",kicker:"n_hc=4 · SINKHORN 20",input:"Xₗ",inputShape:"[B,S,4096]",output:"mixed stream + post/comb",outputShape:"[B,S,4096]"}),
    norm: cloneOp(n.norm,{id:`${prefix}-norm`,kind:"norm",title:"Input RMSNorm"}),
    wqa: cloneOp(n.attn,{id:`${prefix}-wqa`,kind:"linear",title:"Q LoRA Down · wq_a",summary:"H→R_q 的复制线性层，不分 TP。",input:"normalized X",inputShape:"[B,S,4096]",output:"q_a",outputShape:"[B,S,1024]",weights:n.attn.weights.filter(w=>w.key.includes("wq_a"))}),
    qnorm: cloneOp(n.attn,{id:`${prefix}-qnorm`,kind:"norm",title:"Q LoRA RMSNorm",input:"q_a",inputShape:"[B,S,1024]",output:"q̃_a",outputShape:"[B,S,1024]",weights:n.attn.weights.filter(w=>w.key.includes("q_norm"))}),
    wqb: cloneOp(n.attn,{id:`${prefix}-wqb`,kind:"linear",title:"Q LoRA Up · wq_b",input:"q̃_a",inputShape:"[B,S,1024]",output:"Q",outputShape:"[B,S,32768]",weights:n.attn.weights.filter(w=>w.key.includes("wq_b"))}),
    wkv: cloneOp(n.attn,{id:`${prefix}-wkv`,kind:"linear",title:"KV Latent Projection",input:"normalized X",inputShape:"[B,S,4096]",output:"k_v",outputShape:"[B,S,512]",weights:n.attn.weights.filter(w=>w.key.endsWith("wkv.weight"))}),
    kvnorm: cloneOp(n.attn,{id:`${prefix}-kvnorm`,kind:"norm",title:"KV RMSNorm",input:"k_v",inputShape:"[B,S,512]",output:"KṼ",outputShape:"[B,S,512]",weights:n.attn.weights.filter(w=>w.key.includes("kv_norm"))}),
    rope: cloneOp(n.attn,{id:`${prefix}-rope`,kind:"rope",title:"YaRN RoPE · Dᵣ=64",summary:"只旋转每个 head 的前 64 维；CSA/HCA 使用 compress_rope_theta=160000。",input:"Q/KV + positions",inputShape:"Q [B,64,S,512] + [Nq]",output:"Qᵣ / KVᵣ",outputShape:"same",weights:[]}),
    swa: cloneOp(n.attn,{id:`${prefix}-swa`,kind:"cache",title:"SWA Cache · W=128",summary:"三层类型都保留滑动窗口 MLA cache。A5 上可用 --kv-cache-dtype bfloat16；否则 attention KV 为 FP8。Indexer cache 始终独立 FP8。",input:"KṼ + slot_mapping",inputShape:"[B,S,512] + runtime",output:"window KV",outputShape:"W=128 pages",weights:[]}),
    qk: cloneOp(n.attn,{id:`${prefix}-qk`,kind:"matmul",title:prefix==="c"?"Q × selected compressed K":prefix==="h"?"Q × compressed K (dense)":"Q × SWA K"}),
    softmax: cloneOp(n.attn,{id:`${prefix}-softmax`,kind:"softmax",title:"Softmax + Attention Sink",input:"scores",inputShape:"[B,Nₕ,S,*]",output:"P",outputShape:"same",weights:n.attn.weights.filter(w=>w.key.includes("attn_sink"))}),
    pv: cloneOp(n.attn,{id:`${prefix}-pv`,kind:"matmul",title:prefix==="c"?"P × selected V":prefix==="h"?"P × compressed V":"P × SWA V"}),
    oproj: cloneOp(n.oproj,{id:`${prefix}-oproj`,kind:"linear",title:"Grouped O-LoRA"}),
    hcpost: cloneOp(n.mhc,{id:`${prefix}-hcpost`,kind:"add",title:"mHC Post (Attention)",input:"Yattn + residual + post/comb",inputShape:"[B,S,4096]",output:"U",outputShape:"[B,S,4096]"}),
    hcpre2: cloneOp(n.mhc,{id:`${prefix}-hcpre2`,kind:"add",title:"mHC Pre (FFN)",input:"U",inputShape:"[B,S,4096]",output:"mixed U",outputShape:"[B,S,4096]"}),
    postnorm: cloneOp(n.norm,{id:`${prefix}-postnorm`,kind:"norm",title:"Post-attn RMSNorm",input:"mixed U",inputShape:"[B,S,4096]",output:"Û",outputShape:"[B,S,4096]",weights:[{key:`model.layers.${layer}.post_attention_layernorm.weight`,shape:"[4096]",dtype:"BF16",shard:"official DeepSeek-V4-Flash safetensors",params:"4,096"}]}),
    router: cloneOp(n.moe,{id:`${prefix}-router`,kind:"route",title:"FP32 Router Logits",input:"Û",inputShape:"[B,S,4096]",output:"router_logits",outputShape:"[B,S,256]",weights:n.moe.weights.filter(w=>w.key.includes("gate.weight"))}),
    experts: cloneOp(n.moe,{id:`${prefix}-experts`,kind:"activation",title:"Top-6 Routed Experts",input:"Û + router_logits",inputShape:"[B,S,4096] + [B,S,256]",output:"Y_routed",outputShape:"[B,S,4096]"}),
    shared: cloneOp(n.moe,{id:`${prefix}-shared`,kind:"activation",title:"Shared Expert ×1",input:"Û",inputShape:"[B,S,4096]",output:"Y_shared",outputShape:"[B,S,4096]"}),
    sum: cloneOp(n.moe,{id:`${prefix}-sum`,kind:"add",title:"Add Routed + Shared",input:"Y_routed + Y_shared",inputShape:"2 × [B,S,4096]",output:"Ymoe",outputShape:"[B,S,4096]",weights:[]}),
    hcpost2: cloneOp(n.mhc,{id:`${prefix}-hcpost2`,kind:"add",title:"mHC Post (FFN)",input:"Ymoe + residual U",inputShape:"[B,S,4096]",output:"Xₗ₊₁",outputShape:"[B,S,4096]"}),
  };
}

export function csaGraph(layer: number): Record<string, OpNode> {
  const n = layerNodes(layer);
  const g = commonGraph(layer, "c");
  return {
    ...g,
    compress: cloneOp(n.compressor,{id:"c-compress",kind:"linear",title:"CSA Compressor · m=4 overlap"}),
    idxproj: cloneOp(n.indexer,{id:"c-idxproj",kind:"linear",title:"Index Q Projection",input:"q̃_a",inputShape:"[B,S,1024]",output:"Qidx",outputShape:"[B,S,64,128]"}),
    idxcache: cloneOp(n.indexer,{id:"c-idxcache",kind:"cache",title:"Index K Cache · FP8",input:"compressed Index K",inputShape:"[B,⌈S/4⌉,128]",output:"FP8 index pages",outputShape:"independent of attn KV"}),
    indexer: cloneOp(n.indexer,{id:"c-indexer",kind:"matmul",title:"Lightning Indexer Score"}),
    topk: cloneOp(n.indexer,{id:"c-topk",kind:"route",title:"Top-512 Compressed Slots",input:"index scores",inputShape:"[B,S,T_cmp]",output:"topk_indices",outputShape:"[B,S,512]"}),
  };
}

export function hcaGraph(layer: number): Record<string, OpNode> {
  const n = layerNodes(layer);
  return {
    ...commonGraph(layer, "h"),
    compress: cloneOp(n.compressor,{id:"h-compress",kind:"linear",title:"HCA Compressor · m=128"}),
  };
}

export function swaGraph(layer: number): Record<string, OpNode> {
  return commonGraph(layer, "w");
}

export function graphFor(type: "swa"|"csa"|"hca", layer: number) {
  if (type==="csa") return csaGraph(layer);
  if (type==="hca") return hcaGraph(layer);
  return swaGraph(layer);
}
