import type { OpNode, Weight } from "../types";
import type { LayerType, ExpandedStage } from "./types";
import type { GraphEdge } from "../../graph/types";
import { GraphSurface } from "../../graph/surface";
import { GraphPan } from "../../graph/pan";
import { Op, Tensor, AddCircle } from "../../graph/nodes";

function checkpointWeightName(weight?:Weight){
  return weight?.key.replace(/^model\.layers\.\d+\./,"")??"weight";
}

function InputWeightedOp({node,active,onHover,onLeave,onSelect,inputName,inputShape,weightIndex=0,inputGraphId,graphId,weightGraphId,className=""}:{node:OpNode;active:boolean;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;inputName:string;inputShape:string;weightIndex?:number;inputGraphId:string;graphId:string;weightGraphId:string;className?:string}){
  const weight=node.weights[weightIndex];
  const symbolicWeightShape=weight?.shape.replaceAll("4096","H").replaceAll("1024","R_q").replaceAll("512","Dₕ")??"[H]";
  return <div className={`input-weighted-op ${className}`}><div className="co-input-row"><Tensor name={inputName} shape={inputShape} graphId={inputGraphId}/><Tensor name={checkpointWeightName(weight)} shape={symbolicWeightShape} role="weight" graphId={weightGraphId}/></div><Op node={node} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} graphId={graphId}/></div>;
}

function StageZoom({type,stage,g,active,onHover,onLeave,onSelect,onClose}:{type:LayerType;stage:Exclude<ExpandedStage,null>;g:Record<string,OpNode>;active:string;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void;onClose:()=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const N=({id,graphId}:{id:string;graphId?:string})=><Op node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  if(stage==="ffn"){
    const edges:GraphEdge[]=[
      {from:"moe-u",to:"moe-router",fromPort:"bottom-left",approach:28},{from:"moe-wrouter",to:"moe-router",fromPort:"right",toPort:"left"},{from:"moe-router",to:"moe-router-logits"},{from:"moe-u",to:"moe-experts"},{from:"moe-router-logits",to:"moe-experts",toPort:"top-left",approach:28},{from:"moe-wexperts",to:"moe-experts",fromPort:"right",toPort:"left"},{from:"moe-experts",to:"moe-routed"},{from:"moe-u",to:"moe-shared",fromPort:"bottom-right",toPort:"top",approach:28},{from:"moe-wshared",to:"moe-shared",fromPort:"left",toPort:"right"},{from:"moe-shared",to:"moe-shared-out"},{from:"moe-routed",to:"moe-sum",toPort:"top-left",approach:24},{from:"moe-shared-out",to:"moe-sum",toPort:"top-right",approach:24},{from:"moe-sum",to:"moe-y"},
    ];
    return <section className="stage-zoom lesson-zoom"><header><span>DEEPSEEKMOE · TOP-6 / 256 + SHARED EXPERT</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className="moe-node-graph" edges={edges}>
      <Tensor name="Û" shape="[B,S,H]" graphId="moe-u"/>
      <Tensor name="mlp.gate.weight" shape="[E,H]" role="weight" graphId="moe-wrouter"/><N id="router" graphId="moe-router"/>
      <Tensor name="router logits" shape="[B,S,E]" graphId="moe-router-logits"/>
      <Tensor name="routed expert weights · correction bias" shape="E × expert weights · [E]" role="weight" graphId="moe-wexperts"/><N id="experts" graphId="moe-experts"/>
      <Tensor name="weighted routed output" shape="[B,S,H]" graphId="moe-routed"/>
      <N id="shared" graphId="moe-shared"/><Tensor name="shared expert weights ×3" shape="gate / up / down" role="weight" graphId="moe-wshared"/>
      <Tensor name="shared output" shape="[B,S,H]" graphId="moe-shared-out"/>
      <N id="sum" graphId="moe-sum"/><Tensor name="Ymoe" shape="[B,S,H]" graphId="moe-y"/>
    </GraphSurface></GraphPan></section>;
  }
  const csa=type==="csa";
  const hca=type==="hca";
  const variantEdges:GraphEdge[]=csa?[
    {from:"attn-x",to:"attn-compress",fromPort:"bottom-left",toPort:"top"},{from:"attn-compress",to:"attn-ckv"},
    {from:"attn-qat",to:"attn-idxproj"},{from:"attn-idxproj",to:"attn-qidx"},
    {from:"attn-ckv",to:"attn-idxcache"},{from:"attn-qidx",to:"attn-indexer",toPort:"top-left",approach:30},{from:"attn-idxcache",to:"attn-indexer"},
    {from:"attn-indexer",to:"attn-topk"},{from:"attn-topk",to:"attn-topids"},{from:"attn-topids",to:"attn-qk",toPort:"top-right",approach:28},
    {from:"attn-ckv",to:"attn-qk"},{from:"attn-swa",to:"attn-pv",toPort:"top-right",approach:34},
  ]:hca?[
    {from:"attn-x",to:"attn-compress",fromPort:"bottom-left",toPort:"top"},{from:"attn-compress",to:"attn-ckv"},{from:"attn-ckv",to:"attn-qk"},
    {from:"attn-swa",to:"attn-pv",toPort:"top-right",approach:34},
  ]:[
    {from:"attn-swa",to:"attn-qk"},
  ];
  const edges:GraphEdge[]=[
    {from:"attn-x",to:"attn-wqa"},{from:"attn-wqa",to:"attn-qa"},{from:"attn-qa",to:"attn-qnorm"},{from:"attn-qnorm",to:"attn-qat"},{from:"attn-qat",to:"attn-wqb"},{from:"attn-wqb",to:"attn-q"},
    {from:"attn-x",to:"attn-wkv",fromPort:"bottom-right",toPort:"top"},{from:"attn-wkv",to:"attn-kv"},{from:"attn-kv",to:"attn-kvnorm"},{from:"attn-kvnorm",to:"attn-kvt"},
    {from:"attn-q",to:"attn-rope",toPort:"top-left",approach:38},{from:"attn-pos",to:"attn-rope",toPort:"top-right",approach:38},{from:"attn-rope",to:"attn-qr"},
    {from:"attn-kvt",to:"attn-swa",toPort:"top-right",approach:48},{from:"attn-slots",to:"attn-swa",fromPort:"left",toPort:"right"},
    {from:"attn-qr",to:"attn-qk",toPort:"top-left",approach:28},{from:"attn-qk",to:"attn-softmax"},{from:"attn-softmax",to:"attn-p"},{from:"attn-p",to:"attn-pv",toPort:"top-left",approach:34},{from:"attn-pv",to:"attn-heads"},{from:"attn-heads",to:"attn-oproj"},{from:"attn-oproj",to:"attn-y"},
    ...variantEdges,
  ];
  const title=csa?"CSA · COMPRESS-4 + LIGHTNING INDEXER":hca?"HCA · COMPRESS-128 DENSE":"SWA · SLIDING WINDOW + SINK";
  return <section className={`stage-zoom lesson-zoom attention-lesson`}><header><span>{title}</span><button onClick={onClose}>收起 ×</button></header><GraphPan><GraphSurface className={`attention-flowchart connected-attention-graph ${csa?"sparse-attention":"dense-attention"}`} edges={edges}>
    <div className="compact-chain"><Tensor name="X̂" shape="[B,S,H]" graphId="attn-x"/><N id="wqa" graphId="attn-wqa"/><Tensor name="q_a" shape="[B,S,R_q]" graphId="attn-qa"/><N id="qnorm" graphId="attn-qnorm"/><Tensor name="q̃_a" shape="[B,S,R_q]" graphId="attn-qat"/><N id="wqb" graphId="attn-wqb"/><Tensor name="Q" shape="[B,S,Nₕ·Dₕ]" graphId="attn-q"/></div>
    <div className={`attention-branches ${csa?"":"dense"}`}>
      {csa&&<div className="index-ribbon"><header className="index-ribbon-label">LIGHTNING INDEXER · CSA</header>
        <N id="compress" graphId="attn-compress"/><Tensor name="compressed KV" shape="[B,⌈S/4⌉,Dₕ]" graphId="attn-ckv"/>
        <N id="idxproj" graphId="attn-idxproj"/><Tensor name="Qidx" shape="[B,S,N_idx,D_idx]" graphId="attn-qidx"/>
        <N id="idxcache" graphId="attn-idxcache"/><N id="indexer" graphId="attn-indexer"/><N id="topk" graphId="attn-topk"/>
        <Tensor name="topk_indices · 512" shape="[B,S,K_idx]" graphId="attn-topids"/>
      </div>}
      {hca&&<div className="index-ribbon"><header className="index-ribbon-label">HCA COMPRESSOR · m=128</header>
        <N id="compress" graphId="attn-compress"/><Tensor name="compressed KV" shape="[B,⌈S/128⌉,Dₕ]" graphId="attn-ckv"/>
      </div>}
      <div className="attention-data-path">
        <div className="qkv-lanes">
          <section><header>Q PATH</header><Tensor name="Qᵣ" shape="[B,Nₕ,S,Dₕ]" graphId="attn-qr"/><div className="two-source"><Tensor name="positions" shape="[Nq]" role="side" graphId="attn-pos"/></div><N id="rope" graphId="attn-rope"/></section>
          <section><header>KV PATH</header><N id="wkv" graphId="attn-wkv"/><Tensor name="k_v" shape="[B,S,Dₕ]" graphId="attn-kv"/><N id="kvnorm" graphId="attn-kvnorm"/><Tensor name="KṼ" shape="[B,S,Dₕ]" graphId="attn-kvt"/></section>
          <section><header>SWA</header><Tensor name="slot_mapping" shape="runtime" role="side" graphId="attn-slots"/><N id="swa" graphId="attn-swa"/></section>
        </div>
      </div>
    </div>
    <div className="score-pipeline"><header className="score-pipeline-label">ATTENTION SCORE PIPELINE</header><N id="qk" graphId="attn-qk"/><N id="softmax" graphId="attn-softmax"/><Tensor name="P" shape={csa?"[B,Nₕ,S,K_idx]":"[B,Nₕ,S,T_vis]"} graphId="attn-p"/></div>
    <div className="context-pipeline"><N id="pv" graphId="attn-pv"/><Tensor name="heads" shape="[B,S,Nₕ·Dₕ]" graphId="attn-heads"/><N id="oproj" graphId="attn-oproj"/><Tensor name="Yattn" shape="[B,S,H]" graphId="attn-y"/></div>
  </GraphSurface></GraphPan></section>;
}

export function DecoderDiagram({type,g,active,expanded,onExpand,onHover,onLeave,onSelect}:{type:LayerType;g:Record<string,OpNode>;active:string;expanded:ExpandedStage;onExpand:(stage:ExpandedStage)=>void;onHover:(n:OpNode)=>void;onLeave:()=>void;onSelect:(n:OpNode)=>void}){
  const p={active:false,onHover,onLeave,onSelect};
  const IW=({id,inputName,inputShape,inputGraphId,graphId,weightGraphId}:{id:string;inputName:string;inputShape:string;inputGraphId:string;graphId:string;weightGraphId:string})=><InputWeightedOp node={g[id]} {...p} active={active===g[id].id} inputName={inputName} inputShape={inputShape} inputGraphId={inputGraphId} graphId={graphId} weightGraphId={weightGraphId}/>;
  const N=({id,graphId}:{id:string;graphId:string})=><Op node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const A=({id,graphId}:{id:string;graphId:string})=><AddCircle node={g[id]} {...p} active={active===g[id].id} graphId={graphId}/>;
  const attnLabel=type==="csa"?"CSA · Compress-4 + Lightning Indexer":type==="hca"?"HCA · Compress-128 Dense":"SWA · Window 128 + Sink";
  const edges:GraphEdge[]=[
    {from:"main-x",to:"main-hcpre",toPort:"top-left",approach:48},{from:"main-whc",to:"main-hcpre",toPort:"top-right",approach:48},
    {from:"main-hcpre",to:"main-norm"},{from:"main-norm",to:"main-attn"},{from:"main-attn",to:"main-hcpost"},
    {from:"main-x",to:"main-hcpost",fromPort:"left",toPort:"left",route:"side-left"},
    {from:"main-hcpost",to:"main-u",approach:52},{from:"main-u",to:"main-hcpre2",toPort:"top-left",approach:48},{from:"main-whc2",to:"main-hcpre2",toPort:"top-right",approach:48},
    {from:"main-hcpre2",to:"main-post"},{from:"main-post",to:"main-ffn"},{from:"main-ffn",to:"main-hcpost2"},
    {from:"main-u",to:"main-hcpost2",fromPort:"left",toPort:"left",route:"side-left"},{from:"main-hcpost2",to:"main-out",approach:44},
  ];
  return <div className={`decoder-workbench ${expanded?"has-zoom":""}`}>{!expanded&&<GraphPan><GraphSurface className="decoder-column decoder-node-graph" edges={edges}>
    <IW id="hcpre" inputName="Xₗ · residual stream" inputShape="[B,S,H]" inputGraphId="main-x" graphId="main-hcpre" weightGraphId="main-whc"/>
    <N id="norm" graphId="main-norm"/>
    <button data-graph-id="main-attn" className="stage-summary attention-stage" onClick={()=>onExpand(expanded==="attention"?null:"attention")}><small>点击展开</small><b>{attnLabel}</b></button>
    <A id="hcpost" graphId="main-hcpost"/>
    <IW id="hcpre2" inputName="U · after attention mHC" inputShape="[B,S,H]" inputGraphId="main-u" graphId="main-hcpre2" weightGraphId="main-whc2"/>
    <N id="postnorm" graphId="main-post"/>
    <button data-graph-id="main-ffn" className="stage-summary ffn-stage" onClick={()=>onExpand(expanded==="ffn"?null:"ffn")}><small>点击展开</small><b>DeepSeekMoE · Top-6 + Shared</b></button>
    <A id="hcpost2" graphId="main-hcpost2"/>
    <Tensor name="Xₗ₊₁ · hidden_states" shape="[B,S,H]" graphId="main-out"/>
  </GraphSurface></GraphPan>}{expanded&&<StageZoom type={type} stage={expanded} g={g} active={active} onHover={onHover} onLeave={onLeave} onSelect={onSelect} onClose={()=>onExpand(null)}/>}</div>;
}

export function LayerNavigator({type,onChange}:{type:LayerType;onChange:(type:LayerType)=>void}){
  return <div className="layer-nav layer-type-nav"><div className="layer-nav-head"><b>{type==="csa"?"CSA · Compress-4 + Indexer":type==="hca"?"HCA · Compress-128":"SWA · Sliding Window"}</b></div><div className="layer-type-options">
    <button className={type==="swa"?"active swa":"swa"} onClick={()=>onChange("swa")}><span>L0–L1 · L42</span><b>SWA + Attention Sink</b><small>compress_ratio = 0</small></button>
    <button className={type==="csa"?"active csa":"csa"} onClick={()=>onChange("csa")}><span>CSA 层</span><b>Compress-4 + Lightning Indexer</b><small>Top-512 · Index KV = FP8</small></button>
    <button className={type==="hca"?"active hca":"hca"} onClick={()=>onChange("hca")}><span>HCA 层</span><b>Compress-128 Dense Attn</b><small>无 Indexer · 压缩流稠密</small></button>
  </div></div>;
}
