import { useState } from "react";
import type { OpNode, Tab } from "../types";
import type { LayerType, ExpandedStage } from "./types";
import { nextDetailState, type DetailEvent, type DetailState } from "../../detail-selection";
import { graphFor } from "./operators";
import { DecoderDiagram, LayerNavigator } from "./diagrams";
import { DetailPanel } from "./details";
import { Arrow } from "../../graph/nodes";

const LAYER_BY_TYPE: Record<LayerType, number> = { swa: 0, csa: 2, hca: 3 };

export function DeepSeekV4View({onExpandedChange}:{onExpandedChange:(expanded:boolean)=>void}){
  const [layerType,setLayerType]=useState<LayerType>("csa");
  const [expanded,setExpandedState]=useState<ExpandedStage>(null);
  const setExpanded=(stage:ExpandedStage)=>{setExpandedState(stage);onExpandedChange(stage!==null)};
  const [tab,setTab]=useState<Tab>("io");
  const layer=LAYER_BY_TYPE[layerType];
  const graph=graphFor(layerType, layer);
  const [detail,setDetail]=useState<DetailState<OpNode>>({hovered:null,pinned:null});
  const active=detail.pinned??detail.hovered;
  const updateDetail=(event:DetailEvent<OpNode>)=>setDetail(state=>nextDetailState(state,event));
  const changeLayerType=(next:LayerType)=>{setLayerType(next);setExpanded(null);updateDetail({type:"clear"})};
  const headline=layerType==="csa"?"CSA · Compress-4 + Lightning Indexer + DeepSeekMoE":layerType==="hca"?"HCA · Compress-128 Dense Attention + DeepSeekMoE":"SWA · Sliding Window + Attention Sink + DeepSeekMoE";
  return <div className="screen-grid"><section className="map-panel">
    <div className="model-overview"><div className="model-step">Token IDs</div><Arrow/><div className="model-step">Embedding <code>[B,S,H]</code></div><Arrow/><div className="overview-stack"><b>Decoder ×43 · 全 MoE</b><span><i className="dense"/>L0–1 · L42 · SWA（compress=0）</span><span><i className="sparse"/>CSA · Compress-4 + Lightning Indexer Top-512</span><span><i className="sparse"/>HCA · Compress-128 压缩流稠密注意力</span></div><Arrow/><div className="model-step">mHC · n_hc=4</div><Arrow/><div className="model-step">LM Head <code>[B,S,V]</code></div></div>
    <LayerNavigator type={layerType} onChange={changeLayerType}/>
    <section className="layer-canvas"><header><div><span>DECODER LAYER · 按 attention 变体展示</span><h1>{headline}</h1></div><div className="node-legend"><span><i className="tensor-swatch"/>TENSOR</span><span><i className="external-swatch"/>EXTERNAL</span><span><i className="weight-swatch"/>WEIGHT</span><span title="颜色区分算子类型"><i className="operator-swatch"/>OPERATOR</span><code>点击大模块展开 · 按下算子后右侧固定</code></div></header><DecoderDiagram type={layerType} g={graph} active={active?.id??""} expanded={expanded} onExpand={setExpanded} onHover={node=>updateDetail({type:"hover",node})} onLeave={()=>updateDetail({type:"leave"})} onSelect={node=>{updateDetail({type:"pin",node});setTab("io")}}/></section>
  </section><DetailPanel node={active} tab={tab} setTab={setTab} pinned={Boolean(detail.pinned)} onClear={()=>updateDetail({type:"clear"})} expanded={expanded} layerType={layerType}/></div>;
}
