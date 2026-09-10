import type { OpNode, OpKind } from "../types";

export const SIMPLE_FORMULA: Partial<Record<OpKind,string>> = {
  norm:String.raw`y=\operatorname{RMSNorm}(x)`,linear:String.raw`y=xW^{\mathsf T}`,split:String.raw`(a,b,\ldots)=\operatorname{Split}(x)`,rope:String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`,matmul:String.raw`y=a\,b^{\mathsf T}`,scale:String.raw`y=x/\sqrt{d_h}`,mask:String.raw`y=x+\mathrm{mask}`,softmax:String.raw`p=\operatorname{softmax}(x)`,activation:String.raw`y=\operatorname{SwiGLU}_c(g,u)`,route:String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`,cache:String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(K,V)`,add:String.raw`y=\operatorname{mHC}(x,f(x))`,io:String.raw`y=x`,
};

export const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"每个 token 沿 H 或 R_q / Dₕ 做 RMS；shape 不变。",linear:"W 是当前模块绑定的权重。",split:"只切分最后一维。",rope:"只旋转每个 head 的前 Dᵣ=64 维。",matmul:"沿共同 head_dim 相乘求和。",scale:"Dₕ=512。",mask:"不可见位置加 −∞。",softmax:"每行归一化为概率。",activation:"SwiGLU，clamp limit=10。",route:"只选择去哪里计算。",cache:"slot 由 runtime block table 决定。",add:"mHC 用 Sinkhorn 混合多条流后再与 residual 合并。",io:"数据入口或运行时元数据。",
};

export type FormulaTerm = readonly [symbol:string,meaning:string];
export type FormulaStep = { title:string; formula:string; explanation:string };

export const FORMULA_TERMS_BY_KIND: Record<OpKind,readonly FormulaTerm[]> = {
  io:[["x","输入"],["y","输出"]],
  norm:[["x","输入向量"],["y","归一化输出"],["γ","RMSNorm 权重"],["ε","1e−6"]],
  linear:[["x","输入张量"],["W","投影权重"],["y","线性投影输出"]],
  split:[["x","待切分张量"],["a,b,…","沿最后一维得到的输出"]],
  rope:[["q / k","Q 或 K"],["p","token position"],["Dᵣ","64"]],
  matmul:[["a","左输入"],["b","右输入"],["y","矩阵乘输出"]],
  scale:[["x","未缩放分数"],["Dₕ","512"],["y","缩放后分数"]],
  mask:[["x","score"],["M","causal / window / pad"],["y","mask 后 score"]],
  softmax:[["x","score"],["p","概率"]],
  activation:[["g","gate"],["u","up"],["c","swiglu_limit=10"],["y","SwiGLU 输出"]],
  route:[["s","分数"],["K","选择数量"],["I","选中 id"]],
  cache:[["K / V","写入 cache 的张量"],["slot","物理位置"]],
  add:[["x","residual"],["f(x)","计算分支"],["n_hc","hc_mult=4"]],
};

export const FORMULA_TERMS_BY_ID: Partial<Record<string,readonly FormulaTerm[]>> = {
  "c-norm":[["x","hc_pre 后的 hidden"],["H","4096"],["γ","input_layernorm.weight"],["ε","1e−6"]],
  "h-norm":[["x","hc_pre 后的 hidden"],["H","4096"],["γ","input_layernorm.weight"],["ε","1e−6"]],
  "w-norm":[["x","hc_pre 后的 hidden"],["H","4096"],["γ","input_layernorm.weight"],["ε","1e−6"]],
  "c-wqa":[["X","[B,S,H]"],["R_q","q_lora_rank=1024"],["W_qa","wq_a.weight"]],
  "c-qnorm":[["q_a","Q LoRA 中间量"],["R_q","1024"]],
  "c-wqb":[["q̃_a","归一化 Q LoRA"],["Nₕ","64"],["Dₕ","512"]],
  "c-wkv":[["X","hidden"],["Dₕ","512"],["W_kv","wkv.weight"]],
  "c-kvnorm":[["kv","latent KV"],["Dₕ","512"]],
  "c-rope":[["Dᵣ","64"],["Dₕ","512"],["θ_c","compress_rope_theta=160000"]],
  "c-compress":[["m","4"],["coff","2（overlap）"],["APE","ape[m, coff·Dₕ]"]],
  "h-compress":[["m","128"],["coff","1"],["APE","ape[128, Dₕ]"]],
  "c-indexer":[["N_idx","64"],["D_idx","128"],["K_idx","512"],["m","4"]],
  "c-topk":[["K_idx","512"],["I","topk_indices"]],
  "c-qk":[["Q","RoPE 后的 Q"],["K_sel","Indexer 选出的压缩 K"],["K_idx","512"]],
  "c-router":[["Û","post-attn RMSNorm"],["E","256"],["r","router_logits"]],
  "c-experts":[["K","6"],["E","256"],["s_route","1.5"],["c","10"]],
  "c-hcpre":[["n_hc","4"],["T_sink","20"],["H","4096"]],
  "c-hcpost":[["residual","clone(X)"],["post,comb","hc_pre 的辅助流"]],
};

export const FORMULA_STEPS_BY_ID: Partial<Record<string,readonly FormulaStep[]>> = {
  "c-experts":[
    {title:"1 · 路由得分",formula:String.raw`r=\hat U W_{\mathrm{gate}}^\top,\quad s=\mathrm{sqrtsoftplus}(r)`,explanation:"Flash 使用 scoring_func=sqrtsoftplus；前 L_hash=3 层可走 hash gate（依赖 input_ids）。"},
    {title:"2 · Top-6 + 归一化",formula:String.raw`\mathcal E=\operatorname{TopK}_{6}(s+b),\qquad \hat w_e=s_{\mathrm{route}}\frac{s_e}{\sum_{j\in\mathcal E}s_j}`,explanation:"noaux_tc 的 correction bias 只参与选择；混合权重仍用未加 bias 的分数，再乘 s_route=1.5。"},
    {title:"3 · 专家与 shared",formula:String.raw`Y= \sum_{e\in\mathcal E}\hat w_e E_e(\hat U)+E_{\mathrm{shared}}(\hat U)`,explanation:"256 个 routed expert 每次激活 6 个，外加 1 个 shared expert；SwiGLU clamp=10。"},
  ],
};

export function formulaTerms(node:OpNode){
  return FORMULA_TERMS_BY_ID[node.id]??(node.latex?[]:FORMULA_TERMS_BY_KIND[node.kind]);
}

export const LATEX_BY_ID: Record<string,string> = {
  "c-hcpre":String.raw`(\hat X,\mathrm{post},\mathrm{comb})=\operatorname{hc\_pre}(X;W_{\mathrm{hc}},n_{\mathrm{hc}},T_{\mathrm{sink}})`,
  "h-hcpre":String.raw`(\hat X,\mathrm{post},\mathrm{comb})=\operatorname{hc\_pre}(X;W_{\mathrm{hc}},n_{\mathrm{hc}},T_{\mathrm{sink}})`,
  "w-hcpre":String.raw`(\hat X,\mathrm{post},\mathrm{comb})=\operatorname{hc\_pre}(X;W_{\mathrm{hc}},n_{\mathrm{hc}},T_{\mathrm{sink}})`,
  "c-hcpost":String.raw`U=\operatorname{hc\_post}(Y_{\mathrm{attn}},X,\mathrm{post},\mathrm{comb})`,
  "h-hcpost":String.raw`U=\operatorname{hc\_post}(Y_{\mathrm{attn}},X,\mathrm{post},\mathrm{comb})`,
  "w-hcpost":String.raw`U=\operatorname{hc\_post}(Y_{\mathrm{attn}},X,\mathrm{post},\mathrm{comb})`,
  "c-hcpre2":String.raw`(\hat U,\mathrm{post}',\mathrm{comb}')=\operatorname{hc\_pre}(U;W_{\mathrm{hc}}^{\mathrm{ffn}})`,
  "h-hcpre2":String.raw`(\hat U,\mathrm{post}',\mathrm{comb}')=\operatorname{hc\_pre}(U;W_{\mathrm{hc}}^{\mathrm{ffn}})`,
  "w-hcpre2":String.raw`(\hat U,\mathrm{post}',\mathrm{comb}')=\operatorname{hc\_pre}(U;W_{\mathrm{hc}}^{\mathrm{ffn}})`,
  "c-hcpost2":String.raw`X_{l+1}=\operatorname{hc\_post}(Y_{\mathrm{moe}},U,\mathrm{post}',\mathrm{comb}')`,
  "h-hcpost2":String.raw`X_{l+1}=\operatorname{hc\_post}(Y_{\mathrm{moe}},U,\mathrm{post}',\mathrm{comb}')`,
  "w-hcpost2":String.raw`X_{l+1}=\operatorname{hc\_post}(Y_{\mathrm{moe}},U,\mathrm{post}',\mathrm{comb}')`,
  "c-norm":String.raw`y_i=\frac{x_i}{\sqrt{\frac1H\sum_j x_j^2+\varepsilon}}\gamma_i`,
  "h-norm":String.raw`y_i=\frac{x_i}{\sqrt{\frac1H\sum_j x_j^2+\varepsilon}}\gamma_i`,
  "w-norm":String.raw`y_i=\frac{x_i}{\sqrt{\frac1H\sum_j x_j^2+\varepsilon}}\gamma_i`,
  "c-postnorm":String.raw`\hat U_i=\frac{U_i}{\sqrt{\frac1H\sum_j U_j^2+\varepsilon}}\gamma_{\mathrm{post},i}`,
  "h-postnorm":String.raw`\hat U_i=\frac{U_i}{\sqrt{\frac1H\sum_j U_j^2+\varepsilon}}\gamma_{\mathrm{post},i}`,
  "w-postnorm":String.raw`\hat U_i=\frac{U_i}{\sqrt{\frac1H\sum_j U_j^2+\varepsilon}}\gamma_{\mathrm{post},i}`,
  "c-wqa":String.raw`q_a=X W_{qa}^\top\in\mathbb R^{B\times S\times R_q}`,
  "h-wqa":String.raw`q_a=X W_{qa}^\top\in\mathbb R^{B\times S\times R_q}`,
  "w-wqa":String.raw`q_a=X W_{qa}^\top\in\mathbb R^{B\times S\times R_q}`,
  "c-qnorm":String.raw`\tilde q_a=\operatorname{RMSNorm}(q_a)`,
  "h-qnorm":String.raw`\tilde q_a=\operatorname{RMSNorm}(q_a)`,
  "w-qnorm":String.raw`\tilde q_a=\operatorname{RMSNorm}(q_a)`,
  "c-wqb":String.raw`Q=\tilde q_a W_{qb}^\top\in\mathbb R^{B\times S\times N_h D_h}`,
  "h-wqb":String.raw`Q=\tilde q_a W_{qb}^\top\in\mathbb R^{B\times S\times N_h D_h}`,
  "w-wqb":String.raw`Q=\tilde q_a W_{qb}^\top\in\mathbb R^{B\times S\times N_h D_h}`,
  "c-wkv":String.raw`k_v=X W_{kv}^\top\in\mathbb R^{B\times S\times D_h}`,
  "h-wkv":String.raw`k_v=X W_{kv}^\top\in\mathbb R^{B\times S\times D_h}`,
  "w-wkv":String.raw`k_v=X W_{kv}^\top\in\mathbb R^{B\times S\times D_h}`,
  "c-kvnorm":String.raw`\widetilde{KV}=\operatorname{RMSNorm}(k_v)`,
  "h-kvnorm":String.raw`\widetilde{KV}=\operatorname{RMSNorm}(k_v)`,
  "w-kvnorm":String.raw`\widetilde{KV}=\operatorname{RMSNorm}(k_v)`,
  "c-rope":String.raw`(Q_{\mathrm{rot}},Q_{\mathrm{pass}})=\operatorname{Split}(Q;D_r,D_h-D_r),\quad Q^r=\operatorname{Concat}(\operatorname{RoPE}(Q_{\mathrm{rot}},p),Q_{\mathrm{pass}})`,
  "h-rope":String.raw`(Q_{\mathrm{rot}},Q_{\mathrm{pass}})=\operatorname{Split}(Q;D_r,D_h-D_r),\quad Q^r=\operatorname{Concat}(\operatorname{RoPE}(Q_{\mathrm{rot}},p),Q_{\mathrm{pass}})`,
  "w-rope":String.raw`(Q_{\mathrm{rot}},Q_{\mathrm{pass}})=\operatorname{Split}(Q;D_r,D_h-D_r),\quad Q^r=\operatorname{Concat}(\operatorname{RoPE}(Q_{\mathrm{rot}},p),Q_{\mathrm{pass}})`,
  "c-compress":String.raw`C=\operatorname{Compress}_{m=4}^{\mathrm{overlap}}(X;W_{kv},W_{\mathrm{gate}},\mathrm{APE})`,
  "h-compress":String.raw`C=\operatorname{Compress}_{m=128}(X;W_{kv},W_{\mathrm{gate}},\mathrm{APE})`,
  "c-idxproj":String.raw`Q^{\mathrm{idx}}=\tilde q_a W_{qb}^{\mathrm{idx}\top}\in\mathbb R^{B\times S\times N_{\mathrm{idx}} D_{\mathrm{idx}}}`,
  "c-idxcache":String.raw`\mathcal K_{\mathrm{idx}}[\mathrm{slot}]\leftarrow \operatorname{FP8}(\operatorname{Hadamard}(K^{\mathrm{idx}}))`,
  "c-indexer":String.raw`S=\operatorname{Lightning}(Q^{\mathrm{idx}},\mathcal K_{\mathrm{idx}})`,
  "c-topk":String.raw`I=\operatorname{TopK}_{K_{\mathrm{idx}}}(S)`,
  "c-swa":String.raw`\mathcal K_{\mathrm{swa}}[\mathrm{slot}]\leftarrow \widetilde{KV},\quad W=128`,
  "h-swa":String.raw`\mathcal K_{\mathrm{swa}}[\mathrm{slot}]\leftarrow \widetilde{KV},\quad W=128`,
  "w-swa":String.raw`\mathcal K_{\mathrm{swa}}[\mathrm{slot}]\leftarrow \widetilde{KV},\quad W=128`,
  "c-qk":String.raw`A_{i,j}=\langle Q^r_i, C_{I_{i,j}}\rangle/\sqrt{D_h}`,
  "h-qk":String.raw`A_{i,j}=\langle Q^r_i, C_j\rangle/\sqrt{D_h},\quad j<(q+1)/128`,
  "w-qk":String.raw`A_{i,j}=\langle Q^r_i,K_j\rangle/\sqrt{D_h},\quad j\in\mathrm{window}(i,W)`,
  "c-softmax":String.raw`P=\operatorname{softmax}(A+\mathrm{sink})`,
  "h-softmax":String.raw`P=\operatorname{softmax}(A+\mathrm{sink})`,
  "w-softmax":String.raw`P=\operatorname{softmax}(A+\mathrm{sink})`,
  "c-pv":String.raw`O=\sum_{j\in I} P_j V_j`,
  "h-pv":String.raw`O=\sum_j P_j V_j^{\mathrm{c128}}`,
  "w-pv":String.raw`O=\sum_{j\in\mathrm{window}} P_j V_j`,
  "c-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}(W_{ob}\,W_{oa}O)`,
  "h-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}(W_{ob}\,W_{oa}O)`,
  "w-oproj":String.raw`Y_{\mathrm{attn}}=\operatorname{RowParallel}(W_{ob}\,W_{oa}O)`,
  "c-router":String.raw`r=\hat U W_{\mathrm{gate}}^\top\in\mathbb R^{B\times S\times E}`,
  "h-router":String.raw`r=\hat U W_{\mathrm{gate}}^\top\in\mathbb R^{B\times S\times E}`,
  "w-router":String.raw`r=\hat U W_{\mathrm{gate}}^\top\in\mathbb R^{B\times S\times E}`,
  "c-experts":String.raw`\begin{aligned}s&=\mathrm{sqrtsoftplus}(r),\quad\mathcal E=\operatorname{TopK}_K(s+b)\\\hat w_e&=s_{\mathrm{route}}\frac{s_e}{\sum_{j\in\mathcal E}s_j}\\Y_{\mathrm{routed}}&=\sum_{e\in\mathcal E}\hat w_e E_e(\hat U)\end{aligned}`,
  "h-experts":String.raw`\begin{aligned}s&=\mathrm{sqrtsoftplus}(r),\quad\mathcal E=\operatorname{TopK}_K(s+b)\\\hat w_e&=s_{\mathrm{route}}\frac{s_e}{\sum_{j\in\mathcal E}s_j}\\Y_{\mathrm{routed}}&=\sum_{e\in\mathcal E}\hat w_e E_e(\hat U)\end{aligned}`,
  "w-experts":String.raw`\begin{aligned}s&=\mathrm{sqrtsoftplus}(r),\quad\mathcal E=\operatorname{TopK}_K(s+b)\\\hat w_e&=s_{\mathrm{route}}\frac{s_e}{\sum_{j\in\mathcal E}s_j}\\Y_{\mathrm{routed}}&=\sum_{e\in\mathcal E}\hat w_e E_e(\hat U)\end{aligned}`,
  "c-shared":String.raw`E_{\mathrm{shared}}(u)=W_2\operatorname{SwiGLU}_c(W_1 u,W_3 u)`,
  "h-shared":String.raw`E_{\mathrm{shared}}(u)=W_2\operatorname{SwiGLU}_c(W_1 u,W_3 u)`,
  "w-shared":String.raw`E_{\mathrm{shared}}(u)=W_2\operatorname{SwiGLU}_c(W_1 u,W_3 u)`,
  "c-sum":String.raw`Y_{\mathrm{moe}}=s_{\mathrm{route}}Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)`,
  "h-sum":String.raw`Y_{\mathrm{moe}}=s_{\mathrm{route}}Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)`,
  "w-sum":String.raw`Y_{\mathrm{moe}}=s_{\mathrm{route}}Y_{\mathrm{routed}}+E_{\mathrm{shared}}(\hat U)`,
};
