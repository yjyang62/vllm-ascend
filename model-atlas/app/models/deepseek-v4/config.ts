export const CONFIG_GROUPS = [
  {title:"DeepSeek-V4-Flash · 官方 config.json",rows:[
    ["architectures","DeepseekV4ForCausalLM"],["model_type","deepseek_v4"],["torch_dtype","bfloat16"],["expert_dtype","fp4"],["hidden_size","4096"],["num_hidden_layers","43"],["num_attention_heads","64"],["num_key_value_heads","1"],["head_dim","512"],["q_lora_rank","1024"],["o_lora_rank","1024"],["o_groups","8"],["qk_rope_head_dim","64"],["vocab_size","129280"],["max_position_embeddings","1048576"],["rms_norm_eps","1e−6"],["hidden_act","silu"],["swiglu_limit","10.0"],["tie_word_embeddings","false"],
  ]},
  {title:"MoE",rows:[
    ["n_routed_experts","256"],["n_shared_experts","1"],["num_experts_per_tok","6"],["moe_intermediate_size","2048"],["scoring_func","sqrtsoftplus"],["topk_method","noaux_tc"],["norm_topk_prob","true"],["routed_scaling_factor","1.5"],["num_hash_layers","3"],["num_nextn_predict_layers","1"],
  ]},
  {title:"Hybrid attention / CSA / HCA",rows:[
    ["index_n_heads","64"],["index_head_dim","128"],["index_topk","512"],["sliding_window","128"],["compress_rope_theta","160000"],["compress_ratios","L0–1: 0 · 之后 CSA(4) 与 HCA(128) 交错 · 末层 0（含 MTP 槽位）"],["CSA","compress_ratio=4 · Compressor overlap + Lightning Indexer Top-512"],["HCA","compress_ratio=128 · Compressor only，压缩流上稠密注意力"],["SWA","compress_ratio=0 · 纯滑动窗口 + attention sink"],
  ]},
  {title:"mHC / RoPE",rows:[
    ["hc_mult","4"],["hc_sinkhorn_iters","20"],["hc_eps","1e−6"],["rope_theta","10000"],["rope_scaling.type","yarn"],["rope_scaling.factor","16"],["rope_scaling.beta_fast","32"],["rope_scaling.beta_slow","1"],["rope_scaling.original_max_position_embeddings","65536"],
  ]},
  {title:"Pro 变体差异（本页以 Flash 为准）",rows:[
    ["Flash 总参数 / 激活","284B / 13B"],["Pro 总参数 / 激活","1.6T / 49B"],["共享结构","同一套 mHC + CSA/HCA hybrid attention + DeepSeekMoE"],["服务端注意","Indexer KV 独立 FP8；A5 可将 attention KV 设为 bfloat16"],
  ]},
] as const;

export const CONFIG_SYMBOLS: Record<string,string> = {
  "DeepSeek-V4-Flash · 官方 config.json:hidden_size":"H",
  "DeepSeek-V4-Flash · 官方 config.json:num_hidden_layers":"L",
  "DeepSeek-V4-Flash · 官方 config.json:num_attention_heads":"Nₕ",
  "DeepSeek-V4-Flash · 官方 config.json:num_key_value_heads":"Nₖᵥ",
  "DeepSeek-V4-Flash · 官方 config.json:head_dim":"Dₕ",
  "DeepSeek-V4-Flash · 官方 config.json:q_lora_rank":"R_q",
  "DeepSeek-V4-Flash · 官方 config.json:o_lora_rank":"R_o",
  "DeepSeek-V4-Flash · 官方 config.json:o_groups":"G_o",
  "DeepSeek-V4-Flash · 官方 config.json:qk_rope_head_dim":"Dᵣ",
  "DeepSeek-V4-Flash · 官方 config.json:vocab_size":"V",
  "DeepSeek-V4-Flash · 官方 config.json:max_position_embeddings":"S_max",
  "DeepSeek-V4-Flash · 官方 config.json:rms_norm_eps":"ε_rms",
  "DeepSeek-V4-Flash · 官方 config.json:swiglu_limit":"c",
  "MoE:n_routed_experts":"E",
  "MoE:n_shared_experts":"E_shared",
  "MoE:num_experts_per_tok":"K",
  "MoE:moe_intermediate_size":"H_expert",
  "MoE:routed_scaling_factor":"s_route",
  "MoE:num_hash_layers":"L_hash",
  "MoE:num_nextn_predict_layers":"L_mtp",
  "Hybrid attention / CSA / HCA:index_n_heads":"N_idx",
  "Hybrid attention / CSA / HCA:index_head_dim":"D_idx",
  "Hybrid attention / CSA / HCA:index_topk":"K_idx",
  "Hybrid attention / CSA / HCA:sliding_window":"W",
  "mHC / RoPE:hc_mult":"n_hc",
  "mHC / RoPE:hc_sinkhorn_iters":"T_sink",
  "mHC / RoPE:rope_theta":"θ_base",
};

export function configSymbol(group:string,key:string){
  return CONFIG_SYMBOLS[`${group}:${key}`]??"—";
}
