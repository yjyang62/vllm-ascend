# DeepSeek-V4 Model Atlas

Document-Kind: model-knowledge

本页对应交互图：`app/models/deepseek-v4/`。默认变体是 **DeepSeek-V4-Flash**（284B / 13B 激活 / 1M 上下文）。Pro（1.6T / 49B）共用同一套 mHC + CSA/HCA + DeepSeekMoE，本图不展开 Pro 的更大 hidden / expert 配置。

## 范围

| 层类型 | `compress_ratios` | 实现 |
| --- | --- | --- |
| SWA | `0` | L0–L1 与末层：滑动窗口 128 + attention sink，无 Compressor / Indexer |
| CSA | `4` | Compressor overlap（coff=2）+ Lightning Indexer Top-512；Index K cache 独立 FP8 |
| HCA | `128` | 仅 Compressor，压缩流上稠密注意力 |

全部 43 层都是 MoE（Top-6 / 256 + 1 shared）。残差是 mHC（`hc_mult=4`），不是普通 `x+f(x)`。

## 源码钉扎

- vllm-ascend `DeepseekV4Attention` / `DeepseekV2DecoderLayer` / `DeepseekV4MoE`：`vllm_ascend/models/deepseek_v4/model.py`
- Compressor：`vllm_ascend/models/deepseek_v4/compressor.py`
- Lightning Indexer：`vllm_ascend/models/deepseek_v4/indexer.py`
- 层压缩比：`get_dsv4_compress_ratio` in `vllm_ascend/utils.py`
- Attention KV 与 Indexer KV 分离：`vllm_ascend/attention/dsa_attn_kv_plan.py`（Indexer 保持 FP8；A5 可用 `--kv-cache-dtype bfloat16` 改 attention KV）
- 官方配置：https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
- 论文：https://arxiv.org/abs/2606.19348

## 本地打开

```bash
cd model-atlas
npm install
npm run dev
```

浏览器打开后默认是 DeepSeek-V4-Flash。`?model=minimax-m3` 切回 MiniMax 参考实现。
