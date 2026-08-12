# Supported Features

The feature support principle of vLLM Ascend is: **aligned with vLLM**. We are also actively collaborating with the community to accelerate support.

Tool calling: [vLLM documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)

Routing Replay（MoE RL 路由回放 / routed-experts capture）：[Routing Replay][routing_replay]

DP 感知 Router（多 DP 下按 `dp_rank` 采集 MoE 路由）：[DP 感知 Router][dp_aware_router]

DP Router（External DP 外部负载均衡路由）：[DP Router][dp_router]

Token In / Token Out（`/inference/v1/generate`）：[Token In / Token Out][token_in_token_out]

Sleep / Wakeup（RL 分阶段休眠唤醒流程）：[Sleep / Wakeup][sleep_wakeup]

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       |      Status    | Next Step                                                              |
|-------------------------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🟢 Functional    | Functional, see detailed note: [Chunked Prefill][cp]                     |
| Automatic Prefix Caching      | 🟢 Functional    | Functional, see detailed note: [vllm-ascend#732][apc]                    |
| LoRA                          | 🔵 Experimental  | Functional, see detailed note: [LoRA][LoRA]                    |
| Speculative decoding          | 🟢 Functional    | Basic support                                                          |
| Pooling                       | 🔵 Experimental  | CI needed to adapt to more models; V1 support relies on vLLM support.   |
| Enc-dec                       | 🟡 Planned       | vLLM should support this feature first.                                |
| Multi Modality                | 🟢 Functional    | [Multi Modality][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional    | CI needed                                                              |
| Prompt LogProbs               | 🟢 Functional    | CI needed                                                              |
| Async output                  | 🟢 Functional    | CI needed                                                              |
| Beam search                   | 🔵 Experimental  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional    | [vllm-ascend#177][guided_decoding]                                     |
| Tensor Parallel               | 🟢 Functional    | Make TP >4 work with graph mode.                                        |
| Pipeline Parallel             | 🟢 Functional    | Write official guide and tutorial.                                     |
| Expert Parallel               | 🟢 Functional    | Support dynamic EPLB.                                                  |
| Data Parallel                 | 🟢 Functional    | Data Parallel support for Qwen3 MoE.                                   |
| Prefill Decode Disaggregation | 🟢 Functional    | Functional, xPyD is supported.                                         |
| Quantization                  | 🟢 Functional    | W8A8 available; working on more quantization method support (W4A8, etc.) |
| Graph Mode                    | 🟢 Functional    | Functional, see detailed note: [Graph Mode][graph_mode]                 |
| Sleep Mode                    | 🟢 Functional    | Functional, see detailed note: [Sleep Mode][sleep_mode]                 |
| Sleep / Wakeup (RL flow)      | 🟢 Functional    | Functional, see detailed note: [Sleep / Wakeup][sleep_wakeup]           |
| Context Parallel              | 🟢 Functional    | Functional, see detailed note: [Context Parallel][context_parallel]     |
| Routing Replay                | 🔵 Experimental  | Functional, see detailed note: [Routing Replay][routing_replay]         |
| DP-Aware Router               | 🔵 Experimental  | Functional, see detailed note: [DP-Aware Router][dp_aware_router]       |
| DP Router                     | 🔵 Experimental  | Functional, see detailed note: [DP Router][dp_router]                   |
| Token In / Token Out          | 🔵 Experimental  | Functional, see detailed note: [Token In / Token Out][token_in_token_out] |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan/Deprecated: No plan or deprecated by vLLM.

[v1_user_guide]: https://docs.vllm.ai/en/latest/usage/v1_guide/
[multimodal]: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen-VL-Dense.html
[guided_decoding]: https://github.com/vllm-project/vllm-ascend/issues/177
[LoRA]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html
[graph_mode]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html
[apc]: https://github.com/vllm-project/vllm-ascend/issues/732
[cp]: https://docs.vllm.ai/en/stable/configuration/optimization/
[1P1D]: https://github.com/vllm-project/vllm-ascend/pull/950
[context_parallel]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html
[sleep_mode]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sleep_mode.html
[routing_replay]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/routing_replay.html
[dp_aware_router]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dp_aware_router.html
[dp_router]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/dp_router.html
[token_in_token_out]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/token_in_token_out.html
