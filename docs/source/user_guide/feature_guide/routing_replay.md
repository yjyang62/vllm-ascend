# Routing Replay（路由回放）

!!! note

    Routing Replay 基于上游 vLLM 的 routed-experts 采集能力
    （`--enable-return-routed-experts`）。vLLM Ascend 针对 Ascend MoE
    通信布局（DP / EP / SP / AlltoAll / MC2）适配了采集路径。

Routing Replay（也称 **Routed Experts Replay** 或 **R3**）会在推理
rollout 阶段记录每个 token 实际命中的 MoE expert，并将这些 expert ID
随生成结果一并返回。训练框架可以在训练 forward 中**回放**同一套路由决策，
使训练侧 expert 选择与推理侧保持一致。

这对 GRPO、RLHF 等 MoE RL 流程非常关键：训练/推理 router 不一致会放大
policy KL，甚至导致训练不稳定或崩溃。

上游参考：

- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)
- 上游示例：[`examples/rl/routed_experts_e2e.py`](https://github.com/vllm-project/vllm/blob/main/examples/rl/routed_experts_e2e.py)

## 原理

在典型 MoE RL 循环中，rollout 和训练常常使用不同引擎（例如 vLLM 负责生成，
Megatron / FSDP 负责训练）。即便权重已经对齐，两侧 router 仍可能对同一
token 选出不同 expert，从而使 train/infer 概率偏差显著大于 Dense 模型。

| 局限 | 表现 | 后果 |
| --- | --- | --- |
| 训练/推理路由不一致 | 同一 token 激活不同 expert | policy KL 变大，RL 更新不稳定 |
| Router 非确定性 | 多次 forward 的 top-k expert 不一致 | rollout 与梯度难以复现 |
| Off-policy 放大 | importance ratio 出现极端值 | MoE RL 训练崩溃风险升高 |

Routing Replay 直接对准根因：**训练阶段复用推理阶段的路由结果**。

**没有 Routing Replay：**

```text
Rollout（vLLM） → expert 集合 A
训练 forward    → expert 集合 B（可能不同）
                → train/infer logits 发散
```

**启用 Routing Replay：**

```text
Rollout（vLLM） → expert 集合 A，并返回 routed_experts
训练 forward    → 强制使用 expert 集合 A
                → train/infer 路由对齐
```

## 工作流程

1. 推理引擎启动时打开 `--enable-return-routed-experts`。
2. 每一层 MoE forward 中，Ascend 捕获该层每个 token 的 `topk_ids`。
3. 请求完成后，vLLM 在 completion 结果中返回三维 expert ID 张量。
4. 训练侧按约定拼接/重塑张量（例如 Megatron 的
   `rollout_routed_experts`），并在训练 forward 中强制使用这些 expert 索引。

```mermaid
sequenceDiagram
    autonumber
    participant T as 训练侧 / RL Client
    participant S as vLLM Server<br/>Ascend Worker
    participant M as MoE Layers
    participant A as Completions / Generate API

    S->>S: vllm serve MODEL<br/>--enable-return-routed-experts
    T->>S: GET /health
    S-->>T: 200 OK

    T->>A: 推理 / rollout 请求
    A->>S: 调度 generate
    loop 每一层 MoE
        S->>M: select_experts → topk_ids
        M->>S: RoutedExpertsCapturer.capture(layer_id, topk_ids)
    end
    S->>S: 打包 prompt + decode 路由
    A-->>T: text + routed_experts<br/>shape [seq, layers, top_k]

    T->>T: 校验 shape / dtype<br/>构造 rollout_routed_experts
    T->>T: 训练 forward<br/>回放捕获的 expert IDs
    T-->>T: ROUTING_REPLAY=PASS
```

在 Ascend 上，采集挂接在 Ascend fused-MoE 路径中，并通过 patch 上游
`RoutedExpertsCapturer.capture`，在写入 device buffer 前正确处理
DP/EP/SP 下的 token 布局切片。

## 如何启用

### 在线服务

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --enable-return-routed-experts \
  --async-scheduling false
```

然后调用 OpenAI 兼容的 Completions API。功能开启后，已完成请求的每个
choice 会携带 base64 编码的 NumPy `routed_experts`：

```python
import io
import base64

import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    prompt="Hello, please introduce yourself.",
    max_tokens=32,
    temperature=0.0,
    extra_body={"return_token_ids": True},
)

payload = resp.model_dump()["choices"][0]["routed_experts"]
routed_experts = np.load(io.BytesIO(base64.b64decode(payload)))
# routed_experts.shape == [num_tokens, num_moe_layers, top_k]
print(routed_experts.shape, routed_experts.dtype)
```

### 离线 / 进程内 API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    enable_return_routed_experts=True,
    async_scheduling=False,
)

outputs = llm.generate(
    ["Hello, please introduce yourself."],
    SamplingParams(max_tokens=32, temperature=0.0),
)

routed = outputs[0].outputs[0].routed_experts
assert routed is not None and routed.size > 0
print(routed.shape)  # [seq_len, num_moe_layers, top_k]
```

## 返回约定

| 字段 | 位置 | 含义 |
| --- | --- | --- |
| `routed_experts` | `CompletionOutput` / `choices[].routed_experts` | 请求 token 使用的 expert ID |

张量约定：

| 属性 | 值 |
| --- | --- |
| Shape | `[num_tokens, num_moe_layers, top_k]` |
| 典型长度 | `prompt_len + generated_len - 1`（按 next-token 对齐） |
| Dtype（引擎缓冲） | worker 传输缓冲使用 `int32` |
| HTTP 编码 | base64 编码的 `.npy` 字节 |
| 合法 ID | `[0, num_experts)`（prefix cache 场景可能使用 `-1` 哨兵值） |

若训练框架期望单个 Megatron 侧缓冲（`rollout_routed_experts`），请先解码响应
张量，再校验 shape 后赋值：

```text
rollout_routed_experts.shape == (len(tokens) - 1, num_layers, moe_router_topk)
```

以 Qwen3-30B-A3B 为例：`(seq_len - 1, 48, 8)`。

## Ascend 实现说明

vLLM Ascend **不会**重写完整的训练侧 replay kernel，而是实现上游 vLLM 所需的
推理侧采集路径：

| 组件 | 作用 |
| --- | --- |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 在 `select_experts` 后调用 capturer，传入 `topk_ids` |
| `vllm_ascend/patch/worker/patch_routed_experts_capture.py` | patch `RoutedExpertsCapturer.capture`，适配 Ascend DP/SP/AlltoAll/MC2 布局 |
| `NPUModelRunner.init_routed_experts_capturer` | 分配缓冲，并把 capturer 绑定到 Ascend MoE runner |

Ascend capturer patch 覆盖的并行路径包括：

- 单 DP / 多 DP 的 token 归属切片
- padded all-gather 布局
- sequence parallel 分片后的 TP all-gather 重建
- AlltoAll 与 MC2 MoE 通信类型

## 限制

- **仅支持 MoE。** Dense 模型没有可采集的 routed experts。
- **建议关闭 async scheduling（`async_scheduling=False`）。**
  部分 vLLM 版本将 routed-experts 采集与 async scheduling 视为不兼容。
- **仅在请求完成后返回完整张量。** `routed_experts` 在请求 finish 时组装；
  流式分片不会各自携带完整路由张量。
- **需要训练侧接入。** 仅返回 expert ID 不够，训练栈必须在 MoE forward 中
  强制使用这些 ID（例如支持 R3 的框架中的 `--use-rollout-routing-replay`）。

## 已验证模型

当前 CI 覆盖：

- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen3.5-35B-A3B`

参见 `tests/e2e/pull_request/two_card/test_moe_routing_replay.py`。

## 相关功能

- [Batch Invariance](batch_invariance.md)：降低算子非确定性，可与 routing replay
  互补，提升 RL 稳定性。
- [DP 感知 Router](dp_aware_router.md)：External DP 下按负载选择哪一个 DP 实例
  （与本页 MoE routed-experts 采集不同）。
- [Sleep Mode](sleep_mode.md)：同卡 RL 场景下，在 rollout 与训练阶段之间做显存卸载。
- 权重同步示例见 `examples/rl/`（`rlhf_http_npu_ipc.py`、`rlhf_http_hccl.py`），
  用于把更新后的策略权重同步到推理引擎。
