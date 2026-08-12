# DP Router（数据并行感知路由）

!!! note

    DP Router（数据并行感知路由）由上游 **vLLM Router + API Server** 提供：
    将请求精确路由到某个 vLLM 实例内部的具体 DP rank。

    RL 场景下，核心是把同一样本的 rollout **固定到同一个 DP**，保证推理路径
    稳定，支撑**训推一致**。本文先讲该原理，再写 Ascend 如何承接
    `X-data-parallel-rank`；RL / Vime 编排细节见 5.4.1。

上游参考：

- [vllm-project/router](https://github.com/vllm-project/router)
- DP-aware API：[vllm#24945](https://github.com/vllm-project/vllm/pull/24945)
- Token I/O：[Token In / Token Out](token_in_token_out.md)

## 1. 原理

### 1.1 整体原理：DP Router 对 RL 的作用

RL 要的是 **训推一致**：rollout（推理）产出的 token / logprobs 等，必须能
被训练侧在同一设定下复现或对齐。Internal DP 下若同一样本的 generate 落在
不同 DP rank，计算路径与数值环境可能分叉，训推更容易漂。

**DP Router 的核心作用**：让 RL 把相关请求**固定到同一个 DP rank**（感知并
指定 `data_parallel_rank`），从而保证这条样本在推理侧始终走同一 EngineCore，
支撑训推一致；而不是把流量随便打散到各个 DP。

```text
  同一样本的 rollout generate（可多次）
        │
        ▼
  vllm-router  ──►  固定落到同一 DP rank
                        推理路径稳定 → 便于与训练对齐（训推一致）
```

| 没有 DP 感知 | 有 DP Router |
| --- | --- |
| 同一样本可能落到不同 DP | 可绑定**同一个 DP** |
| 推理路径不稳定，训推易不一致 | 同 rank 上路径固定，利于训推一致 |
| session / 样本级落点不可控 | `x-session-id` + consistent_hash 等固定落点 |

手段：Router 注入 `X-data-parallel-rank`（或等价字段），Ascend Internal DP
Engine 按该 rank 派发。权重同步仍走参数面、**不经 Router**
（`/update_weights`、IPC / HCCL），与“固定同一 DP 做 rollout”分开。

### 1.2 Ascend 推理侧如何承接

vLLM Ascend **不实现** vllm-router，而是作为后端 Engine：拉起 Internal DP，
解析 `X-data-parallel-rank`，把请求派发到对应 EngineCore / NPUModelRunner。

```text
HTTP（Router 转发）
  → Ascend API Server 解析 X-data-parallel-rank
  → 指定 DP EngineCore
  → NPUWorker / NPUModelRunner（V1 或 V2）forward / sample
  → 返回 token_ids / logprobs / …
```

1. **拓扑**：`--data-parallel-size N` → 一个 `host:port` 对应 N 个
   EngineCore，Router 才能用 `http://host:port@rank` 点名。
2. **落点**：由请求头决定；Ascend 无额外 env；未带头则默认调度。
3. **Runner**：不感知 Router，只跑已被派发到本 rank 的 batch。
4. **权重**：`/update_weights` 等直连本机 Engine（见 `examples/rl/`）。

## 5. 特性工作流

### 5.2 特性工作流图

同一样本多次 generate 应落到**同一 DP**（训推一致）；权重同步仍直连 Engine。

```mermaid
sequenceDiagram
    autonumber
    participant RM as RolloutManager
    participant R as vllm-router
    participant E as vLLM Ascend Engine<br/>（含多 DP EngineCore）
    participant T as Trainer

    RM->>R: 启动 Router（建议 consistent_hash）
    RM->>E: 启动 vLLM Ascend<br/>vllm serve ... --data-parallel-size N
    E->>R: POST /workers 注册<br/>（或编排写入 worker-urls）

    Note over RM,E: 同一样本 / 同一 session → 固定同一 DP

    RM->>R: POST /inference/v1/generate<br/>x-session-id: sample-A
    R->>R: 按 session 映射到固定 rank<br/>（如 DP1）
    R->>E: 转发 + X-data-parallel-rank: 1
    E->>E: 派发到 EngineCore DP1
    E-->>R: token_ids / logprobs
    R-->>RM: 返回结果

    RM->>R: 同一样本再次 generate<br/>x-session-id: sample-A
    R->>E: 仍转发到 DP1<br/>X-data-parallel-rank: 1
    E-->>R: 结果（推理路径与上次一致）
    R-->>RM: 返回结果

    T->>E: /update_weights 或 IPC/HCCL
    Note over T,E: 权重同步绕过 Router<br/>与「固定 DP 做 rollout」分离
```

### 5.4 使用说明

#### 5.4.1 RL 端（摘要）

目标：同一样本的 rollout **始终打到同一个 DP**，保证训推一致。

- 推理 Client **不直连**某个 Engine，而是访问 Router：

```python
base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
url = f"{base}/inference/v1/generate"
```

- 普通文本（Token In / Token Out）；**务必带稳定的 session / 样本 ID**，以便
  Router 固定 DP：

```python
headers = {
    "x-session-id": sample.session_id,  # 同一样本多次 generate 用同一 ID
}
payload = {
    "model": args.hf_checkpoint,
    "token_ids": prompt_ids,
    "sampling_params": inference_sampling_params,
}
output = await post(
    f"{base}/inference/v1/generate",
    payload,
    headers=headers,
)
```

- RL 推荐 Router policy：`consistent_hash`（或 `cache_aware`），按
  `x-session-id` **绑定同一 Worker/DP**。`random` / `round_robin` 会打散
  落点，不利于训推一致，一般不用于该场景。
- 多模态：`/v1/chat/completions/render` 与 `/inference/v1/generate` 都走
  **同一 Router**，并带**同一** `x-session-id`，避免 render 与 generate 落到
  不同 DP。

#### 5.4.2 推理端（vLLM Ascend）

**拓扑要求：Internal DP**

Ascend 侧需以 Internal DP 暴露可被 Router 点名的多 rank Engine（同一
`host:port` 后挂多个 EngineCore），否则无法「固定同一个 DP」：

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --tensor-parallel-size 1
```

多节点时补充 `--data-parallel-size-local`、`--data-parallel-start-rank`、
`--data-parallel-address`、`--data-parallel-rpc-port` 等（见上游
[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)）。

MoE + EP 场景通常加 `--enable-expert-parallel`，并保证各 DP rank 通信配置一致。

**对接 Router（RL：固定同一 DP）**

Router 不在 vllm-ascend 仓库内，需单独安装
[vllm-router](https://github.com/vllm-project/router)。RL 示意用
`consistent_hash`：

```bash
pip install vllm-router

vllm-router \
  --worker-urls http://127.0.0.1:8000 \
  --intra-node-data-parallel-size 2 \
  --policy consistent_hash \
  --host 0.0.0.0 \
  --port 30000
```

- `--intra-node-data-parallel-size` 需与后端 `--data-parallel-size` 对齐。
- Router 将逻辑后端展开为 `http://host:port@rank`，转发时注入
  `X-data-parallel-rank`；同一 `x-session-id` 应得到**同一 rank**。
- Ascend **无额外 env 开关**；打开 Internal DP 并被 Router 注册即可。

**Engine 侧请求落点**

1. Router 按 session/策略选定**固定** rank 后转发到 Ascend API Server。
2. Serving 解析 `X-data-parallel-rank`，派发到对应 DP EngineCore——同一样本
   多次请求应进**同一个** EngineCore，推理路径稳定，支撑训推一致。
3. 未携带该头时走服务端默认调度，**落点不稳定**，RL 场景应避免。
4. Rollout 常用 `POST /inference/v1/generate`（见
   [Token In / Token Out](token_in_token_out.md)）。

调试时可直连 Engine，手动固定 rank（模拟 Router 行为）：

```bash
# 同一样本的两次请求都指定 rank=1，验证固定 DP
curl http://127.0.0.1:8000/inference/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-data-parallel-rank: 1" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "token_ids": [1, 2, 3],
    "sampling_params": {"max_tokens": 16, "temperature": 0.0}
  }'
```

**权重同步（绕过 Router）**

训练侧应直连 Engine 做参数面操作，与「rollout 固定同一 DP」分开：

```bash
# 启动时按需打开开发态权重传输（示例）
VLLM_SERVER_DEV_MODE=1 vllm serve MODEL \
  --weight-transfer-config '{"backend": "hccl"}' \
  ...
```

Trainer → `http://<engine>/update_weights`（或 IPC / HCCL 数据面），**不要**
经 `vllm-router`。参考：

- `examples/rl/rlhf_http_hccl.py`
- `examples/rl/rlhf_http_npu_ipc.py`
- [Sleep / Wakeup](sleep_wakeup.md)

## 5.3 版本区别 / 支持情况

| 维度 | Model Runner V1 | Model Runner V2 |
| --- | --- | --- |
| 按 `X-data-parallel-rank` 固定 DP | 支持 | 支持 |
| 经 Router 的 `/inference/v1/generate` | 支持 | 支持 |
| Ascend 额外开关 | 无 | 无（`VLLM_USE_V2_MODEL_RUNNER=1` 只切 runner） |

结论：**V1 / V2 均已支持**「指定 / 固定 DP rank」语义，可用于 RL 训推一致
所需的同 DP 落点。差异在 runner 能力本身（多模态、PD 等），不在能否被
Router 点名。

同一 Router 后的 Engine 不要混部 V1/V2（混部本身会破坏路径一致性）。生产
建议默认 V1；V2 跟踪
[MRv2 RFC](https://github.com/vllm-project/vllm-ascend/issues/5208)。

## 限制

- **依赖 Internal DP。** 每 rank 独立 port 的 External DP 应按 host:port 做
  外部均衡，见 [External DP](external_dp.md) / [DP Router 代理](dp_router.md)，
  与本文「单入口 + `X-data-parallel-rank`」不同。
- **RL 勿用会打散落点的 policy。** `random` / `round_robin` 无法保证同一样本
  固定同一 DP，不利训推一致；应用 `consistent_hash` 等并带稳定
  `x-session-id`。
- **Router 实现不在 Ascend 内。** 发现、policy、重试、熔断以 vllm-router 为准。
- **MoE 集合通信仍在。** 指定 rank 只固定请求落点，不取消跨 DP 的 MoE/EP 同步；
  MoE expert 级训推对齐另见 [Routing Replay](routing_replay.md)。
- **参数面与数据面分离。** 权重更新必须直连 Engine；误走 Router 可能导致错误
  后端或状态不一致。

## 相关功能

- [Token In / Token Out](token_in_token_out.md)：RL rollout 常用
  `/inference/v1/generate`
- [External DP](external_dp.md) / [DP Router](dp_router.md)：External DP
  按 endpoint 分发（不同机制）
- [Sleep / Wakeup](sleep_wakeup.md)、`examples/rl/`：权重同步与分阶段 RL
- [Routing Replay](routing_replay.md)：MoE routed-experts 采集（不同机制）
- 上游：[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
