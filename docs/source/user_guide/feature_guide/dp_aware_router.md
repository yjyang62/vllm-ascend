# DP Router（数据并行感知路由）

!!! note

    DP Router（数据并行感知路由）是 **vLLM Router** 中用于将请求精确路由到
    某个 vLLM 实例内部具体 DP rank 的机制。

    在 RL 编排（以 Vime 为例）中：Vime 使用 **vllm-router** 作为 vLLM rollout
    的 HTTP 网关；Router 维护 worker 列表、选择后端并转发推理请求；训练、
    reward、参数更新仍由 Vime / 训练后端负责。

    **本文写 vLLM Ascend 推理侧**：Engine 如何作为 Router 后端被访问、如何按
    DP rank 承接请求，以及权重同步为何绕过 Router。RL 侧接入（RolloutManager
    打 Router、`x-session-id` 等）见编排框架文档；此处只给出与推理侧对齐的
    调用关系与启用方式。

上游参考：

- [vllm-project/router](https://github.com/vllm-project/router)
- DP-aware API：[vllm#24945](https://github.com/vllm-project/vllm/pull/24945)
- Token I/O：[Token In / Token Out](token_in_token_out.md)

## 1. 原理

### 1.1 角色分工

| 组件 | 职责 |
| --- | --- |
| 训练（Megatron Actor / Critic 等） | 参数更新、组训练 batch；权重同步直连 Engine |
| RolloutManager | 编排 rollout、发起 generate、样本 / reward 转换 |
| vllm-router | HTTP 路由 / 负载均衡；维护 worker；把请求打到具体 Worker（及 DP rank） |
| vLLM Ascend Engine 集群 | 执行推理；按 `X-data-parallel-rank`（或等价字段）落到对应 DP EngineCore |

核心边界：

- **推理数据面**：`RolloutManager → vllm-router → vLLM Ascend Engine`
- **训练参数面**：`Trainer → vLLM Engine`（如 `/update_weights`、IPC / HCCL），
  **权重同步不经过 Router**

### 1.2 vllm-router 与训练 / 推理 Manager 的调用关系

```mermaid
flowchart LR
    T["训练<br/>Megatron Actor / Critic"]
    RM["RolloutManager<br/>编排与数据转换"]
    R["vllm-router<br/>HTTP 路由 / 负载均衡"]
    V["vLLM Ascend Engine 集群"]
    D["Rollout 数据<br/>tokens / reward / logprobs"]

    T -->|"1. 训练参数更新"| T
    T -->|"2. 权重同步<br/>IPC / NCCL / HCCL<br/>不经过 Router"| V

    RM -->|"3. 发起 rollout"| R
    R -->|"4. 路由请求"| V
    V -->|"5. 生成结果"| R
    R -->|"6. 返回结果"| RM

    RM -->|"7. reward + 样本转换"| D
    D -->|"8. 训练 batch"| T

    classDef train fill:#e8f1ff,stroke:#4b83d8,color:#17365d
    classDef router fill:#fff3d6,stroke:#d99a00,color:#5c4300
    classDef infer fill:#e9f7ef,stroke:#45a36b,color:#174d2a
    classDef data fill:#f2eafa,stroke:#8a63b8,color:#45265f

    class T train
    class R router
    class V infer
    class RM,D data
```

对 Ascend 推理侧意味着：

1. **Rollout 流量只从 Router 进来**（Client 不直连某个 Engine 做日常 generate）。
2. **Engine 需可被 Router 发现 / 注册**（例如 `POST /workers` 或静态
   `--worker-urls`，以所用 vllm-router 版本为准）。
3. **权重面直连 Engine**（`examples/rl/rlhf_http_hccl.py` /
   `rlhf_http_npu_ipc.py` 中的 `/update_weights` 等），避免 Router 成为参数面瓶颈或状态错位。

## 5. 特性工作流

### 5.2 特性工作流图

```mermaid
sequenceDiagram
    autonumber
    participant RM as RolloutManager
    participant R as vllm-router
    participant E as vLLM Ascend Engine<br/>（含多 DP EngineCore）
    participant T as Trainer

    RM->>R: 启动 Router
    RM->>E: 启动 vLLM Ascend<br/>vllm serve ... --data-parallel-size N
    E->>R: POST /workers 注册自身地址<br/>（或由编排写入 worker-urls）

    RM->>R: POST /inference/v1/generate<br/>（可带 x-session-id）
    R->>R: 按 policy 选 Worker / DP rank<br/>random / round_robin / cache_aware / ...
    R->>E: 转发 + X-data-parallel-rank
    E->>E: API Server 派发到对应 DP EngineCore
    E-->>R: token_ids / logprobs / ...
    R-->>RM: 返回生成结果

    T->>E: /update_weights 或 IPC/HCCL
    Note over T,E: 权重同步绕过 Router
```

### 5.4 使用说明

#### 5.4.1 RL 端（摘要）

RL / Vime 侧要点（推理代码应对齐，细节以编排框架为准）：

- 推理 Client **不直连**某个 Engine，而是访问 Router：

```python
base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
url = f"{base}/inference/v1/generate"
```

- 普通文本（Token In / Token Out）大致为：

```python
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

- Router 按策略选 Worker，例如：`random`、`round_robin`、`cache_aware`、
  `poweroftwo`、`consistent_hash`。
- 多轮请求可设 `x-session-id: <sample.session_id>`；在 `consistent_hash` 下
  尽量打到同一 Worker，便于复用 prefix / KV cache。
- 多模态：先 `/v1/chat/completions/render`，再 `/inference/v1/generate`，
  **都走同一 Router 地址**。

#### 5.4.2 推理端（vLLM Ascend）

**拓扑要求：Internal DP**

Ascend 侧需以 Internal DP 暴露可被 Router 感知的多 rank Engine（同一
`host:port` 后挂多个 EngineCore）：

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

**对接 Router**

Router 不在 vllm-ascend 仓库内，需单独安装
[vllm-router](https://github.com/vllm-project/router)。示意：

```bash
pip install vllm-router

vllm-router \
  --worker-urls http://127.0.0.1:8000 \
  --intra-node-data-parallel-size 2 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000
```

- `--intra-node-data-parallel-size` 需与后端 `--data-parallel-size` 对齐。
- Router 将逻辑后端展开为 `http://host:port@rank`，转发时注入
  `X-data-parallel-rank`。
- Ascend **无额外 env 开关**；打开 Internal DP 并被 Router 注册即可。

**Engine 侧请求落点**

1. Router 选定 Worker / rank 后转发到 Ascend API Server。
2. Serving 解析 `X-data-parallel-rank`，将请求派发到对应 DP EngineCore。
3. 未携带该头时，行为与普通请求一致（服务端默认调度）。
4. Rollout 常用路径为 `POST /inference/v1/generate`（见
   [Token In / Token Out](token_in_token_out.md)）；Chat/Completions /
   render 同样可经 Router 转发。

调试时可直连 Engine 并手动指定 rank：

```bash
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

训练侧应直连 Engine 做参数面操作，例如：

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
| DP 感知路由（API 派发） | 支持 | 支持 |
| 经 Router 的 `/inference/v1/generate` | 支持 | 支持 |
| Ascend 额外开关 | 无 | 无（`VLLM_USE_V2_MODEL_RUNNER=1` 只切 runner） |

结论：**V1 / V2 均已支持** DP 感知路由语义（请求落到指定 DP rank）。差异在
runner 能力本身（多模态、PD 等），不在「能否被 Router 按 rank 点名」。

同一 Router 后的 Engine 不要混部 V1/V2。生产建议默认 V1；V2 跟踪
[MRv2 RFC](https://github.com/vllm-project/vllm-ascend/issues/5208)。

## 限制

- **依赖 Internal DP。** 每 rank 独立 port 的 External DP 应按 host:port 做
  外部均衡，见 [External DP](external_dp.md) / [DP Router 代理](dp_router.md)，
  与本文「单入口 + `X-data-parallel-rank`」不同。
- **Router 实现不在 Ascend 内。** 发现、policy、重试、熔断以 vllm-router 为准。
- **MoE 集合通信仍在。** 指定 rank 只决定请求落点，不取消跨 DP 的 MoE/EP 同步。
- **参数面与数据面分离。** 权重更新必须直连 Engine；误走 Router 可能导致错误
  后端或状态不一致。
- **与 Routing Replay 无关。** Expert 回放见 [Routing Replay](routing_replay.md)。

## 相关功能

- [Token In / Token Out](token_in_token_out.md)：RL rollout 常用
  `/inference/v1/generate`
- [External DP](external_dp.md) / [DP Router](dp_router.md)：External DP
  按 endpoint 分发（不同机制）
- [Sleep / Wakeup](sleep_wakeup.md)、`examples/rl/`：权重同步与分阶段 RL
- [Routing Replay](routing_replay.md)：MoE routed-experts 采集（不同机制）
- 上游：[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
