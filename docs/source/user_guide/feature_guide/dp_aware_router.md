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

vLLM Ascend **不实现** vllm-router 本身，而是作为 Router 后端的推理 Engine：
在 Internal DP 拓扑下，同一 API Server 后挂多个 DP EngineCore；收到带
`X-data-parallel-rank` 的请求后，把 generate 派发到对应 rank 的 NPU 进程。

Ascend 侧链路：

```text
HTTP 请求（可经 vllm-router 转发）
  → Ascend API Server 解析 X-data-parallel-rank
  → 派发到指定 DP EngineCore
  → NPUWorker / NPUModelRunner（V1 或 V2）执行 forward / sample
  → 返回 token_ids / logprobs / ...
```

要点：

1. **拓扑是 Internal DP。** `--data-parallel-size N` 时，一个 `host:port`
   对应 N 个 EngineCore；Router 才能用 `http://host:port@rank` 点名某一 rank。
2. **落点由请求头决定。** 上游 Serving 读 `X-data-parallel-rank`；Ascend
   无额外 env。未带头时走服务端默认调度。
3. **Runner 不感知 Router。** V1（`worker/model_runner_v1.py`）与 V2
   （`worker/v2/model_runner.py`）只执行已被派发到本 rank 的 batch；选 rank
   的策略在 Router 侧。
4. **参数面直连本机 Engine。** `/update_weights`、IPC / HCCL 权重同步打到
   Ascend server 自身，不经过 Router（见 `examples/rl/`、
   `NPUWorker.update_weights`）。

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
