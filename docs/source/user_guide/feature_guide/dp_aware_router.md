# DP Router（数据并行感知路由）

!!! note

    DP Router（数据并行感知路由）是 **vLLM Router** 中用于将请求**精确路由到
    某个 vLLM 实例内部具体 DP rank** 的机制。能力由上游 vLLM API Server
    （`X-data-parallel-rank`）与 [vllm-router](https://github.com/vllm-project/router)
    提供；vLLM Ascend 作为推理后端**直接复用**该路径，无额外开关。

    本文聚焦 **vLLM Ascend 推理侧**：如何拉起带 Internal DP 的 `vllm serve`，
    以及如何与 vllm-router 对接。训练过程、reward 计算和模型参数更新不在本文范围。

## 1. 原理

在 RL / 大规模 Serving 中，常见分工如下（以 Vime 类编排为例）：

| 组件 | 职责 |
| --- | --- |
| Vime / 训练后端 | 训练过程、reward 计算、模型参数更新 |
| vllm-router | HTTP 网关：维护 vLLM worker 列表、选择后端 worker、转发推理请求；在 DP 感知模式下把请求打到指定 DP rank |
| vLLM Ascend（`vllm serve`） | Internal DP 多 engine；按请求携带的 DP rank 把推理调度到对应 EngineCore |

**没有 DP 感知路由：**

```text
Client / Trainer → 某个 vLLM HTTP 入口
                 → API Server 自行（或随机/默认）选 DP engine
                 → 难以做 cache 亲和 / 精确负载控制
```

**启用 DP 感知路由：**

```text
Client / Trainer → vllm-router（统一入口）
                 → 选 worker，并指定 data_parallel_rank
                 → vLLM Ascend API Server 按 X-data-parallel-rank
                   把请求派发到对应 DP EngineCore
```

推理侧关键约定：

1. Ascend 上以 **Internal DP** 拉起：`--data-parallel-size N`，同一 HTTP
   endpoint 后挂多个 DP engine。
2. Router 将逻辑后端表示为 `http://host:port@rank`（例如
   `http://0.0.0.0:8000@4`）。
3. 转发时注入请求头 **`X-data-parallel-rank: <rank>`**（或上游等价字段）。
4. 上游 OpenAI Serving 解析该头，engine 派发到对应 DP rank；未携带时行为与
   普通请求一致（由服务端默认调度）。

!!! warning

    不要与下列能力混淆：

    - [DP Router / External DP 代理](dp_router.md)：按 host:port 分发到**各自独立
      endpoint** 的 External DP 实例（`dp_load_balance_proxy_server.py`），
      **不**注入 `X-data-parallel-rank`。
    - [Routing Replay](routing_replay.md)：MoE expert 路由采集与训练回放，
      与 HTTP 选 DP rank **无关**。

## 5. 特性工作流

### 5.2 特性工作流图

```mermaid
sequenceDiagram
    autonumber
    participant T as Client / Trainer<br/>（如 Vime rollout）
    participant R as vllm-router<br/>DP 感知网关
    participant S as vLLM Ascend<br/>API Server :8000
    participant E0 as EngineCore DP0
    participant E1 as EngineCore DP1

    Note over S,E1: vllm serve MODEL<br/>--data-parallel-size 2

    T->>R: POST /v1/completions<br/>或 /v1/chat/completions
    R->>R: 维护 worker 列表<br/>按 policy 选择后端
    R->>R: 选定逻辑 worker<br/>http://host:8000@rank
    alt 选中 DP0
        R->>S: 转发请求<br/>X-data-parallel-rank: 0
        S->>E0: 派发到 EngineCore DP0
        E0-->>S: 流式 / 完整响应
    else 选中 DP1
        R->>S: 转发请求<br/>X-data-parallel-rank: 1
        S->>E1: 派发到 EngineCore DP1
        E1-->>S: 流式 / 完整响应
    end
    S-->>R: 回传响应
    R-->>T: 回传响应
```

## 如何启用（推理侧）

### 1. 拉起 vLLM Ascend（Internal DP）

单机示例（DP=2，TP=1）：

```bash
vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --tensor-parallel-size 1
```

多节点 Internal DP 时，按上游约定补充本地 rank 与 RPC，例如：

```bash
# 示意：本节点负责一部分 DP rank
vllm serve MODEL \
  --host 0.0.0.0 --port 8000 \
  --data-parallel-size $DP_SIZE \
  --data-parallel-size-local $DP_SIZE_LOCAL \
  --data-parallel-start-rank $DP_START_RANK \
  --data-parallel-address $DP_MASTER_ADDR \
  --data-parallel-rpc-port $DP_RPC_PORT \
  ...
```

MoE + EP 时通常还需 `--enable-expert-parallel`，并保证各 DP rank 的通信地址
一致；具体组合以目标模型与 [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) 为准。

### 2. 启动 vllm-router（DP 感知）

Router **不属于** vllm-ascend 仓库；需单独安装
[vllm-router](https://github.com/vllm-project/router)：

```bash
pip install vllm-router

# 将单个 Ascend 服务的每个 DP rank 视为可调度后端
vllm-router \
  --worker-urls http://127.0.0.1:8000 \
  --intra-node-data-parallel-size 2 \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30000
```

说明：

- `--intra-node-data-parallel-size`（或文档/版本中的等价 DP 感知参数）需与
  后端 `--data-parallel-size` 对齐。
- Router 会展开为逻辑 worker：`http://127.0.0.1:8000@0`、
  `http://127.0.0.1:8000@1`，转发时带上对应 `X-data-parallel-rank`。
- `--policy` 可选 `round_robin`、`consistent_hash`、`cache_aware` 等，用于
  **选哪个 rank**；Ascend 侧只负责按头字段执行派发。

多 worker URL 时：

```bash
vllm-router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --intra-node-data-parallel-size 8 \
  --policy consistent_hash \
  --port 30000
```

### 3. 向 Router 发请求

训练 / rollout Client（如 Vime）只打 Router 入口即可：

```bash
curl http://127.0.0.1:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello",
    "max_tokens": 32,
    "temperature": 0
  }'
```

调试时可直连 Ascend API Server 并手动指定 rank：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-data-parallel-rank: 1" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Hello",
    "max_tokens": 16
  }'
```

非法或越界的 rank 通常被忽略并回退为默认调度（以当前上游实现为准）；上线前请用
目标 vLLM 版本做一次联调确认。

### 与 Vime / 训练侧的边界

| 层级 | 负责方 | 说明 |
| --- | --- | --- |
| Rollout HTTP 入口 | vllm-router | 维护 worker、选后端、注入 DP rank、转发 |
| 推理执行 | vLLM Ascend | Internal DP EngineCore 按 rank 执行 generate |
| 训练 / reward / 权重更新 | Vime 及训练后端 | **不**经 Router；权重回写推理实例见 Sleep/Wake、weight transfer 等文档 |

## Model Runner V1 / V2

DP 感知路由发生在 **API Server → Engine 派发** 层，与 Model Runner V1/V2
的 HTTP 契约相同：均消费 `X-data-parallel-rank`（由上游 Serving 解析）。

| 维度 | 说明 |
| --- | --- |
| 推理侧开关 | 无需 Ascend 专用 env；开 Internal DP + 对接 router 即可 |
| V1 / V2 | 路由语义等价；后端不要混部不同 runner 代际 |
| 建议 | 生产先用 V1；V2 跟进 [MRv2 RFC](https://github.com/vllm-project/vllm-ascend/issues/5208) |

## 限制

- **依赖 Internal DP 拓扑。** 同一 `host:port` 后需有多个 DP EngineCore；纯
  External DP（每 rank 独立 port）应使用 [External DP](external_dp.md) /
  [DP Router 代理](dp_router.md)，而不是本机制。
- **Router 与 Ascend 版本需匹配。** `X-data-parallel-rank` 与
  `--intra-node-data-parallel-size` 以所安装的上游 vLLM / vllm-router 为准。
- **Ascend 不实现 Router 本身。** worker 发现、policy、重试、熔断见
  vllm-router 文档。
- **MoE 跨 DP 同步仍在。** 即使请求打到某一 rank，MoE/EP 下其它 rank 仍可能
  参与通信；DP 感知路由解决的是**请求落点**，不是取消 DP 集合通信。
- **与 Routing Replay 无关。** 需要 expert 回放时另开
  `--enable-return-routed-experts`，见 [Routing Replay](routing_replay.md)。

## 相关功能

- 上游 Data Parallel：[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- 上游 Router：[vllm-project/router](https://github.com/vllm-project/router)
- 上游 API 支持：[vllm#24945](https://github.com/vllm-project/vllm/pull/24945)
- [External DP](external_dp.md) / [DP Router](dp_router.md)：每 rank 独立
  endpoint 的负载均衡（不同机制）
- [Routing Replay](routing_replay.md)：MoE routed-experts 采集（不同机制）
