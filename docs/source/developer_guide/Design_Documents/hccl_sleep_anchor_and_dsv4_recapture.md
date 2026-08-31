# HCCL Sleep Anchor 与 DSV4 ACL Graph 重捕获

## 背景

vLLM Ascend 的 `sleep_mode_extra_cleanup` 会在休眠时释放 ACL Graph、Attention workspace 和 HCCL 通信资源，并在唤醒后恢复通信域、重新捕获 ACL Graph。该机制能够释放更多设备内存，但 DeepSeek V4（DSV4）的 DSA 稀疏 Attention 路径可能在唤醒后的 FULL Graph 重捕获阶段首先暴露底层资源生命周期问题。

一种候选规避方案是在休眠前创建一个临时 HCCL lifecycle anchor：销毁业务通信域时保留这一个真实的多卡 communicator，待业务通信域恢复且 ACL Graph 重捕获完成后再释放 anchor。

本文解释：

- HCCL 全部清空与保留 anchor 对组图环境的区别；
- 为什么重建业务 HCCL communicator 不等于恢复全部运行时状态；
- 为什么当前主要在 DSV4 上观察到问题；
- anchor 方案的生命周期、限制和验证方法。

!!! warning

    “销毁最后一个 HCCL communicator 会破坏 DSV4 所依赖的共享运行时状态”目前是需要真机 A/B 实验验证的假设，不是 CANN 的公开接口契约。Anchor 是诊断性规避方案，不应在验证前描述为确定根因。

## 术语

### HCCL

HCCL（Huawei Collective Communication Library）向框架提供 AllReduce、AllGather、ReduceScatter、AllToAll、Send 和 Receive 等集合通信与点对点通信能力。

### HCOMM

HCOMM（Huawei Communication）是 HCCL 的通信基础库，分为控制面和数据面：

- 控制面负责拓扑查询、通信域以及通信资源管理；
- 数据面负责数据搬运、通信操作、本地操作和算子间同步。

HCOMM 管理的资源可能包括通信连接、buffer、stream、event、notify 和通信引擎上下文。

### HCCP

HCCP 是更接近设备、驱动和网络链路的通信协议与服务层。设备日志中常见 `hccp_service.bin`。它参与设备侧通信服务、RDMA/RoCE agent、连接和链路状态管理。

HCCP、HCOMM 和 Generic AICPU runtime 不是同一个概念。特别是，DSA metadata 是 AICPU 算子，并不能仅凭这一点认定它由 HCCP 提供后台服务。

### Communicator

Communicator 是一组 Rank 的具体通信上下文。它不只对应一块通信 buffer，还可能关联拓扑、连接、stream、event、notify 和底层通信状态。

## HCCL 全清空与保留 Anchor

从模型图的逻辑内容看，两种方式没有直接区别。Anchor 不参与模型 forward，也不会成为模型图中的计算节点。区别在于重捕图所处的底层通信运行时是否可能经历一次最终清理和重新初始化。

| 对比项 | 全部清空 HCCL | 保留一个 anchor group |
| --- | --- | --- |
| TP/EP/DP/MC2 业务 communicator | 全部销毁 | 全部销毁 |
| 临时 anchor communicator | 无 | 保留一个真实多卡 communicator |
| Communicator 总数是否变为 0 | 是 | 否 |
| 是否可能触发 HCCL/HCOMM 最终清理 | 可能 | 避免进入“最后一个 communicator”路径 |
| 业务 communicator 是否在 wake 后重建 | 是 | 是 |
| 业务 communicator handle | 新 handle | 同样是新 handle |
| 进程级通信 runtime | 可能发生代际切换 | 尽量保持连续 |
| 旧 static task/executor/cache | 可能跨 runtime 代际 | 仍处于原 runtime 环境 |
| 模型 Graph 的逻辑节点 | 不变 | 不变 |
| Sleep 释放显存 | 最多 | 少释放 anchor 的资源 |
| 风险 | 恢复链路可能不完整 | 依赖未公开的生命周期行为 |

### 全部清空

```text
Runtime 第 1 代
  → 销毁所有业务 communicator
  → 最后一个 communicator 被销毁
  → 可能执行 HCCL/HCOMM 进程级最终清理
  → 创建 Runtime 第 2 代资源
  → 重建业务 communicator
  → 重捕获 ACL Graph
```

### 保留 Anchor

```text
Runtime 第 1 代
  → 创建并物理初始化 anchor
  → 销毁所有业务 communicator
  → anchor 使 communicator 数量保持大于 0
  → Runtime 第 1 代继续存活
  → 重建业务 communicator
  → 刷新 MC2/HCCL 业务引用
  → 重捕获 ACL Graph
  → 所有 Rank 同步
  → 释放 anchor communicator
```

Anchor 只在管理流程中执行 barrier，模型 forward 不使用它，因此不应被记录进模型图。

## 为什么重建 HCCL 仍可能不够

当前 `restore_hccl()` 主要重新创建 device process group 和 device communicator，并刷新部分 MoE/MC2 元数据。它不是完整的 worker 冷启动。

| 资源 | `restore_hccl()` 是否明确恢复 | 说明 |
| --- | --- | --- |
| `ProcessGroupHCCL` | 是 | 创建新的 device process group |
| HCCL communicator | 是 | 获得新的 communicator handle |
| Device communicator wrapper | 是 | 重新初始化 |
| MC2 communicator 信息 | 部分 | 通过专用 refresh 路径更新 |
| Python `GroupCoordinator` | 否 | 通常复用原对象 |
| 模型层对象 | 否 | 模型不会重新加载 |
| DSA metadata builder | 否 | 仅在 KV cache 初始化时创建 |
| 自定义算子 static 对象 | 否 | 随进程继续存活 |
| CANN aclnn executor/task cache | 否 | 没有统一的 Python 清理接口 |
| ACL Graph 全局 pool | 否 | 清 graph entry 不等于重建全局 pool |
| Python/C++ 全局 stream | 通常否 | 多数对象继续存在 |
| HCCP/HCOMM 内部资源 | 不由 Python 直接控制 | 取决于底层实现 |

因此，业务 HCCL 重建后可能出现如下状态：

```text
旧的 AICPU task/executor/cache → 仍保存第 1 代资源信息
新的业务 communicator        → 使用第 2 代资源
```

这是一种可能的“跨代引用”。如果 CANN 和 vLLM 的资源边界完整，销毁后重建理应正常；若 anchor 才能解决，则说明可能存在：

- vLLM 少清理了某个缓存；
- 自定义算子的 static task 没有刷新；
- HCCL 最终清理范围大于调用方预期；
- restore 没有恢复某项共享状态；
- 销毁与异步任务之间存在竞态；
- 或实际根因根本不在 HCCL 生命周期。

## 为什么直接模仿首次捕图不一定有效

首次捕图发生在一致的全新状态中：

```text
初始化 ACL device context
→ 创建 HCCL communicator
→ 初始化模型和 DSA builder
→ 初始化 KV/control buffer
→ 初始化算子与 task cache
→ 创建 stream/event
→ eager/profile warmup
→ 首次 capture_model()
```

Wake 重捕图则发生在混合状态中：

```text
旧 Python/模型/static 对象继续存在
→ 旧 Graph 被清理
→ CaMem 部分内存被 unmap/remap
→ HCCL communicator 被销毁/重建
→ 部分进程级缓存继续存在
→ capture_model()
```

重新调用 `capture_model()` 只能重放捕图流程，不能重新运行全部构造函数，也不能重建 Python 不可见的 CANN/HCCP/AICPU 内部状态。某些初始化还是一次性的，例如 graph parameter 容器不允许重复初始化。

真正完全模仿冷启动需要重建整个 worker、重新加载模型和 KV cache，这等价于服务重启，会失去 sleep/wake 快速恢复的意义。

## 为什么目前主要是 DSV4

“存在多流”不是区分条件。关键在于流上执行的算子类型、缓存方式以及它在重捕图中的位置。

| 模型类别 | Attention/metadata 特点 | 重捕图早期是否执行自定义 AICPU metadata | 对旧 task/cache 的敏感度 |
| --- | --- | --- | --- |
| Llama/Qwen Dense | 常规 Attention，metadata 多由 Python/device tensor 构造 | 通常否 | 低 |
| 普通 GQA/MHA | 标准 FA/FIA 路径 | 通常否 | 低 |
| DeepSeek V3/R1 MoE | 有 EP/MC2，但不是 DSV4 DSA pipeline | 通常否 | 中 |
| Qwen MoE | 有 EP/AllToAll/MC2 | 通常否 | 中 |
| DSV4 | DSA + Indexer + Compressor + Sparse Attention | 是 | 高 |
| DSV4 + DSA CP | 增加 TP/CP metadata 与通信 | 是 | 更高 |

### DSV4 的特殊调用链

```text
构造 DSA metadata
→ SparseAttnSharedkvMetadata
→ 生成固定 1024 个 int32 的任务切分表
→ Sparse Attention 消费任务表
→ Indexer/Compressor/KV 流程
→ MoE/MC2
```

`SparseAttnSharedkvMetadata` 使用进程级 static task space：

```cpp
static internal::AicpuTaskSpace space(
    "SparseAttnSharedkvMetadata");
```

同时，DSV4 重捕图时会重新生成 DSA metadata，因此该算子会在组图准备阶段很早执行。它一旦失败，worker 会立即退出，排在后面的 AICPU 算子没有机会执行。

所以目前只能确认：

> `SparseAttnSharedkvMetadata` 是重捕图后第一个稳定失败的自定义 AICPU 算子。

不能据此确认：

> 其他模型和其他 AICPU 算子一定没有同类问题。

仓库中其他 static AICPU task 还包括：

- `VllmQuantLightningIndexerMetadata`；
- `StoreKvBlockMetadata`；
- `ScatterNdUpdateV2`。

它们可能不在当前路径中、执行位置更晚，或使用不同的缓存与恢复机制。Anchor 实验通过后仍需覆盖这些路径。

## DSV4 的两条显式计算流

DSV4 DSA overlap 主要使用：

1. 当前主流：普通执行时是 current/default stream，ACL Graph 捕获时是 capture stream；
2. DSV4 辅助流：由 `dsv4_dsa_overlap_stream()` 延迟创建，负责与 Q 路径并行的 KV、RoPE、scatter、Indexer 和 Compressor 工作。

两条流通过 event 和 `wait_stream()` 同步。`SparseAttnSharedkvMetadata` 运行在当前主流，而不是 DSV4 overlap stream。因此仅重建 `_DSV4_DSA_OVERLAP_STREAM` 不能直接修复 metadata 异常。

HCCL 还可能创建内部通信流，但它们不属于 DSV4 Python 代码显式管理的这两条计算流。

## Anchor 的推荐生命周期

专用 anchor 不应永久保留。推荐按 sleep 周期管理：

```text
Sleep:
  创建或恢复 anchor communicator
  → barrier，确保物理初始化
  → 销毁业务 HCCL communicator

Wake:
  恢复业务 communicator
  → 刷新 MC2/HCCL 元数据
  → 完成 ACL Graph 重捕获
  → 所有 Rank 在 anchor 上 barrier
  → 销毁 anchor communicator
```

对于 level-2 分阶段唤醒：

```text
tags=["weights"]  → 保留 anchor，不重捕图
tags=["kv_cache"] → 重捕图完成后释放 anchor
```

下一轮 sleep 应恢复已有 anchor coordinator 的 HCCL communicator，而不是重复注册同名 group。

### TP=1 场景

如果复用业务 group 作为 anchor，不能固定选择 TP group。TP=1 时通常没有真实的多卡 TP communicator，应按以下顺序选择真实 group：

```text
TP world size > 1 → TP
否则 EP world size > 1 → EP
否则 DP device group world size > 1 → DP
否则没有可用的多卡 anchor
```

专用 world-size anchor 则不依赖 TP 大小，但会带来额外通信组开销。

## 风险与代价

| 风险 | 说明 |
| --- | --- |
| 额外显存 | 1 MiB HCCL buffer 不代表总开销只有 1 MiB，还可能包含 stream、notify 和连接资源 |
| 端口资源 | Anchor 可能保持额外连接或监听状态 |
| 集体调用一致性 | 所有 Rank 必须以相同顺序创建、barrier 和销毁 anchor |
| 分阶段唤醒 | 不能在仅恢复 weights 后提前释放 |
| 异常路径 | 若 graph recapture 失败，需决定保留 anchor 便于诊断还是执行容错清理 |
| 实现依赖 | 依赖 HCCL/HCOMM 未公开的生命周期细节 |
| 误判根因 | Anchor 无效则应立即停止该方向，检查 DSA/AICPU/Graph 状态 |

## 验证矩阵

建议至少执行以下对照：

| 实验 | Sleep 行为 | 目的 |
| --- | --- | --- |
| A | 销毁全部 communicator | 复现基线失败 |
| B | 保留专用 anchor | 验证 runtime 连续性假设 |
| C | TP=1，保留 EP/DP 或专用 anchor | 验证方案不依赖 TP group |
| D | 销毁全部后完整重建，再执行 DSA eager | 区分通用 AICPU 与 Graph capture 问题 |
| E | 保留 anchor，连续 10 次 sleep/wake | 排除偶发异步竞态 |

每次实验应确认：

- wake 后推理输出正确；
- FULL ACL Graph 重捕获完成；
- Graph replay 正常；
- 各 Rank 无 HCCL watchdog 错误；
- sleep 前后显存变化符合预期；
- 进程退出时没有遗留端口或 communicator；
- 其他 static AICPU task 路径没有出现新的首个失败点。

## 如何解读结果

| 结果 | 结论 |
| --- | --- |
| 全清空失败，保留 anchor 稳定成功 | 支持“最后一个 communicator 最终清理”假设 |
| 保留 TP 与保留 EP 都成功 | 支持依赖任意真实多卡 communicator 的进程级生命周期 |
| 只有保留特定业务 group 成功 | 更可能是 group-specific 资源或缓存 |
| 保留 anchor 仍失败 | 根因不在 communicator 归零 |
| DSA eager 成功、Graph capture 失败 | 故障更接近 Graph capture/task cache 生命周期 |
| 出现另一个 static AICPU 算子失败 | 原算子只是第一个故障点 |

## 相关实现

- `vllm_ascend/device_allocator/sleep_mem_optimized.py`：extra cleanup、HCCL destroy/restore 与 anchor 生命周期；
- `vllm_ascend/patch/worker/patch_distributed.py`：`GroupCoordinator.destroy_hccl()` / `restore_hccl()`；
- `vllm_ascend/compilation/acl_graph.py`：ACL Graph entry、graph params 和重捕获；
- `vllm_ascend/attention/dsa_v1.py`：DSA metadata、两条显式计算流与 `SparseAttnSharedkvMetadata` 调用；
- `csrc/attention/sparse_attn_sharedkv_metadata/`：自定义 AICPU metadata 算子；
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`：MC2 communicator 元数据刷新。

## 总结

HCCL 全清空与保留 anchor 不改变模型图的逻辑内容。潜在区别是：重捕图时底层 HCCL/HCOMM runtime 是否经历了最后一个 communicator 所触发的最终清理。

业务 communicator 虽然会在 wake 时重建，但模型对象、static AICPU task、executor cache、Graph pool 和内部 runtime 资源不一定随之全部重建。DSV4 比其他模型多了一条在重捕图早期执行的自定义 static AICPU metadata 路径，因此更容易成为第一个暴露潜在生命周期不一致的模型。

只有严格的全清空/anchor A/B 测试能够确认该假设。Anchor 成功意味着应继续寻找未正确恢复的底层资源；anchor 失败则说明应停止 HCCL 生命周期方向，转向 DSA、AICPU task 或 Graph capture 状态排查。
