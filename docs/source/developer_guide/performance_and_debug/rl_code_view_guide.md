# vLLM Ascend RL 场景 — 纯代码视角导读

> 本文是“代码结构导读”，不按日志排障展开。  
> 若你要按日志定位问题，请看 `rl_log_location_guide.md`。

## 1. 范围与目标

本文聚焦 vllm-ascend 中与 RL（尤其在线 RLHF/GRPO/PPO/DPO 训练-推理协同）直接相关的实现：

1. **训练-推理一致性**：Batch Invariance 路径  
2. **训练资源让渡**：Sleep/Wake 内存生命周期  
3. **参数热更新**：HCCL 与 NPU IPC 两条权重传输链路  
4. **工程验证**：examples 与 e2e/ut 测试如何覆盖以上路径

---

## 2. 代码总览（模块地图）

| 模块 | 关键文件 | 代码职责 |
|---|---|---|
| Worker 主入口 | `vllm_ascend/worker/worker.py` | RL 场景主控制面：sleep/wake、weight update 生命周期、batch invariance 初始化触发 |
| Batch Invariance | `vllm_ascend/batch_invariant.py` | 确定性环境覆盖 + torch/torch_npu 算子接管 |
| 采样兼容 | `vllm_ascend/sample/sampler.py` | batch invariance 开启时采样实现回退到 vLLM 原生路径 |
| Sleep 分配器 | `vllm_ascend/device_allocator/camem.py` | 以 tag 管理分配，sleep 时 offload/release，wake 时 remap/restore |
| Sleep 扩展清理 | `vllm_ascend/device_allocator/sleep_mem_optimized.py` | ACL graph / HCCL 的额外清理与恢复 |
| 权重传输（HCCL） | `vllm_ascend/distributed/weight_transfer/hccl_engine.py` | 跨设备 broadcast 传参，支持 packed |
| 权重传输（NPU IPC） | `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py` | 同机同物理卡 IPC handle 传参 |
| packed 工具 | `vllm_ascend/distributed/weight_transfer/packed_tensor.py` | 大模型参数分块打包、双缓冲传输 |
| 配置约束 | `vllm_ascend/ascend_config.py` | RL 相关行为约束（如 batch invariance 与 async exponential） |
| 场景示例 | `examples/rl/rlhf_http_hccl.py`, `examples/rl/rlhf_http_npu_ipc.py` | trainer 侧参考实现 |

---

## 3. 主调用链（按生命周期）

```mermaid
flowchart TD
    A[NPUWorker 初始化] --> B[init_batch_invariance]
    B --> C[在线推理/rollout]
    C --> D[sleep(level)]
    D --> E[wake_up(tags)]
    E --> F[start_weight_update]
    F --> G[update_weights]
    G --> H[finish_weight_update]
    H --> I[resume 推理]
```

### 3.1 初始化阶段

- `NPUWorker._init_worker_distributed_environment()`  
  - 先调用 `init_batch_invariance()`，再初始化 distributed/model parallel。  
  - RL 视角：保证一致性配置尽早生效，避免后续路径混入非确定性行为。

### 3.2 训练让资源阶段（Sleep/Wake）

- `NPUWorker.sleep(level)`  
  - level=1：保留可恢复权重（offload），丢弃 KV。  
  - level=2：权重+KV 都可走更“彻底”的释放路径。  
  - 可选：`enable_sleep_mode_extra_cleanup` 触发 `SleepWakeupManager.sleep()`。

- `NPUWorker.wake_up(tags)`  
  - 调 `CaMemAllocator.wake_up()` 恢复分配。  
  - 对未量化 MoE 权重做布局恢复（`w13_weight/w2_weight` transpose 逻辑）。  
  - 若启用 extra cleanup，走 `SleepWakeupManager.wakeup(tags)` 恢复 HCCL 与 graph 捕获。

### 3.3 权重热更新阶段

- 生命周期 API（服务侧）  
  - `start_weight_update()`  
  - `update_weights(update_info)`  
  - `finish_weight_update()`

- 传输后端  
  - HCCL：`HCCLWeightTransferEngine`  
  - IPC：`NPUIPCWeightTransferEngine`

- 调用约束  
  - 必须先 `start` 再 `update` 再 `finish`（否则 worker 抛 RuntimeError）。  
  - `update_weights` 后会 `torch.npu.synchronize()`，保证下一步看到新权重。

---

## 4. 核心类与函数速查

## 4.1 `vllm_ascend/worker/worker.py`

| 符号 | 类型 | 作用 |
|---|---|---|
| `NPUWorker.sleep` | method | sleep 主流程，统计释放内存 |
| `NPUWorker.wake_up` | method | wake 主流程，含 RL/NZ 风险拦截与 MoE 权重布局恢复 |
| `NPUWorker.start_weight_update` | method | 开启一次更新会话，初始化 layerwise reload 状态 |
| `NPUWorker.update_weights` | method | 接收权重块并加载 |
| `NPUWorker.finish_weight_update` | method | 收尾 layerwise reload |
| `NPUWorker._init_worker_distributed_environment` | method | 初始化分布式前触发 batch invariance |

## 4.2 `vllm_ascend/batch_invariant.py`

| 符号 | 类型 | 作用 |
|---|---|---|
| `override_envs_for_invariance` | function | 覆盖 `weight_nz_mode`、deterministic env 等 |
| `enable_batch_invariant_mode` | function | 注册/替换关键算子实现 |
| `init_batch_invariance` | function | 总入口，按可用后端决定开启或告警 |
| `add_rms_norm` / `reduce_sum` | function | 为一致性重写的算子路径 |

## 4.3 `vllm_ascend/device_allocator/*`

| 符号 | 文件 | 作用 |
|---|---|---|
| `CaMemAllocator.sleep` | `camem.py` | 按 tag offload/release |
| `CaMemAllocator.wake_up` | `camem.py` | remap 并回拷 offload 数据 |
| `SleepWakeupManager` | `sleep_mem_optimized.py` | 统一管理 HCCL/ACL graph 额外清理 |
| `AclGraphSleepWakeupManager` | `sleep_mem_optimized.py` | graph workspace 清理与 recapture |
| `HcclSleepWakeupManager` | `sleep_mem_optimized.py` | process group destroy/restore 与 MoE group refresh |

## 4.4 `vllm_ascend/distributed/weight_transfer/*`

| 符号 | 文件 | 作用 |
|---|---|---|
| `HCCLWeightTransferEngine.receive_weights` | `hccl_engine.py` | HCCL 广播消费并增量 load |
| `HCCLWeightTransferEngine.trainer_send_weights` | `hccl_engine.py` | trainer 侧广播 |
| `NPUIPCWeightTransferEngine.receive_weights` | `npu_ipc_engine.py` | IPC handle 重建 tensor 并 load |
| `NPUIPCWeightTransferEngine.trainer_send_weights` | `npu_ipc_engine.py` | trainer 侧构造/发送 IPC handles |
| `packed_broadcast_*` / `packed_npu_ipc_*` | `packed_tensor.py` | packed 高效传输 |

---

## 5. 关键设计约束（代码层）

1. **RL 精度保护：NZ 模式门禁**  
   - `wake_up()` 与权重更新前均有 RL 场景下 `weight_nz_mode` / `VLLM_ASCEND_ENABLE_NZ` 校验。  

2. **一致性优先：关闭潜在非确定性路径**  
   - Batch invariance 下会覆盖 deterministic 环境并禁用/回退部分路径（如 sampler 分支）。  

3. **生命周期强约束**  
   - 权重更新必须遵循 start → update → finish 状态机。  

4. **NPU IPC 拓扑约束**  
   - trainer 与 worker 必须在同物理 NPU（UUID 匹配）。  
   - HTTP 传 IPC handle 需显式允许反序列化开关。  

5. **Sleep extra cleanup 的权衡**  
   - 释放更多显存，但 wake 时需要恢复 HCCL 并可能重捕获 graph。

---

## 6. 示例脚本与测试映射（从代码读行为）

| 场景 | 示例/测试 | 代码点 |
|---|---|---|
| HCCL 热更新 | `examples/rl/rlhf_http_hccl.py`, `tests/e2e/pull_request/two_card/test_hccl_weight_transfer.py` | control plane + HCCL data plane |
| NPU IPC 热更新 | `examples/rl/rlhf_http_npu_ipc.py`, `tests/e2e/pull_request/one_card/test_npu_ipc_weight_transfer.py` | IPC handle over HTTP |
| Batch 一致性 | `tests/e2e/pull_request/one_card/test_batch_invariant.py`, `tests/ut/test_batch_invariant.py` | 确定性与反例验证 |
| Sleep/Wake | `tests/e2e/pull_request/one_card/test_camem.py`, `tests/ut/device_allocator/test_sleep_mem_optimized.py`, `tests/ut/worker/a2/test_worker_v1.py` | 内存回收与恢复 |

---

## 7. 建议阅读顺序（纯代码）

1. `worker.py`（看生命周期入口）  
2. `batch_invariant.py` + `sample/sampler.py`（看一致性实现）  
3. `camem.py` + `sleep_mem_optimized.py`（看 sleep/wake 细节）  
4. `hccl_engine.py` / `npu_ipc_engine.py` / `packed_tensor.py`（看热更新数据面）  
5. `examples/rl/*` 与 e2e tests（看真实调用方式）

