# vLLM Ascend RL 场景 — 日志定位指南

库内 RL 相关**可 grep 日志 / 异常原文**索引。仅收录 `vllm_ascend` 中已核对条目；**不含** `examples/`、`tests/` 的 `print`。

相关功能说明见：[Batch Invariance](../../user_guide/feature_guide/batch_invariance.md)、[Sleep Mode](../../user_guide/feature_guide/sleep_mode.md)。

## 1. Batch Invariance

入口：`VLLM_BATCH_INVARIANT=1` → `init_batch_invariance()`（`batch_invariant.py`）。

| 级别 | 原文 | 条件 |
| --- | --- | --- |
| INFO | `Enabling batch-invariant mode for vLLM on Ascend NPU.` | Triton 或 AscendC batch-invariant ops 可用 |
| WARNING | `Batch-invariant mode requested but Triton or AscendC batch-invariant ops is not available.skipping batch-invariant initialization.` | 均不可用（`available.` 与 `skipping` 间无空格）；此时无 INFO 成功日志 |
| DEBUG | `Batch-invariant env override: weight_nz_mode=0, HCCL_DETERMINISTIC=strict, LCCL_DETERMINISTIC=1, use_deterministic_algorithms=True` | 成功路径内 |
| DEBUG | `Batch-invariant op registration: Triton=%s, AscendC=%s` | 成功路径内 |
| DEBUG once | `[sample/sampler] BATCH_INVARIANT mode enabled, falling back to vLLM native top-k/top-p implementation.` | 首次采样走 `AscendSampler.forward_native` |

## 2. Sleep / Wake

入口：`enable_sleep_mode`；extra cleanup 需 `enable_sleep_mode_extra_cleanup=True`（默认关闭）。

**sleep 顺序（extra cleanup 开启）**：`Destroyed HCCL`（仅 `num_destroyed > 0`）→ `released HCCL and attention workspace` → `CaMem sleep` → `Sleep mode (level=...) freed`。

**wake 顺序**：`weight_nz_mode` 门禁 → `CaMem wake_up` →（extra cleanup）`Restored HCCL`。

| 级别 | 原文 | 模块 | 条件 |
| --- | --- | --- | --- |
| INFO | `Destroyed %d HCCL process groups for sleep mode.` | `sleep_mem_optimized.py` | extra cleanup 且 `num_destroyed > 0` |
| INFO | `Sleep mode released HCCL and attention workspace memory: %.3f GiB.` | `sleep_mem_optimized.py` | extra cleanup |
| INFO | `CaMem sleep: offloading %s/%s allocations (tags=%s)` | `camem.py` | sleep |
| INFO | `Sleep mode (level=%s) freed %.2f GiB memory, %.2f GiB memory is still in use.` | `worker.py` | sleep 结束 |
| INFO | `CaMem wake_up: restoring %s/%s allocations (tags=%s)` | `camem.py` | wake |
| INFO | `Restored %d HCCL process groups after sleep mode.` | `sleep_mem_optimized.py` | extra cleanup（`num_restored` 为 0 也会打） |
| ValueError | `FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set weight_nz_mode=0 via --additional-config.` | `worker.py` | `wake_up` 且 `weight_nz_mode != 0` |

## 3. Weight Transfer

控制面：`worker.py`（`start_weight_update` / `update_weights` / `finish_weight_update`）。  
数据面：`hccl_engine.py` / `npu_ipc_engine.py`。

**库内无权重更新成功路径 logger。** 以下均为异常原文。

| 原文 | 触发 |
| --- | --- |
| `Weight transfer not configured. Please set weight_transfer_config to enable weight transfer.` | 未配置 transfer engine |
| `FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set VLLM_ASCEND_ENABLE_NZ=0.` | `start_weight_update` 且 `VLLM_ASCEND_ENABLE_NZ != 0` |
| `start_weight_update called while a weight update is already active. Call finish_weight_update first.` | 重复 start |
| `start_weight_update must be called before update_weights.` | 未 start 就 update |
| `start_weight_update must be called before finish_weight_update.` | 未 start 就 finish |
| `HCCL weight transfer not initialized. Call init_transfer_engine() first.` | HCCL 未 `init_transfer_engine` |
| `` Refusing to deserialize `ipc_handles_pickled` without VLLM_ALLOW_INSECURE_SERIALIZATION=1 `` | IPC HTTP 未开 insecure serialization |
| `IPC handle not found for NPU UUID ...` | trainer/worker 物理 NPU UUID 不匹配 |

配置别名（Ascend factory patch）：`backend: "nccl"` → HCCL；`backend: "ipc"` → NPU IPC。
