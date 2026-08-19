# 同卡训推的 Sleep Mode：让出显存，原地换权

> **说明**  
> 本文讲 RL / 同卡训推下的 **Sleep Mode 流程与原理**。Vime 复用上游 vLLM
> 的 sleep / wake，Level 2 灌权重时走 `initialize → reload → finalize`；
> Ascend 承接 NPU 内存与权重传输。基础 API 见
> [Sleep Mode Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sleep_mode.html)。
>
> 上游参考：
>
> - [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
> - [Layerwise (Re)loading](https://docs.vllm.ai/en/latest/training/layerwise/)
> - Ascend 示例：`examples/rl/rlhf_http_hccl.py`、`examples/rl/rlhf_http_npu_ipc.py`

## 1. 原理

强化学习（PPO / GRPO / RLHF 等）包含两个阶段：

1. **Rollout**：策略模型在 vLLM Ascend 上做自回归生成；
2. **Train**：Trainer（如 Vime）基于生成样本更新策略参数。

**同卡训推**指两个阶段共享同一组 NPU。Rollout 期间引擎占用模型权重与 KV cache；进入 Train 前需释放这部分显存，否则训练侧无法完成 forward / backward。Train 结束后策略参数已更新，须写回推理引擎，供下一轮 Rollout 使用。

若每次通过销毁并重建 vLLM 进程完成切换，通信组与图捕获等状态均需重建，开销较大。Sleep Mode 在**不退出进程**的前提下完成显存让渡与权重回写：

> sleep 释放推理显存 → Train → wake 并灌入新权重 → 继续 Rollout。

| 组件 | 职责 |
| --- | --- |
| **Vime** | 编排 sleep / wake 与权重同步时机 |
| **vLLM（上游）** | Sleep 语义、内存 tag、`initialize → reload → finalize` |
| **vLLM Ascend** | `CaMemAllocator`、HCCL/IPC、可选 HCCL/ACLGraph 清理、MoE 布局处理 |

启用 `enable_sleep_mode` 后，权重与 KV 分配进入 sleep 内存池；显存让渡与恢复通过对池内 handle 的 **unmap / remap** 完成，由 Vime 调用上游控制面接口。

### 1.1 Level 1 与 Level 2

引擎提供两级 sleep，差别主要在**权重怎么处理**：

| | Level 1 | Level 2 |
| --- | --- | --- |
| **权重** | 拷到 Host 再 unmap（内容可还原） | 直接丢弃内容 |
| **KV cache** | 丢弃 | 丢弃 |
| **醒来后权重** | CPU backup 自动还原 | 必须 reload / update |
| **Host 压力** | 需放下整模权重 | 几乎不备份大权重 |
| **适用** | 同权重短暂让卡 | **RL 换权重**、Host 紧张 |
| **唤醒后是否走 layerwise** | 一般不需要 | 需要 `initialize → reload → finalize` |

Ascend：L1 为 `sleep(offload_tags=("weights",))`；L2 为 `sleep(offload_tags=())`。
Level 1 权重不变时，`wake_up()` 后即可继续推理。

同卡 RL 更新策略时优先 **Level 2**：训练侧本身也吃 Host/Device，L1 再备份一份
整模权重容易把 Host 打满；且训完本来就要灌**新**权重，旧内容没必要 offload。

因此下文流程图与 layerwise 展开都只画 **Level 2**——这是 Vime 换权路径；
Level 1 调用见文末示例。

### 1.2 `enable_sleep_mode_extra_cleanup`

默认 sleep 仅释放 sleep 内存池管理的分配。同卡 RL 若需进一步把显存归还训练侧，
可通过 `additional_config` 打开 `enable_sleep_mode_extra_cleanup`：

- **sleep**：清理 ACLGraph attention workspace 并失效已捕获图；等待 PP 发送完成后
  同步 NPU，并销毁 HCCL 进程组；
- **wake**：重建 HCCL、刷新 MoE dispatcher 元数据；在权重与状态就绪后按需
  `capture_model()` 重新构图。

这是显存占用与唤醒时延的权衡：同卡显存紧张时可开启；更看重唤醒延迟时保持默认关闭。

## 2. 整体流程（Level 2）

RL 同卡换权的完整路径：

```mermaid
sequenceDiagram
    autonumber
    participant Vm as Vime<br/>RolloutManager / Trainer
    participant E as vLLM Ascend Engine
    participant U as 上游 vLLM<br/>layerwise reload
    participant N as NPU Memory

    Vm->>E: rollout generate
    Vm->>E: sleep(level=2)
    E->>N: discard weights + kv_cache
    Note over E,N: named_buffers CPU 备份<br/>可选 extra_cleanup
    N-->>Vm: 显存归还训练

    Vm->>Vm: train / 得到新策略权重

    Vm->>E: wake_up(tags=["weights"])
    E->>N: remap 权重内存

    Vm->>E: start_weight_update
    E->>U: initialize_layerwise_reload
    Note over U: 保存 kernel tensor 快照<br/>参数放到 meta，包装 loader

    Vm->>E: update_weights / reload_weights
    E->>U: 按层装载 + process_weights
    Note over U: 含量化 / MoE transpose 等布局

    Vm->>E: finish_weight_update
    E->>U: finalize_layerwise_reload
    Note over U: 布局结果 copy 回原 Parameter<br/>地址不变

    Vm->>E: wake_up(tags=["kv_cache"])
    E->>N: 分配 KV cache
    Note over E: 可选 recapture ACLGraph
    Vm->>E: 下一轮 rollout
```

一条因果链：

```text
sleep(level=2) 让出显存 → train → wake / 灌权重
  （initialize → reload → finalize）→ 继续 rollout
```

## 3. 原理展开

### 3.1 内存池与 tag

`CaMemAllocator`（Ascend；上游 GPU 为 CuMem）给分配打 tag。sleep / wake 并不删
Python 侧模型对象，而是：

- **unmap**：物理页归还（内容丢弃，或先拷到 CPU）；
- **remap**：按原 handle 再 map，**虚地址尽量保持稳定**。

因此 `nn.Parameter` 对象仍在，图捕获记住的 `data_ptr` 仍可能对上——这是后面
layerwise「更新数值、不换锚点」的前提。RoPE 等 `named_buffers` 在 sleep 前
CPU clone、wake 后写回；个别必须跨 sleep 存活的分配可用 `sleep_persistent`。

### 3.2 Level 2 灌权重：`initialize → reload → finalize`

Level 2 丢掉的是**权重内容**，不是 Parameter 对象。若醒来后再普通 `load_model`，
`process_weights_after_loading` 常会**换掉** Parameter，ACLGraph 仍钉着旧
`data_ptr`，结果错或踩非法地址。

上游因此提供 layerwise reload——**在不换锚点的前提下**更新数值与布局。
Ascend / Vime 复用同一套生命周期：

```text
start_weight_update  →  initialize_layerwise_reload
update_weights       →  reload / load_weights（含 process_weights_after_loading）
finish_weight_update →  finalize_layerwise_reload
```

也可用 `collective_rpc("reload_weights", ...)` 走检查点，语义仍落在这条链上。

**initialize：保存快照，进入可加载态**

1. **把当前每层 Parameter / buffer 记入 `kernel_tensors`**  
   ACLGraph / CUDA Graph 捕获时绑定的是这些 Parameter 的 storage 地址
   （`data_ptr`）。先把对象引用存成快照，作为后续回写的**锚点**；后面无论
   中间加载是否换过临时 tensor，最终都要把结果写回这些原对象。

2. **按首次 load 的 metadata 恢复到 meta 占位**  
   首次建模时 `record_metadata_for_reloading` 已记下每层参数的名称、shape、
   dtype 等元信息。此处按该元信息把层参数切到 **meta device** 上的占位形态，
   相当于卸下当前 NPU 上的旧内容视图，使层重新处于「可被 weight_loader 装载」
   的状态，同时避免与即将到来的新权重在设备内存上冲突。

3. **包装 `weight_loader`：先缓冲，层内齐套后再 materialize + process**  
   一层往往有多个参数（如 `qkv` / `o_proj`，或 MoE 的 `w13` / `w2`）。包装后的
   loader 不立刻做完整后处理，而是把本次载入的权重先缓存；等该层约定数量的
   权重到齐后，再一次性：materialize 到目标 device → 写入 → 调用
   `process_weights_after_loading`（量化、MoE transpose 等）。这样保证布局处理
   看到的是完整一层，而不是半成品。

目标：钉住原 Parameter 作锚点，同时把逻辑层切到可安全接收新权重的加载态。

**reload：装入新权重并做布局处理**

- 数据来源：Trainer 经 HCCL / NPU IPC，或磁盘 checkpoint；
- 层内：materialize → 原始 loader 写入 → `process_weights_after_loading`。

Ascend 上这里会做量化打包、**MoE `w13`/`w2` 的 `transpose(1, 2)`** 等，
把 checkpoint 布局变成 `npu_grouped_matmul` 需要的运行时布局。
转置属于加载后处理，**不应再塞进 `wake_up`**。

**finalize：布局落稳，地址不变**

1. **补处理未 online 完的层（deferred attention、padding 等）**  
   reload 时「权重到齐 → 立刻 process」并不覆盖全部层，典型残留由 finalize 收尾：

   - **Deferred attention**：如带 KV scale 的 Attention 层。online 路径会显式跳过
     （`is_deferred_attention_layer`），因 `k_scale` / `v_scale` 等需在其它层就绪后，
     再于 finalize 里统一 `_finalize_attention_layer` / `_reload_attention_scales`。
   - **Padding 导致载入量不足**：层参数按对齐创建了略大的 storage（含 padding），
     但 checkpoint 只写入有效元素，于是一直 `load_numel < load_numel_total`，
     online 条件达不到「到齐」。finalize 识别这类 Delayed 层并补做 process。

   若不做这步收尾，这些层会停留在未处理或半处理状态。

2. **`param.data.copy_(processed)` 写回 initialize 保存的原 Parameter / buffer**  
   reload / process 得到的是已完成 **HF / checkpoint 布局 → runtime 布局** 的结果（如 MoE `transpose`、量化重排）。这些结果可能暂存在临时 Parameter 上。此处把数值 **copy 进 initialize 快照里的原 storage**，使锚点对象持有最终 runtime 布局内容，而不是换掉对象本身。

3. **原对象重新挂回 module → `data_ptr` 不变**  
   将快照中的原 Parameter / buffer 重新 `register` 回对应 module 属性。模块对外仍暴露同一批对象，**`data_ptr` 与图捕获时一致**；ACLGraph / 后续 forward 继续引用原地址，读到的已是 runtime 布局后的新权重。

目标：完成 HF → runtime 的布局落地，同时保证图绑定的 Parameter 身份与地址不变。

```text
锚点 Parameter(A)  ←── finalize: A.data.copy_(processed runtime 布局)
临时权重 (B) ────────┘
推理图 / ACLGraph 始终引用 A（地址不变，内容已是 runtime 布局）
```

finalize 保证的是**对象与地址稳定**：数值来自 Trainer，运行时布局来自
`process_weights_after_loading`，地址来自 initialize 保存的锚点。三段缺一，
就又容易退回「在 wake 里补转置」之类的旁路。

## 4. 具体调用方案

### 4.1 Vime 编排（推荐）

```python
engine.sleep(level=2)
trainer.step()

engine.wake_up(tags=["weights"])

engine.start_weight_update()            # initialize
engine.update_weights(update_info)      # reload（含布局 / 转置）
engine.finish_weight_update()           # finalize

engine.wake_up(tags=["kv_cache"])
```

同卡建议：

- 权重同步（`start/update/finish_weight_update` 或 `reload_weights`）**开始前**
  `pause_generation`，**结束后**再 `resume_generation`，避免在飞请求读到半更新权重；
- Level 2 后按需 `reset_prefix_cache`，避免沿用旧权重下的 prefix；
- 关闭 FRACTAL_NZ（`VLLM_ASCEND_ENABLE_NZ=0`，`weight_nz_mode=0`）。

### 4.2 Online HTTP（dev mode）

```bash
export VLLM_SERVER_DEV_MODE=1
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --enable-sleep-mode \
  --additional-config '{"enable_sleep_mode_extra_cleanup": true}' \
  --weight-transfer-config '{"backend": "hccl"}'

curl -X POST 'http://127.0.0.1:8000/sleep?level=2'
curl -X POST 'http://127.0.0.1:8000/wake_up?tags=weights'

# 路径 A：检查点
curl -X POST 'http://127.0.0.1:8000/collective_rpc' \
  -H 'Content-Type: application/json' \
  -d '{"method":"reload_weights","kwargs":{"weights_path":"Qwen/Qwen2.5-0.5B-Instruct"}}'

# 路径 B：在线同步（见 examples/rl/rlhf_http_hccl.py）
# POST /start_weight_update → /update_weights → /finish_weight_update

curl -X POST 'http://127.0.0.1:8000/wake_up?tags=kv_cache'
curl -X GET  'http://127.0.0.1:8000/is_sleeping'
```

### 4.3 离线 Python API

```python
from vllm import LLM

llm = LLM(
    "Qwen/Qwen2.5-0.5B-Instruct",
    enable_sleep_mode=True,
    additional_config={"enable_sleep_mode_extra_cleanup": True},
)

llm.sleep(level=2)
llm.wake_up(tags=["weights"])
llm.collective_rpc("reload_weights", kwargs={"weights_path": "Qwen/Qwen2.5-0.5B-Instruct"})
llm.wake_up(tags=["kv_cache"])
```

### 4.4 Level 1：同权重快速让卡

```python
llm.sleep(level=1)   # weights → CPU，kv 丢弃
# ... 其它任务占用 NPU ...
llm.wake_up()        # 无需 reload / layerwise
```

## 5. 实践补充

1. **Sleep ≠ Weight Transfer ≠ Pause**  
   显存让渡、权重字节、在飞请求窗口是三件事，组合用。

2. **布局处理走 reload/process，不走 wake**  
   MoE transpose 等应在 `process_weights_after_loading` + finalize 完成；
   `wake_up` 只负责 remap / buffer / 通信 / 图。

3. **extra cleanup**  
   见 [1.2](#12-enable_sleep_mode_extra_cleanup)：更低 sleep 显存 ↔ 更长 wakeup。

4. **管理接口仅限 `VLLM_SERVER_DEV_MODE=1`**，内网使用。

5. **与 DP Router 的边界**  
   Sleep / 权重同步直连 Engine；请求落 DP 仍走 Router。

## 6. 相关链接

- [Sleep Mode Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sleep_mode.html)
- 上游 [Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
- 上游 [Layerwise (Re)loading](https://docs.vllm.ai/en/latest/training/layerwise/)
- 代码锚点：
  - `vllm_ascend/worker/worker.py` — `sleep` / `wake_up` / weight update
  - `vllm_ascend/device_allocator/camem.py` — tag offload / discard / remap
  - `vllm_ascend/distributed/weight_transfer/hccl_engine.py` — initialize / finalize
  - `vllm_ascend/ops/fused_moe/routed_experts.py` — MoE 布局在 process 路径
  - `vllm_ascend/device_allocator/sleep_mem_optimized.py` — extra cleanup
- E2E：`tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py`
