# Sleep Mode 代码审查报告

## 结论

**审查决定：🔄 Request Changes**

当前 Sleep Mode 的主流程和 Level 1 单卡场景已有基础测试，但内存分配器、Sleep/Wake
状态转换及用户文档仍存在阻断问题。在修复原生回调生命周期、异常安全、拷贝结果校验和
Level 2 使用契约前，不建议将该功能作为可安全重复调用的生产能力。

本报告是静态审查结果，未在 Ascend NPU 上执行测试。涉及异步执行和 CANN 内存池复用的
风险已明确标记为需要 NPU 验证。

## 审查范围与方法

- 基线提交：`ad04efb9`
- 用户指南：`docs/source/user_guide/feature_guide/sleep_mode.md`
- Python 实现：
    - `vllm_ascend/worker/worker.py`
    - `vllm_ascend/device_allocator/camem.py`
    - `vllm_ascend/device_allocator/sleep_mem_optimized.py`
    - `vllm_ascend/patch/worker/patch_distributed.py`
- 原生实现：`csrc/camem_allocator.cpp`
- 示例和测试：`examples/`、`tests/ut/`、`tests/e2e/`

审查采用
[`awesome-skills/code-review-skill`](https://github.com/awesome-skills/code-review-skill)
的流程：先检查架构和功能契约，再逐行检查正确性、异常与资源生命周期、并发、性能、
安全、测试和文档一致性。严重度定义如下：

- 🔴 `[blocking]`：合入或发布前必须处理；
- 🟡 `[important]`：应处理，若暂缓需明确限制和跟踪项；
- 🟢 `[suggestion]`：非阻断改进。

## 架构与执行链路

```text
LLM.sleep(level)
  -> NPUWorker.sleep()
     -> [可选] 清理 ACL Graph / 销毁 HCCL
     -> CaMemAllocator.sleep()
        -> Level 1: 权重复制到 CPU，释放权重和 KV 的 NPU 映射
        -> Level 2: 释放权重和 KV 的 NPU 映射

LLM.wake_up(tags)
  -> NPUWorker.wake_up()
     -> CaMemAllocator.wake_up(tags)
     -> MoE 权重布局处理 / Buffer 恢复
     -> [可选] 恢复 HCCL / 重新捕获 ACL Graph
```

该分层总体合理：Worker 编排生命周期，`CaMemAllocator` 管理稳定虚拟地址，
`SleepWakeupManager` 管理分配器之外的 ACL/HCCL 资源。主要问题是各层没有共享且可验证的
状态机，也没有跨阶段失败后的回滚协议。

## Findings

### 🔴 1. C++ 保存临时 Python bound method 的借用引用，释放回调可能访问已析构对象

**证据**

- `csrc/camem_allocator.cpp:29-33,268-272` 将回调保存为全局 borrowed reference，
  没有 `Py_INCREF`。
- `vllm_ascend/device_allocator/camem.py:97,277` 传入
  `self.python_malloc_callback` 和 `self.python_free_callback`；每次属性访问会创建临时
  bound-method 对象。
- `allocator_and_pools` 仅保存 pool 和 allocator，没有保存这两个 bound-method 对象。

**触发条件**

`init_module()` 返回、临时 bound method 被回收后，任意 allocator 的 malloc/free 回调。

**影响**

确定性的对象生命周期错误，可能导致崩溃、调用错误对象或进程内存破坏。

**建议**

在持有 GIL 时对新回调执行 `Py_XINCREF`，替换或模块销毁时对旧回调执行
`Py_XDECREF`。Python 侧可额外保存稳定回调引用，但不能替代 C 扩展的所有权管理。

### 🔴 2. 原生回调存在 GIL 泄漏、Python 引用泄漏和跨 C ABI 抛出异常

**证据**

- `csrc/camem_allocator.cpp:206-229` 的两个错误分支在
  `PyGILState_Ensure()` 后直接返回，没有释放 GIL。
- `my_malloc()` 的 `py_result` 以及 `my_free()` 的 `py_ptr`、`py_result` 未
  `Py_DECREF`。
- `csrc/camem_allocator.cpp:62-97,127-248,296,320` 可抛出
  `std::runtime_error`，但 Python 入口和 allocator 的 `extern "C"` 回调均未捕获。

**影响**

错误路径可能永久占用 GIL；正常路径会持续泄漏 Python 对象；ACL 错误可能让 C++ 异常
穿过 CPython 或外部分配器 ABI，最终触发 `std::terminate`。

**建议**

使用 RAII 管理 GIL 和 `PyObject*`。所有 Python/C ABI 边界必须 `catch (...)`：
Python 入口转换为 `PyErr_*` 并返回 `nullptr`，allocator 回调按其失败协议返回且完成资源
回滚。

### 🔴 3. D2H/H2D 拷贝的目标容量参数错误，且失败后仍销毁唯一数据副本

**证据**

- `vllm_ascend/device_allocator/camem.py:210-217` 将 `destMax` 传为
  `cpu_ptr + size * 2`。
- `vllm_ascend/device_allocator/camem.py:244-249` 将 `destMax` 传为
  `ptr + size * 2`。
- 两处均未检查 `acl.rt.memcpy` 返回码。
- D2H 后无条件解除映射；H2D 后无条件清空 CPU backup。

**影响**

拷贝失败时，Sleep 可能释放权重唯一副本，Wake 可能删除唯一 CPU backup，造成不可恢复的
静默权重损坏。

**建议**

`destMax` 应为目标缓冲区容量 `size_in_bytes`。检查每次 ACL 调用结果；仅在 D2H 成功后
unmap，仅在 H2D 成功后清除 backup，并为失败保留可重试状态。

### 🔴 4. Sleep/Wake 没有状态机，重复调用、分阶段调用和失败重试会破坏映射

**证据**

- `AllocationData` 只记录 handle、tag 和 backup，没有 mapped/offloaded/discarded 状态：
  `vllm_ascend/device_allocator/camem.py:78-82`。
- `sleep()` 对所有匹配 allocation 无条件 unmap：
  `vllm_ascend/device_allocator/camem.py:204-217`。
- `wake_up()` 无条件重新 create/map：
  `vllm_ascend/device_allocator/camem.py:234-249`。
- `NPUWorker.sleep()/wake_up()` 没有锁和
  `AWAKE/SLEEPING/ASLEEP/WAKING` 状态：`vllm_ascend/worker/worker.py:219-283`。

**触发条件**

重复或并发 sleep/wake、`wake_up(["weights"])` 后再完整 wake，或任一 allocation /
HCCL / ACL Graph 阶段中途失败后重试。

**影响**

可能出现 double-unmap、同一 VA double-map、physical handle 被覆盖泄漏、各 rank 状态
分叉和 collective 卡死。

**建议**

增加 Worker 级互斥锁与显式状态机；每个 allocation 记录映射状态；分阶段恢复只处理
`UNMAPPED` 项。明确重复调用是幂等还是报错，并为跨 rank 失败提供一致的提交/回滚策略。

### 🔴 5. Level 2 可直接完整 Wake 到未初始化权重，非法 level 也会进入丢弃路径

**证据**

- `vllm_ascend/worker/worker.py:221-231` 仅对 `level == 2` 保存 buffer，但所有
  `level != 1` 都丢弃权重和 KV。
- `vllm_ascend/device_allocator/camem.py:238-249` 对被丢弃的权重只重新映射空 physical
  memory，不会重新加载参数。
- 正确的外部加载流程仅在
  `examples/offline_external_launcher.py:221-229` 展示。

**影响**

`sleep(level=2); wake_up()` 或传入 `level=0/3` 后可能用未初始化权重继续推理，产生静默
错误结果。

**建议**

严格限制 level 为 `{1, 2}`；记录当前 level。Level 2 后，在完成
`weights wake -> load weights -> kv_cache wake` 前禁止完整 wake 和推理。

### 🔴 6. 未量化 MoE 的 Level 1 Wake 会再次转置已处于运行时布局的权重

**证据**

- 初次加载已在 `vllm_ascend/ops/fused_moe/fused_moe.py:112-126` 对
  `w13_weight`、`w2_weight` 执行 `transpose(1, 2)`。
- Level 1 仅备份并恢复 allocation 原始字节，不改变现有 Parameter 的 shape/stride。
- `vllm_ascend/worker/worker.py:254-273` 在每次恢复 `"weights"` 时再次 transpose，
  没有区分 Level 1 与 Level 2 外部重载。

**影响**

Level 1 wake 后权重 shape/stride 被翻回另一布局，可能导致 grouped matmul shape 错误或
静默精度错误。

**建议**

Level 1 不应再次执行布局转换。Level 2 外部加载应调用模型/量化方法自身的
post-load finalize，而不是按参数名猜测布局。增加未量化 MoE 的 sleep/wake 输出一致性
NPU 回归测试。

### 🔴 7. 用户指南中的 Online Level 参数不会按示例生效，Level 2 流程也不安全

**证据**

- `docs/source/user_guide/feature_guide/sleep_mode.md:154-164` 将 `level` 放在 JSON
  body。
- 上游 vLLM Sleep API 将 `level` 定义为 query 参数，body 被忽略；缺省值为 1。
- 文档在 Level 1 尚未 wake 时继续执行 Level 2，并在 Level 2 后直接完整
  `/wake_up`：`sleep_mode.md:154-174`。
- 仓库可运行示例要求 Level 2 分阶段恢复并重新加载权重：
  `examples/offline_external_launcher.py:221-229`。

**影响**

文档中的 Level 2 请求实际执行 Level 1；即使改成 query 参数，后续完整 wake 也会使用
未初始化权重。

**建议**

使用 `/sleep?level=1` 和 `/sleep?level=2`。将两种 level 拆成互斥示例；Level 2 只提供
包含真实权重加载步骤的可执行流程。

### 🟡 8. 默认 Sleep 路径在解除 NPU 映射前没有无条件同步

**证据**

- `my_malloc/my_free` 收到的 stream 参数未保存或使用：
  `csrc/camem_allocator.cpp:127,199`。
- 默认 `NPUWorker.sleep()` 在 allocator unmap 前不调用 `torch.npu.synchronize()`：
  `vllm_ascend/worker/worker.py:219-231`。
- 同步仅存在于 extra cleanup 且 distributed 已初始化的分支：
  `vllm_ascend/device_allocator/sleep_mem_optimized.py:172-178`。

**风险**

若其他 stream 仍在读取权重/KV，解除映射可能造成 device fault 或数据破坏。该项需在
真实 NPU 多 stream/ACL Graph 场景验证。

**建议**

将 device quiescence 作为 allocator sleep 的硬前置条件，至少在 unmap 前无条件同步；
长期方案应跟踪相关 stream/event，避免不必要的全设备同步。

### 🟡 9. 分配 tag 是底层 segment 级，而 `sleep_persistent` 被当作 tensor 级使用

**证据**

- tag 只在 pluggable allocator 申请新底层地址时记录：
  `vllm_ascend/device_allocator/camem.py:163-168`。
- DSA Hadamard tensor 通过临时修改全局 `current_tag` 标记为 persistent：
  `vllm_ascend/attention/dsa_v1.py:441-450` 和
  `vllm_ascend/attention/context_parallel/dsa_cp.py:198-207`。
- PyTorch MemPool 可以复用已有 segment，因此上下文内创建 tensor 不保证触发新的
  malloc callback。

**风险**

Hadamard 可能复用 `kv_cache` segment 而未被持久化；反向情况会让整个含 KV 的 segment
保持映射，降低内存回收效果。该项需通过真实 allocator segment 复用测试确认。

**建议**

为 persistent 数据使用独立 MemPool/allocator domain，避免用全局 tag 模拟 tensor 级
所有权。

### 🟡 10. `use_memory_pool()` 异常退出后不会恢复全局 tag

**证据**

`vllm_ascend/device_allocator/camem.py:275-295` 在 `yield` 后恢复 `current_tag`，但没有
`try/finally`。模型或 KV 初始化抛错后，后续 allocation 会继承错误 tag。

**建议**

使用 `try/finally` 恢复 tag；将 tag 改为 `contextvars.ContextVar` 或 thread-local，并对
pool 创建和生命周期操作串行化。

### 🟡 11. 平台无条件声明 Sleep Mode 可用，与扩展加载结果矛盾

**证据**

- 扩展导入失败时函数被设为 `None`：
  `vllm_ascend/device_allocator/camem.py:56-71`。
- `vllm_ascend/platform.py:148-154` 和
  `vllm_ascend/patch/platform/patch_camem_allocator.py:20-28` 仍无条件返回可用。
- 随后 `camem.py:97` 会直接调用可能为 `None` 的 `init_module`。

**影响**

配置验证通过后才以不可操作的 `NoneType is not callable` 失败，且与“Sleep mode will be
disabled”日志不一致。

**建议**

在 Worker 初始化时执行可重试 capability probe 并 fail fast，给出缺失
`vllm_ascend_C` 的明确安装说明。

### 🟡 12. Extra cleanup 在设备同步前清理 ACL Graph 引用

**证据**

- 调用顺序是 ACL cleanup 后 HCCL cleanup：
  `vllm_ascend/device_allocator/sleep_mem_optimized.py:47-52`。
- Graph/workspace 在 `:116-119` 被清理，而同步直到 `:172-178` 才发生，且受 distributed
  初始化条件限制。

**风险**

异步 graph/kernel 尚未结束时释放 Python 引用，可能形成资源生命周期竞争。

**建议**

先等待 PP work 并同步设备，再清理 ACL Graph，最后销毁 HCCL。增加有在途任务时调用
sleep 的 NPU 测试。

### 🟡 13. 指南还有多项能力与配置描述不一致

- `sleep_mode.md:11` 使用 `"weight"`，实现实际使用 `"weights"`。
- `sleep_mode.md:13` 声称支持 “v0/v1”；当前 V0 已不支持，v2 model runner 也没有
  Sleep Mode E2E 保证。
- `sleep_mode.md:26` 声称必须源码构建；当前官方 wheel 可包含所需扩展，真正要求是
  `vllm_ascend_C` 可用。
- `sleep_mode.md:54` 描述的 Wake 顺序与
  `worker.py:251-283` 实际顺序不一致。
- NZ 限制仅藏在示例环境变量中；默认配置下
  `worker.py:244-250` 会拒绝 wake。
- Online 示例开启所有 dev API，但未显式限制 `--host 127.0.0.1`。

**建议**

将指南改为“能力/限制、Level 1、Level 2、Extra cleanup、Online 安全”五个独立章节；
所有命令都应进入自动化文档测试。Dev API 示例绑定 loopback，并明确不得暴露给不受信任
网络。

## 测试覆盖评价

现有测试主要验证 mock 调用和单卡 dense Level 1：

- `tests/ut/device_allocator/test_camem.py` mock 了 ACL/C++ 边界，无法发现 callback
  生命周期、GIL、ABI 异常和真实映射错误。
- `tests/e2e/pull_request/one_card/test_camem.py` 只覆盖单次
  `Level 1 -> full wake`，且模型为 dense。

建议补充以下阻断级测试：

1. C++ callback 强引用、refcount、异常路径 GIL 释放，以及 ASan/UBSan/LSan；
2. 每个 ACL API 的失败注入、回滚和重试；
3. 重复、并发、乱序、partial sleep/wake 状态组合；
4. Level 2 分阶段权重加载，以及直接 full wake 的拒绝行为；
5. 未量化 MoE Level 1/2 前后 logits 和输出一致性；
6. 多 stream 有在途任务时的 sleep；
7. TP/PP/EP 多 rank 多轮 HCCL destroy/restore 和单 rank 失败传播；
8. ACL Graph、Encoder Graph、speculative decoding 的清理与重捕获；
9. Online API query 参数、状态返回、安全绑定和 Level 2 流程；
10. 多轮 sleep/wake 的 NPU、pinned CPU 内存和延迟基准。

## 做得好的部分

- Worker、allocator、ACL/HCCL 清理职责边界清晰，extra cleanup 默认关闭，降低了默认路径
  的侵入性。
- Level 1 使用 pinned CPU backup，且虚拟地址在 sleep 期间保留，设计目标符合权重地址
  稳定性要求。
- 权重和 KV cache 在主要创建路径使用独立 tag。
- HCCL 正常销毁路径会等待 PP send handle 并同步 NPU。
- 分阶段 Wake 只在恢复 `"kv_cache"` 后重新捕获 ACL Graph，时机设计合理。
- 当前热路径没有新增 `tensor.item()` 引发的逐次 NPU/CPU 同步。

## 建议修复顺序

1. 修复 C++ 回调所有权、GIL/refcount、C ABI 异常和 ACL 返回码处理；
2. 修复 memcpy 参数并建立 allocation/worker 状态机；
3. 明确并强制 Level 2 与 MoE 布局恢复契约；
4. 修复 Online API 和用户指南；
5. 调整同步/ACL/HCCL 顺序及 persistent allocation 隔离；
6. 补齐故障注入、多卡、多轮及性能测试。
