# `extract_hidden_states` 从 Model Runner v1 迁移到 v2

## 1. 背景

`extract_hidden_states` 是一种特殊的 speculative decoding 模式。它不预测多个未来
token，而是提取目标模型指定层的 hidden states，并通过 KV Connector 保存或传输这些
数据。典型用途是收集 EAGLE 类 draft model 的训练数据。

用户配置示例：

```python
speculative_config = {
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": [2, 14, 26],
        }
    },
}
```

该功能借用了 speculative decoding 的调度流程，但不进行真正的 token 推测：
proposer 会把目标模型已经采样出的 token 作为 draft token 返回，因此该 token 在下一次
验证时一定匹配。这个模式的主要产物是 hidden states，而不是推测加速。

## 2. 核心概念

### 2.1 Auxiliary hidden states

目标模型正常只返回最后一层 hidden states。开启
`use_aux_hidden_state_outputs` 后，模型额外返回指定中间层的输出：

```text
model_output = (hidden_states, aux_hidden_states)
```

其中 `aux_hidden_states` 是一个 tensor 列表，每个元素对应一个被选择的模型层。

### 2.2 Cache-only model

extract 模式会加载一个轻量的 `ExtractHiddenStatesModel`。该模型没有正常的
attention 计算，主要包含一个 `CacheOnlyAttentionLayer`，负责把 hidden states 写入
cache。

普通 attention cache 使用 K/V 两个 tensor：

```text
(k_cache, v_cache)
```

hidden-state cache 只需要一个 tensor：

```text
hidden_state_cache
```

因此该 cache 使用 `HiddenStateCacheSpec` 作为类型标记，不能直接复用普通 K/V
cache 的分配和 reshape 逻辑。

## 3. v1 实现

v1 的主要入口位于：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/spec_decode/extract_hidden_states_proposer.py`
- `vllm_ascend/spec_decode/__init__.py`

### 3.1 v1 抽象

v1 使用 `drafter`/`proposer` 抽象：

```text
NPUModelRunner
    └── AscendExtractHiddenStatesProposer
            └── ExtractHiddenStatesModel
                    └── CacheOnlyAttentionLayer
```

初始化时，`get_spec_decode_method()` 根据
`method == "extract_hidden_states"` 创建
`AscendExtractHiddenStatesProposer`。

### 3.2 v1 执行流程

```text
目标模型 forward
    ↓
得到 hidden_states 和 aux_hidden_states
    ↓
正常执行 token sampling
    ↓
propose_draft_token_ids() 进入 extract 分支
    ↓
把多层 aux_hidden_states 传给 proposer
    ↓
proposer 将多层 tensor stack 成
[num_tokens, num_layers, hidden_size]
    ↓
CacheOnlyAttentionLayer 写入 hidden-state cache
    ↓
KV Connector 保存或传输 hidden states
```

v1 的 extract 分支直接位于 `propose_draft_token_ids()` 中。Runner 负责准备
sampled token、attention metadata、padding 以及请求状态，然后调用 proposer。

### 3.3 v1 KV cache

v1 在 `get_kv_cache_spec()` 中识别 `CacheOnlyAttentionLayer`，并生成
`HiddenStateCacheSpec`。在 cache 分配和 reshape 阶段，它会：

1. 为 hidden-state cache 分配一个底层 tensor；
2. 按照 `[num_blocks, block_size, num_layers, hidden_size]` 建立 view；
3. 在 hybrid cache pool 中处理 hidden-state layer 与普通 attention layer
   共享底层内存的情况。

## 4. v2 与 v1 的主要差异

| 维度 | Model Runner v1 | Model Runner v2 |
| --- | --- | --- |
| Spec decode 抽象 | Drafter/Proposer | Speculator |
| extract 调用入口 | `propose_draft_token_ids()` 的专用分支 | 统一调用 `speculator.propose()` |
| 请求状态 | v1 request/input batch | v2 `RequestState`、`InputBatch` |
| sampled token 来源 | Runner 直接传入采样结果 | 从 `last_sampled` 和 `next_prefill_tokens` 读取 |
| attention metadata | Runner 提供 `CommonAttentionMetadata` | Speculator 根据 v2 input batch 重建 |
| KV cache 初始化 | v1 runner 内部实现 | v2 共用 `worker/v2/attn_utils.py` |
| Graph 流程 | Runner dummy run 中调用 proposer | `speculator.capture()` 和统一 dummy propose |
| 上游支持情况 | 上游已有 extract proposer | 上游 v2 没有 extract speculator |

最关键的差异是：v2 runner 不再调用 v1 的
`propose_draft_token_ids()`，而是只认统一的 speculator 协议。因此，仅复制 v1 的
extract 分支不能使功能生效。

## 5. v2 需要修改的地方

### 5.1 增加 v2 Speculator 适配层

新增文件：

```text
vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py
```

`AscendExtractHiddenStatesSpeculator` 使用组合方式复用已有的
`AscendExtractHiddenStatesProposer`：

```text
v2 Model Runner
    ↓ v2 propose 参数
AscendExtractHiddenStatesSpeculator
    ↓ 转换参数
AscendExtractHiddenStatesProposer
    ↓
ExtractHiddenStatesModel
```

没有直接让 proposer 继承 v2 speculator，因为两者的 `propose()` 参数、生命周期和
graph 接口不同。组合方式可以保留已有 Ascend proposer 的 DP/SP 和 cache-only forward
逻辑，同时把 v2 相关变化限制在适配层内。

适配层需要实现：

- `load_model()`：加载 cache-only model；
- `set_attn()`：记录 KV cache group 和 block tables；
- `init_cudagraph_manager()`：初始化 proposer graph keys；
- `capture()`：捕获 cache-only model 的 piecewise graph；
- `propose()`：把 v2 输入转换为 proposer 所需参数。

### 5.2 注册 speculator

修改：

```text
vllm_ascend/worker/v2/spec_decode/__init__.py
```

在 factory 中增加：

```python
if speculative_config.uses_extract_hidden_states():
    return AscendExtractHiddenStatesSpeculator(
        vllm_config,
        device,
        runner,
    )
```

extract speculator 需要 runner 引用，因为 Ascend proposer 使用 runner 的
sequence-parallel padding 和 data-parallel metadata 同步方法。

### 5.3 延迟上游 speculator 初始化

修改：

```text
vllm_ascend/worker/v2/model_runner.py
```

上游 `GPUModelRunner.__init__()` 会先调用上游 `init_speculator()`，但上游 v2
factory 不支持 `extract_hidden_states`，会直接抛出 `NotImplementedError`。

解决方案是在执行父类初始化时暂时跳过 extract speculator 的创建，父类初始化完成后再
用 Ascend factory 创建：

```text
执行父类初始化
    ↓ 暂时让 extract speculator 返回 None
父类基础字段初始化完成
    ↓
调用 Ascend init_speculator(vllm_config, device, runner=self)
```

这个处理只影响 extract 模式，不改变 MTP、EAGLE、DSpark 或 DFlash 的初始化路径。

### 5.4 启用 auxiliary hidden states

extract 模式下必须设置：

```python
self.use_aux_hidden_state_outputs = True
```

这样目标模型加载阶段会配置
`eagle_aux_hidden_state_layer_ids`，forward 阶段会返回：

```python
hidden_states, aux_hidden_states = model_output
```

当前方案明确拒绝 extract 与 pipeline parallel 同时使用，因为 v2 尚未实现跨 PP rank
收集这些中间层输出。

### 5.5 转换 v2 sampled token

v2 在执行 `speculator.propose()` 之前已经完成 request state 更新。适配层根据
`input_batch.idx_mapping` 从全局状态中取得当前请求的 token：

```python
req_indices = input_batch.idx_mapping[:input_batch.num_reqs]
sampled_token_ids = last_sampled[req_indices, 0]
```

chunked prefill 请求可能尚未采样 token，因此需要根据 `num_sampled` 回退到
`next_prefill_tokens`：

```python
sampled_token_ids = torch.where(
    num_sampled > 0,
    sampled_token_ids,
    next_prefill_tokens,
)
```

### 5.6 重建 CommonAttentionMetadata

v1 runner 会直接向 proposer 提供 `CommonAttentionMetadata`；v2 的统一 speculator
接口只提供 input batch、block tables 和 per-layer slot mappings。

适配层需要根据以下数据重建 metadata：

- `query_start_loc`：每个请求在 token tensor 中的起点；
- `seq_lens`：请求当前序列长度；
- `num_scheduled_tokens`：本轮每个请求处理的 token 数；
- `block_table_tensor`：请求对应的 cache blocks；
- `slot_mapping`：每个 token 的 cache 写入位置；
- `positions`：token position。

构造完成后，原 proposer 可以继续调用现有
`AttentionMetadataBuilder.build_for_drafting()`。

### 5.7 支持 HiddenStateCacheSpec

修改：

```text
vllm_ascend/worker/v2/attn_utils.py
```

需要在三个阶段增加 hidden-state cache 支持。

#### KV spec 发现

识别 `CacheOnlyAttentionLayer`，保留
`HiddenStateCacheSpec` 类型：

```python
if isinstance(attn_module, CacheOnlyAttentionLayer):
    kv_cache_spec[layer_name] = HiddenStateCacheSpec(...)
```

不能降级为普通 `AttentionSpec`，否则 cache grouping 和内存布局会错误。

#### 底层内存分配

普通 attention cache 分配 K/V 两个 tensor；hidden-state cache 分配一个完整 tensor。

还需要检查整个 `KVCacheTensor.shared_by`，不能只检查第一个 layer。vLLM 可能让不同
KV group 中相同位置的普通 attention layer 和 hidden-state layer 共用同一块底层
内存。

正确策略是：

```text
共享池包含 HiddenStateCacheSpec
    ↓
只分配一个完整底层 tensor
    ├── hidden-state layer：建立完整 hidden-state view
    └── attention layer：从同一 tensor 建立 K view 和 V view
```

如果仅根据 `shared_by[0]` 决定 tensor 类型，可能出现：

- hidden-state layer 收到 `(K, V)` tuple；
- 普通 attention layer 收到未经拆分的 tensor；
- reshape 阶段类型断言失败。

#### Cache reshape

对 hidden-state spec，直接建立：

```text
[num_blocks, block_size, num_hidden_layers, hidden_size]
```

对共享同一底层 tensor 的普通 attention spec，则根据 backend shape 和 dtype 计算
K/V 大小，再从底层 tensor 中切出 K/V view。

### 5.8 保留 ACL graph 的 slot mapping

修改：

```text
vllm_ascend/spec_decode/extract_hidden_states_proposer.py
```

`CacheOnlyAttentionLayer` 依赖 slot mapping 才能执行 cache update。如果 graph capture
时传入空 mapping：

```python
slot_mapping={}
```

捕获的图可能不包含 hidden-state cache 写入，运行时会得到空数据、零数据或旧数据。

解决方案是让 `speculator.capture()` 把 captured attention state 中的
`slot_mappings` 传给 proposer dummy run，再通过 proposer 的预分配 buffer 构造
cache-only layer 对应的 mapping：

```text
CapturedAttentionState.slot_mappings
    ↓
speculator.capture()
    ↓
proposer.dummy_run(slot_mappings=...)
    ↓
set_forward_context(slot_mapping=...)
```

## 6. v2 完整数据流

```text
用户配置 method=extract_hidden_states
    ↓
Ascend init_speculator 创建 ExtractHiddenStatesSpeculator
    ↓
目标模型启用 auxiliary hidden-state layers
    ↓
目标模型 forward
    ↓
(hidden_states, aux_hidden_states)
    ↓
目标模型正常 sampling
    ↓
v2 runner 更新 RequestState
    ↓
ExtractHiddenStatesSpeculator.propose()
    ├── 选择 last_sampled 或 next_prefill_tokens
    ├── 去除 aux hidden states 的 graph padding
    ├── 重建 CommonAttentionMetadata
    └── 调用已有 Ascend proposer
            ↓
        stack 多层 hidden states
            ↓
        CacheOnlyAttentionLayer 写入 hidden-state cache
            ↓
KV Connector post_forward
    ↓
输出 hidden_states_path
```

## 7. 与 MTP 的边界

迁移 extract 功能不需要迁移 MTP。

| 项目 | extract_hidden_states | MTP |
| --- | --- | --- |
| 目标 | 保存中间层 hidden states | 预测多个未来 token |
| Draft model | Cache-only model | MTP heads/layers |
| Hidden states 来源 | 多层 `aux_hidden_states` | MTP target hidden states |
| Cache | `HiddenStateCacheSpec` 单 tensor | MTP attention/模型 cache |
| 输出 token | 回显已采样 token | 新预测的 draft tokens |

两者只共享 speculative decoding 调度框架。extract 依赖的是 EAGLE3 风格的
auxiliary hidden-state 输出接口，而不是 MTP 模型结构。

因此迁移方案应保持以下边界：

```text
需要迁移：
  Speculator 适配、aux hidden states、cache-only model、
  HiddenStateCacheSpec、slot mapping、DP/SP 协调、ACL graph

不需要迁移：
  MTP model、MTP proposer/speculator、MTP heads、
  get_mtp_target_hidden_states()
```

## 8. 测试方案

### 8.1 单元测试

1. Factory 能为 extract 配置创建 v2 speculator；
2. `propose()` 能正确处理 request index；
3. decode 请求使用 `last_sampled`；
4. prefill 请求使用 `next_prefill_tokens`；
5. 缺少 `aux_hidden_states` 时明确报错；
6. 只传递真实 token 范围，不保存 graph padding；
7. dummy run 和 graph capture 能传递 slot mappings；
8. `HiddenStateCacheSpec` 保持类型不变；
9. hidden-state cache 使用单 tensor；
10. 普通 attention 与 hidden-state group 共享底层 tensor 时，两者都能正确 reshape。

### 8.2 E2E 测试

使用：

```text
VLLM_USE_V2_MODEL_RUNNER=1
```

至少验证：

1. eager 模式能生成 `hidden_states_path`；
2. 输出 shape 为
   `[num_tokens, num_selected_layers, hidden_size]`；
3. token IDs 能正确 round-trip；
4. ACL graph 模式输出不是空数据或旧数据；
5. hybrid 模型不会在共享 cache pool reshape 时失败；
6. DP 大于 1 时不会发生 metadata collective 死锁。

## 9. 迁移结果

完成上述修改后，v2 runner 可以复用成熟的 v1 Ascend proposer，同时遵循 v2 的统一
speculator 生命周期。迁移没有引入 MTP 模型依赖，主要新增复杂度集中在：

1. v1 proposer 与 v2 speculator 的接口转换；
2. hidden-state 单 tensor cache；
3. 跨 KV group 共享底层内存时的多种 view；
4. ACL graph capture 中保留真实 cache 写入路径。
