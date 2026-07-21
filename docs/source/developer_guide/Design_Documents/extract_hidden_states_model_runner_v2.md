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

#### 第一步：根据配置创建 proposer

文件：`vllm_ascend/spec_decode/__init__.py`，`get_spec_decode_method()`。

```python
elif method == "extract_hidden_states":
    return AscendExtractHiddenStatesProposer(
        vllm_config,
        device,
        runner,
    )
```

当配置中的 `method` 是 `extract_hidden_states` 时，factory 返回 v1 使用的
`AscendExtractHiddenStatesProposer`。

#### 第二步：Runner 保存 proposer 并启用 auxiliary hidden states

文件：`vllm_ascend/worker/model_runner_v1.py`，`_set_up_drafter()`，约
587-624 行。

```python
self.drafter: (
    AscendNgramProposer
    | AscendEagleProposer
    | AscendExtractHiddenStatesProposer
    | None
) = None

if self.speculative_config:
    if get_pp_group().is_last_rank:
        self.drafter = self._get_drafter()
        if self.speculative_config.method == "extract_hidden_states":
            assert isinstance(
                self.drafter,
                AscendExtractHiddenStatesProposer,
            )
            self.use_aux_hidden_state_outputs = True
```

这里做了两件事：

1. 将 factory 创建的 proposer 保存到 `self.drafter`；
2. 设置 `use_aux_hidden_state_outputs=True`，要求目标模型 forward 时额外返回指定
   中间层的 hidden states。

#### 第三步：加载 cache-only model 并配置目标模型层

文件：`vllm_ascend/worker/model_runner_v1.py`，`load_model()`，约
3796-3821 行。

```python
if self.drafter:
    logger.info("Loading drafter model...")
    with get_tp_context(self.drafter):
        self.drafter.load_model(self.model)

if should_configure_aux_hidden_states:
    from vllm.model_executor.models.interfaces import supports_eagle3

    if not supports_eagle3(self.model):
        raise RuntimeError(
            "Model does not support EAGLE3 interface but "
            "aux_hidden_state_outputs was requested"
        )

    aux_layers = self._get_eagle3_aux_layers_from_config()
    if not aux_layers:
        aux_layers = (
            self.model.get_eagle3_default_aux_hidden_state_layers()
        )
    self.model.set_aux_hidden_state_layers(aux_layers)
```

`self.drafter.load_model(self.model)` 会加载 `ExtractHiddenStatesModel`。该模型主要包含
一个 `CacheOnlyAttentionLayer`。随后，Runner 根据
`eagle_aux_hidden_state_layer_ids` 配置目标模型应该返回哪些中间层。

#### 第四步：执行目标模型并拆分输出

文件：`vllm_ascend/worker/model_runner_v1.py`，`execute_model()`，约
2354-2369 行。

```python
hidden_states = self._model_forward(
    num_tokens_padded,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
    **model_kwargs,
)

aux_hidden_states = None
if self.use_aux_hidden_state_outputs:
    hidden_states, aux_hidden_states = hidden_states
```

不开启 extract 时，模型通常只返回：

```text
hidden_states
```

开启 extract 后，模型返回：

```text
(hidden_states, aux_hidden_states)
```

其中 `hidden_states` 继续用于计算 logits 和采样；`aux_hidden_states` 用于写入
hidden-state cache。

#### 第五步：采样完成后进入 extract 专用分支

文件：`vllm_ascend/worker/model_runner_v1.py`，
`propose_draft_token_ids()`，约 1750-1779 行。

```python
elif self.speculative_config.uses_extract_hidden_states():
    assert isinstance(
        self.drafter,
        AscendExtractHiddenStatesProposer,
    )
    if (
        not self.use_aux_hidden_state_outputs
        or aux_hidden_states is None
    ):
        raise ValueError(
            "aux_hidden_states are required when using "
            "`extract_hidden_states`"
        )

    common_attn_metadata = spec_decode_common_attn_metadata
    target_hidden_states = [
        h[:num_scheduled_tokens]
        for h in aux_hidden_states
    ]

    draft_token_ids = self.drafter.propose(
        self.speculative_config.num_speculative_tokens,
        sampled_token_ids=valid_sampled_token_ids,
        target_hidden_states=target_hidden_states,
        common_attn_metadata=common_attn_metadata,
    )
```

`h[:num_scheduled_tokens]` 会移除 graph padding，只保留本轮真实 token 对应的 hidden
states。之后 Runner 把以下数据交给 proposer：

- `valid_sampled_token_ids`：目标模型已经采样出的 token；
- `target_hidden_states`：选择层的 hidden states；
- `common_attn_metadata`：block table、slot mapping、序列长度等 cache 写入信息。

完成 proposer 调用后，Runner 还会更新下一轮 token：

```python
next_token_ids, valid_sampled_tokens_count = (
    self.drafter.prepare_next_token_ids_padded(
        valid_sampled_token_ids,
        self.requests,
        self.input_batch,
        self.discard_request_indices.gpu,
        self.num_discarded_requests,
    )
)
self._copy_valid_sampled_token_count(
    next_token_ids,
    valid_sampled_tokens_count,
)
```

#### 第六步：Proposer 堆叠多层 hidden states

上游文件：`vllm/v1/spec_decode/extract_hidden_states.py`，
`ExtractHiddenStatesProposer.propose()`。

```python
stacked_hidden_states = torch.stack(
    target_hidden_states,
    dim=1,
)
num_tokens = stacked_hidden_states.shape[0]
self.hidden_states[:num_tokens] = stacked_hidden_states

attn_metadata = self.attn_metadata_builder.build_for_drafting(
    common_attn_metadata=common_attn_metadata,
    draft_index=0,
)
per_layer_attn_metadata = {
    layer_name: attn_metadata
    for layer_name in self.attn_layer_names
}
```

假设：

```text
num_tokens = 10
选择的模型层数 = 3
hidden_size = 1024
```

stack 后的 shape 是：

```text
[10, 3, 1024]
```

随后 proposer 建立 cache-only layer 所需的 attention metadata，并执行
`ExtractHiddenStatesModel`：

```python
with set_forward_context(
    per_layer_attn_metadata,
    self.vllm_config,
    num_tokens=num_input_tokens,
    num_tokens_across_dp=num_tokens_across_dp,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    slot_mapping=self._get_slot_mapping(
        num_input_tokens,
        common_attn_metadata.slot_mapping,
    ),
):
    self.model(
        hidden_states=self.hidden_states[:num_input_tokens],
    )

return sampled_token_ids[:, :1]
```

返回 `sampled_token_ids[:, :1]` 表示它不预测新 token，而是把目标模型已经采样出的
token 当作 draft token。

#### 第七步：CacheOnlyAttentionLayer 写入 cache

上游文件：`vllm/model_executor/models/extract_hidden_states.py`，
`CacheOnlyAttentionLayer.forward()`。

```python
def forward(self, to_cache: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        0,
        device=to_cache.device,
        dtype=to_cache.dtype,
    )
    dummy_out = unified_kv_cache_update(
        to_cache,
        self.layer_name,
    )
    _ = dummy_attention(self.layer_name, dummy_out)
    return output
```

真正的基础写入操作为：

```python
def basic_cache(to_cache, kv_cache, slot_mapping):
    block_size = kv_cache.shape[1]
    kv_cache[
        slot_mapping // block_size,
        slot_mapping % block_size,
    ] = to_cache
```

slot mapping 将每个 token 映射为：

```text
block index = slot // block_size
block offset = slot % block_size
```

因此每个 token 的多层 hidden states 会写入正确的 cache block。

#### 第八步：KV Connector 保存或传输结果

`CacheOnlyAttentionLayer.forward()` 中的 `dummy_attention()` 使用
`@maybe_transfer_kv_layer` 装饰器。cache 更新完成后，该装饰器会通知 KV Connector
处理当前 layer。使用 `ExampleHiddenStatesConnector` 时，hidden states 最终被保存为
safetensors 文件，并通过输出中的 `hidden_states_path` 返回给用户。

完整调用关系如下：

```text
get_spec_decode_method()
    ↓
AscendExtractHiddenStatesProposer
    ↓
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

#### v1 初始化调用栈

下面的调用栈描述 Worker 创建 runner 和 proposer 的过程。`类名.方法名()` 后括号内是
对应文件。

```text
NPUWorker.init_device()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.__init__()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
NPUModelRunner._set_up_drafter()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
NPUModelRunner._get_drafter()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
get_spec_decode_method(
    method="extract_hidden_states"
)
  (vllm_ascend/spec_decode/__init__.py)
    ↓
AscendExtractHiddenStatesProposer.__init__()
  (vllm_ascend/spec_decode/extract_hidden_states_proposer.py)
    ↓
ExtractHiddenStatesProposer.__init__()
  (vllm/v1/spec_decode/extract_hidden_states.py)
```

模型加载在 Worker 的另一个阶段执行：

```text
NPUWorker.load_model()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.load_model()
  (vllm_ascend/worker/model_runner_v1.py)
    ├── 加载 target model
    ├── self.drafter.load_model(self.model)
    │     ↓
    │   ExtractHiddenStatesProposer.load_model()
    │     (vllm/v1/spec_decode/extract_hidden_states.py)
    │     ↓
    │   get_model(draft_model_config)
    │     ↓
    │   ExtractHiddenStatesModel
    │     ↓
    │   CacheOnlyAttentionLayer
    │
    └── self.model.set_aux_hidden_state_layers(aux_layers)
          配置 target model 需要输出的中间层
```

#### v1 KV cache 初始化调用栈

加载 target model 和 cache-only model 后，Engine 会查询所有 layer 的 KV cache
规格，再分配实际 cache：

```text
NPUWorker.get_kv_cache_spec()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.get_kv_cache_spec()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
遍历 AttentionLayerBase
    ↓
发现 CacheOnlyAttentionLayer
    ↓
CacheOnlyAttentionLayer.get_kv_cache_spec()
  (vllm/model_executor/models/extract_hidden_states.py)
    ↓
生成 HiddenStateCacheSpec
```

分配阶段：

```text
NPUWorker.initialize_from_config()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.initialize_kv_cache()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
NPUModelRunner.initialize_kv_cache_tensors()
    ↓
NPUModelRunner._allocate_kv_cache_tensors()
    ↓
为 HiddenStateCacheSpec 分配单 tensor
    ↓
reshape 为
[num_blocks, block_size, num_layers, hidden_size]
    ↓
bind_kv_cache()
    ↓
CacheOnlyAttentionLayer.kv_cache
```

#### v1 单步推理调用栈

v1 将模型执行和采样拆成两个 Worker 调用。`execute_model()` 先执行 target model 并
临时保存结果，`sample_tokens()` 再完成采样和 hidden-state 提取。

```text
NPUWorker.execute_model()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.execute_model()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
NPUModelRunner._model_forward()
    ↓
TargetModel.forward()
    ↓
(hidden_states, aux_hidden_states)
    ↓
保存到 self.execute_model_state
```

随后执行：

```text
NPUWorker.sample_tokens()
  (vllm_ascend/worker/worker.py)
    ↓
NPUModelRunner.sample_tokens()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
从 self.execute_model_state 取出
hidden_states 和 aux_hidden_states
    ↓
NPUModelRunner._sample()
    ↓
得到 sampler_output.sampled_token_ids
    ↓
sample_tokens() 内部的 propose_draft_token_ids()
    ↓
NPUModelRunner.propose_draft_token_ids()
  (vllm_ascend/worker/model_runner_v1.py)
    ↓
uses_extract_hidden_states() 分支
    ↓
AscendExtractHiddenStatesProposer.propose()
  继承自：
  vllm/v1/spec_decode/extract_hidden_states.py
    ↓
torch.stack(target_hidden_states, dim=1)
    ↓
ExtractHiddenStatesModel.forward()
  (vllm/model_executor/models/extract_hidden_states.py)
    ↓
CacheOnlyAttentionLayer.forward()
    ↓
unified_kv_cache_update()
    ↓
CacheOnlyAttentionImpl.do_kv_cache_update()
    ↓
basic_cache()
    ↓
hidden states 写入 cache
    ↓
dummy_attention()
    ↓
@maybe_transfer_kv_layer
    ↓
ExampleHiddenStatesConnector
    ↓
生成 hidden_states_path
```

这里容易混淆的地方是存在两个同名方法：

1. `sample_tokens()` 内部定义的局部函数 `propose_draft_token_ids()`，负责整理参数；
2. `NPUModelRunner.propose_draft_token_ids()`，负责根据 speculative method 进入
   extract、EAGLE、MTP 等具体分支。

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

### 5.1 增加原生 v2 Speculator

新增文件：

```text
vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py
```

`AscendExtractHiddenStatesSpeculator` 在 v2 目录中独立实现完整生命周期：

```text
v2 Model Runner
    ↓
AscendExtractHiddenStatesSpeculator
    ↓
ExtractHiddenStatesModel
    ↓
CacheOnlyAttentionLayer
```

v2 speculator 不继承、不持有、也不调用 v1 的
`AscendExtractHiddenStatesProposer`。模型加载、buffer 管理、hidden states stack、
metadata builder、cache-only forward、DP/SP 协调和 graph capture 都由 v2
speculator 自己负责。

需要区分“v1 Model Runner 实现”和上游 Python 包名。当前上游 Model Runner v2 本身仍
位于 `vllm.v1.worker.gpu` 命名空间，因此 v2 代码仍会导入其中的公共类型，例如
`InputBatch`、`BlockTables` 和 `KVCacheConfig`。这些是 v2 runner 正在使用的公共
基础设施，不表示继续依赖 v1 proposer 或 `model_runner_v1.py`。

原生 speculator 需要实现：

- `load_model()`：加载 cache-only model；
- `set_attn()`：记录 KV cache group 和 block tables；
- `init_cudagraph_manager()`：初始化自己的 graph dispatcher；
- `capture()`：捕获 cache-only model 的 piecewise graph；
- `propose()`：直接执行 hidden states stack 和 cache-only forward；
- `_dispatch_and_sync()`：执行 SP padding、graph dispatch 和 DP 同步；
- `_run_cache_only_model()`：建立 forward context 并执行 cache 写入。

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

extract speculator 需要 runner 引用，以调用 v2 runner 的 sequence-parallel padding
和 data-parallel metadata 同步方法。

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

v2 在执行 `speculator.propose()` 之前已经完成 request state 更新。原生 speculator 根据
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

v2 speculator 需要根据以下数据重建 metadata：

- `query_start_loc`：每个请求在 token tensor 中的起点；
- `seq_lens`：请求当前序列长度；
- `num_scheduled_tokens`：本轮每个请求处理的 token 数；
- `block_table_tensor`：请求对应的 cache blocks；
- `slot_mapping`：每个 token 的 cache 写入位置；
- `positions`：token position。

构造完成后，v2 speculator 直接调用
`AttentionMetadataBuilder.build_for_drafting()`，再执行 cache-only model。

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
vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py
```

`CacheOnlyAttentionLayer` 依赖 slot mapping 才能执行 cache update。如果 graph capture
时传入空 mapping：

```python
slot_mapping={}
```

捕获的图可能不包含 hidden-state cache 写入，运行时会得到空数据、零数据或旧数据。

解决方案是让 `speculator.capture()` 把 captured attention state 中的
`slot_mappings` 传给 v2 speculator 自己的 `_dummy_run()`，再通过 v2 的预分配
buffer 构造 cache-only layer 对应的 mapping：

```text
CapturedAttentionState.slot_mappings
    ↓
speculator.capture()
    ↓
speculator._dummy_run(slot_mappings=...)
    ↓
set_forward_context(slot_mapping=...)
```

Graph 和 DP padding 使用 `-1` 表示无效 slot。上游 cache-only update 直接使用 tensor
索引，`-1` 会被解释为最后一个 cache 位置，而不是“跳过”。因此 v2 speculator 在加载
cache-only layer 时为该实例安装只写入 `slot_mapping >= 0` 的 update 方法，防止
padding hidden states 覆盖真实 cache。NPU 路径使用固定 launch shape 的 Triton
kernel，在设备端通过 mask 跳过负 slot，避免 ACL graph replay 依赖动态 tensor shape。

普通 profile/dummy run 没有真实 slot mapping，此时必须向 forward context 传入空字典
并跳过 cache update，不能复用 slot mapping buffer 中上一批请求的旧值。

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
    ├── stack 多层 hidden states
    ├── 执行 graph dispatch 和 DP/SP 协调
    └── CacheOnlyAttentionLayer 写入 hidden-state cache
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
  原生 v2 Speculator、aux hidden states、cache-only model、
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

完成上述修改后，v2 runner 使用独立的原生 speculator，不依赖 v1 proposer 或
`model_runner_v1.py`。迁移没有引入 MTP 模型依赖，主要新增复杂度集中在：

1. 在 v2 speculator 内完整实现模型加载、buffer、metadata 和 cache-only forward；
2. hidden-state 单 tensor cache；
3. 跨 KV group 共享底层内存时的多种 view；
4. ACL graph capture 中保留真实 cache 写入路径。
