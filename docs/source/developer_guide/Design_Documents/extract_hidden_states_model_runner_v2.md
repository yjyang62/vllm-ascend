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

### 1.1 本文中的 extract 是什么

本文后面为了简短，会把 `extract_hidden_states` 写成 **extract**。

这里的 extract 不是 Python 关键字，也不是另一个类名，它只是英文“提取”的意思，完整
名称始终是：

```text
extract_hidden_states
```

模型处理一个 token 时，每一层都会产生一份中间计算结果：

```text
输入 token
    ↓
第 1 层 hidden states
    ↓
第 2 层 hidden states
    ↓
...
    ↓
最后一层 hidden states
```

正常生成文本时，用户通常只关心最后一层结果。extract 功能会把用户指定的某几层中间
结果额外复制出来。例如：

```python
"eagle_aux_hidden_state_layer_ids": [2, 14, 26]
```

表示：

```text
保留第 2 层 hidden states
保留第 14 层 hidden states
保留第 26 层 hidden states
```

这些中间结果会组成一个 tensor：

```text
[num_tokens, num_selected_layers, hidden_size]
```

然后写入 hidden-state cache，最后由 KV Connector 保存或传输。

所以 extract 的完整动作是：

```text
目标模型执行
    ↓
取得指定中间层输出
    ↓
把多层输出 stack 在一起
    ↓
写入 hidden-state cache
    ↓
保存到文件或传给其他组件
```

extract 不会从模型中“删除”数据，也不是把文本中的某一段截取出来。它只是把模型内部
原本就会产生的中间 tensor 额外复制并保存。目标模型仍然会正常生成 token。

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

`NPUModelRunner.propose_draft_token_ids()` 不是由 `execute_model()` 直接调用的，而是由
`sample_tokens()` 中的局部函数调用。

调用方文件：`vllm_ascend/worker/model_runner_v1.py`，
`sample_tokens()`，约 2498-2513 行。

```python
def propose_draft_token_ids(sampled_token_ids):
    assert spec_decode_common_attn_metadata is not None
    self._draft_token_ids = self.propose_draft_token_ids(
        sampled_token_ids,
        self.input_batch.sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        scheduler_output.total_num_scheduled_tokens,
        hidden_states,
        aux_hidden_states,
        sample_hidden_states,
        batch_desc,
    )
    self._copy_draft_token_ids_to_cpu(scheduler_output)
```

注意这里有两个名字相同、但作用不同的函数：

```text
sample_tokens() 内部的 propose_draft_token_ids()
    ↓ 调用
self.propose_draft_token_ids()
    ↓ 实际是
NPUModelRunner.propose_draft_token_ids()
```

extract 模式会被归类为 padded-batch drafter，判断代码约在 2518-2524 行：

```python
use_padded_batch = (
    self.speculative_config.use_eagle()
    or self.speculative_config.uses_draft_model()
    or self.speculative_config.uses_extract_hidden_states()
    or self.speculative_config.use_ngram_gpu()
) and not self.speculative_config.disable_padded_drafter_batch
```

在通常的非 PP 场景中，真正触发局部函数调用的位置约在 2557-2560 行：

```python
if use_padded_batch and not early_pp_padded_drafter:
    propose_draft_token_ids(
        sampler_output.sampled_token_ids
    )
```

因此通常的完整调用关系是：

```text
NPUWorker.sample_tokens()
    ↓
NPUModelRunner.sample_tokens()
    ↓
self._sample()
    ↓ 得到 sampler_output.sampled_token_ids
sample_tokens() 局部函数 propose_draft_token_ids(
    sampler_output.sampled_token_ids
)
    ↓
self.propose_draft_token_ids(...)
    ↓
NPUModelRunner.propose_draft_token_ids(...)
    ↓
uses_extract_hidden_states() 分支
```

如果是 pipeline parallel 且需要提前运行 padded drafter，则调用发生在约
2530-2534 行：

```python
if early_pp_padded_drafter:
    with record_function_or_nullcontext("draft_token"):
        propose_draft_token_ids(
            sampler_output.sampled_token_ids
        )
```

只有非 padded-batch 的 speculative method 才会在 bookkeeping 之后使用
`valid_sampled_token_ids` 调用，约在 2561-2564 行：

```python
if self.speculative_config and not use_padded_batch:
    propose_draft_token_ids(valid_sampled_token_ids)
```

extract 默认属于 padded-batch，所以通常走
`sampler_output.sampled_token_ids` 那条路径，不走最后这个非 padded 分支。

被调用方文件：`vllm_ascend/worker/model_runner_v1.py`，
成员方法 `NPUModelRunner.propose_draft_token_ids()`，其 extract 分支约
1750-1779 行。

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

先用一个简单比喻理解：

```text
v1 像一家旧餐厅：
  主厨 Model Runner 自己判断客人点了什么，
  如果是 extract_hidden_states，
  主厨就直接叫专门的助手 Proposer 来处理。

v2 像一家重新分工的餐厅：
  主厨 Model Runner 不再自己处理每种特殊订单，
  而是把所有 speculative decoding 工作交给
  统一岗位 Speculator。
```

所以，v1 和 v2 的区别并不只是函数改了名字，而是“谁负责组织这项工作”发生了变化。

### 4.1 谁负责调用 extract

v1 中，Runner 自己有一大段 `if/elif`：

```python
if method == "ngram":
    ...
elif method == "eagle":
    ...
elif method == "extract_hidden_states":
    self.drafter.propose(...)
```

对应文件：

```text
vllm_ascend/worker/model_runner_v1.py
└── NPUModelRunner.propose_draft_token_ids()
```

准确地说，v1 Runner 知道的是 extract 的**调用和编排流程**，不是所有底层实现。
Runner 自己写了以下代码：

```text
判断当前 method 是不是 extract_hidden_states
    ↓
检查 aux_hidden_states 是否存在
    ↓
去掉 aux_hidden_states 中的 graph padding
    ↓
准备 sampled token 和 CommonAttentionMetadata
    ↓
调用 AscendExtractHiddenStatesProposer.propose()
    ↓
准备下一轮使用的 token
```

这些步骤都能在
`vllm_ascend/worker/model_runner_v1.py` 的
`NPUModelRunner.propose_draft_token_ids()` extract 分支中看到。

真正的实现分成三层：

```text
第一层：Ascend v1 Runner
  文件：vllm_ascend/worker/model_runner_v1.py
  负责：
    - 什么时候执行 extract
    - 从哪里取得 aux_hidden_states
    - 传哪些参数
    - 调用完成后如何更新请求状态

第二层：vLLM ExtractHiddenStatesProposer
  文件：vllm/v1/spec_decode/extract_hidden_states.py
  负责：
    - stack 多层 hidden states
    - 构造 cache-only attention metadata
    - 执行 ExtractHiddenStatesModel
    - 返回 sampled token 作为 draft token

第三层：vLLM CacheOnlyAttentionLayer
  文件：vllm/model_executor/models/extract_hidden_states.py
  负责：
    - 根据 slot mapping 找到 cache 地址
    - 真正把 hidden states 写入 cache
    - 触发 KV Connector
```

所以“细节不是在 vLLM 里面吗”的答案是：

```text
底层写入细节确实在 vLLM 里；
但 Ascend v1 Runner 中仍然存在 extract 专用的调用和编排代码。
```

迁移到 v2 时，不需要重新实现通用的 `ExtractHiddenStatesModel` 和
`CacheOnlyAttentionLayer`；需要迁移的是原来放在 v1 Runner/Proposer 路径中的
Ascend 编排和 v2 生命周期接入。

v2 中，Runner 不再写 extract 专用分支。它只做统一调用：

```python
draft_tokens = self.speculator.propose(...)
```

然后由具体的 `AscendExtractHiddenStatesSpeculator` 处理 v2 侧的编排：

```text
vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py
└── AscendExtractHiddenStatesSpeculator.propose()
```

直白地说：

```text
v1：Runner 包含 extract 专用编排，然后调用 vLLM Proposer。
v2：Runner 统一转交给原生 v2 Speculator；
    通用模型和 cache 写入仍然使用 vLLM 提供的组件。
```

这就是文档中“Drafter/Proposer”和“Speculator”的含义。它们都是负责 speculative
decoding 的组件，但属于两套不同的 Runner 组织方式。

### 4.2 为什么不能把 v1 Proposer 直接放进 v2

v1 Proposer 希望调用方直接给它这些参数：

```text
sampled_token_ids
target_hidden_states
CommonAttentionMetadata
```

而 v2 Runner 统一传给 Speculator 的是：

```text
InputBatch
last_sampled
next_prefill_tokens
aux_hidden_states
block tables
slot mappings
```

两边收到的参数不是同一套。更重要的是，v2 还要求 Speculator 自己参与：

- 模型加载；
- KV cache 绑定；
- dummy run；
- ACL graph capture；
- DP/SP 同步。

因此，不能简单地写成：

```text
v2 Speculator
    ↓ 转一下参数
v1 Proposer
```

本方案是在 v2 目录中实现原生 Speculator，不继承、不持有、也不调用 v1 Proposer。

### 4.3 请求数据放在哪里不同

一次推理中会同时处理多个请求，例如：

```text
请求 A："你好"
请求 B："介绍一下北京"
```

Runner 需要记录每个请求已经处理了多少 token、上次生成了什么 token、当前在 batch
中的位置等信息。

v1 使用自己的一套 request 和 input batch 对象，`propose_draft_token_ids()` 可以直接
拿到整理好的采样结果。

v2 将长期状态放在 `RequestState` 中，将本轮输入放在 `InputBatch` 中：

```text
RequestState：保存跨多轮不变或持续更新的请求状态
InputBatch：保存当前这一轮实际参与计算的请求
```

因此 v2 Speculator 需要使用：

```python
req_indices = input_batch.idx_mapping[:input_batch.num_reqs]
```

把“当前 batch 第几个请求”转换成“RequestState 中第几个请求”。

### 4.4 sampled token 的取得方式不同

sampled token 就是目标模型刚刚生成的 token。

v1 的调用链已经把采样结果整理成 `valid_sampled_token_ids`，然后直接传给 proposer：

```python
self.drafter.propose(
    sampled_token_ids=valid_sampled_token_ids,
    ...
)
```

v2 在调用 Speculator 之前已经更新了 `RequestState`，因此 Speculator 从
`last_sampled` 中取：

```python
sampled_token_ids = last_sampled[req_indices, 0]
```

如果请求仍在 prefill，还没有生成新 token，则使用：

```python
next_prefill_tokens[req_indices]
```

直白地说：

```text
v1：Runner 把 token 直接递给 Proposer。
v2：Speculator 根据请求编号去状态表中取 token。
```

### 4.5 cache 写入说明书的来源不同

把 hidden states 写进 cache 时，需要知道：

- 写入哪个 cache block；
- 写入 block 中的哪个位置；
- 当前有多少请求和 token；
- 每个请求的序列长度是多少。

这些信息合起来叫 `CommonAttentionMetadata`。可以把它理解为“cache 写入地址说明书”。

v1 Runner 已经准备好了这份说明书，直接传给 proposer：

```python
common_attn_metadata = spec_decode_common_attn_metadata
```

v2 的统一 `speculator.propose()` 接口没有直接提供完整的
`CommonAttentionMetadata`。因此 v2 Speculator 需要根据以下数据重新组成一份：

```text
InputBatch
block tables
slot mappings
seq_lens
query_start_loc
positions
```

这就是“Speculator 根据 v2 input batch 重建 metadata”的意思，并不是重新计算
hidden states。

### 4.6 KV cache 在哪里创建不同

KV cache 可以理解为模型运行时使用的一块大内存。

普通 attention 需要两份：

```text
K cache + V cache
```

extract 只保存 hidden states，所以只需要一份：

```text
hidden-state cache
```

v1 的 cache 发现、分配和 reshape 主要写在：

```text
vllm_ascend/worker/model_runner_v1.py
```

v2 把这些公共操作拆到了：

```text
vllm_ascend/worker/v2/attn_utils.py
```

所以迁移时不能只增加 Speculator，还必须让 v2 的 `attn_utils.py` 认识
`HiddenStateCacheSpec`，并知道这种 cache 是单 tensor，不是 `(K, V)` 两个 tensor。

### 4.7 ACL graph 的执行方式不同

ACL graph 可以粗略理解为：

```text
先把一段计算过程录下来，
之后遇到相同形状的输入时直接重放。
```

v1 在 Runner 的 dummy run 中顺便调用 proposer：

```text
Runner dummy run
    ↓
Proposer dummy run
```

v2 为 Speculator 规定了独立的生命周期：

```text
Speculator.init_cudagraph_manager()
    ↓
Speculator.capture()
    ↓
运行时 Speculator.propose()
```

因此原生 v2 Speculator 必须自己实现 graph 初始化、捕获和 dummy run，不能依赖 v1
Runner 帮它完成。

### 4.8 “上游 v2 没有实现”是什么意思

“上游”指 vLLM 主仓库。

上游已经提供：

```text
ExtractHiddenStatesModel
CacheOnlyAttentionLayer
HiddenStateCacheSpec
v1 ExtractHiddenStatesProposer
```

但是上游没有提供可以直接给 Model Runner v2 使用的
`ExtractHiddenStatesSpeculator`。

所以 vLLM Ascend 需要补充自己的原生 v2 实现。可以继续使用通用的
`ExtractHiddenStatesModel`、`CacheOnlyAttentionLayer` 和 `HiddenStateCacheSpec`，
但不能继续调用 v1 Proposer。

### 4.9 最后用两条流程对比

v1：

```text
Runner.sample_tokens()
    ↓
Runner.propose_draft_token_ids()
    ↓
AscendExtractHiddenStatesProposer.propose()
    ↓
CacheOnlyAttentionLayer
```

v2：

```text
Runner.sample_tokens()
    ↓
统一调用 speculator.propose()
    ↓
AscendExtractHiddenStatesSpeculator.propose()
    ↓
CacheOnlyAttentionLayer
```

最核心的一句话：

```text
v1 的 extract 逻辑分散在 Runner 和 Proposer 中；
v2 的 extract 逻辑集中在原生 Speculator 中。
```

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
