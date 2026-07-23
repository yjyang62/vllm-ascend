# Model Runner v2 中 `extract_hidden_states` 支持情况与上游对接指南

本文说明三件事：

1. 上游 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 Model Runner v1 / v2 对 `extract_hidden_states` 的支持现状。
2. `vllm-ascend` 在 **MRv1** 上已经做了哪些适配。
3. 如果未来上游 **MRv2** 原生支持了该特性，`vllm-ascend` 应该如何收敛改造。

配套设计细节（Ascend 原生 Speculator 实现、v1/v2 对比、调用栈）见：

- [`extract_hidden_states` Model Runner v1/v2 comparison and migration design](extract_hidden_states_model_runner_v2.md)

---

## 1. 一句话结论

| 路径 | 支持情况 |
|---|---|
| 上游 MRv1（`vllm/v1/worker/gpu_model_runner.py`） | **已支持**，通过 `ExtractHiddenStatesProposer` |
| 上游 MRv2（`vllm/v1/worker/gpu/model_runner.py`） | **未支持**，`init_speculator()` 直接 `NotImplementedError` |
| Ascend MRv2（本 PR） | **已补齐**，原生 `AscendExtractHiddenStatesSpeculator` |

因此：当前上游 GPU MRv2 不能直接跑 `method="extract_hidden_states"`；Ascend 侧是在补上游缺口，而不是跟上游已有 MRv2 Speculator 对齐。

---

## 2. 上游支持现状（截至本文编写时）

核对仓库：`https://github.com/vllm-project/vllm`  
示例 commit：`27ffbfd`（随 `main` 前进，结论以工厂注册与目录结构为准）。

### 2.1 MRv2 工厂未注册 extract

上游文件：`vllm/v1/worker/gpu/spec_decode/__init__.py`

MRv2 的 `init_speculator()` 当前只注册了：

- `dflash`
- `dspark`
- Gemma4 MTP
- `mtp`
- eagle 家族（`use_eagle()`）

其余 method（包含 `extract_hidden_states`）走：

```python
raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
```

对比上游 MRv1（`vllm/v1/worker/gpu_model_runner.py` 里创建 `self.drafter`），同一批常见 method 的支持情况如下：

| method / 能力 | 上游 MRv1（创建 `drafter`） | 上游 MRv2（`init_speculator()`） |
|---|---|---|
| `dflash` | 支持（`DFlashProposer`） | 支持（`DFlashSpeculator`） |
| `dspark` | 未看到独立注册 | 支持（`DSparkSpeculator`） |
| Gemma4 MTP | 支持（`Gemma4Proposer`） | 支持（`Gemma4Speculator`） |
| `mtp` / 其他 MTP 变体 | 支持（如 Step3.5 MTP 等） | 支持（`MTPSpeculator`） |
| eagle 家族（`use_eagle()`） | 支持（`EagleProposer`） | 支持（`EagleSpeculator`） |
| **`extract_hidden_states`** | **支持（`ExtractHiddenStatesProposer`）** | **未注册，直接 `NotImplementedError`** |
| `ngram` / `suffix` / `medusa` / draft model 等 | 支持 | 当前工厂未覆盖 |

MRv1 对 extract 的注册代码：

```python
elif self.speculative_config.method == "extract_hidden_states":
    self.drafter = ExtractHiddenStatesProposer(
        vllm_config=self.vllm_config, device=self.device
    )
    self.use_aux_hidden_state_outputs = True
```

结论：`extract_hidden_states` 已在上游 MRv1 落地，但尚未迁入 MRv2 工厂；
这是 Ascend 需要在 MRv2 侧自行补齐的直接原因。

### 2.2 MRv2 目录中没有 extract Speculator

上游 MRv2 speculative 目录大致如下：

```text
vllm/v1/worker/gpu/spec_decode/
├── autoregressive/
├── dflash/
├── dspark/
├── eagle/
├── gemma4/
├── mtp/
├── __init__.py          # init_speculator()
└── speculator.py        # BaseSpeculator / DraftModelSpeculator
```

**没有** `extract_hidden_states/` 子目录，也没有
`ExtractHiddenStatesSpeculator` 之类的类。

### 2.3 完整实现仍挂在 MRv1

上游已有能力全部位于 MRv1 / 共享层：

| 组件 | 路径 | 作用 |
|---|---|---|
| Proposer | `vllm/v1/spec_decode/extract_hidden_states.py` | MRv1 侧编排 |
| Model | `vllm/model_executor/models/extract_hidden_states.py` | `ExtractHiddenStatesModel` / `CacheOnlyAttentionLayer` |
| Config | `vllm/config/speculative.py` + `ExtractHiddenStatesConfig` | `uses_extract_hidden_states()` |
| Runner 接线 | `vllm/v1/worker/gpu_model_runner.py` | 创建 drafter、aux hidden states、propose |
| Cache Spec | `HiddenStateCacheSpec`（`kv_cache_interface`） | 把 hidden state 当 KV 写 |
| Connector 示例 | `example_hidden_states_connector.py` | 导出提取结果 |
| 文档/示例 | `docs/features/speculative_decoding/extract_hidden_states.md` | 用户用法 |

MRv1 接线示例：

```python
elif self.speculative_config.method == "extract_hidden_states":
    self.drafter = ExtractHiddenStatesProposer(...)
    self.use_aux_hidden_state_outputs = True
```

以及 propose 阶段：

```python
elif spec_config.uses_extract_hidden_states():
    assert isinstance(self.drafter, ExtractHiddenStatesProposer)
    # 要求 aux_hidden_states 存在，再调用 drafter.propose(...)
```

### 2.4 相关上游议题，但不等于 MRv2 已落地

- RFC：[Hidden States Extraction #33118](https://github.com/vllm-project/vllm/issues/33118)
- 近期相关 PR 多在修 MRv1 / hybrid / CUDA graph / connector，**尚未看到**已合并的
  “port `extract_hidden_states` to Model Runner V2” 实现。

判断是否已支持，请优先看这两个硬条件，而不是 release note 措辞：

1. `vllm/v1/worker/gpu/spec_decode/` 是否出现 extract Speculator。
2. `init_speculator()` 是否对 `uses_extract_hidden_states()` 有正式分支，而不再抛
   `NotImplementedError`。

---

## 3. Ascend 在 MRv1 上做了什么适配

上游 MRv1 已有完整 `ExtractHiddenStatesProposer`。Ascend **没有重写整套 extract**，
而是“继承上游 Proposer + 打 NPU 差异补丁 + Runner/KV 侧接线”。

### 3.1 核心文件

| 文件 | 作用 |
|---|---|
| `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` | `AscendExtractHiddenStatesProposer`，继承上游 Proposer |
| `vllm_ascend/spec_decode/__init__.py` | `get_spec_decode_method()` 工厂注册 extract |
| `vllm_ascend/worker/model_runner_v1.py` | aux hidden states、propose、KV 分配/reshape、ACL graph keys |
| `vllm_ascend/utils.py` | `is_drafter_moe_model()` / `is_hidden_state_cache_spec()` |
| `vllm_ascend/patch/platform/patch_mamba_config.py` | hybrid + extract 时跳过强制 mamba align |

### 3.2 复用上游 vs Ascend 自研

| 类别 | 内容 |
|---|---|
| **直接复用上游** | `ExtractHiddenStatesProposer.propose()` / `load_model()`、`ExtractHiddenStatesModel`、`CacheOnlyAttentionLayer`、`HiddenStateCacheSpec`、connector 协议 |
| **Ascend 薄适配** | DP/SP 同步、ACL graph `dummy_run`、discard token API、factory、runner 接线、hybrid KV pool |

### 3.3 具体适配点

1. **工厂注册**
   - `get_spec_decode_method("extract_hidden_states", ...)` 返回
     `AscendExtractHiddenStatesProposer(vllm_config, device, runner)`
   - 需要传入 `runner`，供 DP/SP hook 使用

2. **薄子类覆盖（不改 propose 主逻辑）**
   - `_determine_batch_execution_and_padding`
     - 先 `_pad_for_sequence_parallelism`
     - 再用 `runner._sync_metadata_across_dp(..., is_draft_model=True)`
     - **不用**上游 `coordinate_batch_across_dp`（collective tensor 形状与主 runner 不一致，会弄坏 gloo）
   - `dummy_run`
     - 适配 ACL graph capture 签名
     - idle DP rank 也必须走同一套 DP sync，避免死锁
   - `prepare_next_token_ids_padded`
     - 适配 Ascend 的 `discard_request_indices` / count API（不是 GPU 布尔 mask）

3. **Runner 接线（`model_runner_v1.py`）**
   - `_set_up_drafter()`：`method == "extract_hidden_states"` 时
     `use_aux_hidden_state_outputs = True`
   - `propose_draft_token_ids()`：切 `aux_hidden_states` 后调用 `drafter.propose(...)`
   - ACL graph：`initialize_cudagraph_keys(...)` 覆盖 extract
   - KV：
     - `CacheOnlyAttentionLayer` → `HiddenStateCacheSpec`
     - 单 tensor 分配 / reshape（不要按普通 K/V 拆）
     - hybrid 共享 pool（如 Qwen3.5）用 `page_size_padded` 等特殊处理

4. **平台/配置补丁**
   - `is_drafter_moe_model()`：extract 强制视为非 MoE draft，避免错误 DP all_reduce
   - `patch_mamba_config`：extract + hybrid 时不要强制 `mamba_cache_mode="align"`
     （否则会走到 Ascend 编不过的 GPU Triton 路径）

5. **明确未做 / 限制**
   - PP：v1 也没有完整支持（aux 层配置在 PP>1 时基本走不通）
   - padding slot `-1` 的 Triton mask：主要是本 PR 的 **MRv2** 防护，不是 v1 Proposer 主体

一句话概括 MRv1 适配模式：

> **上游负责 extract 语义；Ascend 只修 NPU 运行时差异（ACL graph / DP-SP / discard API / hybrid KV）。**

---

## 4. 如果上游把 MRv2 做好了，Ascend 需要做什么适配

对应关系应是：**把现在的“完整自研 Speculator”收敛成“像 MRv1 一样的薄适配”。**

### 4.1 目标形态：对齐 MRv1 适配模式

| 维度 | 今天 Ascend MRv2（本 PR） | 上游 MRv2 落地后的目标 |
|---|---|---|
| Speculator | 完整自研 `AscendExtractHiddenStatesSpeculator` | 继承上游 `ExtractHiddenStatesSpeculator`，只覆盖 NPU 差异 |
| 工厂 | Ascend `init_speculator()` 自己创建 + 需 `runner=` | 继续替换为 Ascend 子类；尽量不再强依赖临时 monkeypatch |
| init 绕过 | `upstream_extract_hidden_states_init_wrapper` 必需 | **删除** |
| propose / load_model / cache write | Ascend 自己实现大量编排 | 默认 `super()`，只保留必要 override |
| ACL graph / DP-SP / padding | Ascend 全量实现 | 审查后保留最小 NPU 补丁 |
| `attn_utils` / HiddenStateCacheSpec | Ascend 补 discovery + reshape | 上游已有则删除重复，只留 NPU pool/dtype 差异 |

### 4.2 建议保留的 Ascend 适配（大概率仍需要）

这些是 MRv1 已经证明“上游 GPU 路径不够用”的点，MRv2 仍要逐项核对。
总表如下；其后按条目展开讲清楚“为什么不够用 / 例子 / MRv2 怎么核”。

| 适配点 | 为什么可能还要留 | 建议做法 |
|---|---|---|
| ACL graph `dummy_run` / capture | NPU graph 与 CUDA graph 不同 | 覆盖 graph 相关方法，或接 Ascend graph manager |
| DP sync 形状 / `is_draft_model=True` | 避免 collective 形状或 MoE all_reduce 误触发 | 若上游仍用 GPU 专用 DP API，继续走 runner sync |
| SP padding | `num_tokens % TP == 0` | 保留 pad-before-sync |
| discard / sampled token API | Ascend runner 状态字段不同 | 覆盖 token 准备相关 helper |
| hybrid KV / HiddenStateCacheSpec reshape | 共享 pool、单 tensor、page padding | 上游若未覆盖 hybrid，保留 `attn_utils` 分支 |
| mamba/hybrid 配置补丁 | 防止误走 GPU Triton | 保留或迁移到 MRv2 配置路径 |
| PP 限制 | 若上游仍不传 aux HS 跨 PP | 继续显式报错；上游支持后再验证 |

#### 4.2.1 ACL graph `dummy_run` / capture

**上游 GPU 路径假设什么**

- CUDA Graph capture 的函数签名、runtime mode、warmup/dummy 路径是 GPU 约定。
- upstream Proposer 的 `dummy_run(...)` 参数名/语义围绕 CUDA graph。

**Ascend 为什么不够用**

- NPU 用的是 **ACL graph**，不是 CUDA graph。
- capture 时参数叫 `aclgraph_runtime_mode`，并且 idle DP rank 也必须走同一套 dummy 路径。
- 直接调用上游 GPU `dummy_run`，签名对不上，或 idle rank 漏 sync 会挂死。

**例子**

假设 DP=2 做 ACL graph capture：

- Rank0（busy）：propose 路径会调用 `_sync_metadata_across_dp(..., is_draft_model=True)`
- Rank1（idle）：如果只跑上游 GPU dummy，**不发同样的 DP sync**
- 结果：DP cpu_group 上 collective 次数不一致 → **死锁**

Ascend 修法（MRv1）：

```python
def dummy_run(self, num_tokens, ..., aclgraph_runtime_mode=None, ...):
    (num_tokens, num_tokens_across_dp, _) = self.runner._sync_metadata_across_dp(
        num_tokens, is_draft_model=True
    )
    with set_forward_context(..., cudagraph_runtime_mode=aclgraph_runtime_mode or CUDAGraphMode.NONE):
        self.model(hidden_states=self.hidden_states[:num_tokens])
```

**MRv2 怎么核**

- 上游若提供 Speculator.capture / dummy，先看签名是否 NPU 可用。
- 不可用则继续覆盖 `_dummy_run` / `capture`，并保留 idle-rank DP sync。
- 回归：开启 ACL graph 的单卡 + DP extract E2E。

#### 4.2.2 DP sync 形状 / `is_draft_model=True`

**上游 GPU 路径假设什么**

- draft 侧可调用 `coordinate_batch_across_dp`，同步 tensor 形状偏 GPU 约定（历史上是 `[4, dp]` 一类）。
- 若 draft 被判定为 MoE，还可能额外 all_reduce。

**Ascend 为什么不够用**

1. **形状不一致**  
   Ascend 主 runner 的 `_sync_metadata_across_dp` 使用另一套形状（如 `[2, dp]`）。  
   同一 `cpu_group` 上，一边 post `[4, dp]`、一边 post `[2, dp]`，gloo 会报类似：

   ```text
   op.preamble.length 8 vs 4
   ```

2. **误判 MoE draft**  
   extract 的 `hf_config` 常从 target 拷贝。若 target 是 MoE（如 DeepSeek / Qwen-MoE），
   朴素扫描 `expert` 字段会把 extract drafter 误判成 MoE，于是 busy rank 多发 all_reduce，
   idle rank 没有对应操作 → 再死锁。

**例子 A：形状**

- DP=2，主 forward 已用 runner sync（`[2,2]`）
- extract 若改走上游 `coordinate_batch_across_dp`（`[4,2]`）
- 同 group 上长度对不上 → gloo 失败

**例子 B：MoE 误判**

- target=`Qwen3-MoE`，`method=extract_hidden_states`
- extract 实际只是 cache-only attention，**根本没有 expert**
- 但 draft hf_config 里残留 `num_experts` 等字段
- 不短路的话：`is_drafter_moe_model()==True` → 多余 collective

Ascend 修法：

```python
# proposer: 强制走 runner sync
self.runner._sync_metadata_across_dp(num_tokens=..., is_draft_model=True, ...)

# utils: extract 永不当 MoE draft
if speculative_config.method == "extract_hidden_states":
    _IS_DRAFTER_MOE_MODEL = False
```

**MRv2 怎么核**

- 上游 Speculator 是否仍调用 GPU 专用 DP API？
- Ascend runner 是否仍要求统一 `_sync_metadata_across_dp`？
- 是则保留 override + `is_draft_model=True`，并保留 extract 非 MoE 短路。

#### 4.2.3 SP padding（先 pad，再 DP sync）

**上游 GPU 路径假设什么**

- token 数直接进 draft forward / graph dispatch，不一定先按 TP 对齐。

**Ascend 为什么不够用**

- Sequence Parallel / TP reduce_scatter 要求：

  ```text
  num_tokens % tensor_parallel_size == 0
  ```

- 若先 DP sync 再 pad，各 rank 可能先对一个“未对齐”的数达成一致，随后主路径再 pad，
  反而和已同步值冲突；或 reduce_scatter 直接 shape error。

**例子**

- TP=4，本 step 真实 tokens=`6`
- 正确顺序：

  ```text
  6 --SP pad--> 8 --DP sync--> 各 rank 以 8 对齐 --cache-only forward-->
  ```

- 错误顺序：

  ```text
  6 --DP sync--> 大家都同意 6 --forward--> reduce_scatter 需要 %4==0 → 失败
  ```

Ascend 修法：

```python
num_tokens = self.runner._pad_for_sequence_parallelism(num_tokens)  # 6 -> 8
# 然后再 cudagraph dispatch / DP sync
```

单测语义：`test_determine_batch_execution_and_padding_dp1_sp_pads_and_skips_sync`、
`..._dp2_keeps_tp_aligned_for_main_forward`。

**MRv2 怎么核**

- 上游 Speculator 是否保证 SP/TP 对齐？
- 若否，继续在 Ascend Speculator 入口保留 pad-before-sync。

#### 4.2.4 discard / sampled token API

**上游 GPU 路径假设什么**

- `prepare_next_token_ids_padded` 使用 **boolean discard mask**（长度 = batch）。

**Ascend 为什么不够用**

- Ascend MRv1 runner 维护的是：

  ```text
  discard_request_indices: Tensor[int64]   # 被丢弃请求的下标
  num_discarded_requests: int             # 有效下标个数
  ```

- 直接把上游“吃 boolean mask”的函数接过来，参数对不上。

**例子**

batch 4 个请求，第 4 个被 discard：

```text
sampled_token_ids = [[10], [20], [-1], [40]]   # 第3个无效，第4个 discard
discard_request_indices = [3]
num_discarded_requests = 1
```

Ascend 先把 indices 扩成 mask，再决定用 sampled 还是 backup：

```python
discard_mask = zeros(num_reqs, dtype=bool)
discard_mask[discard_request_indices[:num_discarded_requests]] = True
use_sampled = is_valid & ~discard_mask
next_token_ids = where(use_sampled, sampled, backup_tokens)
```

期望：

- req0/1：用 sampled
- req2：token 无效 → backup
- req3：discard → backup

**MRv2 怎么核**

- 看 MRv2 Ascend runner / InputBatch 是否仍用 indices/count。
- 若上游 Speculator 仍假设 boolean mask，就保留 Ascend override 或做薄适配层。

#### 4.2.5 hybrid KV / `HiddenStateCacheSpec` reshape

**上游 GPU 路径假设什么**

- 普通 attention KV 常按 **K/V 两个 tensor**（或固定 layout）分配、reshape。
- hybrid（attention + mamba）共享 pool 时，GPU 侧有自己的 page 对齐假设。

**Ascend 为什么不够用**

extract 的 cache-only 层是 **单 tensor hidden-state cache**，不能走“拆成 K/V”的普通分支。
hybrid 模型（如 Qwen3.5）还可能：

- 与 attention 共享 KV pool
- 需要 `page_size_padded` 做 strided view
- 必须保持 `HiddenStateCacheSpec` 类型，避免被降级成普通 MLA/AttentionSpec

**例子**

Qwen3.5 hybrid + extract：

```text
pool A:
  - full attention layers (K/V)
  - cache_only_layers.0  (HiddenStateCacheSpec, 单 tensor)
```

错误路径：

```python
raw_k, raw_v = kv_cache_raw_tensors[layer]   # cache-only 根本不是 tuple
```

正确路径：

```python
if is_hidden_state_cache_spec(spec) or "cache_only_layers" in layer_name:
    raw = kv_cache_raw_tensors[layer]          # 单 tensor
    # 必要时按 page_size_padded 做 view，而不是 K/V split
```

**MRv2 怎么核**

- 上游 MRv2 attn/KV 是否已原生识别 `HiddenStateCacheSpec` 与 shared pool。
- 若只覆盖 dense、未覆盖 hybrid shared pool，Ascend `attn_utils.py` 分支必须保留。
- 回归：Qwen3.5 / hybrid + extract E2E，检查 connector 导出 shape。

#### 4.2.6 mamba/hybrid 配置补丁

**上游 GPU 路径假设什么**

- 开 KV transfer + hybrid 时，常强制 `mamba_cache_mode="align"`，以便跨实例迁移 mamba block。
- align 模式会走到 GPU fused postprocess Triton kernel。

**Ascend 为什么不够用**

- extract 用的 `ExampleHiddenStatesConnector` **只导出 hidden-state cache-only 层**，
  并不迁移 mamba KV blocks。
- 若仍强制 `align`，hybrid 模型会进入 Ascend Triton **编不过** 的 GPU kernel 路径。

**例子**

```text
model = Qwen3.5-hybrid
method = extract_hidden_states
kv_connector = ExampleHiddenStatesConnector / AscendStoreConnector
```

错误结果：

```text
强制 mamba_cache_mode=align
  → 调用 vLLM GPU Triton postprocess
  → Ascend Triton backend compile fail
```

Ascend 修法（`patch_mamba_config.py`）：

```python
is_extract_hidden_states = (spec_config.method == "extract_hidden_states")
if using_kv_store_with_hybrid and not is_extract_hidden_states:
    cache_config.mamba_cache_mode = "align"
# extract 时保持上游推导值（如 none）
```

**MRv2 怎么核**

- 上游 MRv2 是否仍对 hybrid+KV transfer 强制 align。
- 若是，继续保留 extract 例外；或把例外迁到 MRv2 配置更新路径。

#### 4.2.7 PP 限制

**上游 GPU 路径假设什么**

- speculative / Eagle3 的 aux hidden states，在 PP 下通常只有 **最后一级 PP rank**
  做 sampling / drafting。
- 中间层 hidden states 若产生在非 last PP rank，需要跨 PP 传到 last rank。

**Ascend / 上游为什么现在不够用**

- extract 依赖多份 `aux_hidden_states`。
- 当前 MRv1/MRv2 都没有把“非 last PP rank 的 aux HS”完整送到 last rank。
- MRv1 甚至在 PP>1 时，aux 层配置容易被 Eagle3 专用逻辑挡住；MRv2 Ascend 选择 **启动即明确报错**。

**例子**

```text
PP=2, layers=32
layer_ids = [1, 2, 30, 31]
```

- Rank0 持有 layer 1/2 的 activation
- Rank1 持有 layer 30/31，并负责 sample + extract propose
- 若没有跨 PP 传输，Rank1 的 `aux_hidden_states` 缺 layer 1/2
  → extract 无法正确堆叠/写入 cache

Ascend MRv2 现状：

```python
if self.speculative_config.uses_extract_hidden_states() and self.use_pp:
    raise ValueError(
        "extract_hidden_states with pipeline parallelism is not supported by model runner v2."
    )
```

**MRv2 怎么核**

- 上游若仍不支持跨 PP 传 aux HS：继续保留显式 `ValueError`。
- 上游若支持：再补 NPU PP 传输/测试，不能只删报错。

### 4.3 建议删除或收敛的临时逻辑

| 当前逻辑 | 上游 MRv2 支持后 |
|---|---|
| 完整自建 Speculator 编排 | 删除重复，改为继承上游 |
| `upstream_extract_hidden_states_init_wrapper` | 删除 |
| 强制 `runner=` 才能创建 Speculator | 若上游接口不需要，去掉；若 DP/SP 仍需 runner，可保留但文档化 |
| 重复设置 `use_aux_hidden_state_outputs` | 上游已设则删 |
| 与上游重复的 `HiddenStateCacheSpec` 发现/分配 | 收敛到上游实现 |

### 4.4 推荐落地步骤（按顺序）

1. **确认上游 MRv2 extract 可跑**
   - `init_speculator()` 已注册
   - GPU UT/E2E 覆盖 propose + cache +（如有）connector
2. **改继承**
   - `AscendExtractHiddenStatesSpeculator(ExtractHiddenStatesSpeculator)`
3. **逐项搬迁 MRv1 经验**
   - 优先移植：DP/SP sync、ACL graph dummy、discard API、hybrid KV
4. **删临时绕过**
   - 去掉 init monkeypatch，按 Eagle 模式 `del self.speculator` 后重建 Ascend 子类
5. **回归**
   - 单卡 E2E、ACL graph、DP、hybrid（如 Qwen3.5）、padding slot 不污染 cache

### 4.5 一句话对照

| Runner | Ascend 适配模式 |
|---|---|
| **MRv1（已做）** | 上游 Proposer 可用 → Ascend **薄子类 + runner/KV 补丁** |
| **MRv2（上游未做 / 本 PR）** | 上游 Speculator 缺失 → Ascend **先完整自研补齐** |
| **MRv2（上游做好后）** | 上游 Speculator 可用 → Ascend **收回自研，回到薄适配** |

---

## 5. 当前 Ascend 为什么要自己做（本 PR 过渡方案）

因为上游 MRv2 工厂对 extract 直接失败，Ascend 不能“等上游再启用”。本 PR 的策略是：

1. **原生实现** `AscendExtractHiddenStatesSpeculator`
   - 不继承 / 不包装 / 不调用 v1 `*Proposer`
   - 自己负责 load_model、buffer、metadata、cache-only forward、DP/SP、ACL graph
2. **复用上游共享 primitives**
   - `ExtractHiddenStatesModel`
   - `CacheOnlyAttentionLayer`
   - `HiddenStateCacheSpec`
3. **绕过上游工厂缺口**
   - `upstream_extract_hidden_states_init_wrapper()` 在 `super().__init__()` 期间临时把
     upstream `init_speculator` 置空，避免 `NotImplementedError`
   - 父类初始化完成后，再调用 Ascend 自己的 `init_speculator(..., runner=self)`

关键代码位置：

```text
vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py
vllm_ascend/worker/v2/spec_decode/__init__.py
vllm_ascend/worker/v2/model_runner.py
vllm_ascend/worker/v2/attn_utils.py
```

注意：路径里出现 `vllm.v1.worker.gpu.*` 是上游 **MRv2 命名空间**，不等于“继续用 v1 Proposer”。

---

## 6. 过渡期决策表

| 上游状态 | Ascend 策略 |
|---|---|
| 仍 `NotImplementedError`（当前） | 保持本 PR 原生 Speculator + init wrapper |
| 上游合并但接口不稳 / 测试不全 | 继续以 Ascend 原生实现为主，开始做接口对照，不急删 |
| 上游可跑且接口稳定 | 改为继承上游 Speculator，删除 wrapper 与重复编排 |
| 上游实现已覆盖全部 NPU 所需行为 | 进一步评估是否可直接使用上游 Speculator，不做替换 |

---

## 7. 快速自检清单（给后续 PR）

上游已支持后，开收敛 PR 前先打勾：

- [ ] 上游 `init_speculator()` 已支持 `extract_hidden_states`
- [ ] 上游存在 MRv2 `ExtractHiddenStatesSpeculator`（或等价类）
- [ ] Ascend Speculator 改为继承上游实现
- [ ] 删除 `upstream_extract_hidden_states_init_wrapper`
- [ ] factory 仅保留 NPU 必要替换
- [ ] `attn_utils` / KV pool 特殊分支已按上游能力收敛
- [ ] UT / E2E 全绿，尤其 padding slot 与 aux hidden states
- [ ] 设计文档更新为“已对接上游 MRv2”

---

## 8. 相关文件索引

### Ascend MRv1 适配

- `vllm_ascend/spec_decode/extract_hidden_states_proposer.py`
- `vllm_ascend/spec_decode/__init__.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/utils.py`
- `vllm_ascend/patch/platform/patch_mamba_config.py`
- `tests/ut/spec_decode/test_extract_hidden_states_proposer.py`

### Ascend MRv2（本 PR）

- `vllm_ascend/worker/v2/spec_decode/extract_hidden_states/speculator.py`
- `vllm_ascend/worker/v2/spec_decode/__init__.py`
- `vllm_ascend/worker/v2/model_runner.py`
- `vllm_ascend/worker/v2/attn_utils.py`
- `tests/ut/spec_decode/test_extract_hidden_states_speculator_v2.py`
- `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py`

### 上游对照

- `vllm/v1/worker/gpu/spec_decode/__init__.py`（MRv2 工厂，当前不支持 extract）
- `vllm/v1/worker/gpu_model_runner.py`（MRv1，已支持）
- `vllm/v1/spec_decode/extract_hidden_states.py`（MRv1 Proposer）
- `vllm/model_executor/models/extract_hidden_states.py`（共享模型）
- `docs/features/speculative_decoding/extract_hidden_states.md`
