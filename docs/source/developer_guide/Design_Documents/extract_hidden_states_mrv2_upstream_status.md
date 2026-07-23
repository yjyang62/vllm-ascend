# Model Runner v2 中 `extract_hidden_states` 支持情况与上游对接指南

本文说明两件事：

1. 上游 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 Model Runner v2（简称 MRv2）对 `extract_hidden_states` 的支持现状。
2. 如果未来上游 MRv2 原生支持了该特性，`vllm-ascend` 应该如何收敛改造。

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

## 3. 当前 Ascend 为什么要自己做

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

## 4. 如果上游 MRv2 支持了，Ascend 应该怎么改

上游一旦合并 MRv2 extract，通常会出现下面一类结构（名称可能略有差异）：

```text
vllm/v1/worker/gpu/spec_decode/
├── extract_hidden_states/
│   └── speculator.py          # ExtractHiddenStatesSpeculator
└── __init__.py                # init_speculator() 增加 extract 分支
```

并在 MRv2 runner 中正式设置：

- `use_aux_hidden_state_outputs = True`
- propose 时传入 `aux_hidden_states`
- hidden-state cache / connector 生命周期与其它 Speculator 对齐

届时 Ascend 建议按“能删尽删、只保留 NPU 差异”收敛。

### 4.1 改造总原则

| 原则 | 说明 |
|---|---|
| 优先继承上游 Speculator | 与 Eagle / DFlash / DSpark 一样，做 thin Ascend wrapper |
| 只保留 NPU 必要差异 | ACL graph、NPU kernel、padding slot 处理、Ascend attention metadata |
| 删除临时绕过逻辑 | 上游工厂可用后，不再需要 monkeypatch `init_speculator` |
| 不要回退到 v1 Proposer | 即使上游也保留 MRv1 Proposer，Ascend MRv2 仍应走 Speculator 路径 |
| 保持行为兼容 | 配置字段、connector 协议、测试语义尽量不变 |

### 4.2 建议改造步骤

#### Step A：确认上游真正可用

先在上游 `main` 验证：

1. `init_speculator()` 对 `extract_hidden_states` 返回 Speculator，而不是抛错。
2. GPU MRv2 e2e / UT 覆盖 extract +（如有）KV connector。
3. 上游 Speculator 接口稳定：`load_model` / `set_attn` / `propose` / `capture` /
   `init_cudagraph_manager` 等与现有 DraftModelSpeculator 家族一致，或有明确文档。

若上游只是“部分骨架、尚未可跑”，不要急着删除 Ascend 原生实现。

#### Step B：把 Ascend Speculator 改成继承上游实现

目标形态（示意）：

```python
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states.speculator import (
    ExtractHiddenStatesSpeculator,
)

class AscendExtractHiddenStatesSpeculator(ExtractHiddenStatesSpeculator):
    """Only override NPU-specific pieces."""

    def propose(self, ...):
        # 默认直接 super().propose(...)
        # 仅在 metadata / graph / padding 等 NPU 差异处覆写
        return super().propose(...)
```

优先审查并决定去留的 Ascend 专有逻辑：

| Ascend 现状 | 上游支持后建议 |
|---|---|
| 完整自建 Speculator | 改为继承上游，删除重复编排 |
| Triton / NPU padding slot mask（`slot >= 0`） | 若上游已正确处理 padding，可删；否则保留 NPU override |
| ACL graph capture / dispatcher | 保留 Ascend graph 适配 |
| Ascend attention metadata 构造 | 保留，对接上游 `set_attn` / metadata builder |
| DP/SP token 对齐、dummy batch | 若上游已抽象，尽量复用；否则保留最小补丁 |
| `HiddenStateCacheSpec` 共享 pool reshape（`attn_utils.py`） | 若上游 MRv2 attn/KV pool 已原生支持，收敛到上游路径 |

#### Step C：简化 factory

`vllm_ascend/worker/v2/spec_decode/__init__.py` 建议变成：

```python
if speculative_config.uses_extract_hidden_states():
    from vllm_ascend.worker.v2.spec_decode.extract_hidden_states import (
        AscendExtractHiddenStatesSpeculator,
    )
    return AscendExtractHiddenStatesSpeculator(vllm_config, device)
```

重点变化：

- 尽量不再强制 `runner=` 参数（若上游 Speculator 不需要 runner 反向依赖）。
- 若上游工厂已经足够，甚至可进一步评估：仅在有 NPU override 时才替换 Speculator；否则直接用上游。

#### Step D：删除临时 init wrapper

`vllm_ascend/worker/v2/model_runner.py` 中：

```python
upstream_extract_hidden_states_init_wrapper(...)
```

以及对应 contextmanager，应删除或缩成空操作。

父类 `super().__init__()` 期间应能正常创建 upstream Speculator；Ascend 再按现有 Eagle 模式：

1. `del self.speculator`
2. 用 Ascend `init_speculator()` 重建 NPU 版本

同时复核：

- `use_aux_hidden_state_outputs = True` 是否已由上游设置；若是，Ascend 可去掉重复赋值。
- PP 限制：若上游仍不支持 PP，保留明确报错；若上游支持，再跟进验证 NPU PP。

#### Step E：收敛 `attn_utils` / cache 分配

当前 Ascend 为 extract 做了 `HiddenStateCacheSpec` 与共享 pool 特殊处理。上游 MRv2 支持后：

1. 对比上游 `attn_utils` / KV cache 分配是否已覆盖 hidden-state cache。
2. 若上游已支持，删除 Ascend 重复分支，只保留 NPU dtype / device / pool 差异。
3. 若上游实现与 Ascend connector / hybrid 行为不一致，先补测试再删代码。

#### Step F：测试与文档同步

至少更新：

- UT：`tests/ut/spec_decode/test_extract_hidden_states_speculator_v2.py`
- UT：`tests/ut/worker/test_attn_utils_v2.py`
- E2E：`tests/e2e/.../test_extract_hidden_states.py`
- 设计文档：本文 + `extract_hidden_states_model_runner_v2.md`

测试重点：

1. 工厂不再依赖 monkeypatch。
2. propose 输出语义不变（尤其 padding slot 不污染 cache）。
3. aux hidden states 层配置仍生效。
4. ACL graph / DP 场景不回归。
5. 若使用 KV connector，导出结果与升级前一致。

### 4.3 建议的目标架构

```text
上游 MRv2
└── ExtractHiddenStatesSpeculator          # 通用编排

Ascend MRv2
└── AscendExtractHiddenStatesSpeculator    # 仅 NPU 差异
        ├── ACL graph / NPU kernel overrides
        ├── Ascend attention metadata
        └── 必要的 DP/SP 或 padding 补丁

共享 primitives（继续直接复用上游）
├── ExtractHiddenStatesModel
├── CacheOnlyAttentionLayer
└── HiddenStateCacheSpec
```

### 4.4 不建议的改法

1. **不要**在上游已有 MRv2 Speculator 后，继续长期维护一份完整分叉实现。
2. **不要**把 Ascend MRv2 重新接到 v1 `ExtractHiddenStatesProposer`。
3. **不要**在未验证上游 padding / aux / connector 行为前，盲目删除 NPU 防护逻辑。
4. **不要**把 MTP / Eagle 逻辑与 extract 耦在一起迁移；extract 不是 MTP。

---

## 5. 过渡期决策表

| 上游状态 | Ascend 策略 |
|---|---|
| 仍 `NotImplementedError`（当前） | 保持本 PR 原生 Speculator + init wrapper |
| 上游合并但接口不稳 / 测试不全 | 继续以 Ascend 原生实现为主，开始做接口对照，不急删 |
| 上游可跑且接口稳定 | 改为继承上游 Speculator，删除 wrapper 与重复编排 |
| 上游实现已覆盖全部 NPU 所需行为 | 进一步评估是否可直接使用上游 Speculator，不做替换 |

---

## 6. 快速自检清单（给后续 PR）

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

## 7. 相关文件索引

### 当前 Ascend（本 PR）

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
