# DeepSeek-V4 BF16 SparseFlashMla 适配设计

## 1. 背景

DeepSeek-V4 使用稀疏注意力、滑动窗口 KV Cache 和分层压缩 KV Cache。Ascend 950
上的原有实现主要面向量化 KV Cache，通过 vLLM Ascend 自带的量化稀疏注意力接口完成
metadata 生成、KV 写入和 attention 计算。

BF16 权重版本有以下差异：

- Sparse attention 使用 BF16 KV Cache，而不是带 scale 的 FP8 KV Cache。
- BF16 SparseFlashMla 算子由 CANN `ops-transformer` 项目提供。
- BF16 cache 使用 `PA_BBND` 布局和二维 block/offset slot mapping。
- BF16 checkpoint 的普通线性层没有 `weight_scale`。
- ACLGraph 捕获阶段不允许 `.item()` 等 NPU 到 CPU 的同步操作。

本适配的目标是让 vLLM Ascend 只负责路由和参数适配，直接调用已安装的
`cann_ops_transformer` 算子包，不复制 SparseFlashMla 的 host、tiling 或 kernel
实现。

## 2. 设计目标与非目标

### 2.1 设计目标

- Ascend 950 上的 DeepSeek-V4 BF16 KV Cache 使用 SparseFlashMla。
- 保留已有 FP8 KV Cache 路径，不改变其接口和行为。
- 根据最终解析出的 KV Cache dtype 自动选择实现，不增加模型专用开关。
- eager 和 ACLGraph 使用相同的 attention 计算路径。
- 支持 prefill、decode、压缩比 1/4/128 和 context parallel。
- BF16 输出投影不依赖量化参数。
- 算子版本由 `ops-transformer` 和 CANN 管理。

### 2.2 非目标

- 不在 vLLM Ascend 中维护 SparseFlashMla kernel。
- 不替换 DSV4 indexer 的量化 cache。
- 不改变量化 checkpoint 的输出投影快速路径。
- 不在 attention 热路径中检查或打印输出 Tensor。

## 3. 总体调用链

```text
vllm_config.cache_config.cache_dtype
                 |
                 v
      kv_cache_dtype_str_to_dtype
                 |
       +---------+---------+
       |                   |
       v                   v
  torch.bfloat16       FP8/其他 dtype
       |                   |
       v                   v
  SparseFlashMla       原量化稀疏注意力
  PA_BBND cache        PA_ND cache
  block/offset slot    flat slot
       |
       v
torch.ops.cann_ops_transformer.sparse_flash_mla
```

路由的核心原则是使用“实际 KV Cache dtype”，而不是模型名称、环境变量或某个局部
cache spec 的 dtype。

## 4. 外部算子调用

### 4.1 Python 适配层

文件：`vllm_ascend/ops/sparse_flash_mla.py`

适配层首先导入 `cann_ops_transformer`：

```python
import cann_ops_transformer
```

这个 import 的作用不是在运行时编译 NPU kernel，而是加载 PyTorch 扩展并注册
`torch.ops.cann_ops_transformer` namespace。随后获取两个接口：

```python
attention_op = torch.ops.cann_ops_transformer.sparse_flash_mla
metadata_op = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata
```

算子查找结果通过 `lru_cache` 缓存。若 Python 包不存在，或安装版本未注册两个接口，
适配层会抛出包含安装建议的明确异常。

### 4.2 为什么不复制算子

SparseFlashMla 的可执行能力不仅包含 Python/C++ 符号，还包含：

- OpDef 和 `AddConfig("ascend950", ...)`。
- `binary_info_config.json`。
- Ascend 950 kernel binary。
- ACLNN API 动态库。
- 与当前 CANN 版本匹配的 tiling 和 runtime 描述。

如果把源码复制到 vLLM Ascend，就需要在构建 vLLM Ascend 时重新编译并维护这些内容，
同时增加 CANN 版本不匹配的风险。直接依赖 `ops-transformer` custom OPP 可以保持职责
边界：

```text
vLLM Ascend       参数、缓存和执行路由
ops-transformer   算子定义、tiling 和 kernel
CANN              runtime 和设备执行
```

## 5. 参数协议适配

DSA 原量化接口和 SparseFlashMla 使用不同的参数名称。

### 5.1 参数重命名

```text
seqused_kv      -> seqused_ori_kv
max_seqlen_kv   -> max_seqlen_ori_kv
```

### 5.2 删除旧接口专用参数

SparseFlashMla 不接受以下原量化接口参数：

```text
kv_quant_mode
tile_size
rope_head_dim
device
```

metadata wrapper 和 attention wrapper 会在调用外部算子前移除这些参数。

### 5.3 压缩 KV 长度

设原始 KV 长度为 `S`，压缩比为 `R`：

```text
seqused_cmp_kv  = S // R
cmp_residual_kv = S % R
```

最大压缩长度为：

```text
max_seqlen_cmp_kv = max_seqlen_ori_kv // R
```

压缩比小于等于 1 时没有 compressed KV，适配层不会生成这些参数。

## 6. KV Cache dtype 路由

### 6.1 统一使用全局解析结果

文件：

- `vllm_ascend/models/deepseek_v4.py`
- `vllm_ascend/attention/dsa_v1.py`
- `vllm_ascend/attention/context_parallel/dsa_cp.py`
- `vllm_ascend/models/layer/attention/layer.py`

dtype 通过下面的标准函数解析：

```python
kv_cache_dtype_str_to_dtype(
    vllm_config.cache_config.cache_dtype,
    vllm_config.model_config,
)
```

不能直接使用某个 grouped `kv_cache_spec.dtype` 决定 SparseFlashMla 路由。DeepSeek-V4
同时存在 attention KV Cache 和 indexer cache，后者在 Ascend 950 上仍可能是 FP8。
如果 metadata builder 恰好读取 indexer spec，就会错误地为 BF16 attention 生成 FP8
metadata。

### 6.2 `cache_dtype_str`

KV Cache spec 同时保存 `torch.dtype` 和字符串形式：

```python
cache_dtype_str = str(dtype).removeprefix("torch.")
```

例如：

```text
torch.bfloat16      -> bfloat16
torch.float8_e4m3fn -> float8_e4m3fn
```

字符串必须描述最终分配的 dtype，而不能直接沿用原始配置。原因是配置可能为 `auto`，
而 Ascend 950 路径会根据模型和硬件将实际 cache dtype 调整为 BF16 或 FP8。

`cache_dtype_str` 用于 KV Cache spec 合并、兼容性检查和后续 worker 传递。

## 7. 算子、布局和 metadata 选择

文件：`vllm_ascend/device/device_op.py`

Ascend 950 的选择规则如下：

| KV Cache dtype | Metadata | Attention | Layout |
| --- | --- | --- | --- |
| BF16 | `sparse_flash_mla_metadata` | `sparse_flash_mla` | `PA_BBND` |
| FP8 | 原量化 metadata op | 原量化 sparse attention op | `PA_ND` |

metadata 和 attention 必须使用相同 dtype 路由。若二者协议不一致，可能在真正执行
attention 时出现 shape、layout 或必选参数错误。

## 8. BF16 Cache 布局和 slot mapping

### 8.1 布局差异

量化路径使用 flat slot id：

```text
flat_slot = block_id * block_size + block_offset
```

BF16 `PA_BBND` cache 使用二维位置：

```text
[block_id, block_offset]
```

因此 BF16 的 slot mapping shape 必须为：

```text
[num_tokens, 2]
```

### 8.2 Cache 写入

BF16 写入逻辑为：

```python
block_indices = slot_mapping[:, 0]
block_offsets = slot_mapping[:, 1]
cache[block_indices, block_offsets] = values
```

不能将 BF16 cache 简单展平后使用 `index_copy_`。当物理 page stride 大于逻辑
`block_size` 时，展平索引会忽略 padding，导致写入错误位置。

FP8 路径保持原有 `kv_compress_epilog`，由融合算子完成量化、压缩和 scatter。

## 9. DSV4 多种 Cache 的边界

DeepSeek-V4 至少涉及以下不同用途的 cache：

- SWA/ori KV Cache。
- Compress-4 KV Cache。
- Compress-128 KV Cache。
- Compressor state cache。
- Indexer key cache 和 scale cache。

BF16 SparseFlashMla 只改变 attention 使用的 ori/cmp KV Cache。Indexer 仍采用原来的
量化流程，因为 indexer 的计算和 SparseFlashMla 的 KV 输入不是同一个协议。

因此不能把“DSV4 使用 BF16”理解为所有 cache 都必须变成 BF16。

## 10. BF16 输出投影修复

### 10.1 问题

SparseFlashMla attention 成功后，执行 DSA `o_proj` 时曾出现：

```text
AttributeError:
'AscendColumnParallelLinear' object has no attribute 'weight_scale'
```

BF16 checkpoint 使用普通线性层：

```text
wo_a.weight       存在
wo_a.weight_scale 不存在
```

原 Ascend 950 路径只判断设备类型，并无条件使用量化 batch matmul，因此错误读取
`weight_scale`。

### 10.2 路由条件

量化输出投影现在要求：

```python
device_is_a5 and weight_scale is not None
```

满足条件时继续使用：

- dynamic MX quant。
- `npu_transpose_quant_batchmatmul`。
- checkpoint 中加载的 `weight_scale`。

BF16 普通线性层则进入非量化 grouped matmul。

### 10.3 Grouped matmul

输入 shape：

```text
[T, G, D]
```

普通 BF16 `wo_a.weight` shape：

```text
[G * R, D]
```

首先恢复 group 维：

```text
[G * R, D] -> [G, R, D]
```

计算公式：

```text
output[t, g, r] = sum(input[t, g, d] * weight[g, r, d], d)
```

它等价于：

```python
torch.einsum("tgd,grd->tgr", input, grouped_weight)
```

实现使用 `torch.matmul`。这样避免普通二维 BF16 权重进入要求特定三维布局的量化
NPU batch matmul。

### 10.4 CVLinear

BF16 DSV4 会关闭当前未验证的 DSA multistream overlap，因此不会进入
`CVLinearWrapper` 的拆分量化路径。`CVLinearWrapper` 保持原有仅依据 quant method
类型判断 W8A8 的行为，不额外依赖 `weight_scale`。

## 11. Context Parallel

Context parallel 需要与普通 DSA 路径保持以下一致：

- 使用全局解析后的 KV Cache dtype。
- BF16 选择 SparseFlashMla metadata 和 attention。
- BF16 使用 `PA_BBND`。
- 输出投影仅在 scale 存在时进入量化路径。
- 非量化输出投影使用相同 grouped matmul。

如果只修改普通 DSA 而遗漏 context parallel，启用 `enable_dsa_cp` 后仍可能复现
layout 或 `weight_scale` 问题。

## 12. ACLGraph 约束

调试过程中曾对 SparseFlashMla 输出执行 finite、NaN 和 Inf 检查。检查需要：

```python
tensor.sum().item()
tensor.max().item()
```

`.item()` 会调用 `LocalScalarDenseNpu`，把设备标量同步到 CPU。ACLGraph 捕获要求
stream 上的操作可异步捕获，因此不能在图内执行这种同步，典型错误为：

```text
LocalScalarDenseNpu.cpp
device error 107027
```

最终实现已完全删除输出检查、日志、状态缓存和相关环境变量。SparseFlashMla wrapper
现在直接返回外部算子结果，eager 和 ACLGraph 不再因为诊断代码产生不同执行路径。

重要原则：

- attention 热路径不要调用 `.item()`。
- 不要打印设备 Tensor 内容。
- 不要在图捕获期间执行 CPU 条件判断所需的设备同步。
- 调试信息应放在图外、离线测试或专用 profiling 工具中。

## 13. 算子安装与容器持久化

### 13.1 Python 扩展和 OPP 的区别

下面的检查只证明 PyTorch 接口已注册：

```python
hasattr(torch.ops.cann_ops_transformer, "sparse_flash_mla")
```

它不能证明 Ascend 950 kernel 已安装。运行还需要 custom OPP 中存在：

- Ascend 950 `binary_info_config.json`。
- `SparseFlashMla` 配置项。
- kernel binary。
- ACLNN API library。

### 13.2 启动前验证

```bash
CUSTOM_OPP=/opt/cann-ops-transformer-smla/vendors/custom_transformer

test -d "$CUSTOM_OPP"

grep -R '"SparseFlashMla"' \
    "$CUSTOM_OPP/op_impl/ai_core/tbe/kernel/config/ascend950"
```

启动 vLLM 的同一个 shell 或 entrypoint 中执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /opt/cann-ops-transformer-smla/vendors/custom_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH%:}"
```

### 13.3 为什么 Pod 重建后可能丢失

`docker start` 会保留原容器 writable layer，但以下操作会创建新文件系统：

- 再次执行 `docker run`。
- Kubernetes Pod 重建。
- Deployment 滚动更新。

若算子只安装在运行中的容器 `/opt` 下，新 Pod 中不会存在。生产环境应将 run 包安装
步骤写入 Dockerfile，或将完整 custom OPP 目录挂载为持久卷。

## 14. 推荐启动方式

先用 eager 验证权重、算子和通信：

```bash
vllm serve /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
    --host 0.0.0.0 \
    --port 8008 \
    --served-model-name auto \
    --max-model-len 2048 \
    --max-num-seqs 48 \
    --data-parallel-size 2 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --safetensors-load-strategy prefetch \
    --async-scheduling \
    --enforce-eager
```

eager 验证通过后，删除 `--enforce-eager` 并增加：

```text
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

不要同时把 `--enforce-eager` 和图模式配置作为最终验证命令，否则无法明确实际执行
模式。

## 15. 常见错误与判断方法

### 15.1 `ninja: no work to do`

这是 PyTorch 扩展构建缓存提示，表示现有产物没有需要重新编译的内容，不是错误。

### 15.2 `EZ1009` 和 `ascend950 verification failed`

优先检查：

- custom OPP 目录是否存在。
- `ASCEND_CUSTOM_OPP_PATH` 是否传递给 vLLM 主进程和 Worker。
- `binary_info_config.json` 是否包含 `SparseFlashMla`。
- 算子包是否使用 `--soc=ascend950` 构建。
- CANN、ops-transformer 和 PyTorch 扩展版本是否匹配。

### 15.3 HCCL 端口绑定失败

默认 HCCL 可能使用 `60000-60031`。多个服务或残留进程会导致冲突。可以为不同任务
配置不同的 `HCCL_IF_BASE_PORT`，并确保所有 rank 使用一致配置。

### 15.4 `weight_scale` 不存在

这说明普通 BF16 线性层误进入了量化输出投影。检查 `o_proj` 的路由条件，而不是重新
安装 SparseFlashMla。

### 15.5 DP Coordinator 启动失败

这是汇总错误，不是根因。需要查找它之前第一个 Worker traceback，常见原因包括：

- OPP 不可见。
- HCCL 端口冲突。
- 可见 NPU 数量小于 `DP * TP`。
- Worker 中的动态库环境与主进程不同。

## 16. 测试覆盖

当前测试重点包括：

- BF16 和 FP8 的算子选择。
- Metadata 参数重命名。
- compressed KV 长度推导。
- BF16 block/offset scatter。
- BF16 grouped matmul 与 einsum 的数值一致性。
- BF16 普通线性层不读取 `weight_scale`。
- 异常 weight rank 的错误处理。

静态检查包括：

```text
Python byte compilation
Ruff lint
Ruff formatting
codespell
typos
markdownlint
shellcheck
repository custom checks
```

真实 A5 验证仍应覆盖：

- eager 启动和首个请求。
- `FULL_DECODE_ONLY` 图模式启动和首个请求。
- prefill 和 decode。
- DP2/TP4/EP 拓扑。
- BF16 KV Cache 的长序列和多并发。

## 17. 最重要的工程结论

1. 接口注册成功不等于设备 kernel 已安装。
2. DSV4 同时存在多种 cache，不能用任意一个局部 spec 的 dtype 决定全局路由。
3. Attention、metadata、layout 和 slot mapping 必须作为一个协议整体切换。
4. BF16 checkpoint 不能假设存在量化 scale。
5. ACLGraph 热路径禁止 `.item()` 和设备到 CPU 的同步。
6. vLLM Ascend 应调用 `ops-transformer`，而不是复制和维护算子源码。
7. custom OPP 必须进入镜像或持久卷，不能依赖临时容器文件系统。
