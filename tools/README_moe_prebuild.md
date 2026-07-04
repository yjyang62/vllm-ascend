# MoE Runtime Prebuild Workflow (Avoid compile during production serve)

This repo provides `tools/prebuild_moe_runtime.sh` to trigger JIT build for
MoE runtime extensions (for example `npu_mega_moe.so`) **before** you start the
production server.

The key point is to use a persistent cache path:

```bash
export TORCH_EXTENSIONS_DIR=/mnt/share/torch_extensions
```

If this path is reused by the same Python / torch / torch_npu / CANN stack,
production `vllm serve` will load cached artifacts instead of compiling again.

## Step 1: Prebuild once

```bash
export TORCH_EXTENSIONS_DIR=/mnt/share/torch_extensions

bash tools/prebuild_moe_runtime.sh \
  --model /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
  --served-model-name auto \
  --port 18008 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1
```

Notes:
- The script starts a temporary local server, sends a warmup request to force
  JIT compile, then exits.
- Artifacts are written into `TORCH_EXTENSIONS_DIR`.

## Step 2: Start production server with the same cache path

```bash
export TORCH_EXTENSIONS_DIR=/mnt/share/torch_extensions

vllm serve /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
  --host 0.0.0.0 \
  --max_model_len 32768 \
  --served-model-name auto \
  --gpu-memory-utilization 0.99 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-num-seqs 48 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --enable-expert-parallel \
  --port 8008 \
  --safetensors-load-strategy prefetch \
  --async-scheduling \
  --enforce-eager
```

## Why compile can still happen

JIT may recompile if any of the following changes:
- Python version
- torch / torch_npu version
- CANN toolkit version
- compile flags
- `TORCH_EXTENSIONS_DIR` path

So prebuild and production startup must run with the same software stack and
the same cache directory.
