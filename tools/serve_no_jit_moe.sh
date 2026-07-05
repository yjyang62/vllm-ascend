#!/usr/bin/env bash
set -euo pipefail

# Production serve wrapper that forbids runtime JIT compile for mega_moe.
# It requires prebuilt npu_mega_moe.so to already exist in TORCH_EXTENSIONS_DIR.
#
# Example:
#   export TORCH_EXTENSIONS_DIR=/mnt/share/torch_extensions
#   bash tools/serve_no_jit_moe.sh -- \
#     /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
#     --host 0.0.0.0 --port 8008 --served-model-name auto \
#     --data-parallel-size 8 --tensor-parallel-size 1 \
#     --enable-expert-parallel --enforce-eager

if [[ "${1:-}" != "--" ]]; then
  echo "Usage: $0 -- <vllm serve args>" >&2
  exit 1
fi
shift

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/share/torch_extensions}"

if [[ ! -d "${TORCH_EXTENSIONS_DIR}" ]]; then
  echo "[serve-no-jit] TORCH_EXTENSIONS_DIR does not exist: ${TORCH_EXTENSIONS_DIR}" >&2
  echo "[serve-no-jit] run tools/prebuild_moe_runtime.sh first." >&2
  exit 1
fi

if ! rg -n "npu_mega_moe\\.so" "${TORCH_EXTENSIONS_DIR}" >/dev/null 2>&1; then
  echo "[serve-no-jit] npu_mega_moe.so not found under ${TORCH_EXTENSIONS_DIR}" >&2
  echo "[serve-no-jit] run tools/prebuild_moe_runtime.sh first, then retry." >&2
  exit 1
fi

echo "[serve-no-jit] using prebuilt extension cache: ${TORCH_EXTENSIONS_DIR}"
echo "[serve-no-jit] starting vllm serve (runtime JIT is not expected)."

exec vllm serve "$@"
