#!/usr/bin/env bash
set -euo pipefail

# Prebuild runtime JIT extensions (e.g. npu_mega_moe.so) before production serve.
# The generated artifacts are persisted in TORCH_EXTENSIONS_DIR.
#
# Usage:
#   bash tools/prebuild_moe_runtime.sh \
#     --model "/mnt/share/weight/DeepSeek-V4-Flash-BF16" \
#     --served-model-name auto \
#     --port 18008 \
#     --tensor-parallel-size 1 \
#     --data-parallel-size 1
#
# Required environment:
#   TORCH_EXTENSIONS_DIR: persistent directory for torch cpp extension cache.
#                         default: /mnt/share/torch_extensions

MODEL_PATH=""
SERVED_MODEL_NAME="auto"
PREBUILD_PORT="18008"
TP_SIZE="1"
DP_SIZE="1"
MAX_MODEL_LEN="2048"
MAX_NUM_SEQS="1"
GPU_MEM_UTIL="0.8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      PREBUILD_PORT="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TP_SIZE="$2"
      shift 2
      ;;
    --data-parallel-size)
      DP_SIZE="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "--model is required" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/share/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

echo "[prebuild] TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[prebuild] starting temporary vllm serve on port ${PREBUILD_PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PREBUILD_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code &

SERVER_PID=$!

READY=0
for _ in $(seq 1 180); do
  if curl -s "http://127.0.0.1:${PREBUILD_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "[prebuild] server did not become ready in time" >&2
  exit 1
fi

echo "[prebuild] sending warmup request to trigger JIT compile"
curl -s -X POST "http://127.0.0.1:${PREBUILD_PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SERVED_MODEL_NAME}\",
    \"prompt\": \"warmup\",
    \"max_tokens\": 1,
    \"temperature\": 0,
    \"top_p\": 1
  }" >/dev/null

echo "[prebuild] stopping temporary server"
cleanup
trap - EXIT

echo "[prebuild] done. Cached extensions are under: ${TORCH_EXTENSIONS_DIR}"
echo "[prebuild] verify artifact:"
echo "  ls -l \"${TORCH_EXTENSIONS_DIR}\" | rg \"mega_moe|npu_mega_moe\" || true"
