#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: $0 /path/to/local/model [extra offline_external_launcher.py args...]"
    echo
    echo "Or set MODEL_PATH:"
    echo "  MODEL_PATH=/path/to/local/model $0"
    exit 1
fi
shift || true

PYTHON_BIN="${PYTHON_BIN:-python3}"
TP_SIZE="${TP_SIZE:-1}"
NODE_SIZE="${NODE_SIZE:-1}"
NODE_RANK="${NODE_RANK:-0}"
PROC_PER_NODE="${PROC_PER_NODE:-2}"
MODEL_WEIGHT_GIB="${MODEL_WEIGHT_GIB:-16}"
TEMPERATURE="${TEMPERATURE:-0}"

export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/offline_external_launcher.py" \
    --model "${MODEL_PATH}" \
    --tp-size "${TP_SIZE}" \
    --node-size "${NODE_SIZE}" \
    --node-rank "${NODE_RANK}" \
    --proc-per-node "${PROC_PER_NODE}" \
    --trust-remote-code \
    --enable-sleep-mode \
    --temperature "${TEMPERATURE}" \
    --model-weight-gib "${MODEL_WEIGHT_GIB}" \
    --sleep-mode-level 2 \
    "$@"
