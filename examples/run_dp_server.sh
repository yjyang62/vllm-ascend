
export HCCL_IF_IP=2.0.0.0
export GLOO_SOCKET_IFNAME="eth0"
export TP_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10

export VLLM_USE_MODELSCOPE=True

export ASCEND_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=/mnt/share/torch_extensions

# Optional but recommended:
# Prebuild and persist JIT extensions (e.g. npu_mega_moe.so) before production serve.
# This avoids on-the-fly compilation in the real serving startup path.
# bash tools/prebuild_moe_runtime.sh \
#   --model Qwen/Qwen1.5-MoE-A2.7B \
#   --served-model-name Qwen \
#   --port 18008 \
#   --tensor-parallel-size 4 \
#   --data-parallel-size 2

vllm serve Qwen/Qwen1.5-MoE-A2.7B  \
  --host 0.0.0.0 \
  --port 20002 \
  --served-model-name Qwen \
  --data-parallel-size 2 \
  --data-parallel-size-local 2 \
  --data-parallel-address 2.0.0.0 \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enforce-eager
