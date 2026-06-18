#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/data_parallel.py

# Note: This script is designed to run with e2e test,
# please be careful to modify it.
"""
Usage:
Single node:
    Dense models:
        python examples/offline_external_launcher.py \
                --model="Qwen/Qwen2.5-0.5B-Instruct" \
                --tp-size=1 \
                --proc-per-node=2
    MOE models:
        python examples/offline_external_launcher.py \
                --model="Qwen/Qwen3-30B-A3B" \
                --tp-size=2 \
                --proc-per-node=2 \
                --enable-expert-parallel
              
Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_external_launcher.py \
                    --model="Qwen/Qwen3-30B-A3B" \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --proc-per-node=2 \
                    --enable-expert-parallel \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_external_launcher.py \
                    --model="Qwen/Qwen3-30B-A3B" \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --enable-expert-parallel \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import argparse
import contextlib
import gc
import os
from multiprocessing import Process
from time import monotonic, sleep

import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (  # noqa E402
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.network_utils import get_open_port

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

PROCESS_JOIN_TIMEOUT_SECONDS = 15 * 60
PROCESS_POLL_INTERVAL_SECONDS = 1.0
PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def patch_vllm_moe_model_weight_loader(model):
    model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")
    for layer in model.layers:
        mlp_attr = "mlp"
        mlp = getattr(layer, mlp_attr)
        param_dict = dict(mlp.named_parameters())
        for name, param in param_dict.items():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = mlp.experts.weight_loader


def load_and_merge_safetensors(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"The provided directory does not exist: {directory}")
    merged_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".safetensors"):
            file_path = os.path.join(directory, filename)
            print(f"loading file: {file_path}")
            f = load_file(file_path)
            merged_dict.update(f)
    return merged_dict


def parse_args():
    parser = argparse.ArgumentParser(description="External launcher Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--node-size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of the current node")
    parser.add_argument("--proc-per-node", type=int, default=1, help="Number of processes per node")
    parser.add_argument("--master-addr", type=str, default="", help="Master node IP address")
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode execution.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code.")
    parser.add_argument(
        "--enable-expert-parallel", action="store_true", help="Enable expert parallel, used in MOE models."
    )
    parser.add_argument("--enable-sleep-mode", action="store_true", help="Enable sleep mode for the engine.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Float that controls the randomness of the sampling."
    )
    parser.add_argument(
        "--model-weight-gib",
        type=float,
        default=None,
        help="Model weight memory usage in GiB (e.g., 1.0 for 0.5B model).",
    )
    parser.add_argument(
        "--sleep-mode-level",
        type=int,
        choices=[1, 2],
        default=1,
        help="Sleep mode level: 1 or 2. This example of level 2 is only supported for dense model.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model sequence length to pass to LLM.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of sequences to pass to LLM.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Maximum number of generated tokens per prompt.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=10,
        help="Number of times to repeat the base prompt set.",
    )

    args = parser.parse_args()
    if args.max_model_len is not None and args.max_model_len <= 0:
        parser.error("max-model-len must be greater than 0 when provided.")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("max-num-seqs must be greater than 0 when provided.")
    if args.max_tokens <= 0:
        parser.error("max-tokens must be greater than 0.")
    if args.prompt_repeat <= 0:
        parser.error("prompt-repeat must be greater than 0.")
    if args.enable_sleep_mode:
        if args.model_weight_gib is None or args.temperature != 0:
            parser.error(
                "model-weight-gib must be provided, and temperature must be zero when enable-sleep-mode is set."
            )
        if args.model_weight_gib <= 0:
            parser.error("model-weight-gib must be greater than 0 when enable-sleep-mode is set.")
        if args.model == parser.get_default("model") and args.model_weight_gib is None:
            parser.error("model-weight-gib must be provided for default model when enable-sleep-mode is set.")

    return args


def main(
    local_rank: int,
    rank: int,
    master_addr: str,
    master_port: int,
    model_weight_gib: float | None,
    model: str = "Qwen/Qwen3-0.6B",
    world_size: int = 4,
    tensor_parallel_size: int = 2,
    enable_expert_parallel: bool = False,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
    enable_sleep_mode: bool = False,
    temperature: float = 0.8,
    sleep_mode_level: int = 1,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_tokens: int = 10,
    prompt_repeat: int = 10,
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="cpu:gloo,npu:hccl",
            world_size=world_size,
            rank=rank,
        )
    base_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = base_prompts * prompt_repeat
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    llm_kwargs = {}
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        distributed_executor_backend="external_launcher",
        seed=0,
        enable_sleep_mode=enable_sleep_mode,
        **llm_kwargs,
    )
    tp_ranks = get_tp_group().ranks
    print(f"TP RANKS: {tp_ranks}")

    outputs = llm.generate(prompts, sampling_params)

    if enable_sleep_mode:
        if rank == 0:
            free_bytes_before_sleep, _ = torch.npu.mem_get_info()
        llm.sleep(level=sleep_mode_level)
        if rank == 0:
            free_bytes_after_sleep, _ = torch.npu.mem_get_info()
            freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
            print(f"Freed memory: {freed_bytes / 1024**3:.2f} GiB")
            # now the freed memory should be larger than the model weights
            assert freed_bytes >= model_weight_gib / tensor_parallel_size * GiB_bytes

        if sleep_mode_level == 1:
            llm.wake_up()
        else:
            llm.wake_up(tags=["weights"])
            run_model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(run_model)
            sd = load_and_merge_safetensors(model)
            run_model.load_weights(sd.items())
            llm.wake_up(tags=["kv_cache"])

        outputs_after_wakeup = llm.generate(prompts, sampling_params)
        if rank == 0:
            # cmp output
            assert outputs[0].outputs[0].text == outputs_after_wakeup[0].outputs[0].text
            print("Sleep and wake up successfully!!")

    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Global rank: {rank}, Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Give engines time to pause their processing loops before exiting.
    sleep(5)
    del llm
    cleanup_env_and_memory()


def cleanup_env_and_memory():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def stop_processes(procs: list[Process]):
    for proc in procs:
        if proc.exitcode is None:
            proc.kill()
    for proc in procs:
        proc.join(timeout=PROCESS_SHUTDOWN_TIMEOUT_SECONDS)


def wait_for_processes(procs: list[Process], timeout_seconds: float) -> int:
    deadline = monotonic() + timeout_seconds
    pending = list(procs)

    while pending:
        for proc in pending[:]:
            proc.join(timeout=0)
            if proc.exitcode is None:
                continue

            pending.remove(proc)
            if proc.exitcode:
                print(f"Process {proc.pid} exited with code {proc.exitcode}; stopping remaining ranks.")
                stop_processes(pending)
                return proc.exitcode

        if not pending:
            return 0

        remaining_seconds = deadline - monotonic()
        if remaining_seconds <= 0:
            for proc in pending:
                print(f"Killing process {proc.pid} that did not stop within {timeout_seconds / 60:.0f} minutes.")
            stop_processes(pending)
            return 1

        sleep(min(PROCESS_POLL_INTERVAL_SECONDS, remaining_seconds))

    return 0


if __name__ == "__main__":
    args = parse_args()

    tp_size = args.tp_size
    node_size = args.node_size
    proc_per_node = args.proc_per_node
    node_rank = args.node_rank

    if node_size == 1:
        master_addr = "127.0.0.1"
        master_port = get_open_port()
    else:
        master_addr = args.master_addr
        master_port = args.master_port

    world_size = node_size * proc_per_node

    procs = []
    for local_rank, rank in enumerate(range(proc_per_node * node_rank, proc_per_node * (node_rank + 1))):
        proc = Process(
            target=main,
            args=(
                local_rank,
                rank,
                master_addr,
                master_port,
                args.model_weight_gib,
                args.model,
                world_size,
                tp_size,
                args.enable_expert_parallel,
                args.enforce_eager,
                args.trust_remote_code,
                args.enable_sleep_mode,
                args.temperature,
                args.sleep_mode_level,
                args.max_model_len,
                args.max_num_seqs,
                args.max_tokens,
                args.prompt_repeat,
            ),
        )

        proc.start()
        procs.append(proc)

    raise SystemExit(wait_for_processes(procs, PROCESS_JOIN_TIMEOUT_SECONDS))
