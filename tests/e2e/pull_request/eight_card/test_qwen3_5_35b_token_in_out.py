#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Verify Token In / Token Out for Qwen3.5-35B-A3B.

The serve flags match the functional verification command:

    vllm serve /mnt/share/weights/Qwen3.5-35B-A3B \\
      --tensor-parallel-size 8 --enforce-eager --max_model_len 4096 \\
      --gpu-memory-utilization 0.9 --served-model-name auto \\
      --max-num-seqs 16 --port 8008
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tools.verify_token_in_token_out import run_verification

LOCAL_MODEL = "/mnt/share/weights/Qwen3.5-35B-A3B"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
SERVED_MODEL_NAME = "auto"
MAX_TOKENS = 16


def _model_path() -> str:
    override = os.environ.get("QWEN35_MOE_MODEL")
    if override:
        return override
    if Path(LOCAL_MODEL).is_dir():
        return LOCAL_MODEL
    return DEFAULT_MODEL


@pytest.mark.e2e_model(DEFAULT_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="token_in_token_out",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_qwen3_5_35b_a3b_token_in_token_out():
    model = _model_path()
    port = get_open_port()
    server_args = [
        "--tensor-parallel-size",
        "8",
        "--enforce-eager",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--max-num-seqs",
        "16",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    env_dict = {
        "HCCL_BUFFSIZE": "1024",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    with RemoteOpenAIServer(
        model,
        server_args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict=env_dict,
    ) as server:
        results = run_verification(
            server.url_root,
            SERVED_MODEL_NAME,
            max_tokens=MAX_TOKENS,
        )
    assert results
    assert all(item.passed for item in results)
