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

import multiprocessing
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.logger import logger
from vllm.profiler.wrapper import WorkerProfiler

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self.profiler: Any = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.")

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        profile_kwargs = {
            "activities": [
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            "with_stack": False,
            "profile_memory": profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            "with_modules": profiler_config.torch_profiler_with_stack,
            "experimental_config": experimental_config,
            "on_trace_ready": TorchNPUProfilerWrapper._create_trace_handler(profiler_config, trace_name),
        }
        schedule = TorchNPUProfilerWrapper._create_schedule(profiler_config)
        if schedule is not None:
            profile_kwargs["schedule"] = schedule
        return torch_npu.profiler.profile(**profile_kwargs)

    @staticmethod
    def _create_schedule(profiler_config: ProfilerConfig) -> Any | None:
        active_iterations = profiler_config.max_iterations
        if active_iterations <= 0:
            return None
        return torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=active_iterations,
            repeat=1,
            skip_first=0,
        )

    @staticmethod
    def _create_trace_handler(profiler_config: ProfilerConfig, trace_name: str) -> Callable[[Any], None]:
        trace_handler = torch_npu.profiler.tensorboard_trace_handler(
            profiler_config.torch_profiler_dir,
            worker_name=trace_name,
        )

        def on_trace_ready(profiler: Any) -> None:
            if multiprocessing.current_process().daemon:
                logger.info(
                    "Scheduling torch_npu profiler offline analysis outside daemon process for %r.",
                    profiler_config.torch_profiler_dir,
                )
                TorchNPUProfilerWrapper._start_offline_analyse(profiler_config)
                return
            trace_handler(profiler)

        return on_trace_ready

    @staticmethod
    def _start_offline_analyse(profiler_config: ProfilerConfig) -> None:
        profiler_dir = Path(profiler_config.torch_profiler_dir)
        try:
            profiler_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = profiler_dir / "offline_analyse.log"
            with log_file_path.open("ab") as log_file:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "from torch_npu.profiler.profiler import analyse; "
                        "import sys; "
                        "analyse(sys.argv[1])",
                        str(profiler_dir),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            logger.info(
                "Started torch_npu profiler offline analysis process pid=%s for %r. "
                "Logs are written to %s.",
                process.pid,
                str(profiler_dir),
                str(log_file_path),
            )
        except OSError as e:
            logger.warning(
                "Failed to start torch_npu profiler offline analysis for %r: %s",
                str(profiler_dir),
                e,
            )

    def _start(self) -> None:
        self.profiler.start()

    def _stop(self) -> None:
        self.profiler.stop()

    def _profiler_step(self) -> bool:
        self.profiler.step()
        return True
