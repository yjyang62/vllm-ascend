#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as envs_vllm
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

def _validate_v2_model_runner(vllm_config: VllmConfig) -> None:
    """No-op replacement for the upstream V2 model runner validation.

    Ascend defaults to V2 and uses ``VLLM_USE_V2_MODEL_RUNNER`` as its only
    runner-selection control. Upstream GPU-specific model, feature, and Triton
    checks do not apply to the Ascend runner.
    """


def apply_v2_model_runner_config_patch() -> None:
    """Apply the Ascend V2 model runner overrides to VllmConfig.

    Installs two overrides on the ``VllmConfig`` class:

    * ``use_v2_model_runner`` defaults to V2 and honors an explicit
      ``VLLM_USE_V2_MODEL_RUNNER`` override (see :func:`use_v2_model_runner`).
    * ``_validate_v2_model_runner`` is neutralized because the upstream checks
      describe the upstream GPU runner and do not apply to the Ascend runner.

    Must run wherever ``VllmConfig`` is (re)created or its properties are read
    in a separate process -- the frontend during config construction, each
    worker process, and the engine-core process (the scheduler reads
    ``use_v2_model_runner`` there from a pickled config, so the class-level
    patch does not carry over from the frontend). Repeated application is
    harmless: it just re-assigns the same overrides.
    """
    from vllm.config.vllm import VllmConfig

    VllmConfig.use_v2_model_runner = property(use_v2_model_runner)
    VllmConfig._validate_v2_model_runner = _validate_v2_model_runner


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Return whether the V2 model runner should be used on Ascend.

    V2 is the default for every configuration. Set
    ``VLLM_USE_V2_MODEL_RUNNER=0`` to select V1 explicitly.
    """
    use_v2_model_runner = envs_vllm.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        logger.info_once(
            "VLLM_USE_V2_MODEL_RUNNER=%s is set; using Model Runner %s.",
            use_v2_model_runner,
            "V2" if use_v2_model_runner else "V1",
        )
        return use_v2_model_runner

    logger.info_once("VLLM_USE_V2_MODEL_RUNNER is unset; using Model Runner V2 by default.")
    return True
