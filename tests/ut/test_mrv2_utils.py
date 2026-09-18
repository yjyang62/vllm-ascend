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

from types import SimpleNamespace

import pytest

import vllm_ascend.mrv2_utils as mrv2_utils
from vllm_ascend.mrv2_utils import use_v2_model_runner


@pytest.mark.parametrize("env_value", [True, False])
def test_environment_override_wins(monkeypatch, env_value):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", env_value)

    assert use_v2_model_runner(SimpleNamespace()) is env_value


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(model_config=None),
        SimpleNamespace(
            model_config=SimpleNamespace(
                runner_type="embedding",
                is_attention_free=True,
                architectures=["UnknownModel"],
            )
        ),
        SimpleNamespace(lora_config=object()),
        SimpleNamespace(
            speculative_config=SimpleNamespace(
                method="unknown_method",
                num_speculative_tokens_per_batch_size=[[1, 256, 4]],
            )
        ),
        SimpleNamespace(
            speculative_config=SimpleNamespace(method="dspark"),
            additional_config={"draft_window_size": 512},
        ),
    ],
    ids=[
        "empty-config",
        "no-model",
        "unknown-attention-free-embedding",
        "lora",
        "dynamic-unsupported-speculation",
        "dspark-sliding-window",
    ],
)
def test_v2_is_default_for_every_configuration(monkeypatch, config):
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)

    assert use_v2_model_runner(config) is True


def test_default_v2_logs_selection(monkeypatch):
    info_calls = []
    monkeypatch.setattr(mrv2_utils.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", None)
    monkeypatch.setattr(mrv2_utils.logger, "info_once", lambda *args: info_calls.append(args))

    assert use_v2_model_runner(SimpleNamespace()) is True
    assert len(info_calls) == 1


def test_validation_is_decoupled_from_upstream():
    mrv2_utils._validate_v2_model_runner(object())
    mrv2_utils._validate_v2_model_runner(SimpleNamespace())


def test_apply_config_patch_is_wired(monkeypatch):
    from vllm.config.vllm import VllmConfig

    original_property = VllmConfig.use_v2_model_runner
    original_validate = VllmConfig._validate_v2_model_runner

    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", False)

    mrv2_utils.apply_v2_model_runner_config_patch()
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    assert VllmConfig.use_v2_model_runner.fget is mrv2_utils.use_v2_model_runner

    # Upstream GPU-specific validation must not change Ascend's default.
    VllmConfig._validate_v2_model_runner(object())

    # Re-applying is harmless.
    mrv2_utils.apply_v2_model_runner_config_patch()
    VllmConfig._validate_v2_model_runner(object())

    # Restore the upstream class state so later tests are unaffected.
    monkeypatch.setattr(VllmConfig, "use_v2_model_runner", original_property)
    monkeypatch.setattr(VllmConfig, "_validate_v2_model_runner", original_validate)
