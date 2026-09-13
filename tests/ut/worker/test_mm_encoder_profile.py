#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.worker.mm_encoder_profile import profile_mm_encoder_cache, skip_parent_mm_profiling
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2


def _make_budget(*, encoder_budget=16384, modality="vision_chunk", items=3):
    budget = MagicMock()
    budget.get_encoder_budget.return_value = encoder_budget
    budget.mm_max_toks_per_item = {modality: 5461}
    budget.get_modality_with_max_tokens.return_value = modality
    budget.mm_max_items_per_batch = {modality: items}
    return budget


def _make_mm_runner(*, skip_mm_profiling=False, encoder_cache=None):
    runner = SimpleNamespace()
    runner.supports_mm_inputs = True
    runner.model_config = SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=skip_mm_profiling))
    runner.mm_budget = _make_budget()
    runner.encoder_cache = {} if encoder_cache is None else encoder_cache
    runner._get_mm_dummy_batch = MagicMock(return_value={"pixel_values": torch.zeros(1)})
    outputs = [torch.ones(4), torch.ones(4), torch.ones(4)]
    runner.model = SimpleNamespace(embed_multimodal=MagicMock(return_value=outputs))
    return runner


def test_profile_mm_encoder_cache_stores_outputs_and_skips_when_disabled():
    runner = _make_mm_runner()
    profile_mm_encoder_cache(runner)

    runner._get_mm_dummy_batch.assert_called_once_with("vision_chunk", 3)
    runner.model.embed_multimodal.assert_called_once()
    assert set(runner.encoder_cache) == {"tmp_0", "tmp_1", "tmp_2"}

    skipped = _make_mm_runner(skip_mm_profiling=True)
    profile_mm_encoder_cache(skipped)
    skipped.model.embed_multimodal.assert_not_called()


def test_profile_mm_encoder_cache_uses_encoder_outputs_attr():
    runner = _make_mm_runner(encoder_cache=SimpleNamespace(encoder_outputs={}))
    profile_mm_encoder_cache(runner)
    assert "tmp_0" in runner.encoder_cache.encoder_outputs


def test_skip_parent_mm_profiling_restores_flag():
    model_config = SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=False))
    with skip_parent_mm_profiling(model_config):
        assert model_config.multimodal_config.skip_mm_profiling is True
    assert model_config.multimodal_config.skip_mm_profiling is False


def test_npu_profile_run_profiles_encoder_before_dummy_compile():
    call_order: list[str] = []
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=False))
    runner.sparse_kv_offload_enabled = False
    runner.max_num_tokens = 16384
    runner.vllm_config = SimpleNamespace()

    def fake_encoder(_runner):
        call_order.append("encoder")

    def fake_dummy(*_args, **_kwargs):
        call_order.append("dummy")
        return None, None

    def fake_super(_self):
        call_order.append("super")
        assert _self.model_config.multimodal_config.skip_mm_profiling is True

    with (
        patch.object(runner, "eplb_warmup"),
        patch("vllm_ascend.worker.model_runner_v1.profile_mm_encoder_cache", side_effect=fake_encoder),
        patch("vllm_ascend.worker.model_runner_v1.get_mc2_tokens_capacity", return_value=512),
        patch(
            "vllm_ascend.worker.model_runner_v1.select_moe_comm_method",
            return_value=MoECommType.MC2,
        ),
        patch.object(runner, "_dummy_run", side_effect=fake_dummy),
        patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.profile_run", fake_super),
    ):
        runner.profile_run()

    assert call_order == ["encoder", "dummy", "super"]
    assert runner.model_config.multimodal_config.skip_mm_profiling is False


def test_v2_profile_run_profiles_encoder_before_dummy_compile():
    call_order: list[str] = []
    runner = NPUModelRunnerV2.__new__(NPUModelRunnerV2)
    runner.model_config = SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=False))
    runner.max_num_tokens = 16384
    runner.vllm_config = SimpleNamespace()

    def fake_encoder(_runner):
        call_order.append("encoder")

    def fake_dummy(*_args, **_kwargs):
        call_order.append("dummy")
        return None

    def fake_super(_self):
        call_order.append("super")
        assert _self.model_config.multimodal_config.skip_mm_profiling is True

    with (
        patch("vllm_ascend.worker.v2.model_runner.profile_mm_encoder_cache", side_effect=fake_encoder),
        patch("vllm_ascend.worker.v2.model_runner.get_mc2_tokens_capacity", return_value=512),
        patch("vllm_ascend.worker.v2.model_runner.select_moe_comm_method", return_value=MoECommType.MC2),
        patch.object(runner, "_dummy_run", side_effect=fake_dummy),
        patch("vllm.v1.worker.gpu.model_runner.GPUModelRunner.profile_run", fake_super),
    ):
        runner.profile_run()

    assert call_order == ["encoder", "dummy", "super"]
    assert runner.model_config.multimodal_config.skip_mm_profiling is False
