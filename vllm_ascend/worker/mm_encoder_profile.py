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

"""Profile the multimodal encoder before language-model compile dummy runs.

``NPUModelRunner.profile_run`` used to call an extra MC2 ``_dummy_run``
before ``GPUModelRunner.profile_run``. That dummy run torch-compiles the
backbone and speculative draft (dflash/eagle). Compiled graphs stay
resident, so the later encoder dummy of several max-size ``vision_chunk``
items (Kimi-K2.5/K2.6 uses 3000x3000 frames) can OOM even though the same
allocation would fit before compile.

Keep encoder-cache outputs around for the subsequent LM dummy so peak
memory still includes encoder-cache storage; only the ViT workspace is
released.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

try:
    from vllm.logger import logger
except ImportError:  # pragma: no cover - UT fallback when vLLM is not installed
    import logging

    logger = logging.getLogger(__name__)


def _release_encoder_workspace() -> None:
    empty_cache = getattr(getattr(torch, "npu", None), "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def _store_encoder_outputs(encoder_cache: Any, outputs: Any) -> None:
    encoder_outputs = getattr(encoder_cache, "encoder_outputs", None)
    if encoder_outputs is not None and hasattr(encoder_outputs, "update"):
        encoder_outputs.update((f"tmp_{i}", output) for i, output in enumerate(outputs))
        return
    for i, output in enumerate(outputs):
        encoder_cache[f"tmp_{i}"] = output


def profile_mm_encoder_cache(runner: Any) -> None:
    """Run max-size multimodal encoder profiling before any compile dummy run."""
    if not getattr(runner, "supports_mm_inputs", False):
        return

    model_config = getattr(runner, "model_config", None)
    mm_config = getattr(model_config, "multimodal_config", None) if model_config is not None else None
    if mm_config is not None and getattr(mm_config, "skip_mm_profiling", False):
        logger.info("Skipping memory profiling for multimodal encoder and encoder cache.")
        return

    mm_budget = getattr(runner, "mm_budget", None)
    if mm_budget is None:
        return

    encoder_budget = mm_budget.get_encoder_budget()
    if encoder_budget <= 0:
        return
    if not mm_budget.mm_max_toks_per_item:
        logger.info(
            "Skipping encoder profiling for embedding-only mode "
            "(all modality limits=0 with enable_mm_embeds=True).",
        )
        return

    dummy_modality = mm_budget.get_modality_with_max_tokens()
    max_mm_items_per_batch = mm_budget.mm_max_items_per_batch[dummy_modality]
    logger.info(
        "Encoder cache will be initialized with a budget of %s tokens, "
        "and profiled with %s %s items of the maximum feature size "
        "(before language-model compile dummy run).",
        encoder_budget,
        max_mm_items_per_batch,
        dummy_modality,
    )

    batched_dummy_mm_inputs = runner._get_mm_dummy_batch(
        dummy_modality,
        max_mm_items_per_batch,
    )
    dummy_encoder_outputs = runner.model.embed_multimodal(**batched_dummy_mm_inputs)
    if dummy_encoder_outputs is None:
        _release_encoder_workspace()
        return

    encoder_cache = getattr(runner, "encoder_cache", None)
    if encoder_cache is not None:
        _store_encoder_outputs(encoder_cache, dummy_encoder_outputs)
    _release_encoder_workspace()


@contextmanager
def skip_parent_mm_profiling(model_config: Any) -> Iterator[None]:
    """Prevent ``GPUModelRunner.profile_run`` from profiling the encoder again."""
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None or not hasattr(mm_config, "skip_mm_profiling"):
        yield
        return

    previous = mm_config.skip_mm_profiling
    mm_config.skip_mm_profiling = True
    try:
        yield
    finally:
        mm_config.skip_mm_profiling = previous
