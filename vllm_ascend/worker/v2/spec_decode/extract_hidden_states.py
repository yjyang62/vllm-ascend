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
"""Ascend MRV2 speculator for extract_hidden_states."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.cudagraph_utils import AttentionStatePair, BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata

PADDING_SLOT_ID = -1


class AscendExtractHiddenStatesSpeculator(DraftModelSpeculator):
    """Cache target aux hidden states through the v2 speculator interface.

    The extract-hidden-states method is not a real drafter: it writes the
    selected target hidden states into a cache-only attention layer and returns
    the target sampled token as the single draft token, so verification always
    succeeds while the KV connector can export the hidden states.
    """

    supports_mm_inputs = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        assert self.num_speculative_steps == 1
        if self.speculative_config.disable_padded_drafter_batch:
            raise ValueError(
                "disable_padded_drafter_batch is not supported with "
                "extract_hidden_states method"
            )

        hf_config = self.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )

        self.num_hidden_states = len(layer_ids)
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.num_hidden_states, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        self._slot_mapping_buffer = torch.full(
            (self.max_num_tokens,),
            PADDING_SLOT_ID,
            dtype=torch.int64,
            device=device,
        )
        self.kv_cache_gid = -1

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # The cache-only forward is intentionally kept eager; the target model
        # can still use the configured graph mode.
        return

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        return

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_model, target_attn_layer_names
        with set_model_tag("extract_hidden_states"):
            return get_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
            )

    def set_attn(self, model_state, kv_cache_config, block_tables) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)
        self._validate_same_kv_cache_group(kv_cache_config)

    def _validate_same_kv_cache_group(self, kv_cache_config) -> None:
        assert len(self.draft_attn_layer_names) == 1, (
            "ExtractHiddenStatesModel should have exactly one attention "
            f"layer, found {len(self.draft_attn_layer_names)}"
        )
        layer = next(iter(self.draft_attn_layer_names))
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if layer in group.layer_names:
                self.kv_cache_gid = gid
                return
        raise ValueError(f"Cache-only layer {layer!r} not in any KV cache group")

    def _get_slot_mapping(self, num_tokens: int, source: torch.Tensor) -> dict[str, torch.Tensor]:
        num_actual = min(source.shape[0], num_tokens)
        self._slot_mapping_buffer[:num_actual].copy_(source[:num_actual])
        if num_tokens > num_actual:
            self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)
        slot_mapping = self._slot_mapping_buffer[:num_tokens]
        return {name: slot_mapping for name in self.draft_attn_layer_names}

    def _build_common_attn_metadata(
        self,
        input_batch: InputBatch,
        num_input_tokens: int,
    ) -> AscendCommonAttentionMetadata:
        num_reqs = input_batch.num_reqs
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np[: num_reqs + 1])
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        max_query_len = int(query_lens.max()) if query_lens.size > 0 else 0
        slot_mapping = self.block_tables.slot_mappings[self.kv_cache_gid]
        slot_mapping = slot_mapping[:num_input_tokens]

        return AscendCommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=input_batch.seq_lens[:num_reqs],
            seq_lens_cpu=torch.from_numpy(input_batch.seq_lens_np[:num_reqs]),
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=input_batch.num_tokens,
            max_query_len=max_query_len,
            block_table_tensor=self.block_tables.input_block_tables[self.kv_cache_gid][:num_reqs],
            slot_mapping=slot_mapping,
            positions=input_batch.positions[:num_input_tokens],
            attn_state=input_batch.attn_state,
            graph_pad_size=-1,
            num_input_tokens=num_input_tokens,
            max_seq_len=self.max_model_len,
            causal=True,
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            last_hidden_states,
            num_rejected,
            temperature,
            seeds,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        if aux_hidden_states is None:
            raise ValueError("aux_hidden_states are required when using `extract_hidden_states`")
        assert self.model is not None

        num_tokens = input_batch.num_tokens
        num_input_tokens = input_batch.num_tokens_after_padding
        target_hidden_states = [h[:num_tokens] for h in aux_hidden_states]
        self.hidden_states[:num_tokens].copy_(torch.stack(target_hidden_states, dim=1))
        if num_input_tokens > num_tokens:
            self.hidden_states[num_tokens:num_input_tokens].zero_()

        common_attn_metadata = self._build_common_attn_metadata(input_batch, num_input_tokens)
        per_layer_attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=input_batch.num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu,
            max_query_len=common_attn_metadata.max_query_len,
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=self.block_tables.input_block_tables,
            slot_mappings=self.block_tables.slot_mappings[:, :num_input_tokens],
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=input_batch.seq_lens_np,
            positions=common_attn_metadata.positions,
            attn_state=common_attn_metadata.attn_state,
            num_input_tokens=num_input_tokens,
            causal=True,
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=self._get_slot_mapping(num_input_tokens, common_attn_metadata.slot_mapping),
        ):
            self.model(hidden_states=self.hidden_states[:num_input_tokens])

        sampled_tokens = last_sampled[input_batch.idx_mapping]
        prefill_tokens = next_prefill_tokens[input_batch.idx_mapping]
        draft_tokens = torch.where(num_sampled > 0, sampled_tokens, prefill_tokens)
        return draft_tokens.to(torch.int64).unsqueeze(1)
