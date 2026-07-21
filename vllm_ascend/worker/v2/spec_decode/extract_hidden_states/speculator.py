from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)


class AscendExtractHiddenStatesSpeculator:
    """Adapt the v1 hidden-state proposer to the model runner v2 protocol."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: Any,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.proposer = AscendExtractHiddenStatesProposer(
            vllm_config,
            device,
            runner=runner,
        )
        self.block_tables: BlockTables | None = None

    @property
    def model(self) -> nn.Module | None:
        return self.proposer.model

    def load_model(self, target_model: nn.Module) -> None:
        self.proposer.load_model(target_model)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self.proposer.initialize_cudagraph_keys(cudagraph_mode)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        del model_state
        self.proposer.validate_same_kv_cache_group(kv_cache_config)
        self.block_tables = block_tables

    def capture(self, attn_states: dict[Any, Any]) -> None:
        """Capture the cache-only model's piecewise graph for target sizes."""
        captured_sizes: set[int] = set()
        for batch_desc, attention_state in attn_states.items():
            num_tokens = getattr(batch_desc, "num_tokens", None)
            if num_tokens is None or num_tokens in captured_sizes:
                continue
            captured_sizes.add(num_tokens)
            self.proposer.dummy_run(
                num_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                slot_mappings=attention_state.slot_mappings,
            )

    def _build_common_attn_metadata(
        self,
        input_batch: InputBatch,
        slot_mappings: dict[str, torch.Tensor],
    ) -> AscendCommonAttentionMetadata:
        assert self.block_tables is not None
        assert self.proposer.kv_cache_gid >= 0
        assert self.proposer.attn_layer_names

        num_reqs = input_batch.num_reqs
        layer_name = self.proposer.attn_layer_names[0]
        seq_lens_cpu = torch.from_numpy(input_batch.seq_lens_np)[:num_reqs]
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound[:num_reqs]

        return AscendCommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            seq_lens=input_batch.seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            max_seq_len=int(seq_lens_cpu_upper_bound.max().item()),
            num_reqs=num_reqs,
            num_actual_tokens=input_batch.num_tokens,
            max_query_len=int(input_batch.num_scheduled_tokens.max()),
            block_table_tensor=self.block_tables.input_block_tables[self.proposer.kv_cache_gid],
            slot_mapping=slot_mappings[layer_name],
            positions=input_batch.positions,
            attn_state=getattr(input_batch, "attn_state", None),
            num_input_tokens=input_batch.num_tokens_after_padding,
            causal=True,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
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
            num_tokens_across_dp,
            mm_inputs,
        )

        if dummy_run:
            self.proposer.dummy_run(
                num_tokens=input_batch.num_tokens_after_padding,
                aclgraph_runtime_mode=CUDAGraphMode.NONE if is_profile or skip_attn_for_dummy_run else None,
                is_profile=is_profile,
            )
            return torch.zeros(
                (input_batch.num_reqs, 1),
                dtype=torch.int64,
                device=self.device,
            )

        if aux_hidden_states is None:
            raise ValueError("aux_hidden_states are required when using `extract_hidden_states`")
        if slot_mappings is None:
            raise ValueError("slot_mappings are required when using `extract_hidden_states`")

        req_indices = input_batch.idx_mapping[: input_batch.num_reqs].long()
        sampled_token_ids = last_sampled[req_indices, 0]
        prefill_token_ids = next_prefill_tokens[req_indices]
        sampled_token_ids = torch.where(
            num_sampled[: input_batch.num_reqs] > 0,
            sampled_token_ids,
            prefill_token_ids,
        ).unsqueeze(1)

        common_attn_metadata = self._build_common_attn_metadata(
            input_batch,
            slot_mappings,
        )
        target_hidden_states = [hidden_states[: input_batch.num_tokens] for hidden_states in aux_hidden_states]
        return self.proposer.propose(
            sampled_token_ids=sampled_token_ids,
            target_hidden_states=target_hidden_states,
            common_attn_metadata=common_attn_metadata,
            slot_mappings=slot_mappings,
        )
