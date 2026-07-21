from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from vllm.compilation.backends import set_model_tag
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

PADDING_SLOT_ID = -1


@triton.jit
def _cache_hidden_states_kernel(
    to_cache_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    to_cache_stride_0,
    to_cache_stride_1,
    to_cache_stride_2,
    kv_cache_stride_0,
    kv_cache_stride_1,
    kv_cache_stride_2,
    kv_cache_stride_3,
    block_size,
    num_heads,
    head_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_width = num_heads * head_size
    slot = tl.load(slot_mapping_ptr + token_idx)
    valid = (slot >= 0) & (hidden_offset < hidden_width)

    head_idx = hidden_offset // head_size
    head_offset = hidden_offset % head_size
    value = tl.load(
        to_cache_ptr + token_idx * to_cache_stride_0 + head_idx * to_cache_stride_1 + head_offset * to_cache_stride_2,
        mask=hidden_offset < hidden_width,
    )

    block_idx = slot // block_size
    block_offset = slot % block_size
    tl.store(
        kv_cache_ptr
        + block_idx * kv_cache_stride_0
        + block_offset * kv_cache_stride_1
        + head_idx * kv_cache_stride_2
        + head_offset * kv_cache_stride_3,
        value,
        mask=valid,
    )


def _update_valid_hidden_state_slots(
    impl,
    layer,
    to_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write real tokens only; negative slots are graph/DP padding."""
    del layer
    assert to_cache.dtype == impl.kv_cache_torch_dtype
    assert kv_cache.dtype == impl.kv_cache_torch_dtype
    if to_cache.device.type == "cpu":
        valid_mask = slot_mapping >= 0
        valid_slots = slot_mapping[valid_mask]
        block_size = kv_cache.shape[1]
        kv_cache[
            valid_slots // block_size,
            valid_slots % block_size,
        ] = to_cache[valid_mask]
        return

    num_tokens, num_heads, head_size = to_cache.shape
    block_size = kv_cache.shape[1]
    block_width = 256
    grid = (
        num_tokens,
        triton.cdiv(num_heads * head_size, block_width),
    )
    _cache_hidden_states_kernel[grid](
        to_cache,
        kv_cache,
        slot_mapping,
        *to_cache.stride(),
        *kv_cache.stride(),
        block_size,
        num_heads,
        head_size,
        BLOCK_SIZE=block_width,
    )


class AscendExtractHiddenStatesSpeculator:
    """Native model runner v2 speculator for hidden-state extraction."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: Any,
    ) -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        assert speculative_config.num_speculative_tokens == 1
        if speculative_config.disable_padded_drafter_batch:
            raise ValueError("disable_padded_drafter_batch is not supported with extract_hidden_states")

        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        self.dtype = vllm_config.model_config.dtype
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.kv_cache_gid = -1
        self.block_tables: BlockTables | None = None

        hf_config = speculative_config.draft_model_config.hf_config
        layer_ids = getattr(
            hf_config,
            "eagle_aux_hidden_state_layer_ids",
            None,
        )
        if not layer_ids:
            raise ValueError("eagle_aux_hidden_state_layer_ids must be set for extract_hidden_states")

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens + vllm_config.scheduler_config.max_num_seqs
        self.hidden_states = torch.zeros(
            (
                max_num_tokens,
                len(layer_ids),
                vllm_config.model_config.get_hidden_size(),
            ),
            dtype=self.dtype,
            device=device,
        )
        self.slot_mapping_buffer = torch.zeros(
            max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.cudagraph_dispatcher = CudagraphDispatcher(vllm_config)

    def load_model(self, target_model: nn.Module) -> None:
        del target_model
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
            )
        )

        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        with set_model_tag("extract_hidden_states"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=speculative_config.draft_model_config,
            )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,
        )
        draft_attn_layers = {
            name: layer for name, layer in all_attn_layers.items() if name not in target_attn_layer_names
        }
        if len(draft_attn_layers) != 1:
            raise ValueError(
                f"ExtractHiddenStatesModel must have exactly one attention layer, found {len(draft_attn_layers)}"
            )

        self.attn_layer_names = list(draft_attn_layers)
        draft_layer = next(iter(draft_attn_layers.values()))
        draft_layer.impl.do_kv_cache_update = MethodType(
            _update_valid_hidden_state_slots,
            draft_layer.impl,
        )
        attn_backend = draft_layer.get_attn_backend()
        self.attn_metadata_builder = attn_backend.get_builder_cls()(
            draft_layer.get_kv_cache_spec(self.vllm_config),
            self.attn_layer_names,
            self.vllm_config,
            self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        speculative_config = self.vllm_config.speculative_config
        assert speculative_config is not None
        if not speculative_config.enforce_eager and cudagraph_mode.mixed_mode() in (
            CUDAGraphMode.PIECEWISE,
            CUDAGraphMode.FULL,
        ):
            speculator_mode = CUDAGraphMode.PIECEWISE
        else:
            speculator_mode = CUDAGraphMode.NONE
        self.cudagraph_dispatcher.initialize_cudagraph_keys(speculator_mode)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        del model_state
        if len(self.attn_layer_names) != 1:
            raise ValueError("ExtractHiddenStatesModel must have one cache-only layer")
        layer_name = self.attn_layer_names[0]
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if layer_name in group.layer_names:
                self.kv_cache_gid = gid
                self.block_tables = block_tables
                return
        raise ValueError(f"Cache-only layer {layer_name!r} is not in a KV cache group")

    def capture(self, attn_states: dict[Any, Any]) -> None:
        captured_sizes: set[int] = set()
        for batch_desc, attention_state in attn_states.items():
            num_tokens = getattr(batch_desc, "num_tokens", None)
            if num_tokens is None or num_tokens in captured_sizes:
                continue
            captured_sizes.add(num_tokens)
            self._dummy_run(
                num_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                slot_mappings=attention_state.slot_mappings,
            )

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if slot_mapping is not None:
            num_actual_tokens = slot_mapping.shape[0]
            self.slot_mapping_buffer[:num_actual_tokens].copy_(slot_mapping)
            if num_tokens > num_actual_tokens:
                self.slot_mapping_buffer[num_actual_tokens:num_tokens].fill_(PADDING_SLOT_ID)
        view = self.slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names}

    def _dispatch_and_sync(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        num_tokens = self.runner._pad_for_sequence_parallelism(num_tokens)
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=None if use_cudagraphs else {CUDAGraphMode.NONE},
        )
        num_tokens_padded = batch_desc.num_tokens
        num_tokens_across_dp = None

        if self.vllm_config.parallel_config.data_parallel_size > 1:
            (
                _,
                num_tokens_across_dp,
                synced_cudagraph_mode,
            ) = self.runner._sync_metadata_across_dp(
                num_tokens=num_tokens_padded,
                is_draft_model=True,
                cudagraph_mode=cudagraph_mode,
                allow_dp_padding=use_cudagraphs,
            )
            if num_tokens_across_dp is not None:
                num_tokens_padded = int(num_tokens_across_dp[self.dp_rank].item())
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={synced_cudagraph_mode},
                )
                assert batch_desc.num_tokens == num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp

    def _run_cache_only_model(
        self,
        num_tokens: int,
        common_attn_metadata: CommonAttentionMetadata | None,
        slot_mapping: dict[str, torch.Tensor],
        cudagraph_runtime_mode: CUDAGraphMode,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        assert self.model is not None
        per_layer_attn_metadata = None
        if common_attn_metadata is not None:
            assert self.attn_metadata_builder is not None
            metadata = self.attn_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=0,
            )
            per_layer_attn_metadata = {name: metadata for name in self.attn_layer_names}

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping,
        ):
            self.model(hidden_states=self.hidden_states[:num_tokens])

    def _dummy_run(
        self,
        num_tokens: int,
        aclgraph_runtime_mode: CUDAGraphMode,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> None:
        del is_profile
        (
            num_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_tokens,
            is_draft_model=True,
        )
        layer_slot_mapping = None
        if self.attn_layer_names and slot_mappings is not None:
            layer_slot_mapping = slot_mappings.get(self.attn_layer_names[0])
        slot_mapping = self._get_slot_mapping(num_tokens, layer_slot_mapping) if layer_slot_mapping is not None else {}
        self._run_cache_only_model(
            num_tokens=num_tokens,
            common_attn_metadata=None,
            slot_mapping=slot_mapping,
            cudagraph_runtime_mode=aclgraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
        )

    def _build_common_attn_metadata(
        self,
        input_batch: InputBatch,
        slot_mappings: dict[str, torch.Tensor],
    ) -> AscendCommonAttentionMetadata:
        assert self.block_tables is not None
        assert self.kv_cache_gid >= 0
        layer_name = self.attn_layer_names[0]
        num_reqs = input_batch.num_reqs
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
            block_table_tensor=self.block_tables.input_block_tables[self.kv_cache_gid],
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
            self._dummy_run(
                num_tokens=input_batch.num_tokens_after_padding,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                is_profile=is_profile,
            )
            return torch.zeros(
                (input_batch.num_reqs, 1),
                dtype=torch.int64,
                device=self.device,
            )

        if aux_hidden_states is None:
            raise ValueError("aux_hidden_states are required when using extract_hidden_states")
        if slot_mappings is None:
            raise ValueError("slot_mappings are required when using extract_hidden_states")

        req_indices = input_batch.idx_mapping[: input_batch.num_reqs].long()
        sampled_token_ids = last_sampled[req_indices, 0]
        sampled_token_ids = torch.where(
            num_sampled[: input_batch.num_reqs] > 0,
            sampled_token_ids,
            next_prefill_tokens[req_indices],
        ).unsqueeze(1)

        stacked_hidden_states = torch.stack(
            [hidden_states[: input_batch.num_tokens] for hidden_states in aux_hidden_states],
            dim=1,
        )
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        (
            cudagraph_runtime_mode,
            num_tokens_padded,
            num_tokens_across_dp,
        ) = self._dispatch_and_sync(num_tokens)
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_tokens_padded

        common_attn_metadata = self._build_common_attn_metadata(
            input_batch,
            slot_mappings,
        )
        layer_slot_mapping = slot_mappings[self.attn_layer_names[0]]
        self._run_cache_only_model(
            num_tokens=num_tokens_padded,
            common_attn_metadata=common_attn_metadata,
            slot_mapping=self._get_slot_mapping(
                num_tokens_padded,
                layer_slot_mapping,
            ),
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        return sampled_token_ids[:, :1]
