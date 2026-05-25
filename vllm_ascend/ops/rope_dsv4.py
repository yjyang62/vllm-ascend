import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.platforms import current_platform


@dataclass(frozen=True)
class RopeCacheConfig:
    rotary_dim: int
    max_position_embeddings: int
    base: float
    scaling_factor: float
    beta_fast: float
    beta_slow: float
    dtype: torch.dtype
    device: str
    max_batch_size: int


class RopeGlobalState:
    def __init__(self):
        self.static_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.runtime_buffer: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.layer_info: dict[str, tuple[str, list[str]]] = {}
        self.registry_summary: dict[str, set] = {}
        self.cache_configs: dict[str, RopeCacheConfig] = {}


_ROPE_STATE = RopeGlobalState()


class RopeDataProxy:
    def __init__(self, data_map, is_cos=True):
        self._data = data_map
        self.idx = 0 if is_cos else 1

    def __getitem__(self, index):
        if not isinstance(index, str):
            new_map: dict = {}
            for config_k, groups_map in self._data.items():
                new_map[config_k] = {}
                for group_name, item in groups_map.items():
                    c_val = item[0][index]
                    s_val = item[1][index]
                    new_map[config_k][group_name] = (c_val, s_val)

            return RopeDataProxy(new_map, is_cos=(self.idx == 0))

        else:
            layername = index
            info = _ROPE_STATE.layer_info.get(layername)
            if info is None:
                raise KeyError(f"Layer {layername} not registered.")

            config_key, required_groups = info

            config_data = self._data.get(config_key, {})

            layer_result = {}
            for grp in required_groups:
                if grp in config_data:
                    layer_result[grp] = config_data[grp][self.idx]
                else:
                    pass
            if len(layer_result) == 1:
                return list(layer_result.values())[0]

            return layer_result


def _precompute_freqs_cis(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    def yarn_find_correction_dim(
        num_rotations: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048,
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048,
        truncate: bool = True,
    ) -> tuple[float | int, float | int]:
        low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    def yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> torch.Tensor:
        if low == high:
            high += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        base,
        original_seq_len,
    )
    inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32)) * 1
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq


def _build_static_cache(config: RopeCacheConfig) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _precompute_freqs_cis(
        config.rotary_dim,
        config.max_position_embeddings,
        config.base,
        config.scaling_factor,
        config.beta_fast,
        config.beta_slow,
    )
    positions = torch.arange(
        config.max_position_embeddings * config.scaling_factor,
        device=config.device,
        dtype=torch.float32,
    )
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(config.dtype)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(config.dtype)
    cos = cos.to(config.device)
    sin = sin.to(config.device)
    return cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1)


def _ensure_dsa_rope_cache(config_key: str) -> bool:
    config = _ROPE_STATE.cache_configs.get(config_key)
    if config is None:
        return False

    restored = False
    if config_key not in _ROPE_STATE.static_cache:
        _ROPE_STATE.static_cache[config_key] = _build_static_cache(config)
        restored = True

    runtime_buffers = _ROPE_STATE.runtime_buffer.setdefault(config_key, {})
    for group_name in _ROPE_STATE.registry_summary.get(config_key, set()):
        if group_name in runtime_buffers:
            continue
        buf_cos = torch.ones(config.max_batch_size, 1, 1, config.rotary_dim, dtype=config.dtype, device=config.device)
        buf_sin = torch.zeros(config.max_batch_size, 1, 1, config.rotary_dim, dtype=config.dtype, device=config.device)
        runtime_buffers[group_name] = (buf_cos, buf_sin)
        restored = True
    return restored


def clear_global_dsa_rope_cache() -> bool:
    cleared = False
    if _ROPE_STATE.static_cache:
        _ROPE_STATE.static_cache.clear()
        cleared = True
    if _ROPE_STATE.runtime_buffer:
        _ROPE_STATE.runtime_buffer.clear()
        cleared = True
    return cleared


def restore_global_dsa_rope_cache() -> bool:
    restored = False
    for config_key in list(_ROPE_STATE.cache_configs):
        restored = _ensure_dsa_rope_cache(config_key) or restored
    return restored


def get_cos_and_sin_dsa(positions: torch.Tensor | dict[str, torch.Tensor], use_cache: bool = False):
    if isinstance(positions, torch.Tensor):
        pos_map = {"default": positions}
    else:
        pos_map = positions

    batch_result: dict[Any, Any] = {}

    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():
        _ensure_dsa_rope_cache(config_key)
        if config_key not in _ROPE_STATE.static_cache:
            continue
        static_cos, static_sin = _ROPE_STATE.static_cache[config_key]

        batch_result[config_key] = {}

        for group_name, pos_tensor in pos_map.items():
            if group_name not in registered_groups:
                continue

            curr_cos = static_cos[pos_tensor]
            curr_sin = static_sin[pos_tensor]

            if use_cache:
                group_buffers = _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name)

                if group_buffers is None:
                    continue

                buf_cos, buf_sin = group_buffers
                num_tokens = pos_tensor.size(0)

                buf_cos[:num_tokens].copy_(curr_cos)
                buf_sin[:num_tokens].copy_(curr_sin)

                batch_result[config_key][group_name] = (buf_cos[:num_tokens], buf_sin[:num_tokens])
            else:
                batch_result[config_key][group_name] = (curr_cos, curr_sin)

    return RopeDataProxy(batch_result, is_cos=True), RopeDataProxy(batch_result, is_cos=False)


class ComplexExpRotaryEmbedding(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layername: str,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        rope_groups: list[str] | None = None,
        **extra_kwargs,
    ) -> None:
        super().__init__()
        if rope_groups is None:
            rope_groups = ["default"]
        self.layername = layername
        self.rotary_dim = rotary_dim
        dtype = torch.get_default_dtype()
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        config_key = (
            f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
            f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}"
        )
        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)

        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)
        _ROPE_STATE.cache_configs[config_key] = RopeCacheConfig(
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scaling_factor=scaling_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            dtype=dtype,
            device=current_platform.device_type,
            max_batch_size=vllm_config.scheduler_config.max_num_batched_tokens,
        )

        if config_key not in _ROPE_STATE.static_cache:
            _ROPE_STATE.static_cache[config_key] = _build_static_cache(_ROPE_STATE.cache_configs[config_key])

        if config_key not in _ROPE_STATE.runtime_buffer:
            _ROPE_STATE.runtime_buffer[config_key] = {}

        target_device = current_platform.device_type
        max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        for grp in rope_groups:
            if grp not in _ROPE_STATE.runtime_buffer[config_key]:
                buf_cos = torch.ones(max_batch_size, 1, 1, rotary_dim, dtype=dtype, device=target_device)
                buf_sin = torch.zeros(max_batch_size, 1, 1, rotary_dim, dtype=dtype, device=target_device)
                _ROPE_STATE.runtime_buffer[config_key][grp] = (buf_cos, buf_sin)

    @staticmethod
    def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
        return _precompute_freqs_cis(dim, original_seq_len, base, factor, beta_fast, beta_slow)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        ori_shape = x.shape
        y = x

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

        y.copy_(x.view(ori_shape))
        return y

    def extra_repr(self) -> str:
        return f"layername={self.layername}, rotary_dim={self.rotary_dim}"
