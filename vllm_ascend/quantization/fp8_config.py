from __future__ import annotations

import json
from typing import Any, Optional, cast

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import MoERunner, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm_ascend.utils import FP8_METHOD

from .methods import get_scheme_class

# DeepSeek-V4 expert storage dtypes that must not allocate MXFP4 (hidden//2) MoE buffers.
_UNQUANTIZED_EXPERT_DTYPES = frozenset(
    {
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "fp32",
        "float32",
        "none",
        "null",
    }
)
_PACKED_EXPERT_DTYPES = frozenset({"fp4", "fp8", "mxfp4", "nvfp4"})


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, (MoERunner, RoutedExperts))


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


def remove_quantization_method():
    if FP8_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(FP8_METHOD)
    if "deepseek_v4_fp8" in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove("deepseek_v4_fp8")


remove_quantization_method()


def create_scheme_for_layer(
    quant_description: dict[str, Any],
    prefix: str,
    layer_type: str,
    packed_modules_mapping: dict[str, Any] | None = None,
):
    """Create a quantization scheme instance for a layer.

    Args:
        quant_description: The quantization description dictionary.
        prefix: The layer prefix.
        layer_type: The type of layer ("linear", "moe", "attention").
        packed_modules_mapping: Mapping for packed/fused modules.

    Returns:
        An instance of the appropriate quantization scheme class.
    """
    logger.info_once("Using the vLLM Ascend fp8 Quantization now!")
    quant_type = "FP8"

    # Use registry to get scheme class
    scheme_cls = get_scheme_class(quant_type, layer_type)
    if scheme_cls is not None:
        return scheme_cls(quant_description)

    raise NotImplementedError(f"Currently, vLLM Ascend doesn't support {quant_type} for {layer_type}.")


def _normalize_expert_dtype(expert_dtype: Any) -> str | None:
    if expert_dtype is None:
        return None
    text = str(expert_dtype).strip().lower()
    return text or None


def _iter_checkpoint_weight_names(model: str, revision: str | None = None) -> list[str] | None:
    """Best-effort list of on-disk parameter names for *model*."""
    from vllm_ascend.quantization.utils import get_model_file

    index_path = get_model_file(model, "model.safetensors.index.json", revision=revision)
    if index_path is not None:
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
            return list(weight_map.keys())
        except Exception as exc:
            logger.warning_once("Failed to read safetensors index for %s: %s", model, exc)
            return None

    single = get_model_file(model, "model.safetensors", revision=revision)
    if single is None:
        return None
    try:
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return list(handle.keys())
    except Exception as exc:
        logger.warning_once("Failed to inspect safetensors keys for %s: %s", model, exc)
        return None


def checkpoint_experts_look_unquantized(model: str, revision: str | None = None) -> bool | None:
    """Return whether DeepSeek-V4 expert tensors look like dense BF16/FP weights.

    True  – expert weights exist and no expert scale tensors are present.
    False – expert scale tensors are present (MXFP4/FP8 style).
    None  – cannot decide (missing files / no expert keys).
    """
    names = _iter_checkpoint_weight_names(model, revision=revision)
    if not names:
        return None

    expert_names = [name for name in names if ".experts." in name or ".mlp.experts" in name]
    if not expert_names:
        return None

    has_expert_scale = any(
        ("scale" in name.lower()) and ("expert" in name.lower() or ".experts." in name) for name in expert_names
    )
    return not has_expert_scale


def should_use_unquantized_dsv4_moe(hf_config: Any, model: str | None = None, revision: str | None = None) -> bool:
    """Whether Ascend should allocate dense (unquantized) MoE expert buffers.

    Flash MXFP4 allocates ``w13`` with ``hidden_size // 2``. Loading BF16
    ``gate_proj`` weights shaped ``[moe_intermediate, hidden]`` into that
    buffer fails with:
      Target sizes: [I, H/2].  Tensor sizes: [I, H]
    """
    expert_dtype = _normalize_expert_dtype(getattr(hf_config, "expert_dtype", None))
    if expert_dtype in _UNQUANTIZED_EXPERT_DTYPES:
        return True
    if expert_dtype in _PACKED_EXPERT_DTYPES:
        # Some BF16 HF exports keep expert_dtype=fp4 by mistake. Fall back to
        # checkpoint inspection when the model path is available.
        if model is None:
            return False
        looks_unquantized = checkpoint_experts_look_unquantized(model, revision=revision)
        if looks_unquantized:
            logger.warning_once(
                "DeepSeek-V4 hf_config.expert_dtype=%r but checkpoint expert "
                "weights have no scale tensors; using unquantized MoE buffers "
                "so BF16 gate/up/down weights can load (avoid hidden//2 MXFP4 layout).",
                expert_dtype,
            )
            return True
        return False

    # expert_dtype missing: prefer checkpoint probe, else keep MXFP4 default
    # for legacy Flash configs that omit the field.
    if model is not None:
        looks_unquantized = checkpoint_experts_look_unquantized(model, revision=revision)
        if looks_unquantized is not None:
            if looks_unquantized:
                logger.info_once(
                    "DeepSeek-V4 expert_dtype is unset and checkpoint experts "
                    "have no scale tensors; using unquantized MoE buffers."
                )
            return looks_unquantized
    return False


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(QuantizationConfig):
    def __init__(
        self,
        ignore: list[str],
        quant_format: str,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_format = quant_format
        self.quant_description = config if config is not None else {}
        self._resolved_use_unquantized_moe: bool | None = None

    def __repr__(self) -> str:
        return "Fp8Config:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return FP8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float8_e4m3fn, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendFp8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        quant_format = cast(str, config.get("format"))

        return cls(
            ignore=ignore,
            quant_format=quant_format,
            config=config,
        )

    def _should_use_unquantized_moe(self) -> bool:
        if self._resolved_use_unquantized_moe is not None:
            return self._resolved_use_unquantized_moe
        try:
            vllm_config = get_current_vllm_config()
            model_config = vllm_config.model_config
            hf_config = model_config.hf_config
            model = getattr(model_config, "model", None)
            revision = getattr(model_config, "revision", None)
        except Exception:
            self._resolved_use_unquantized_moe = False
            return False

        model_type = getattr(hf_config, "model_type", None)
        if model_type != "deepseek_v4":
            self._resolved_use_unquantized_moe = False
            return False

        self._resolved_use_unquantized_moe = should_use_unquantized_dsv4_moe(
            hf_config,
            model=model,
            revision=revision,
        )
        return self._resolved_use_unquantized_moe

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = FP8_METHOD

            scheme = create_scheme_for_layer(self.quant_description, prefix, "ds_linear", self.packed_modules_mapping)
            quant_method = AscendLinearMethod(scheme)
            return quant_method
        if _is_fused_moe_layer(layer):
            # BF16 / dense expert checkpoints must not use MXFP4 w13 layout
            # (hidden_size // 2); that causes gate_proj copy_ shape errors.
            if self._should_use_unquantized_moe():
                from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

                layer.ascend_quant_method = "unquantized"
                return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid=tid2eid)

            layer.ascend_quant_method = FP8_METHOD
            scheme = create_scheme_for_layer(self.quant_description, prefix, "w4a8_moe", self.packed_modules_mapping)
            quant_method = AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


# deepseek_v4_fp8 is handled identically to fp8 on Ascend — reuse the same config.
register_quantization_config("deepseek_v4_fp8")(AscendFp8Config)
