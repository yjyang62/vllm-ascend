# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest import mock

from vllm_ascend.attention.dsa_kv_mode import resolve_dsv4_cache_dtype, uses_explicit_bf16_kv
from vllm_ascend.utils import AscendDeviceType


def _config(cache_dtype: str = "auto"):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


def _on(device_type):
    return mock.patch("vllm_ascend.attention.dsa_kv_mode.get_ascend_device_type", return_value=device_type)


def test_only_explicit_bfloat16_selects_bf16_kv_on_a5():
    with _on(AscendDeviceType.A5):
        assert uses_explicit_bf16_kv(_config("bfloat16"))
        assert not uses_explicit_bf16_kv(_config())
        assert not uses_explicit_bf16_kv(_config("fp8"))
        # The A5 spec path rewrites cache_dtype once FP8 KV is chosen.
        assert not uses_explicit_bf16_kv(_config("float8_e4m3fn"))


def test_non_a5_never_uses_bf16_kv():
    with _on(AscendDeviceType.A3):
        assert not uses_explicit_bf16_kv(_config("bfloat16"))


def test_non_a5_pins_cache_dtype_to_the_model_dtype():
    with _on(AscendDeviceType.A3):
        for launch in ("auto", "bfloat16", "fp8"):
            assert resolve_dsv4_cache_dtype(launch, "bfloat16") == "bfloat16"


def test_a5_collapses_non_bfloat16_requests_to_auto():
    # "auto" resolves to the model dtype everywhere downstream, so it carries
    # the FP8 mode without changing any value upstream would have computed.
    with _on(AscendDeviceType.A5):
        assert resolve_dsv4_cache_dtype("bfloat16", "bfloat16") == "bfloat16"
        assert resolve_dsv4_cache_dtype("auto", "bfloat16") == "auto"
        assert resolve_dsv4_cache_dtype("fp8", "bfloat16") == "auto"


def test_a5_mode_survives_the_spec_path_rewrite():
    with _on(AscendDeviceType.A5):
        for launch in ("auto", "fp8"):
            pinned = resolve_dsv4_cache_dtype(launch, "bfloat16")
            assert not uses_explicit_bf16_kv(_config(pinned))
            # layer.get_kv_cache_spec pins FP8 once it has picked the mode.
            assert not uses_explicit_bf16_kv(_config("float8_e4m3fn"))

        pinned = resolve_dsv4_cache_dtype("bfloat16", "bfloat16")
        assert uses_explicit_bf16_kv(_config(pinned))
