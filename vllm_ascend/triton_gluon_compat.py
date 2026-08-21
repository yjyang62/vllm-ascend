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
"""Import-time shield for NVIDIA-only Triton gluon under triton-ascend.

Upstream vLLM does::

    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

whenever ``HAS_TRITON`` is true. ``triton-ascend`` 3.2.x depends on the public
``triton==3.5.0`` wheel, whose gluon imports ``constexpr_type`` from
``triton.language.core``. The triton-ascend overlay of ``language.core`` does
not provide that symbol, so ``vllm serve`` dies during ``import vllm.config``.

This module is imported from ``vllm_ascend/__init__.py`` at plugin discovery
time, which is before ``vllm.triton_utils`` runs those imports. A bare
``ModuleType`` in ``sys.modules`` can satisfy ``from triton.experimental
import gluon``, but it does not set ``gluon.language``. This installer
creates a package placeholder with ``__path__`` and a ``language``
attribute, and adds ``triton.language.core._aggregate`` when missing.

Keep this module free of ``vllm`` imports so it can run at
``vllm_ascend`` import time without pulling ``vllm.config`` / ``triton_utils``.
"""

from __future__ import annotations

import logging
import sys
import types
from importlib import import_module

logger = logging.getLogger(__name__)

_GLUON_MODULE = "triton.experimental.gluon"
_GLUON_LANGUAGE_MODULE = "triton.experimental.gluon.language"
_STUB_FLAG = "_VLLM_ASCEND_GLUON_STUB"


def _ensure_module(name: str, *, is_package: bool = True) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = _ensure_module(parent_name, is_package=True)
            setattr(parent, attr, module)
    if is_package and not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _purge_gluon_modules() -> None:
    for name in list(sys.modules):
        if name == _GLUON_MODULE or name.startswith(_GLUON_MODULE + "."):
            sys.modules.pop(name, None)


def _install_gluon_stubs() -> None:
    gluon = _ensure_module(_GLUON_MODULE, is_package=True)
    language = _ensure_module(_GLUON_LANGUAGE_MODULE, is_package=True)
    gluon.language = language
    setattr(gluon, _STUB_FLAG, True)
    setattr(language, _STUB_FLAG, True)


def _ensure_aggregate() -> None:
    try:
        from triton.language.core import _aggregate  # noqa: F401

        return
    except Exception:
        pass
    try:
        core = import_module("triton.language.core")
    except Exception:
        return
    if hasattr(core, "_aggregate"):
        return

    def _aggregate(cls):
        return cls

    core._aggregate = _aggregate


def _triton_available() -> bool:
    if "triton" in sys.modules:
        return True
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _core_lacks_constexpr_type() -> bool:
    try:
        core = import_module("triton.language.core")
    except Exception:
        return False
    return not hasattr(core, "constexpr_type")


def _is_complete_stub() -> bool:
    gluon = sys.modules.get(_GLUON_MODULE)
    language = sys.modules.get(_GLUON_LANGUAGE_MODULE)
    if gluon is None or language is None:
        return False
    if not getattr(gluon, _STUB_FLAG, False):
        return False
    if not hasattr(gluon, "__path__"):
        return False
    return getattr(gluon, "language", None) is language


def _gluon_importable() -> bool:
    try:
        gluon = import_module(_GLUON_MODULE)
        language = import_module(_GLUON_LANGUAGE_MODULE)
    except Exception:
        return False
    return getattr(gluon, "language", None) is language


def install_triton_gluon_compat() -> None:
    """Install NVIDIA gluon placeholders when the real module cannot import."""
    if _is_complete_stub():
        _ensure_aggregate()
        return
    if not _triton_available():
        return

    need_stub = _core_lacks_constexpr_type() or not _gluon_importable()
    if need_stub:
        _purge_gluon_modules()
        _install_gluon_stubs()
        logger.info(
            "NVIDIA Triton gluon is unavailable on this Ascend/triton-ascend "
            "stack. Installing a placeholder so vLLM can import HAS_TRITON "
            "without crashing."
        )

    _ensure_aggregate()
