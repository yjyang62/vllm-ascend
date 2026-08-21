# SPDX-License-Identifier: Apache-2.0
"""Reproduce the triton-ascend + vLLM gluon ImportError and verify the shield.

Loaded by file path so this test does not import ``vllm_ascend.__init__``
(which pulls ``vllm`` via the logger sidecar).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPAT_PATH = _REPO_ROOT / "vllm_ascend" / "triton_gluon_compat.py"


def _load_compat():
    spec = importlib.util.spec_from_file_location("triton_gluon_compat_under_test", _COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_compat()
install_triton_gluon_compat = compat.install_triton_gluon_compat
_GLUON_MODULE = compat._GLUON_MODULE
_GLUON_LANGUAGE_MODULE = compat._GLUON_LANGUAGE_MODULE
_STUB_FLAG = compat._STUB_FLAG

_BROKEN_GLUON_LAYOUTS = textwrap.dedent(
    """
    from triton.language.core import _unwrap_if_constexpr, _unwrap_shape, constexpr_type


    class SharedLayout:
        pass


    class DistributedLayout:
        pass


    class NVMMASharedLayout:
        pass
    """
).lstrip()


def _write_broken_triton(root: Path) -> None:
    """Mirror the triton 3.5.0 + triton-ascend overlay that crashes vLLM."""
    (root / "triton").mkdir()
    (root / "triton" / "__init__.py").write_text("__version__ = '3.2.0'\n")
    (root / "triton" / "language").mkdir()
    (root / "triton" / "language" / "__init__.py").write_text("")
    # triton-ascend overlay: no constexpr_type, no _aggregate.
    (root / "triton" / "language" / "core.py").write_text(
        "def _unwrap_if_constexpr(x):\n    return x\n\ndef _unwrap_shape(x):\n    return x\n"
    )
    (root / "triton" / "experimental").mkdir()
    (root / "triton" / "experimental" / "__init__.py").write_text("")
    gluon = root / "triton" / "experimental" / "gluon"
    gluon.mkdir()
    (gluon / "__init__.py").write_text("from . import nvidia\n")
    (gluon / "nvidia").mkdir()
    (gluon / "nvidia" / "__init__.py").write_text("from . import hopper\n")
    (gluon / "nvidia" / "hopper.py").write_text(
        "from triton.experimental.gluon.language._layouts import NVMMASharedLayout\n"
    )
    (gluon / "language").mkdir()
    (gluon / "language" / "__init__.py").write_text("from ._core import *  # noqa: F403\n")
    (gluon / "language" / "_core.py").write_text("from ._layouts import SharedLayout, DistributedLayout\n")
    (gluon / "language" / "_layouts.py").write_text(_BROKEN_GLUON_LAYOUTS)


def _drop_triton_modules() -> dict[str, object]:
    saved = {name: module for name, module in sys.modules.items() if name == "triton" or name.startswith("triton.")}
    for name in saved:
        del sys.modules[name]
    importlib.invalidate_caches()
    return saved


@pytest.fixture
def broken_triton(tmp_path, monkeypatch):
    _write_broken_triton(tmp_path)
    saved = _drop_triton_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield tmp_path
    _drop_triton_modules()
    sys.modules.update(saved)


def _vllm_triton_utils_imports():
    """The exact import block from upstream vllm/triton_utils/__init__.py."""
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.language.core import _aggregate as aggregate

    return gluon, gl, aggregate


def test_broken_gluon_matches_user_importerror(broken_triton):
    with pytest.raises(ImportError, match="constexpr_type"):
        from triton.experimental import gluon  # noqa: F401


def test_bare_moduletype_stub_missing_language_attr(broken_triton):
    """The previous main stub put both names in sys.modules without .language."""
    sys.modules[_GLUON_MODULE] = types.ModuleType(_GLUON_MODULE)
    sys.modules[_GLUON_LANGUAGE_MODULE] = types.ModuleType(_GLUON_LANGUAGE_MODULE)
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    assert getattr(gluon, "language", None) is None
    assert gl is sys.modules[_GLUON_LANGUAGE_MODULE]

    install_triton_gluon_compat()
    gluon2, gl2, _aggregate = _vllm_triton_utils_imports()
    assert gluon2.language is gl2
    assert hasattr(gluon2, "__path__")


def test_install_unblocks_vllm_triton_utils_imports(broken_triton):
    with pytest.raises(ImportError, match="constexpr_type"):
        _vllm_triton_utils_imports()

    install_triton_gluon_compat()
    gluon, gl, aggregate = _vllm_triton_utils_imports()

    assert getattr(gluon, _STUB_FLAG)
    assert gluon.language is gl
    assert aggregate is not None
    assert aggregate(int) is int


def test_install_upgrades_incomplete_stub(broken_triton):
    sys.modules[_GLUON_MODULE] = types.ModuleType(_GLUON_MODULE)
    sys.modules[_GLUON_LANGUAGE_MODULE] = types.ModuleType(_GLUON_LANGUAGE_MODULE)

    install_triton_gluon_compat()
    gluon, gl, _aggregate = _vllm_triton_utils_imports()
    assert hasattr(gluon, "__path__")
    assert gluon.language is gl


def test_install_is_idempotent(broken_triton):
    install_triton_gluon_compat()
    first = sys.modules[_GLUON_MODULE]
    install_triton_gluon_compat()
    assert sys.modules[_GLUON_MODULE] is first


def test_install_noop_without_triton(monkeypatch):
    saved = _drop_triton_modules()
    try:
        install_triton_gluon_compat()
        assert _GLUON_MODULE not in sys.modules
    finally:
        _drop_triton_modules()
        sys.modules.update(saved)


def test_vllm_version_026_still_stubs(broken_triton, monkeypatch):
    monkeypatch.setenv("VLLM_VERSION", "0.26.0")
    install_triton_gluon_compat()
    gluon, gl, _aggregate = _vllm_triton_utils_imports()
    assert gluon.language is gl


def test_package_hooks_call_installer():
    init_src = (_REPO_ROOT / "vllm_ascend" / "__init__.py").read_text()
    platform_src = (_REPO_ROOT / "vllm_ascend" / "platform.py").read_text()
    assert "install_triton_gluon_compat()" in init_src
    assert "install_triton_gluon_compat()" in platform_src
    assert 'os.getenv("VLLM_VERSION", "") != "0.26.0"' not in init_src
