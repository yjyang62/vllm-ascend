# SPDX-License-Identifier: Apache-2.0
"""Guard the A5 custom-op list and csrc/build CMake cache invalidation."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_ACLNN = REPO_ROOT / "csrc" / "build_aclnn.sh"
CMAKE_LISTS = REPO_ROOT / "csrc" / "CMakeLists.txt"
SYMBOL_CMAKE = REPO_ROOT / "csrc" / "cmake" / "symbol.cmake"

A3_ONLY_METADATA_OP = "sparse_attn_sharedkv_metadata"
A5_METADATA_OP = "kv_quant_sparse_attn_sharedkv_metadata"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_custom_ops_arrays(script: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current: str | None = None
    in_array = False
    ops: list[str] = []
    for line in script.splitlines():
        match = re.search(r'matched SOC branch: (\S+)"', line)
        if match:
            current = match.group(1)
        if current is not None and "CUSTOM_OPS_ARRAY=(" in line:
            in_array = True
            ops = []
            continue
        if in_array:
            stripped = line.strip()
            if stripped.startswith(")"):
                result[current] = ops
                in_array = False
                current = None
                continue
            if stripped.startswith("#") or not stripped:
                continue
            ops.append(stripped.strip('"'))
    return result


def _helper_script() -> str:
    text = _read(BUILD_ACLNN)
    start = text.index("resolve_cann_package_path()")
    end = text.index('\nlog "start:')
    return 'log() { echo "[build_aclnn] $*"; }\n' + text[start:end]


def _run_bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        check=False,
    )


class BuildAclnnSocOpsTest(unittest.TestCase):
    def test_a5_uses_kv_quant_metadata_not_a3_sharedkv_metadata(self):
        ops = _parse_custom_ops_arrays(_read(BUILD_ACLNN))
        self.assertIn("ascend950", ops)
        self.assertIn("ascend910_93", ops)
        self.assertIn(A5_METADATA_OP, ops["ascend950"])
        self.assertNotIn(A3_ONLY_METADATA_OP, ops["ascend950"])
        self.assertIn(A3_ONLY_METADATA_OP, ops["ascend910_93"])


class BuildAclnnCmakeGuardTest(unittest.TestCase):
    def test_cmake_resets_aicpu_object_target_cache_each_configure(self):
        cmake = _read(CMAKE_LISTS)
        self.assertIn(
            'set(AICPU_CUST_OBJ_TARGETS "" CACHE INTERNAL "All aicpu cust obj targets" FORCE)',
            cmake,
        )

    def test_symbol_skips_stale_aicpu_object_targets(self):
        symbol = _read(SYMBOL_CMAKE)
        self.assertIn("Skipping stale AICPU object target", symbol)
        self.assertIn("EXISTING_AICPU_TARGETS", symbol)


class BuildAclnnCacheInvalidationTest(unittest.TestCase):
    def test_wipes_build_dir_without_cache_key(self):
        helpers = _helper_script()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").write_text(
                "CUSTOM_ASCEND_CANN_PACKAGE_PATH:PATH=/usr/local/Ascend/cann-9.1.0\n",
                encoding="utf-8",
            )
            marker = build_dir / "stale.ninja"
            marker.write_text("leftover", encoding="utf-8")
            script = (
                helpers
                + f"""
SOC_ARG=ascend950
CUSTOM_OPS=kv_quant_sparse_attn_sharedkv_metadata
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
maybe_invalidate_csrc_build "{build_dir}"
test ! -e "{marker}"
"""
            )
            result = _run_bash(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("has no cache key", result.stdout)

    def test_wipes_build_dir_when_cached_cann_path_differs(self):
        helpers = _helper_script()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            build_dir.mkdir()
            (build_dir / "CMakeCache.txt").write_text(
                "CUSTOM_ASCEND_CANN_PACKAGE_PATH:PATH=/usr/local/Ascend/cann-9.1.T560\n",
                encoding="utf-8",
            )
            (build_dir / ".aclnn_cache_key").write_text(
                "SOC_ARG=ascend950\nCUSTOM_OPS=foo\nASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0\n",
                encoding="utf-8",
            )
            script = (
                helpers
                + f"""
SOC_ARG=ascend950
CUSTOM_OPS=foo
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
maybe_invalidate_csrc_build "{build_dir}"
test ! -d "{build_dir}"
"""
            )
            result = _run_bash(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CMakeCache CANN path", result.stdout)

    def test_reuses_build_dir_when_cache_key_matches(self):
        helpers = _helper_script()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            build_dir.mkdir()
            (build_dir / "keep.txt").write_text("ok", encoding="utf-8")
            key = "SOC_ARG=ascend950\nCUSTOM_OPS=foo\nASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0\n"
            (build_dir / ".aclnn_cache_key").write_text(key, encoding="utf-8")
            (build_dir / "CMakeCache.txt").write_text(
                "CUSTOM_ASCEND_CANN_PACKAGE_PATH:PATH=/usr/local/Ascend/cann-9.1.0\n",
                encoding="utf-8",
            )
            script = (
                helpers
                + f"""
SOC_ARG=ascend950
CUSTOM_OPS=foo
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
maybe_invalidate_csrc_build "{build_dir}"
test -f "{build_dir}/keep.txt"
"""
            )
            result = _run_bash(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("reusing", result.stdout)

    def test_check_cann_package_rejects_tree_without_include(self):
        helpers = _helper_script()
        with tempfile.TemporaryDirectory() as tmp:
            cann = Path(tmp) / "cann-9.1.T560"
            (cann / "lib64").mkdir(parents=True)
            script = helpers + "\ncheck_cann_package\n"
            result = _run_bash(script, env={"ASCEND_HOME_PATH": str(cann)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing include/", result.stdout)

    def test_check_cann_package_accepts_tree_with_include(self):
        helpers = _helper_script()
        with tempfile.TemporaryDirectory() as tmp:
            cann = Path(tmp) / "cann-9.1.0"
            (cann / "include").mkdir(parents=True)
            script = helpers + "\ncheck_cann_package\n"
            result = _run_bash(script, env={"ASCEND_HOME_PATH": str(cann)})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("resolved CANN package path=", result.stdout)


if __name__ == "__main__":
    unittest.main()
