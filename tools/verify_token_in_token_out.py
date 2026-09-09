#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Verify Token In / Token Out against a running vLLM server.

Default target matches the Qwen3.5-35B-A3B serve command:

    vllm serve /mnt/share/weights/Qwen3.5-35B-A3B \\
      --tensor-parallel-size 8 --enforce-eager --max_model_len 4096 \\
      --gpu-memory-utilization 0.9 --served-model-name auto \\
      --max-num-seqs 16 --port 8008

Checks:

1. ``GET /v1/models`` is ready and exposes the served model name.
2. ``POST /tokenize`` converts a text prompt into token IDs (token in).
3. ``POST /inference/v1/generate`` accepts those IDs and returns
   ``choices[].token_ids`` (token out), non-streaming and streaming.
4. OpenAI ``/v1/completions`` accepts a token-id prompt with
   ``return_token_ids=true`` and reports matching usage counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running `python tools/verify_token_in_token_out.py` puts this directory on
# sys.path[0] and shadows the stdlib `bisect` module with tools/bisect.
_TOOLS_DIR = Path(__file__).resolve().parent
if __package__ in {None, ""} and sys.path and Path(sys.path[0]).resolve() == _TOOLS_DIR:
    sys.path.pop(0)
    _repo_root = str(_TOOLS_DIR.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import requests  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8008"
DEFAULT_MODEL = "auto"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 16
DEFAULT_TIMEOUT = 600.0


class TokenIOError(RuntimeError):
    """Raised when a Token In / Token Out check fails."""


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


def _join_url(base_url: str, *parts: str) -> str:
    return base_url.rstrip("/") + "/" + "/".join(parts)


def _request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    json_body: dict[str, Any] | None = None,
    stream: bool = False,
) -> requests.Response:
    response = requests.request(method, url, json=json_body, timeout=timeout, stream=stream)
    if response.status_code >= 400:
        body = response.text[:2000]
        raise TokenIOError(f"{method} {url} failed: HTTP {response.status_code}: {body}")
    return response


def list_models(base_url: str, timeout: float) -> list[str]:
    response = _request_json("GET", _join_url(base_url, "v1", "models"), timeout=timeout)
    data = response.json()
    models = [item.get("id") for item in data.get("data", []) if item.get("id")]
    if not models:
        raise TokenIOError(f"/v1/models returned no model ids: {data}")
    return models


def tokenize_prompt(base_url: str, prompt: str, timeout: float) -> list[int]:
    response = _request_json(
        "POST",
        _join_url(base_url, "tokenize"),
        timeout=timeout,
        json_body={"prompt": prompt},
    )
    body = response.json()
    token_ids = body.get("tokens") or body.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise TokenIOError(f"/tokenize returned no token ids: {body}")
    if not all(isinstance(token, int) and token >= 0 for token in token_ids):
        raise TokenIOError(f"/tokenize returned invalid token ids: {token_ids!r}")
    return token_ids


def _assert_token_ids(token_ids: Any, *, expected_len: int | None = None, where: str) -> list[int]:
    if not isinstance(token_ids, list) or not token_ids:
        raise TokenIOError(f"{where} returned empty token_ids: {token_ids!r}")
    if not all(isinstance(token, int) and token >= 0 for token in token_ids):
        raise TokenIOError(f"{where} returned invalid token_ids: {token_ids!r}")
    if expected_len is not None and len(token_ids) != expected_len:
        raise TokenIOError(f"{where} token_ids length {len(token_ids)} != expected {expected_len}")
    return token_ids


def generate_tokens(
    base_url: str,
    model: str,
    token_ids: Sequence[int],
    *,
    max_tokens: int,
    timeout: float,
    ignore_eos: bool = True,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "token_ids": list(token_ids),
        "sampling_params": {
            "max_tokens": max_tokens,
            "min_tokens": max_tokens if ignore_eos else 0,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "detokenize": False,
        },
        "stream": False,
    }
    response = _request_json(
        "POST",
        _join_url(base_url, "inference", "v1", "generate"),
        timeout=timeout,
        json_body=payload,
    )
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise TokenIOError(f"/inference/v1/generate returned no choices: {body}")
    out_ids = _assert_token_ids(
        choices[0].get("token_ids"),
        expected_len=max_tokens if ignore_eos else None,
        where="/inference/v1/generate",
    )
    body["_output_token_ids"] = out_ids
    return body


def generate_tokens_stream(
    base_url: str,
    model: str,
    token_ids: Sequence[int],
    *,
    max_tokens: int,
    timeout: float,
    ignore_eos: bool = True,
    temperature: float = 0.0,
) -> list[int]:
    payload = {
        "model": model,
        "token_ids": list(token_ids),
        "sampling_params": {
            "max_tokens": max_tokens,
            "min_tokens": max_tokens if ignore_eos else 0,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "detokenize": False,
        },
        "stream": True,
    }
    response = _request_json(
        "POST",
        _join_url(base_url, "inference", "v1", "generate"),
        timeout=timeout,
        json_body=payload,
        stream=True,
    )
    collected: list[int] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if not line or line == "[DONE]":
            continue
        chunk = json.loads(line)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        piece = choices[0].get("token_ids") or []
        if isinstance(piece, list):
            collected.extend(int(token) for token in piece)
    _assert_token_ids(
        collected,
        expected_len=max_tokens if ignore_eos else None,
        where="/inference/v1/generate (stream)",
    )
    return collected


def completions_token_ids(
    base_url: str,
    model: str,
    token_ids: Sequence[int],
    *,
    max_tokens: int,
    timeout: float,
    ignore_eos: bool = True,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": list(token_ids),
        "max_tokens": max_tokens,
        "min_tokens": max_tokens if ignore_eos else 0,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "return_token_ids": True,
    }
    response = _request_json(
        "POST",
        _join_url(base_url, "v1", "completions"),
        timeout=timeout,
        json_body=payload,
    )
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise TokenIOError(f"/v1/completions returned no choices: {body}")
    out_ids = _assert_token_ids(
        choices[0].get("token_ids"),
        expected_len=max_tokens if ignore_eos else None,
        where="/v1/completions",
    )
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens != len(token_ids):
        raise TokenIOError(f"usage.prompt_tokens={prompt_tokens} != token-in length {len(token_ids)}")
    if ignore_eos and completion_tokens != max_tokens:
        raise TokenIOError(f"usage.completion_tokens={completion_tokens} != token-out length {max_tokens}")
    body["_output_token_ids"] = out_ids
    return body


def run_verification(
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    *,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
    skip_stream: bool = False,
) -> list[CheckResult]:
    """Run Token In / Token Out checks. Raises TokenIOError on the first failure."""
    results: list[CheckResult] = []

    models = list_models(base_url, timeout)
    if model not in models:
        raise TokenIOError(f"served model {model!r} not in /v1/models: {models}")
    results.append(CheckResult("v1_models", True, f"models={models}"))

    prompt_token_ids = tokenize_prompt(base_url, prompt, timeout)
    results.append(
        CheckResult(
            "tokenize",
            True,
            f"prompt={prompt!r} token_in={len(prompt_token_ids)}",
            {"token_ids": prompt_token_ids},
        )
    )

    generate_body = generate_tokens(
        base_url,
        model,
        prompt_token_ids,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    results.append(
        CheckResult(
            "generate",
            True,
            f"token_out={len(generate_body['_output_token_ids'])} "
            f"finish_reason={generate_body['choices'][0].get('finish_reason')}",
            {"token_ids": generate_body["_output_token_ids"]},
        )
    )

    if not skip_stream:
        stream_ids = generate_tokens_stream(
            base_url,
            model,
            prompt_token_ids,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        results.append(
            CheckResult(
                "generate_stream",
                True,
                f"stream_token_out={len(stream_ids)}",
                {"token_ids": stream_ids},
            )
        )

    completion_body = completions_token_ids(
        base_url,
        model,
        prompt_token_ids,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    usage = completion_body.get("usage") or {}
    results.append(
        CheckResult(
            "completions_return_token_ids",
            True,
            f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}",
            {"token_ids": completion_body["_output_token_ids"], "usage": usage},
        )
    )
    return results


def _print_results(results: Sequence[CheckResult]) -> None:
    print("Token In / Token Out verification")
    print("-" * 48)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--skip-stream", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = run_verification(
            args.base_url,
            args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            skip_stream=args.skip_stream,
        )
    except (TokenIOError, requests.RequestException, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    _print_results(results)
    print("All Token In / Token Out checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
