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
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.verify_token_in_token_out import (
    TokenIOError,
    completions_token_ids,
    generate_tokens,
    generate_tokens_stream,
    list_models,
    main,
    run_verification,
    tokenize_prompt,
)

PROMPT_IDS = [151644, 8948, 198]
OUTPUT_IDS = [123, 456, 789]


def _json_response(payload: dict, status: int = 200, *, stream_lines: list[str] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.text = json.dumps(payload)
    response.json.return_value = payload
    if stream_lines is not None:
        response.iter_lines.return_value = stream_lines
    return response


def test_list_models_and_tokenize():
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.side_effect = [
            _json_response({"data": [{"id": "auto"}]}),
            _json_response({"tokens": PROMPT_IDS, "count": len(PROMPT_IDS)}),
        ]
        assert list_models("http://127.0.0.1:8008", timeout=5) == ["auto"]
        assert tokenize_prompt("http://127.0.0.1:8008", "hi", timeout=5) == PROMPT_IDS


def test_generate_and_completions_match_token_in_out():
    generate_payload = {"choices": [{"token_ids": OUTPUT_IDS, "finish_reason": "length"}]}
    completion_payload = {
        "choices": [{"token_ids": OUTPUT_IDS, "text": "Paris"}],
        "usage": {"prompt_tokens": len(PROMPT_IDS), "completion_tokens": len(OUTPUT_IDS)},
    }
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.side_effect = [_json_response(generate_payload), _json_response(completion_payload)]
        generate_body = generate_tokens(
            "http://127.0.0.1:8008",
            "auto",
            PROMPT_IDS,
            max_tokens=len(OUTPUT_IDS),
            timeout=5,
        )
        completion_body = completions_token_ids(
            "http://127.0.0.1:8008",
            "auto",
            PROMPT_IDS,
            max_tokens=len(OUTPUT_IDS),
            timeout=5,
        )
    assert generate_body["_output_token_ids"] == OUTPUT_IDS
    assert completion_body["usage"]["prompt_tokens"] == len(PROMPT_IDS)
    assert completion_body["usage"]["completion_tokens"] == len(OUTPUT_IDS)


def test_generate_stream_concatenates_incremental_token_ids():
    lines = [
        "data: " + json.dumps({"choices": [{"token_ids": [123]}]}),
        "data: " + json.dumps({"choices": [{"token_ids": [456, 789]}]}),
        "data: [DONE]",
    ]
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.return_value = _json_response({}, stream_lines=lines)
        collected = generate_tokens_stream(
            "http://127.0.0.1:8008",
            "auto",
            PROMPT_IDS,
            max_tokens=len(OUTPUT_IDS),
            timeout=5,
        )
    assert collected == OUTPUT_IDS


def test_run_verification_happy_path():
    generate_payload = {"choices": [{"token_ids": OUTPUT_IDS, "finish_reason": "length"}]}
    completion_payload = {
        "choices": [{"token_ids": OUTPUT_IDS}],
        "usage": {"prompt_tokens": len(PROMPT_IDS), "completion_tokens": len(OUTPUT_IDS)},
    }
    stream_lines = [
        "data: " + json.dumps({"choices": [{"token_ids": OUTPUT_IDS}]}),
        "data: [DONE]",
    ]
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.side_effect = [
            _json_response({"data": [{"id": "auto"}]}),
            _json_response({"tokens": PROMPT_IDS}),
            _json_response(generate_payload),
            _json_response({}, stream_lines=stream_lines),
            _json_response(completion_payload),
        ]
        results = run_verification("http://127.0.0.1:8008", "auto", max_tokens=len(OUTPUT_IDS), timeout=5)
    assert [item.name for item in results] == [
        "v1_models",
        "tokenize",
        "generate",
        "generate_stream",
        "completions_return_token_ids",
    ]
    assert all(item.passed for item in results)


def test_prompt_token_mismatch_fails():
    completion_payload = {
        "choices": [{"token_ids": OUTPUT_IDS}],
        "usage": {"prompt_tokens": 1, "completion_tokens": len(OUTPUT_IDS)},
    }
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.return_value = _json_response(completion_payload)
        with pytest.raises(TokenIOError, match="prompt_tokens"):
            completions_token_ids(
                "http://127.0.0.1:8008",
                "auto",
                PROMPT_IDS,
                max_tokens=len(OUTPUT_IDS),
                timeout=5,
            )


def test_http_error_is_token_io_error():
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.return_value = _json_response({"error": "missing"}, status=404)
        with pytest.raises(TokenIOError, match="HTTP 404"):
            list_models("http://127.0.0.1:8008", timeout=5)


def test_main_returns_nonzero_on_failure(capsys):
    with patch("tools.verify_token_in_token_out.requests.request") as request:
        request.side_effect = __import__("requests").RequestException("connection refused")
        assert main(["--base-url", "http://127.0.0.1:8008", "--model", "auto"]) == 1
    assert "FAIL" in capsys.readouterr().err
