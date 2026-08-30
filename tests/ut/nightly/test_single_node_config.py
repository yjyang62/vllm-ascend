# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import os

from tests.e2e.nightly.single_node.models.scripts.local_proxy import (
    LOCAL_NO_PROXY_HOSTS,
    extend_no_proxy_for_local_server,
)


def test_extend_no_proxy_adds_local_hosts(monkeypatch):
    monkeypatch.setenv("no_proxy", "local,.local,*.huawei.com")
    monkeypatch.delenv("NO_PROXY", raising=False)

    extend_no_proxy_for_local_server()

    no_proxy = [p.strip() for p in os.environ["no_proxy"].split(",") if p.strip()]
    no_proxy_upper = [p.strip() for p in os.environ["NO_PROXY"].split(",") if p.strip()]
    for host in LOCAL_NO_PROXY_HOSTS:
        assert host in no_proxy
        assert host in no_proxy_upper
    assert "local" in no_proxy
    assert "*.huawei.com" in no_proxy
