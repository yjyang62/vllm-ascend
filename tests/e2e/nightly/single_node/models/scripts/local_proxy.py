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

LOCAL_NO_PROXY_HOSTS = ("127.0.0.1", "0.0.0.0", "localhost")


def extend_no_proxy_for_local_server() -> None:
    """Keep local serve URLs off a corporate HTTP proxy such as Squid."""
    for key in ("no_proxy", "NO_PROXY"):
        parts = [p.strip() for p in os.environ.get(key, "").split(",") if p.strip()]
        for host in LOCAL_NO_PROXY_HOSTS:
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)
