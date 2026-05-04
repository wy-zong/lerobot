#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

DEFAULT_FPS = 30
DEFAULT_REQUEST_TIMEOUT = 10.0


@dataclass
class PolicyServerConfig:
    """Configuration for the synchronous remote policy server."""

    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})
    request_timeout: float = field(
        default=DEFAULT_REQUEST_TIMEOUT,
        metadata={"help": "Seconds to wait for a pending observation request"},
    )

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")
        if self.request_timeout < 0:
            raise ValueError(f"request_timeout must be non-negative, got {self.request_timeout}")


@dataclass
class RobotClientConfig:
    """Configuration for the synchronous remote robot client."""

    policy_type: str = field(metadata={"help": "Type of policy to run on the server"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    task: str = field(default="", metadata={"help": "Task instruction for VLA policies"})
    server_address: str = field(
        default="localhost:8080", metadata={"help": "Policy server address as host:port"}
    )
    policy_device: str = field(default="cpu", metadata={"help": "Device for server-side policy inference"})
    client_device: str = field(default="cpu", metadata={"help": "Device for client-side action tensors"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Robot control loop frequency"})
    n_action_steps: int | None = field(
        default=None,
        metadata={
            "help": "Optional action chunk length override. Defaults to policy.config.n_action_steps."
        },
    )
    rename_map: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Observation feature rename map applied by the server preprocessor"},
    )

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps

    def __post_init__(self) -> None:
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")
        if not self.client_device:
            raise ValueError("client_device cannot be empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.n_action_steps is not None and self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be positive when set, got {self.n_action_steps}")

