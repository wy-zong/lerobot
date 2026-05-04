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

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict
from pprint import pformat
from typing import Protocol

import draccus
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

from .configs import RobotClientConfig
from .helpers import (
    RemotePolicyConfig,
    SyncActionChunk,
    SyncObservationRequest,
    action_tensor_to_robot_action,
    build_observation_frame,
    get_logger,
    make_dataset_features,
    make_robot_side_processors,
    ordered_action_keys_from_dataset_features,
    reset_pipeline,
)
from .transport import SyncPolicyClient


class SyncTransport(Protocol):
    def ready(self) -> None: ...

    def send_policy_config(self, policy_config: RemotePolicyConfig) -> None: ...

    def request_action_chunk(self, request: SyncObservationRequest) -> SyncActionChunk: ...

    def close(self) -> None: ...


class SyncRobotClient:
    prefix = "sync_robot_client"
    logger = get_logger(prefix)

    def __init__(
        self,
        config: RobotClientConfig,
        *,
        transport: SyncTransport | None = None,
        robot_action_processor=None,
        robot_observation_processor=None,
    ) -> None:
        self.config = config
        self.robot: Robot = make_robot_from_config(config.robot)
        self.robot_action_processor, self.robot_observation_processor = make_robot_side_processors(
            robot_action_processor, robot_observation_processor
        )
        self.dataset_features = make_dataset_features(
            self.robot,
            robot_action_processor=self.robot_action_processor,
            robot_observation_processor=self.robot_observation_processor,
        )
        self.ordered_action_keys = ordered_action_keys_from_dataset_features(self.dataset_features)
        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            dataset_features=self.dataset_features,
            device=config.policy_device,
            n_action_steps=config.n_action_steps,
            rename_map=config.rename_map,
        )

        self.transport = transport or SyncPolicyClient(
            config.server_address, initial_backoff=f"{config.environment_dt:.4f}s"
        )
        self.actions: deque[torch.Tensor] = deque()
        self.shutdown_event = threading.Event()
        self.last_raw_observation: RobotObservation | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        try:
            self.transport.ready()
            self.transport.send_policy_config(self.policy_config)
            self.robot.connect()
            self.shutdown_event.clear()
            self.logger.info("Sync robot client connected to %s", self.config.server_address)
            return True
        except Exception as e:
            self.logger.error("Failed to start sync robot client: %s", e)
            return False

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.robot.is_connected:
            self.robot.disconnect()
        self.transport.close()

    def reset(self) -> None:
        self.actions.clear()
        reset_pipeline(self.robot_action_processor)
        reset_pipeline(self.robot_observation_processor)
        self.transport.ready()

    def request_next_chunk(self) -> int:
        raw_observation = self.robot.get_observation()
        self.last_raw_observation = raw_observation
        observation = build_observation_frame(
            raw_observation=raw_observation,
            robot_observation_processor=self.robot_observation_processor,
            dataset_features=self.dataset_features,
        )
        request = SyncObservationRequest(
            observation=observation,
            task=self.config.task,
            robot_type=self.robot.robot_type,
        )
        chunk = self.transport.request_action_chunk(request)
        self.actions.extend(action.to(self.config.client_device) for action in chunk.actions)
        return len(chunk.actions)

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> RobotAction:
        return action_tensor_to_robot_action(
            action_tensor,
            dataset_features=self.dataset_features,
            ordered_action_keys=self.ordered_action_keys,
        )

    def execute_next_action(self) -> RobotAction | None:
        if not self.actions:
            return None
        action_tensor = self.actions.popleft()
        action = self._action_tensor_to_action_dict(action_tensor)
        observation = self.last_raw_observation or {}
        processed_action = self.robot_action_processor((action, observation))
        return self.robot.send_action(processed_action)

    def step(self) -> RobotAction | None:
        if not self.actions:
            self.request_next_chunk()
        return self.execute_next_action()

    def control_loop(self) -> None:
        while self.running:
            loop_start = time.perf_counter()
            self.step()
            precise_sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - loop_start)))


@draccus.wrap()
def sync_client(cfg: RobotClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    client = SyncRobotClient(cfg)
    if not client.start():
        return
    try:
        client.control_loop()
    finally:
        client.stop()


def main() -> None:
    register_third_party_plugins()
    sync_client()


if __name__ == "__main__":
    main()

