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

from collections.abc import Iterable

import pytest
import torch

pytest.importorskip("grpc", reason="grpcio is required by sync inference transport")
pytest.importorskip("serial", reason="pyserial is required by builtin robot client imports")

from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.sync_inference.configs import RobotClientConfig
from lerobot.sync_inference.helpers import RemotePolicyConfig, SyncActionChunk, SyncObservationRequest
from lerobot.sync_inference.robot_client import SyncRobotClient
from tests.mocks.mock_robot import MockRobotConfig


class FakeTransport:
    def __init__(self, chunks: Iterable[list[torch.Tensor]]):
        self.chunks = [SyncActionChunk(actions=chunk) for chunk in chunks]
        self.ready_calls = 0
        self.policy_config: RemotePolicyConfig | None = None
        self.requests: list[SyncObservationRequest] = []
        self.closed = False

    def ready(self) -> None:
        self.ready_calls += 1

    def send_policy_config(self, policy_config: RemotePolicyConfig) -> None:
        self.policy_config = policy_config

    def request_action_chunk(self, request: SyncObservationRequest) -> SyncActionChunk:
        self.requests.append(request)
        return self.chunks.pop(0)

    def close(self) -> None:
        self.closed = True


def _config() -> RobotClientConfig:
    return RobotClientConfig(
        robot=MockRobotConfig(n_motors=3, random_values=False, static_values=[0.0, 0.0, 0.0]),
        server_address="localhost:9999",
        policy_type="smolvla",
        pretrained_name_or_path="mock-policy",
        task="pick",
    )


def _make_client(
    chunks: list[list[torch.Tensor]],
    *,
    robot_action_processor=None,
) -> tuple[SyncRobotClient, FakeTransport, list[dict[str, float]]]:
    transport = FakeTransport(chunks)
    client = SyncRobotClient(_config(), transport=transport, robot_action_processor=robot_action_processor)
    sent_actions = []

    def record_action(action):
        sent_actions.append(action)
        return action

    client.robot.send_action = record_action
    assert client.start()
    return client, transport, sent_actions


def test_client_executes_first_action_from_new_chunk() -> None:
    client, transport, sent_actions = _make_client(
        [[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]]
    )

    try:
        client.step()
    finally:
        client.stop()

    assert len(transport.requests) == 1
    assert sent_actions == [{"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0}]
    assert len(client.actions) == 1


def test_client_requests_next_chunk_only_after_current_chunk_is_empty() -> None:
    client, transport, sent_actions = _make_client(
        [
            [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])],
            [torch.tensor([3.0, 3.0, 3.0])],
        ]
    )

    try:
        client.step()
        assert len(transport.requests) == 1
        client.step()
        assert len(transport.requests) == 1
        client.step()
        assert len(transport.requests) == 2
    finally:
        client.stop()

    assert [action["motor_1.pos"] for action in sent_actions] == [1.0, 2.0, 3.0]


def test_client_runs_robot_action_processor_before_send_action() -> None:
    class MultiplyActionStep(RobotActionProcessorStep):
        def action(self, action):
            return {key: value * 10 for key, value in action.items()}

        def transform_features(self, features):
            return features

    robot_action_processor = RobotProcessorPipeline(
        steps=[MultiplyActionStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    client, _transport, sent_actions = _make_client(
        [[torch.tensor([1.0, 2.0, 3.0])]],
        robot_action_processor=robot_action_processor,
    )

    try:
        client.step()
    finally:
        client.stop()

    assert sent_actions == [{"motor_1.pos": 10.0, "motor_2.pos": 20.0, "motor_3.pos": 30.0}]


def test_client_reset_clears_local_queue_and_resets_server() -> None:
    client, transport, _sent_actions = _make_client([[torch.tensor([1.0, 2.0, 3.0])]])

    try:
        client.actions.extend([torch.ones(3)])
        client.reset()
    finally:
        client.stop()

    assert len(client.actions) == 0
    assert transport.ready_calls == 2
