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

from collections import deque

import numpy as np
import pytest
import torch

pytest.importorskip("grpc", reason="grpcio is required by sync inference transport")

from lerobot.sync_inference.configs import PolicyServerConfig
from lerobot.sync_inference.helpers import SyncObservationRequest
from lerobot.sync_inference.policy_server import SyncPolicyServer
from lerobot.utils.constants import OBS_STATE


class _Config:
    n_action_steps = 3
    use_amp = False


class MockChunkPolicy:
    def __init__(self, chunk: torch.Tensor):
        self.config = _Config()
        self.chunk = chunk
        self.predict_calls = 0
        self.reset_calls = 0

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.predict_calls += 1
        assert batch["task"] == "pick"
        assert batch["robot_type"] == "mock_robot"
        return self.chunk.clone()

    def reset(self) -> None:
        self.reset_calls += 1


class MockSelectPolicy(MockChunkPolicy):
    def __init__(self, chunk: torch.Tensor):
        super().__init__(chunk)
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


@pytest.fixture()
def server() -> SyncPolicyServer:
    sync_server = SyncPolicyServer(PolicyServerConfig(host="localhost", port=9999))
    sync_server.device = torch.device("cpu")
    sync_server.preprocessor = lambda obs: obs
    sync_server.postprocessor = lambda action: action
    return sync_server


def _request() -> SyncObservationRequest:
    return SyncObservationRequest(
        observation={OBS_STATE: np.zeros(2, dtype=np.float32)},
        task="pick",
        robot_type="mock_robot",
    )


@pytest.mark.parametrize("policy_type", ["smolvla", "pi0", "pi05"])
def test_server_uses_predict_action_chunk_and_truncates(server: SyncPolicyServer, policy_type: str) -> None:
    chunk = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]])
    policy = MockChunkPolicy(chunk)
    server.policy = policy
    server.policy_type = policy_type
    server.n_action_steps = 3

    result = server.predict_action_chunk(_request())

    assert policy.predict_calls == 1
    assert len(result.actions) == 3
    assert torch.equal(result.actions[0], torch.tensor([0.0, 1.0]))
    assert torch.equal(result.actions[1], torch.tensor([2.0, 3.0]))
    assert torch.equal(result.actions[2], torch.tensor([4.0, 5.0]))


def test_remote_chunk_matches_local_select_action_order(server: SyncPolicyServer) -> None:
    chunk = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
    local_policy = MockSelectPolicy(chunk)
    remote_policy = MockChunkPolicy(chunk)
    server.policy = remote_policy
    server.n_action_steps = 3

    batch = {
        OBS_STATE: torch.zeros(1, 2),
        "task": "pick",
        "robot_type": "mock_robot",
    }
    local_actions = [local_policy.select_action(batch).squeeze(0) for _ in range(3)]
    remote_actions = server.predict_action_chunk(_request()).actions

    assert len(remote_actions) == len(local_actions)
    for remote, local in zip(remote_actions, local_actions, strict=True):
        assert torch.equal(remote, local)


def test_reset_resets_policy_and_processors(server: SyncPolicyServer) -> None:
    class Resettable:
        def __init__(self):
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    policy = MockChunkPolicy(torch.zeros(1, 3, 2))
    preprocessor = Resettable()
    postprocessor = Resettable()
    server.policy = policy
    server.preprocessor = preprocessor
    server.postprocessor = postprocessor
    server.observation_queue.put(_request())

    server.reset()

    assert policy.reset_calls == 1
    assert preprocessor.reset_calls == 1
    assert postprocessor.reset_calls == 1
    assert server.observation_queue.empty()
