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

import pickle  # nosec B403: Internal robot/server transport only.
import threading
from typing import Any

from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, receive_bytes_in_chunks, send_bytes_in_chunks

from .helpers import RemotePolicyConfig, SyncActionChunk, SyncObservationRequest


def serialize(value: Any) -> bytes:
    return pickle.dumps(value)  # nosec B301


def deserialize(buffer: bytes) -> Any:
    return pickle.loads(buffer)  # nosec B301


class SyncPolicyClient:
    """Synchronous request/response client built on the existing transport service."""

    def __init__(self, server_address: str, *, initial_backoff: str = "0.1s") -> None:
        import grpc

        self._grpc = grpc
        self.channel = grpc.insecure_channel(
            server_address, grpc_channel_options(initial_backoff=initial_backoff)
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

    def ready(self) -> None:
        self.stub.Ready(services_pb2.Empty())

    def send_policy_config(self, policy_config: RemotePolicyConfig) -> None:
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=serialize(policy_config)))

    def request_action_chunk(self, request: SyncObservationRequest) -> SyncActionChunk:
        observation_iterator = send_bytes_in_chunks(
            serialize(request),
            services_pb2.Observation,
            log_prefix="[SYNC_CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator)
        response = self.stub.GetActions(services_pb2.Empty())
        if not response.data:
            raise RuntimeError("Policy server returned an empty action response")
        result = deserialize(response.data)
        if not isinstance(result, SyncActionChunk):
            raise TypeError(f"Expected SyncActionChunk from server, got {type(result)}")
        return result

    def close(self) -> None:
        self.channel.close()


def receive_observation_request(request_iterator, shutdown_event: threading.Event) -> SyncObservationRequest:
    received_bytes = receive_bytes_in_chunks(request_iterator, None, shutdown_event, "[SYNC_SERVER]")
    request = deserialize(received_bytes)
    if not isinstance(request, SyncObservationRequest):
        raise TypeError(f"Expected SyncObservationRequest, got {type(request)}")
    return request

