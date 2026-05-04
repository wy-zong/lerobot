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

import logging
import threading
from concurrent import futures
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .helpers import (
    SUPPORTED_SYNC_POLICIES,
    RemotePolicyConfig,
    SyncActionChunk,
    SyncObservationRequest,
    get_logger,
    get_n_action_steps,
    make_policy_observation,
    postprocess_action_chunk,
    reset_pipeline,
)
from .transport import deserialize, receive_observation_request, serialize


class SyncPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "sync_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig) -> None:
        self.config = config
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue[SyncObservationRequest] = Queue(maxsize=1)

        self.device = torch.device("cpu")
        self.policy_type: str | None = None
        self.dataset_features: dict[str, dict] | None = None
        self.n_action_steps: int | None = None
        self.policy: Any | None = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _clear_pending_requests(self) -> None:
        self.observation_queue = Queue(maxsize=1)

    def reset(self) -> None:
        self.logger.info("Resetting sync policy server state")
        self._clear_pending_requests()
        if self.policy is not None:
            self.policy.reset()
        reset_pipeline(self.preprocessor)
        reset_pipeline(self.postprocessor)

    def Ready(self, request, context):  # noqa: N802
        self.shutdown_event.clear()
        self.reset()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        policy_specs = deserialize(request.data)
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be RemotePolicyConfig, got {type(policy_specs)}")
        if policy_specs.policy_type not in SUPPORTED_SYNC_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} is not supported by sync inference. "
                f"Supported policies: {SUPPORTED_SYNC_POLICIES}"
            )

        self.policy_type = policy_specs.policy_type
        self.dataset_features = policy_specs.dataset_features
        self.device = torch.device(policy_specs.device)

        self.logger.info(
            "Loading sync policy (type=%s, path=%s, device=%s)",
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            self.device,
        )

        policy_class = get_policy_class(policy_specs.policy_type)
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)
        self.policy.eval()
        self.n_action_steps = get_n_action_steps(self.policy, policy_specs.n_action_steps)

        device_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        self.logger.info("Policy ready; sync chunk length=%d", self.n_action_steps)
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        observation_request = receive_observation_request(request_iterator, self.shutdown_event)
        self._put_latest_request(observation_request)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            observation_request = self.observation_queue.get(timeout=self.config.request_timeout)
        except Empty:
            return services_pb2.Empty()

        action_chunk = self.predict_action_chunk(observation_request)
        return services_pb2.Actions(data=serialize(action_chunk))

    def _put_latest_request(self, request: SyncObservationRequest) -> None:
        if self.observation_queue.full():
            _ = self.observation_queue.get_nowait()
        self.observation_queue.put(request)

    def _get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("Policy has not been loaded. Send policy instructions first.")

        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 3:
            raise ValueError(f"predict_action_chunk must return 3D chunk, got shape {tuple(chunk.shape)}")

        n_action_steps = get_n_action_steps(self.policy, self.n_action_steps)
        return chunk[:, :n_action_steps, :]

    def predict_action_chunk(self, request: SyncObservationRequest) -> SyncActionChunk:
        if self.policy is None:
            raise RuntimeError("Policy has not been loaded. Send policy instructions first.")

        autocast_ctx = (
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and getattr(self.policy.config, "use_amp", False)
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = make_policy_observation(
                request.observation,
                device=self.device,
                task=request.task,
                robot_type=request.robot_type,
            )
            if self.preprocessor is not None:
                observation = self.preprocessor(observation)
            action_tensor = self._get_action_chunk(observation)
            actions = postprocess_action_chunk(action_tensor, self.postprocessor)
        return SyncActionChunk(actions=actions)

    def stop(self) -> None:
        self.shutdown_event.set()
        self._clear_pending_requests()


@draccus.wrap()
def serve(cfg: PolicyServerConfig) -> None:
    import grpc

    logging.info(pformat(asdict(cfg)))
    policy_server = SyncPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info("SyncPolicyServer started on %s:%d", cfg.host, cfg.port)
    server.start()
    server.wait_for_termination()


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
