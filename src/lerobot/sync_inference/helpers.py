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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.utils import init_logging

SUPPORTED_SYNC_POLICIES = ("smolvla", "pi0", "pi05")

PolicyObservation = dict[str, Any]
ActionChunk = list[torch.Tensor]


@dataclass
class RemotePolicyConfig:
    """Policy setup sent by the robot client to the policy server."""

    policy_type: str
    pretrained_name_or_path: str
    dataset_features: dict[str, dict]
    device: str = "cpu"
    n_action_steps: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class SyncObservationRequest:
    """One synchronous chunk request from the robot host."""

    observation: PolicyObservation
    task: str = ""
    robot_type: str = ""


@dataclass
class SyncActionChunk:
    """A complete action chunk returned by the policy host."""

    actions: ActionChunk


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None
    init_logging(log_file=log_file, display_pid=False)
    return logging.getLogger(name)


def reset_pipeline(pipeline: Any | None) -> None:
    if pipeline is not None and hasattr(pipeline, "reset"):
        pipeline.reset()


def get_n_action_steps(policy: Any, requested_n_action_steps: int | None = None) -> int:
    if requested_n_action_steps is not None:
        return requested_n_action_steps
    n_action_steps = getattr(getattr(policy, "config", None), "n_action_steps", None)
    if n_action_steps is None:
        raise ValueError(
            "n_action_steps was not provided and policy.config.n_action_steps is not available"
        )
    if n_action_steps <= 0:
        raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
    return int(n_action_steps)


def make_policy_observation(
    observation: PolicyObservation,
    *,
    device: torch.device,
    task: str,
    robot_type: str,
) -> PolicyObservation:
    return prepare_observation_for_inference(dict(observation), device, task, robot_type)


def postprocess_action_chunk(action_tensor: torch.Tensor, postprocessor: Any) -> ActionChunk:
    if action_tensor.ndim == 2:
        action_tensor = action_tensor.unsqueeze(0)
    if action_tensor.ndim != 3:
        raise ValueError(
            "Action chunk must have shape (batch, chunk, action_dim) or (chunk, action_dim), "
            f"got {tuple(action_tensor.shape)}"
        )

    processed_actions = []
    for i in range(action_tensor.shape[1]):
        action = action_tensor[:, i, :]
        action = postprocessor(action) if postprocessor is not None else action
        processed_actions.append(action.squeeze(0).detach().cpu())
    return processed_actions


def _policy_robot_hardware_features(robot: Robot) -> tuple[dict[str, Any], dict[str, Any]]:
    observation_features = {
        key: value
        for key, value in robot.observation_features.items()
        if isinstance(value, tuple) or (value is float and key.endswith(".pos"))
    }
    action_features = {
        key: value for key, value in robot.action_features.items() if key.endswith(".pos")
    }
    return observation_features, action_features


def make_robot_side_processors(
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]
    | None = None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation] | None = None,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    _, default_action_processor, default_observation_processor = make_default_processors()
    return (
        robot_action_processor or default_action_processor,
        robot_observation_processor or default_observation_processor,
    )


def make_dataset_features(
    robot: Robot,
    *,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    use_videos: bool = True,
) -> dict[str, dict]:
    observation_features, action_features = _policy_robot_hardware_features(robot)
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_action_processor,
        initial_features=create_initial_features(action=action_features),
        use_videos=use_videos,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features),
        use_videos=use_videos,
    )
    return combine_feature_dicts(action_dataset_features, observation_dataset_features)


def build_observation_frame(
    *,
    raw_observation: RobotObservation,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset_features: dict[str, dict],
) -> PolicyObservation:
    processed_observation = robot_observation_processor(raw_observation)
    return build_dataset_frame(dataset_features, processed_observation, prefix=OBS_STR)


def action_tensor_to_robot_action(
    action_tensor: torch.Tensor,
    *,
    dataset_features: dict[str, dict],
    ordered_action_keys: list[str],
) -> RobotAction:
    action_dict = make_robot_action(action_tensor, dataset_features)
    return {key: action_dict[key] for key in ordered_action_keys}


def ordered_action_keys_from_dataset_features(dataset_features: dict[str, dict]) -> list[str]:
    return list(dataset_features[ACTION]["names"])

