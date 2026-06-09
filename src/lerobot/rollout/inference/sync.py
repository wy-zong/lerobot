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

"""Synchronous inference engine: inline policy call per control tick."""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class SyncInferenceEngine(InferenceEngine):
    """Inline synchronous inference: compute one action per call.

    Non-relative policies keep the legacy per-tick ``select_action`` path.
    Relative-action policies refill a local FIFO with one postprocessed
    ``predict_action_chunk`` result so the whole chunk is anchored to the
    same observation state.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        sarm_predictor: Any | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._sarm_predictor = sarm_predictor
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._relative_actions_enabled = any(
            isinstance(step, RelativeActionsProcessorStep) and step.enabled
            for step in getattr(preprocessor, "steps", ())
        )
        self._action_queue: deque[torch.Tensor] = deque()
        if self._relative_actions_enabled:
            self._ensure_relative_action_names()
        logger.info(
            "SyncInferenceEngine initialized (device=%s, action_keys=%d, relative_actions=%s)",
            self._device,
            len(ordered_action_keys),
            self._relative_actions_enabled,
        )

    def start(self) -> None:
        """No background resources to start."""
        logger.info("SyncInferenceEngine started (inline mode — no background thread)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("SyncInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy and pre/post-processors."""
        logger.info("Resetting sync inference state (policy + processors)")
        self._action_queue.clear()
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._sarm_predictor is not None:
            self._sarm_predictor.reset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Run the full inference pipeline on ``obs_frame`` and return an action tensor."""
        if self._relative_actions_enabled and self._action_queue:
            return self._action_queue.popleft()
        if obs_frame is None:
            return None

        if self._sarm_predictor is not None:
            self._task = self._sarm_predictor.predict_subtask(obs_frame, self._task)

        if self._relative_actions_enabled:
            return self._get_relative_action(obs_frame)

        return self._get_select_action(obs_frame)

    def _get_select_action(self, obs_frame: dict) -> torch.Tensor:
        """Run the legacy per-tick ``select_action`` sync inference path."""
        # Shallow copy is intentional: the caller (`send_next_action`) builds
        # ``obs_frame`` fresh per tick via ``build_dataset_frame``, so the
        # tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action = self._policy.select_action(observation)
            action = self._postprocessor(action)
        action_tensor = action.squeeze(0).cpu()
        return self._action_tensor_to_ordered_action(action_tensor)

    def _get_relative_action(self, obs_frame: dict) -> torch.Tensor | None:
        """Refill the local action FIFO from one postprocessed relative-action chunk."""
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            action_chunk = self._policy.predict_action_chunk(observation)
            action_chunk = self._postprocessor(action_chunk)

        self._enqueue_action_chunk(action_chunk)
        if not self._action_queue:
            return None
        return self._action_queue.popleft()

    def _enqueue_action_chunk(self, action_chunk: torch.Tensor) -> None:
        """Store postprocessed absolute actions in execution order."""
        action_chunk = action_chunk.squeeze(0).cpu()
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.unsqueeze(0)

        n_action_steps = getattr(self._policy.config, "n_action_steps", action_chunk.shape[0])
        for action_tensor in action_chunk[:n_action_steps]:
            self._action_queue.append(self._action_tensor_to_ordered_action(action_tensor))

    def _action_tensor_to_ordered_action(self, action_tensor: torch.Tensor) -> torch.Tensor:
        # Reorder to match dataset action ordering so the caller can treat
        # the returned tensor uniformly across backends.
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    def _ensure_relative_action_names(self) -> None:
        for step in self._preprocessor.steps:
            if not isinstance(step, RelativeActionsProcessorStep) or not step.enabled:
                continue
            if step.action_names is not None:
                continue
            cfg_names = getattr(self._policy.config, "action_feature_names", None)
            if cfg_names:
                step.action_names = list(cfg_names)
            elif ACTION in self._dataset_features:
                step.action_names = list(self._dataset_features[ACTION]["names"])
