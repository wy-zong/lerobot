# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Processor for RECAP-SARM value targets on top of the SARM encoding path."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    from_tensor_to_numpy,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from ..sarm.processor_sarm import SARMEncodingProcessorStep
from ..sarm.sarm_utils import apply_rewind_augmentation, compute_absolute_indices, pad_state_to_max_dim
from .configuration_recap_sarm import RECAPSARMConfig

SUCCESS_COLUMNS = ("episode_success", "success", "is_success", "successful", "succeeded", "outcome")
TASK_MAX_LENGTH_COLUMNS = ("task_max_episode_length", "max_episode_length")


class RECAPSARMEncodingProcessorStep(SARMEncodingProcessorStep):
    """SARM encoder with RECAP-style return target generation."""

    def __init__(
        self,
        config: RECAPSARMConfig,
        image_key: str | None = None,
        dataset_meta=None,
        dataset_stats: dict | None = None,
    ):
        if dataset_meta is None:
            raise ValueError("RECAP-SARM processor requires dataset_meta to compute episode returns")
        super().__init__(config=config, image_key=image_key, dataset_meta=dataset_meta, dataset_stats=dataset_stats)
        self._episodes_df = self.dataset_meta.episodes.to_pandas()
        self._task_max_lengths = self._build_task_max_lengths(self._episodes_df)
        self._global_max_episode_length = max(self._task_max_lengths.values(), default=1)

    def _normalize_task_name(self, task: Any) -> str | None:
        if task is None:
            return None
        if isinstance(task, list):
            return self._normalize_task_name(task[0] if task else None)
        if isinstance(task, tuple):
            return self._normalize_task_name(task[0] if task else None)
        if isinstance(task, np.ndarray):
            if task.size == 0:
                return None
            return self._normalize_task_name(task.flat[0])
        if isinstance(task, str):
            return task
        return str(task)

    def _episode_length(self, episode: dict | Any) -> int:
        for key in TASK_MAX_LENGTH_COLUMNS:
            value = episode.get(key) if hasattr(episode, "get") else episode[key]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return max(1, int(value))

        if hasattr(episode, "get"):
            length = episode.get("length")
            if length is not None and not (isinstance(length, float) and math.isnan(length)):
                return max(1, int(length))

        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        return max(1, end - start)

    def _build_task_max_lengths(self, episodes_df) -> dict[str, int]:
        task_max_lengths: dict[str, int] = {}
        for _, episode in episodes_df.iterrows():
            task_name = self._normalize_task_name(episode.get("task", episode.get("tasks")))
            if task_name is None:
                continue
            episode_length = self._episode_length(episode)
            task_max_lengths[task_name] = max(task_max_lengths.get(task_name, 0), episode_length)
        return task_max_lengths

    def _resolve_task_name(self, episode, comp_data: dict[str, Any]) -> str | None:
        task = comp_data.get("task")
        if isinstance(task, list):
            task = task[0] if task else None
        task_name = self._normalize_task_name(task)
        if task_name is not None:
            return task_name
        return self._normalize_task_name(episode.get("task", episode.get("tasks")))

    def _resolve_episode_success(self, episode, comp_data: dict[str, Any]) -> bool:
        for key in SUCCESS_COLUMNS:
            if key in comp_data:
                value = comp_data[key]
                break
        else:
            value = None
            for key in SUCCESS_COLUMNS:
                if key in episode:
                    value = episode[key]
                    break

        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(
                "RECAP-SARM requires episode success metadata. Expected one of "
                f"{SUCCESS_COLUMNS} in dataset metadata."
            )

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"success", "successful", "true", "1", "yes"}:
                return True
            if normalized in {"failure", "failed", "false", "0", "no"}:
                return False
            raise ValueError(f"Unsupported success string value: {value}")

        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("Empty success metadata is not supported for RECAP-SARM")
            value = value.flat[0]

        return bool(value)

    def _resolve_task_max_episode_length(self, episode, task_name: str | None) -> int:
        if not self.config.normalize_per_task or self.config.task_max_episode_length_source == "episode_length":
            return self._episode_length(episode)

        for key in TASK_MAX_LENGTH_COLUMNS:
            value = episode.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return max(1, int(value))

        if task_name is not None and task_name in self._task_max_lengths:
            return self._task_max_lengths[task_name]

        return self._global_max_episode_length

    def _sampled_frame_indices(
        self,
        frame_idx: int,
        ep_start: int,
        ep_end: int,
        rewind_step: int,
    ) -> list[int]:
        obs_indices, _ = compute_absolute_indices(
            frame_idx,
            ep_start,
            ep_end,
            self.config.n_obs_steps,
            frame_gap=self.config.frame_gap,
        )
        sampled_indices = obs_indices.tolist()

        if rewind_step > 0:
            _, rewind_indices = apply_rewind_augmentation(
                frame_idx,
                ep_start,
                self.config.n_obs_steps,
                self.config.max_rewind_steps,
                frame_gap=self.config.frame_gap,
                rewind_step=rewind_step,
            )
            sampled_indices.extend(rewind_indices[:rewind_step])

        return sampled_indices

    def _continuous_value_to_bin(self, values: torch.Tensor) -> torch.Tensor:
        scale = (self.config.num_value_bins - 1) / (self.config.value_max - self.config.value_min)
        bin_indices = torch.round((values - self.config.value_min) * scale)
        return bin_indices.clamp(0, self.config.num_value_bins - 1).long()

    def _compute_recap_targets(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
        lengths: torch.Tensor,
        rewind_steps: torch.Tensor,
        comp_data: dict[str, Any],
        apply_perturbation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(frame_indices)
        total_frames = self.config.num_frames

        continuous_targets = torch.full(
            (batch_size, total_frames),
            fill_value=self.config.value_max,
            dtype=torch.float32,
        )
        remaining_steps = torch.zeros((batch_size, total_frames), dtype=torch.int64)
        episode_success = torch.zeros((batch_size,), dtype=torch.bool)

        if apply_perturbation:
            for b_idx in range(batch_size):
                valid_len = int(lengths[b_idx].item())
                continuous_targets[b_idx, :valid_len] = self.config.value_min
            return (
                continuous_targets,
                self._continuous_value_to_bin(continuous_targets),
                episode_success,
                remaining_steps,
            )

        for b_idx in range(batch_size):
            ep_idx = int(episode_indices[b_idx])
            frame_idx = int(frame_indices[b_idx])
            rewind_step = int(rewind_steps[b_idx].item())

            episode = self._episodes_df.iloc[ep_idx]
            ep_start = int(episode["dataset_from_index"])
            ep_end = int(episode["dataset_to_index"])
            success = self._resolve_episode_success(episode, comp_data)
            task_name = self._resolve_task_name(episode, comp_data)
            max_episode_length = self._resolve_task_max_episode_length(episode, task_name)

            sampled_indices = self._sampled_frame_indices(frame_idx, ep_start, ep_end, rewind_step)
            episode_success[b_idx] = success

            for t_idx, abs_idx in enumerate(sampled_indices):
                steps_left = max(1, ep_end - int(abs_idx))
                raw_return = -float(steps_left)
                if not success:
                    raw_return -= self.config.failure_penalty
                normalized_value = raw_return / float(max_episode_length)
                continuous_targets[b_idx, t_idx] = float(
                    np.clip(normalized_value, self.config.value_min, self.config.value_max)
                )
                remaining_steps[b_idx, t_idx] = steps_left

        return (
            continuous_targets,
            self._continuous_value_to_bin(continuous_targets),
            episode_success,
            remaining_steps,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy() if hasattr(transition, "copy") else dict(transition)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        comp_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        frame_index = comp_data.get("index")
        episode_index = comp_data.get("episode_index")

        if frame_index is None:
            raise ValueError("Frame index ('index') not found in COMPLEMENTARY_DATA")
        if episode_index is None:
            raise ValueError("Episode index ('episode_index') not found in COMPLEMENTARY_DATA")

        frame_indices = np.atleast_1d(np.asarray(from_tensor_to_numpy(frame_index)))
        episode_indices = self._get_episode_indices(frame_indices, episode_index)

        image = None
        if self.precomputed_image_features is None:
            image = observation.get(self.image_key)
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if image.ndim == 4:
                image = image[np.newaxis, ...]
            elif image.ndim == 3:
                image = image[np.newaxis, np.newaxis, ...]

            batch_size = image.shape[0]
            total_frames = image.shape[1]
        else:
            batch_size = len(frame_indices)
            total_frames = self.config.num_frames

        n_obs_frames = 1 + self.config.n_obs_steps
        rewind_steps = torch.zeros(batch_size, dtype=torch.int32)
        apply_rewind = self.training and random.random() < self.config.rewind_probability

        if apply_rewind and self.dataset_meta is not None:
            for b_idx, (ep_idx, frame_idx) in enumerate(
                zip(episode_indices.tolist(), frame_indices.tolist(), strict=True)
            ):
                episode = self.dataset_meta.episodes[int(ep_idx)]
                rewind_step, _ = apply_rewind_augmentation(
                    int(frame_idx),
                    int(episode["dataset_from_index"]),
                    self.config.n_obs_steps,
                    self.config.max_rewind_steps,
                    frame_gap=self.config.frame_gap,
                )
                rewind_steps[b_idx] = rewind_step

        lengths = n_obs_frames + rewind_steps

        if image is not None:
            for b_idx in range(batch_size):
                valid_len = lengths[b_idx].item()
                if valid_len < total_frames:
                    image[b_idx, valid_len:] = 0

        if self.precomputed_image_features is None:
            video_features = self._encode_images_batch(image)
        else:
            delta_indices = self.config.observation_delta_indices
            abs_indices = np.empty((batch_size, total_frames), dtype=np.int64)
            for b_idx in range(batch_size):
                center = int(frame_indices[b_idx])
                ep_idx = int(episode_indices[b_idx])
                episode = self.dataset_meta.episodes[ep_idx]
                ep_start = int(episode["dataset_from_index"])
                ep_end = int(episode["dataset_to_index"])
                for t_idx, delta in enumerate(delta_indices):
                    idx = center + int(delta)
                    if idx < ep_start:
                        idx = ep_start
                    elif idx >= ep_end:
                        idx = ep_end - 1
                    abs_indices[b_idx, t_idx] = idx

            feats = np.asarray(self.precomputed_image_features[abs_indices]).copy()
            for b_idx in range(batch_size):
                valid_len = lengths[b_idx].item()
                if valid_len < total_frames:
                    feats[b_idx, valid_len:] = self.clip_zero_pad_vec
            video_features = torch.from_numpy(feats).float()

        observation["video_features"] = video_features

        state_data = observation.get(self.config.state_key)
        if isinstance(state_data, torch.Tensor):
            state_tensor = state_data.float()
        else:
            state_tensor = torch.tensor(state_data, dtype=torch.float32)

        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        elif state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)

        for b_idx in range(batch_size):
            valid_len = lengths[b_idx].item()
            if valid_len < state_tensor.shape[1]:
                state_tensor[b_idx, valid_len:] = 0

        observation["state_features"] = pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

        task = comp_data.get("task")
        if isinstance(task, list):
            task = task[0] if task else ""

        apply_perturbation = self.training and random.random() < self.config.language_perturbation_probability
        if apply_perturbation:
            task = self._generate_perturbed_task()

        observation["text_features"] = self._encode_text_clip(task, batch_size)
        observation["lengths"] = lengths

        if self.training:
            (
                value_targets_continuous,
                value_targets_bin,
                episode_success,
                remaining_steps,
            ) = self._compute_recap_targets(
                frame_indices,
                episode_indices,
                lengths,
                rewind_steps,
                comp_data,
                apply_perturbation,
            )
            observation["value_targets_continuous"] = value_targets_continuous
            observation["value_targets_bin"] = value_targets_bin
            observation["episode_success"] = episode_success
            observation["remaining_steps"] = remaining_steps

        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = super().transform_features(features)
        observation_features = features[PipelineFeatureType.OBSERVATION]
        observation_features["value_targets_continuous"] = PolicyFeature(
            type=FeatureType.REWARD,
            shape=(self.config.num_frames,),
        )
        observation_features["value_targets_bin"] = PolicyFeature(
            type=FeatureType.REWARD,
            shape=(self.config.num_frames,),
        )
        observation_features["episode_success"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        observation_features["remaining_steps"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.config.num_frames,),
        )
        return features


def make_recap_sarm_pre_post_processors(
    config: RECAPSARMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta=None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create pre-processor and post-processor pipelines for RECAP-SARM."""

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                AddBatchDimensionProcessorStep(),
                RenameObservationsProcessorStep(rename_map={}),
                NormalizerProcessorStep(
                    features={**config.input_features, **config.output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                RECAPSARMEncodingProcessorStep(
                    config=config,
                    dataset_meta=dataset_meta,
                    dataset_stats=dataset_stats,
                ),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
