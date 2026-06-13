#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_smolvla import SmolVLAConfig


def _without_state_features(features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
    return {key: feature for key, feature in features.items() if feature.type is not FeatureType.STATE}


@dataclass
@ProcessorStepRegistry.register(name="smolvla_drop_state_processor")
class DropStateProcessorStep(ObservationProcessorStep):
    state_keys: tuple[str, ...] = field(default_factory=lambda: (OBS_STATE,))

    def __post_init__(self):
        self.state_keys = tuple(self.state_keys)

    def observation(self, observation):
        return {key: value for key, value in observation.items() if key not in self.state_keys}

    def get_config(self) -> dict[str, Any]:
        return {"state_keys": list(self.state_keys)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = features.copy()
        observation_features = features.get(PipelineFeatureType.OBSERVATION, {})
        new_features[PipelineFeatureType.OBSERVATION] = {
            key: feature
            for key, feature in observation_features.items()
            if key not in self.state_keys and feature.type is not FeatureType.STATE
        }
        return new_features


@dataclass
@ProcessorStepRegistry.register(name="smolvla_discrete_state_prompt_processor")
class SmolVLADiscreteStatePromptProcessorStep(ProcessorStep):
    """Append PI0.5-style discretized state values to the task prompt."""

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = (transition.get(TransitionKey.OBSERVATION) or {}).get(OBS_STATE)
        if state is None:
            raise ValueError(f"`{OBS_STATE}` is required when `discrete_state_in_language=True`.")

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is required to build the SmolVLA state prompt.")
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError(f"`{self.task_key}` is required to build the SmolVLA state prompt.")
        if isinstance(tasks, str):
            tasks = [tasks]

        state = deepcopy(state)
        if state.ndim > 2:
            state = state[:, -1, :]
        elif state.ndim == 1:
            state = state.unsqueeze(0)

        state_np = state.detach().cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        if len(tasks) != len(discretized_states):
            raise ValueError(
                "The number of task prompts must match the state batch size. "
                f"Got {len(tasks)} task(s) and {len(discretized_states)} state row(s)."
            )

        full_prompts = []
        for task, discrete_state in zip(tasks, discretized_states, strict=True):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discrete_state))
            full_prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim, "task_key": self.task_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Adding a batch dimension.
    3.  Ensuring the language task description ends with a newline character.
    4.  Tokenizing the language task description.
    5.  Moving all data to the specified device.
    6.  Optionally converting absolute actions to relative actions.
    7.  Optionally removing state from the model inputs.
    8.  Normalizing input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1.  Unnormalizing the output actions to their original scale.
    2.  Optionally converting relative actions back to absolute actions.
    3.  Moving data to the CPU.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_features = config.input_features or {}
    output_features = config.output_features or {}
    if not config.use_state:
        input_features = _without_state_features(input_features)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
    ]
    if config.discrete_state_in_language:
        input_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                NewLineTaskProcessorStep(),
                DeviceProcessorStep(device=config.device),
                relative_step,
                NormalizerProcessorStep(
                    features={**input_features, **output_features},
                    norm_map=config.normalization_mapping,
                    stats=dataset_stats,
                ),
                SmolVLADiscreteStatePromptProcessorStep(max_state_dim=config.max_state_dim),
                TokenizerProcessorStep(
                    tokenizer_name=config.vlm_model_name,
                    padding=config.pad_language_to,
                    padding_side="right",
                    max_length=config.tokenizer_max_length,
                ),
                DropStateProcessorStep(),
            ]
        )
    else:
        input_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                NewLineTaskProcessorStep(),
                TokenizerProcessorStep(
                    tokenizer_name=config.vlm_model_name,
                    padding=config.pad_language_to,
                    padding_side="right",
                    max_length=config.tokenizer_max_length,
                ),
                DeviceProcessorStep(device=config.device),
                relative_step,
            ]
        )
        if not config.use_state:
            input_steps.append(DropStateProcessorStep())
        input_steps.append(
            NormalizerProcessorStep(
                features={**input_features, **output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            )
        )
    output_steps = [
        UnnormalizerProcessorStep(
            features=output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
