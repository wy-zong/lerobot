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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
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
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

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

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
    ]
    if not config.use_state:
        input_steps.append(DropStateProcessorStep())
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
            NormalizerProcessorStep(
                features={**input_features, **output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        ]
    )
    output_steps = [
        UnnormalizerProcessorStep(
            features=output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
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
