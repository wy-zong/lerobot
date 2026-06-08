#!/usr/bin/env python

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

from dataclasses import dataclass

from .train import TrainPipelineConfig


@dataclass
class TrainStagePipelineConfig(TrainPipelineConfig):
    """Config for SmolVLA PI0.5-style subtask post-training."""

    flow_loss_weight: float = 10.0
    subtask_required: bool = True
    policy_type_required: str = "smolvla"

    def validate(self) -> None:
        super().validate()

        if self.is_reward_model_training:
            raise ValueError("lerobot-train-stage trains policies only; reward_model is not supported.")

        if self.policy is None:
            raise ValueError("lerobot-train-stage requires a policy configuration.")

        if self.policy.type != self.policy_type_required:
            raise ValueError(
                "lerobot-train-stage currently supports SmolVLA policies only. "
                f"Expected policy.type='{self.policy_type_required}', got '{self.policy.type}'."
            )

        if self.flow_loss_weight < 0:
            raise ValueError(f"flow_loss_weight must be non-negative. Got {self.flow_loss_weight}.")

        if self.sample_weighting is not None:
            raise ValueError("sample_weighting is not supported by lerobot-train-stage.")
