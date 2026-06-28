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

"""Configuration for the RECAP-style SARM value model."""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.rewards import RewardModelConfig

from ..sarm.configuration_sarm import SARMConfig


@RewardModelConfig.register_subclass("recap_sarm")
@dataclass
class RECAPSARMConfig(SARMConfig):
    """Independent RECAP-style distributional value model built on the SARM backbone."""

    num_value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    failure_penalty: float = 50.0
    normalize_per_task: bool = True
    task_max_episode_length_source: str = "dataset_meta"
    advantage_lookahead: int = 50
    use_stage_aux_loss: bool = False
    stage_loss_weight: float = 0.0

    output_features: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        super().__post_init__()

        if self.num_value_bins < 2:
            raise ValueError(f"num_value_bins must be at least 2, got {self.num_value_bins}")
        if self.value_min >= self.value_max:
            raise ValueError(
                f"value_min must be strictly smaller than value_max, got {self.value_min} >= {self.value_max}"
            )
        if self.failure_penalty < 0:
            raise ValueError(f"failure_penalty must be non-negative, got {self.failure_penalty}")
        if self.advantage_lookahead < 1:
            raise ValueError(f"advantage_lookahead must be positive, got {self.advantage_lookahead}")
        if self.task_max_episode_length_source not in {"dataset_meta", "episode_length"}:
            raise ValueError(
                "task_max_episode_length_source must be either 'dataset_meta' or 'episode_length', "
                f"got {self.task_max_episode_length_source}"
            )

        self.output_features = {
            "value_logits": PolicyFeature(
                shape=(self.num_frames, self.num_value_bins),
                type=FeatureType.REWARD,
            ),
            "value": PolicyFeature(
                shape=(self.num_frames, 1),
                type=FeatureType.REWARD,
            ),
        }
