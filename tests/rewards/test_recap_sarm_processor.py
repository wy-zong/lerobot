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

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.rewards.recap_sarm.configuration_recap_sarm import RECAPSARMConfig
from lerobot.types import TransitionKey


class MockDatasetMeta:
    def __init__(self, episodes: list[dict]):
        self._episodes = episodes

    @property
    def episodes(self):
        mock = MagicMock()
        mock.__len__ = lambda s: len(self._episodes)
        mock.__getitem__ = lambda s, idx: self._episodes[idx]
        mock.to_pandas = lambda: pd.DataFrame(self._episodes)
        return mock


@pytest.fixture
def mock_clip_model():
    with (
        patch("lerobot.rewards.sarm.processor_sarm.require_package"),
        patch("lerobot.rewards.sarm.processor_sarm.Faker") as mock_faker_cls,
        patch("lerobot.rewards.sarm.processor_sarm.CLIPModel") as mock_model_cls,
        patch("lerobot.rewards.sarm.processor_sarm.CLIPProcessor") as mock_processor_cls,
    ):
        mock_model = MagicMock()

        def get_image_features_side_effect(**kwargs):
            pixel_values = kwargs.get("pixel_values")
            batch_size = pixel_values.shape[0] if pixel_values is not None else 1
            return torch.randn(batch_size, 512)

        mock_model.get_image_features.side_effect = get_image_features_side_effect
        mock_model.get_text_features.return_value = torch.randn(1, 512)
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = MagicMock()

        def processor_side_effect(images=None, **kwargs):
            num_images = len(images) if images is not None else 1
            return {"pixel_values": torch.randn(num_images, 3, 224, 224)}

        mock_processor.side_effect = processor_side_effect
        mock_processor.tokenizer.return_value = {
            "input_ids": torch.ones(1, 77, dtype=torch.long),
            "attention_mask": torch.ones(1, 77, dtype=torch.long),
        }
        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_faker = MagicMock()
        mock_faker.words.return_value = ["irrelevant", "task"]
        mock_faker_cls.return_value = mock_faker

        yield


def make_transition(config: RECAPSARMConfig, frame_index: int, episode_index: int, task: str):
    return {
        TransitionKey.OBSERVATION: {
            config.image_key: np.random.rand(config.num_frames, 3, 224, 224).astype(np.float32),
            config.state_key: np.random.rand(config.num_frames, 6).astype(np.float32),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "index": frame_index,
            "episode_index": episode_index,
            "task": task,
        },
    }


def test_recap_sarm_processor_generates_success_and_failure_targets(mock_clip_model):
    from lerobot.rewards.recap_sarm.processor_recap_sarm import RECAPSARMEncodingProcessorStep

    config = RECAPSARMConfig(
        temporal_window_mode="current_only",
        rewind_probability=0.0,
        language_perturbation_probability=0.0,
        failure_penalty=25.0,
    )
    dataset_meta = MockDatasetMeta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 100,
                "length": 100,
                "task": "stack the cube",
                "success": True,
            },
            {
                "dataset_from_index": 100,
                "dataset_to_index": 200,
                "length": 100,
                "task": "stack the cube",
                "success": False,
            },
        ]
    )
    processor = RECAPSARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
    processor.train()

    success_result = processor(
        make_transition(config, frame_index=50, episode_index=0, task="stack the cube")
    )
    failure_result = processor(
        make_transition(config, frame_index=150, episode_index=1, task="stack the cube")
    )

    success_obs = success_result[TransitionKey.OBSERVATION]
    failure_obs = failure_result[TransitionKey.OBSERVATION]

    success_value = success_obs["value_targets_continuous"][0, 0].item()
    failure_value = failure_obs["value_targets_continuous"][0, 0].item()

    assert -1.0 <= success_value <= 0.0
    assert -1.0 <= failure_value <= 0.0
    assert failure_value < success_value
    assert int(success_obs["value_targets_bin"][0, 0].item()) in range(config.num_value_bins)
    assert int(failure_obs["value_targets_bin"][0, 0].item()) in range(config.num_value_bins)
    assert bool(success_obs["episode_success"][0].item()) is True
    assert bool(failure_obs["episode_success"][0].item()) is False
    assert int(success_obs["remaining_steps"][0, 0].item()) == 50
    assert int(failure_obs["remaining_steps"][0, 0].item()) == 50


def test_recap_sarm_processor_requires_success_metadata(mock_clip_model):
    from lerobot.rewards.recap_sarm.processor_recap_sarm import RECAPSARMEncodingProcessorStep

    config = RECAPSARMConfig(
        temporal_window_mode="current_only",
        rewind_probability=0.0,
        language_perturbation_probability=0.0,
    )
    dataset_meta = MockDatasetMeta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 100,
                "length": 100,
                "task": "stack the cube",
            }
        ]
    )
    processor = RECAPSARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
    processor.train()

    with pytest.raises(ValueError, match="episode success metadata"):
        processor(make_transition(config, frame_index=50, episode_index=0, task="stack the cube"))


def test_recap_sarm_processor_inference_does_not_require_success_metadata(mock_clip_model):
    from lerobot.rewards.recap_sarm.processor_recap_sarm import RECAPSARMEncodingProcessorStep

    config = RECAPSARMConfig(
        temporal_window_mode="current_only",
        rewind_probability=0.0,
        language_perturbation_probability=0.0,
    )
    dataset_meta = MockDatasetMeta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 10,
                "length": 10,
                "task": "stack the cube",
            }
        ]
    )
    processor = RECAPSARMEncodingProcessorStep(config=config, dataset_meta=dataset_meta)
    processor.eval()

    output = processor(make_transition(config, 5, 0, "stack the cube"))[TransitionKey.OBSERVATION]

    assert "video_features" in output
    assert "value_targets_bin" not in output
    assert "episode_success" not in output
