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

import pytest

from lerobot.rewards.sarm.configuration_sarm import SARMConfig


def test_sarm_current_only_temporal_window_config():
    cfg = SARMConfig(temporal_window_mode="current_only", n_obs_steps=8, max_rewind_steps=4)

    assert cfg.n_obs_steps == 0
    assert cfg.max_rewind_steps == 0
    assert cfg.num_frames == 1
    assert cfg.observation_delta_indices == [0]
    assert cfg.output_features["sparse_stage"].shape[0] == 1
    assert cfg.output_features["sparse_progress"].shape[0] == 1


def test_sarm_bidirectional_temporal_window_config_unchanged():
    cfg = SARMConfig(
        temporal_window_mode="bidirectional",
        n_obs_steps=8,
        max_rewind_steps=4,
        frame_gap=30,
    )

    assert cfg.num_frames == 13
    assert cfg.observation_delta_indices == [-120, -90, -60, -30, 0, 30, 60, 90, 120, -30, -60, -90, -120]
    assert cfg.max_rewind_steps == 4


def test_sarm_rejects_unknown_temporal_window_mode():
    with pytest.raises(ValueError, match="temporal_window_mode"):
        SARMConfig(temporal_window_mode="latest_frame")
