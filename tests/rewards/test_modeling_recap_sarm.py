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

import torch

from lerobot.rewards.recap_sarm.configuration_recap_sarm import RECAPSARMConfig
from lerobot.rewards.recap_sarm.modeling_recap_sarm import RECAPSARMRewardModel, estimate_advantage
from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel


def make_recap_batch(config: RECAPSARMConfig) -> dict:
    batch_size = 2
    seq_len = config.num_frames
    bin_centers = torch.linspace(config.value_min, config.value_max, config.num_value_bins)
    value_targets_bin = torch.tensor(
        [
            [0, 20, 40, 60],
            [10, 30, 50, 0],
        ],
        dtype=torch.long,
    )
    value_targets_continuous = bin_centers[value_targets_bin]

    return {
        "observation": {
            "video_features": torch.randn(batch_size, seq_len, config.image_dim),
            "text_features": torch.randn(batch_size, config.text_dim),
            "state_features": torch.randn(batch_size, seq_len, 6),
            "lengths": torch.tensor([seq_len, seq_len - 1], dtype=torch.int32),
            "value_targets_bin": value_targets_bin,
            "value_targets_continuous": value_targets_continuous,
        }
    }


def test_recap_sarm_forward_predict_and_compute_reward():
    config = RECAPSARMConfig(
        n_obs_steps=2,
        max_rewind_steps=1,
        num_layers=1,
        num_heads=2,
        hidden_dim=64,
        image_dim=32,
        text_dim=32,
        max_state_dim=8,
        num_value_bins=201,
        dropout=0.0,
        device="cpu",
    )
    model = RECAPSARMRewardModel(config)
    batch = make_recap_batch(config)
    batch["observation"]["video_features"] = torch.randn(2, config.num_frames, config.image_dim)
    batch["observation"]["text_features"] = torch.randn(2, config.text_dim)

    outputs = model.predict_value_distribution(batch)
    assert outputs["value_logits"].shape == (2, config.num_frames, config.num_value_bins)
    assert outputs["value_expectation"].shape == (2, config.num_frames)

    loss, metrics = model.forward(batch)
    loss.backward()

    assert loss.requires_grad
    assert "value_loss" in metrics
    assert "value_mae" in metrics
    assert "value_expectation_mean" in metrics
    assert model.value_head.weight.grad is not None

    rewards = model.compute_reward(batch)
    assert rewards.shape == (2,)
    assert torch.allclose(rewards, outputs["value_expectation"][:, config.n_obs_steps], atol=1e-5)


def test_recap_sarm_advantage_helper_bootstraps_and_clips_tail():
    value_t = torch.tensor([-0.6, -0.4])
    rewards = torch.tensor([[-0.1, -0.05], [-0.2, 0.0]])
    value_t_plus_n = torch.tensor([-0.3, -0.1])

    advantage = estimate_advantage(value_t, rewards, value_t_plus_n)
    expected = rewards.sum(dim=-1) + value_t_plus_n - value_t
    assert torch.allclose(advantage, expected)

    tail_advantage = estimate_advantage(torch.tensor(-0.5), torch.tensor([-0.1]), torch.tensor(-0.2))
    assert torch.isclose(tail_advantage, torch.tensor(0.2))


def test_recap_sarm_and_sarm_state_dict_keys_do_not_overlap():
    sarm_cfg = SARMConfig(
        temporal_window_mode="current_only",
        num_layers=1,
        num_heads=2,
        hidden_dim=64,
        image_dim=32,
        text_dim=32,
        max_state_dim=8,
        device="cpu",
    )
    recap_cfg = RECAPSARMConfig(
        temporal_window_mode="current_only",
        num_layers=1,
        num_heads=2,
        hidden_dim=64,
        image_dim=32,
        text_dim=32,
        max_state_dim=8,
        device="cpu",
    )

    sarm_model = SARMRewardModel(sarm_cfg)
    recap_model = RECAPSARMRewardModel(recap_cfg)

    assert set(sarm_model.state_dict()).isdisjoint(set(recap_model.state_dict()))
