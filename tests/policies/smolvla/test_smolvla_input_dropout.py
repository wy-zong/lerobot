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

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _CapturingModel:
    def __init__(self):
        self.state = None

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
        self.state = None if state is None else state.detach().clone()
        return torch.zeros_like(actions)


def _make_config(input_dropout_prob: float, input_dropout_features: list[str]) -> SmolVLAConfig:
    return SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        max_state_dim=4,
        max_action_dim=3,
        device="cpu",
        resize_imgs_with_padding=None,
        input_dropout_prob=input_dropout_prob,
        input_dropout_features=input_dropout_features,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
    )


def _make_policy(
    input_dropout_prob: float, input_dropout_features: list[str], training: bool
) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = _make_config(input_dropout_prob, input_dropout_features)
    policy.model = _CapturingModel()
    policy.train(training)
    return policy


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[5.0, 6.0, 7.0, 8.0]],
            ]
        ),
        OBS_IMAGE: torch.ones(2, 3, 4, 4),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION: torch.ones(2, 2, 3),
    }


@pytest.mark.parametrize("input_dropout_prob", [-0.1, 1.1])
def test_smolvla_input_dropout_prob_must_be_between_zero_and_one(input_dropout_prob):
    with pytest.raises(ValueError, match="input_dropout_prob"):
        SmolVLAConfig(device="cpu", input_dropout_prob=input_dropout_prob)


def test_smolvla_input_dropout_rejects_unsupported_features():
    with pytest.raises(ValueError, match="unsupported feature"):
        SmolVLAConfig(
            device="cpu",
            input_dropout_prob=0.5,
            input_dropout_features=["observation.images.front"],
        )


def test_smolvla_input_dropout_rejects_state_when_state_disabled():
    with pytest.raises(ValueError, match="use_state=False"):
        SmolVLAConfig(
            device="cpu",
            use_state=False,
            input_dropout_prob=0.5,
            input_dropout_features=[OBS_STATE],
        )


def test_smolvla_input_dropout_prob_zero_keeps_state_unchanged():
    policy = _make_policy(input_dropout_prob=0.0, input_dropout_features=[OBS_STATE], training=True)
    batch = _make_batch()
    expected_state = batch[OBS_STATE][:, -1, :]

    _, metrics = policy.forward(batch)

    assert torch.equal(policy.model.state, expected_state)
    assert metrics["state_dropout_fraction"] == 0.0
    assert torch.equal(batch[OBS_STATE][:, -1, :], expected_state)


def test_smolvla_input_dropout_prob_one_passes_zero_state_to_inner_model():
    policy = _make_policy(input_dropout_prob=1.0, input_dropout_features=[OBS_STATE], training=True)
    batch = _make_batch()

    _, metrics = policy.forward(batch)

    assert torch.equal(policy.model.state, torch.zeros_like(batch[OBS_STATE][:, -1, :]))
    assert metrics["state_dropout_fraction"] == 1.0
    assert torch.equal(
        batch[OBS_STATE][:, -1, :],
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
    )


def test_smolvla_input_dropout_is_disabled_in_eval_mode():
    policy = _make_policy(input_dropout_prob=1.0, input_dropout_features=[OBS_STATE], training=False)
    batch = _make_batch()
    expected_state = batch[OBS_STATE][:, -1, :]

    _, metrics = policy.forward(batch)

    assert torch.equal(policy.model.state, expected_state)
    assert metrics["state_dropout_fraction"] == 0.0
