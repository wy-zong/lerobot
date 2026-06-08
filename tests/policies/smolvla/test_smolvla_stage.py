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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _StageLossModel:
    def __init__(self):
        self.ce_lang_tokens = None
        self.ce_subtask_tokens = None
        self.flow_lang_tokens = None

    def forward_subtask_prediction(
        self, images, img_masks, lang_tokens, lang_masks, state, subtask_tokens, subtask_masks
    ):
        self.ce_lang_tokens = lang_tokens.detach().clone()
        self.ce_subtask_tokens = subtask_tokens.detach().clone()
        return torch.tensor(2.0)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, **kwargs):
        self.flow_lang_tokens = lang_tokens.detach().clone()
        return torch.full_like(actions, 0.25)


def _make_policy() -> tuple[SmolVLAPolicy, _StageLossModel]:
    config = SmolVLAConfig(
        chunk_size=2,
        n_action_steps=2,
        max_state_dim=4,
        max_action_dim=3,
        device="cpu",
        resize_imgs_with_padding=None,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
    )
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    model = _StageLossModel()
    policy.model = model
    policy.train()
    return policy, model


def _make_stage_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.ones(2, 1, 4),
        OBS_IMAGE: torch.ones(2, 3, 4, 4),
        OBS_LANGUAGE_TOKENS: torch.tensor([[10, 11], [12, 13]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
        OBS_LANGUAGE_SUBTASK_TOKENS: torch.tensor([[20, 21], [22, 23]], dtype=torch.long),
        OBS_LANGUAGE_SUBTASK_ATTENTION_MASK: torch.ones(2, 2, dtype=torch.bool),
        ACTION: torch.ones(2, 2, 3),
    }


def test_smolvla_forward_stage_combines_subtask_ce_and_action_flow_loss():
    policy, model = _make_policy()
    batch = _make_stage_batch()

    loss, metrics = policy.forward_stage(batch, flow_loss_weight=10.0)

    torch.testing.assert_close(loss, torch.tensor(4.5))
    assert metrics["subtask_ce_loss"] == 2.0
    assert metrics["action_flow_loss"] == 0.25
    assert metrics["flow_loss_weight"] == 10.0
    assert metrics["loss"] == 4.5
    torch.testing.assert_close(model.ce_lang_tokens, batch[OBS_LANGUAGE_TOKENS])
    torch.testing.assert_close(model.ce_subtask_tokens, batch[OBS_LANGUAGE_SUBTASK_TOKENS])
    torch.testing.assert_close(model.flow_lang_tokens, batch[OBS_LANGUAGE_SUBTASK_TOKENS])


def test_smolvla_forward_stage_requires_subtask_tokens():
    policy, _model = _make_policy()
    batch = _make_stage_batch()
    del batch[OBS_LANGUAGE_SUBTASK_TOKENS]

    with pytest.raises(KeyError, match="tokenized subtasks"):
        policy.forward_stage(batch)


def test_smolvla_forward_stage_rejects_reduction_none():
    policy, _model = _make_policy()

    with pytest.raises(ValueError, match="reduction='mean'"):
        policy.forward_stage(_make_stage_batch(), reduction="none")
