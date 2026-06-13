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

from types import SimpleNamespace

import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching


class _FailingStateProjection:
    def __call__(self, state):
        raise AssertionError("state_proj should not be called when use_state=False")


class _FakeVLMWithExpert:
    config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))

    def embed_image(self, image):
        return torch.ones(image.shape[0], 2, 4, dtype=image.dtype, device=image.device)

    def embed_language_tokens(self, tokens):
        return torch.ones(tokens.shape[0], tokens.shape[1], 4, dtype=torch.float32, device=tokens.device)


def test_smolvla_prepare_state_returns_none_when_state_disabled():
    policy = SimpleNamespace(config=SimpleNamespace(use_state=False))

    assert SmolVLAPolicy.prepare_state(policy, {}) is None


def test_smolvla_embed_prefix_skips_state_token_when_state_disabled():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    model.config = SimpleNamespace(use_state=False)
    model.vlm_with_expert = _FakeVLMWithExpert()
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = _FailingStateProjection()

    images = [torch.zeros(2, 3, 8, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    language_tokens = torch.ones(2, 3, dtype=torch.long)
    language_masks = torch.ones(2, 3, dtype=torch.bool)

    embeddings, pad_masks, attention_masks = VLAFlowMatching.embed_prefix(
        model,
        images,
        image_masks,
        language_tokens,
        language_masks,
        state=None,
    )

    assert embeddings.shape == (2, 5, 4)
    assert pad_masks.shape == (2, 5)
    assert attention_masks.shape == (2, 5)
    assert not attention_masks.any()


def test_smolvla_state_projection_not_trainable_when_state_disabled():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_state=False, train_state_proj=True)
    model.state_proj = torch.nn.Linear(2, 2)

    VLAFlowMatching.set_requires_grad(model)

    assert all(not parameter.requires_grad for parameter in model.state_proj.parameters())


def test_smolvla_prepare_state_returns_none_when_discrete_state_in_language_enabled():
    policy = SimpleNamespace(config=SimpleNamespace(use_state=True, discrete_state_in_language=True))

    assert SmolVLAPolicy.prepare_state(policy, {"observation.state": torch.ones(2, 1, 4)}) is None


def test_smolvla_embed_prefix_skips_state_token_when_discrete_state_in_language_enabled():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    model.config = SimpleNamespace(use_state=True, discrete_state_in_language=True)
    model.vlm_with_expert = _FakeVLMWithExpert()
    model.add_image_special_tokens = False
    model.prefix_length = -1
    model.state_proj = _FailingStateProjection()

    images = [torch.zeros(2, 3, 8, 8)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    language_tokens = torch.ones(2, 3, dtype=torch.long)
    language_masks = torch.ones(2, 3, dtype=torch.bool)
    state = torch.ones(2, 4)

    embeddings, pad_masks, attention_masks = VLAFlowMatching.embed_prefix(
        model,
        images,
        image_masks,
        language_tokens,
        language_masks,
        state=state,
    )

    assert embeddings.shape == (2, 5, 4)
    assert pad_masks.shape == (2, 5)
    assert attention_masks.shape == (2, 5)
    assert not attention_masks.any()


def test_smolvla_state_projection_not_trainable_when_discrete_state_in_language_enabled():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_state=True,
        train_state_proj=True,
        discrete_state_in_language=True,
    )
    model.state_proj = torch.nn.Linear(2, 2)

    VLAFlowMatching.set_requires_grad(model)

    assert all(not parameter.requires_grad for parameter in model.state_proj.parameters())


def test_smolvla_default_peft_targets_exclude_state_projection_for_discrete_state_language():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(use_state=True, discrete_state_in_language=True)

    peft_defaults = SmolVLAPolicy._get_default_peft_targets(policy)

    assert "state_proj" not in peft_defaults["target_modules"]
    assert "action_in_proj" in peft_defaults["target_modules"]
