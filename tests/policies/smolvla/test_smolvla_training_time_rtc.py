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

from collections import deque
from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _RTCLossModel:
    def __init__(self, mode: str = "ones"):
        self.mode = mode
        self.kwargs = None

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise, time, **kwargs):
        self.kwargs = kwargs
        if self.mode == "prefix_high":
            prefix_mask = kwargs["training_time_rtc_prefix_mask"]
            return torch.where(
                prefix_mask[:, :, None], torch.full_like(actions, 100.0), torch.ones_like(actions)
            )
        if self.mode == "position":
            position_loss = torch.arange(1, actions.shape[1] + 1, dtype=actions.dtype, device=actions.device)
            return position_loss[None, :, None].expand_as(actions)
        return torch.ones_like(actions)


def _make_config(
    *,
    chunk_size: int = 10,
    action_dim: int = 2,
    training_time_rtc_enabled: bool = False,
    training_time_rtc_max_delay_steps: int = 6,
) -> SmolVLAConfig:
    return SmolVLAConfig(
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        max_state_dim=4,
        max_action_dim=action_dim,
        device="cpu",
        resize_imgs_with_padding=None,
        training_time_rtc_enabled=training_time_rtc_enabled,
        training_time_rtc_max_delay_steps=training_time_rtc_max_delay_steps,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
    )


def _make_policy(config: SmolVLAConfig, model: _RTCLossModel, training: bool = True) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    policy.train(training)
    return policy


def _make_batch(batch_size: int = 2, chunk_size: int = 10, action_dim: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.ones(batch_size, 1, 4),
        OBS_IMAGE: torch.ones(batch_size, 3, 4, 4),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 3, dtype=torch.bool),
        ACTION: torch.ones(batch_size, chunk_size, action_dim),
    }


def _get_norm_mode(config: SmolVLAConfig, feature_type: FeatureType) -> NormalizationMode:
    for key, mode in config.normalization_mapping.items():
        key_value = key.value if isinstance(key, FeatureType) else key
        if key_value == feature_type.value:
            return NormalizationMode(mode)
    raise AssertionError(f"{feature_type} normalization is missing")


def test_smolvla_training_time_rtc_default_disabled_and_forward_contract():
    default_config = SmolVLAConfig(device="cpu")
    assert default_config.training_time_rtc_enabled is False
    assert default_config.rtc_config is None

    config = _make_config(chunk_size=4, training_time_rtc_enabled=False)
    model = _RTCLossModel()
    policy = _make_policy(config, model)
    batch = _make_batch(batch_size=2, chunk_size=4)

    loss, metrics = policy.forward(batch)
    per_sample_loss, per_sample_metrics = policy.forward(batch, reduction="none")

    assert loss.shape == torch.Size([])
    assert torch.equal(per_sample_loss, torch.ones(2))
    assert metrics["training_time_rtc_delay_mean"] == 0.0
    assert metrics["training_time_rtc_prefix_fraction"] == 0.0
    assert per_sample_metrics["training_time_rtc_prefix_fraction"] == 0.0
    assert model.kwargs == {}


@pytest.mark.parametrize(
    ("chunk_size", "max_delay_steps", "match"),
    [
        (50, -1, "greater than or equal to 0"),
        (4, 4, "smaller than `chunk_size`"),
        (4, 5, "smaller than `chunk_size`"),
    ],
)
def test_smolvla_training_time_rtc_rejects_invalid_max_delay(chunk_size, max_delay_steps, match):
    with pytest.raises(ValueError, match=match):
        SmolVLAConfig(
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            device="cpu",
            training_time_rtc_enabled=True,
            training_time_rtc_max_delay_steps=max_delay_steps,
        )


def test_smolvla_training_time_rtc_deterministic_delay_masks_prefix_loss(monkeypatch):
    def fake_randint(*, low, high, size, device):
        assert low == 0
        assert high == 7
        assert size == (1,)
        return torch.tensor([6], device=device)

    monkeypatch.setattr(torch, "randint", fake_randint)

    config = _make_config(chunk_size=10, training_time_rtc_enabled=True, training_time_rtc_max_delay_steps=6)
    model = _RTCLossModel(mode="prefix_high")
    policy = _make_policy(config, model)
    batch = _make_batch(batch_size=1, chunk_size=10)

    loss, metrics = policy.forward(batch)

    assert torch.isclose(loss, torch.tensor(1.0))
    assert metrics["training_time_rtc_delay_mean"] == 6.0
    assert metrics["training_time_rtc_prefix_fraction"] == pytest.approx(0.6)
    torch.testing.assert_close(
        model.kwargs["training_time_rtc_prefix_mask"],
        torch.tensor([[True, True, True, True, True, True, False, False, False, False]]),
    )


def test_smolvla_training_time_rtc_reduction_none_excludes_prefix_and_padding(monkeypatch):
    def fake_randint(*, low, high, size, device):
        assert size == (2,)
        return torch.tensor([2, 4], device=device)

    monkeypatch.setattr(torch, "randint", fake_randint)

    config = _make_config(chunk_size=8, training_time_rtc_enabled=True, training_time_rtc_max_delay_steps=4)
    model = _RTCLossModel(mode="position")
    policy = _make_policy(config, model)
    batch = _make_batch(batch_size=2, chunk_size=8)
    batch["action_is_pad"] = torch.tensor(
        [
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, True],
        ]
    )

    per_sample_loss, metrics = policy.forward(batch, reduction="none")

    torch.testing.assert_close(per_sample_loss, torch.tensor([4.0, 6.0]))
    assert metrics["loss"] == 5.0


def test_smolvla_flow_matching_training_time_rtc_uses_clean_prefix_and_zero_prefix_time():
    model = VLAFlowMatching.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=8)
    captured = {}
    hidden_dim = 3
    action_dim = 2

    def fake_embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
        batch_size = lang_tokens.shape[0]
        return (
            torch.zeros(batch_size, 1, hidden_dim),
            torch.ones(batch_size, 1, dtype=torch.bool),
            torch.zeros(batch_size, 1, dtype=torch.bool),
        )

    def fake_embed_suffix(noisy_actions, timestep):
        captured["x_t"] = noisy_actions.detach().clone()
        captured["timestep"] = timestep.detach().clone()
        batch_size, chunk_size = noisy_actions.shape[:2]
        return (
            torch.zeros(batch_size, chunk_size, hidden_dim),
            torch.ones(batch_size, chunk_size, dtype=torch.bool),
            torch.ones(batch_size, chunk_size, dtype=torch.bool),
        )

    class _FakeVLMWithExpert:
        def forward(
            self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache
        ):
            batch_size = attention_mask.shape[0]
            return (
                torch.zeros(batch_size, 1, hidden_dim),
                torch.zeros(batch_size, model.config.chunk_size, hidden_dim),
            ), None

    class _ZeroActionOut:
        def __call__(self, suffix_out):
            return torch.zeros(suffix_out.shape[0], suffix_out.shape[1], action_dim)

    model.embed_prefix = fake_embed_prefix
    model.embed_suffix = fake_embed_suffix
    model.vlm_with_expert = _FakeVLMWithExpert()
    model.action_out_proj = _ZeroActionOut()

    actions = torch.arange(2 * 8 * action_dim, dtype=torch.float32).reshape(2, 8, action_dim)
    noise = actions + 10.0
    time = torch.tensor([0.25, 0.75])
    delay_steps = torch.tensor([3, 0])

    VLAFlowMatching.forward(
        model,
        images=[],
        img_masks=[],
        lang_tokens=torch.ones(2, 3, dtype=torch.long),
        lang_masks=torch.ones(2, 3, dtype=torch.bool),
        state=None,
        actions=actions,
        noise=noise,
        time=time,
        training_time_rtc_delay_steps=delay_steps,
    )

    expected_x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    expected_x_t[0, :3] = actions[0, :3]
    expected_timestep = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
        ]
    )

    torch.testing.assert_close(captured["x_t"], expected_x_t)
    torch.testing.assert_close(captured["timestep"], expected_timestep)


def test_smolvla_training_time_rtc_config_does_not_change_predict_or_select_action():
    fixed_actions = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    outputs = []

    for enabled in (False, True):
        policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = _make_config(
            chunk_size=4,
            training_time_rtc_enabled=enabled,
            training_time_rtc_max_delay_steps=2,
        )
        policy._queues = {ACTION: deque(maxlen=policy.config.n_action_steps)}
        policy._get_action_chunk = lambda batch, noise=None, **kwargs: fixed_actions.clone()

        chunk = SmolVLAPolicy.predict_action_chunk(policy, {})
        selected_action = SmolVLAPolicy.select_action(policy, {})
        outputs.append((chunk, selected_action, policy.config.rtc_config))

    torch.testing.assert_close(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])
    assert outputs[0][2] is None
    assert outputs[1][2] is None


def test_smolvla_training_time_rtc_policy_cli_overrides_parse():
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=lerobot/pusht",
            "--policy.type=smolvla",
            "--policy.training_time_rtc_enabled=true",
            "--policy.training_time_rtc_max_delay_steps=12",
        ],
    )

    assert isinstance(cfg.dataset, DatasetConfig)
    assert isinstance(cfg.policy, SmolVLAConfig)
    assert cfg.policy.training_time_rtc_enabled is True
    assert cfg.policy.training_time_rtc_max_delay_steps == 12


def test_smolvla_discrete_state_language_config_defaults_and_overrides():
    default_config = SmolVLAConfig(device="cpu")
    assert default_config.discrete_state_in_language is False
    assert default_config.tokenizer_max_length == 48
    assert _get_norm_mode(default_config, FeatureType.STATE) == NormalizationMode.MEAN_STD

    discrete_config = SmolVLAConfig(device="cpu", discrete_state_in_language=True)
    assert discrete_config.discrete_state_in_language is True
    assert discrete_config.tokenizer_max_length == 200
    assert _get_norm_mode(discrete_config, FeatureType.STATE) == NormalizationMode.QUANTILES

    explicit_length_config = SmolVLAConfig(
        device="cpu",
        discrete_state_in_language=True,
        tokenizer_max_length=128,
    )
    assert explicit_length_config.tokenizer_max_length == 128


def test_smolvla_discrete_state_language_policy_cli_overrides_parse():
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=lerobot/pusht",
            "--policy.type=smolvla",
            "--policy.discrete_state_in_language=true",
        ],
    )

    assert isinstance(cfg.dataset, DatasetConfig)
    assert isinstance(cfg.policy, SmolVLAConfig)
    assert cfg.policy.discrete_state_in_language is True
    assert cfg.policy.tokenizer_max_length == 200
    assert _get_norm_mode(cfg.policy, FeatureType.STATE) == NormalizationMode.QUANTILES
