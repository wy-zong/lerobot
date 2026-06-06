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

import numpy as np
import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_vlm_inspect import (
    VLMInspectConfig,
    _resolve_pretrained_name_or_path,
    _run_feature_snapshot,
    answer_with_smolvlm,
    ensure_required_state,
    ensure_smolvla_policy_config,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def test_config_accepts_feature_mode_as_plain_string():
    cfg = VLMInspectConfig(
        robot=SimpleNamespace(type="mock"),
        policy=SmolVLAConfig(device="cpu"),
        device="cpu",
        mode="feature",
    )

    assert cfg.mode == "feature"


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="--mode"):
        VLMInspectConfig(
            robot=SimpleNamespace(type="mock"),
            policy=SmolVLAConfig(device="cpu"),
            device="cpu",
            mode="debug",
        )


def test_config_disables_compile_model_for_inspection():
    cfg = VLMInspectConfig(
        robot=SimpleNamespace(type="mock"),
        policy=SmolVLAConfig(device="cpu", compile_model=True),
        device="cpu",
    )

    assert cfg.policy is not None
    assert not cfg.policy.compile_model


def test_hub_policy_path_preserves_forward_slash():
    assert _resolve_pretrained_name_or_path("wuc1/bi_so101_ffp_4cam_60fps_tRTC") == (
        "wuc1/bi_so101_ffp_4cam_60fps_tRTC"
    )


def test_local_policy_path_uses_path_object(tmp_path):
    assert _resolve_pretrained_name_or_path(tmp_path) == tmp_path


def test_non_smolvla_policy_config_rejected():
    with pytest.raises(ValueError, match="only SmolVLA"):
        ensure_smolvla_policy_config(SimpleNamespace(type="act"))


def test_feature_mode_requires_real_state_when_policy_uses_state():
    policy = SimpleNamespace(config=SimpleNamespace(use_state=True))

    with pytest.raises(ValueError, match="does not synthesize"):
        ensure_required_state(policy, {f"{OBS_IMAGES}.front": torch.zeros(1, 3, 8, 8)})


def test_feature_snapshot_sends_camera_task_and_real_state_to_preprocessor(monkeypatch):
    captured = {}

    class FakePolicy:
        config = SimpleNamespace(use_state=True, use_amp=False)

    def fake_trace(policy, batch, *, baseline_action=None, previous_action=None):
        captured["trace_batch"] = batch
        return (
            {
                "image_tokens": [{"key": f"{OBS_IMAGES}.front"}],
                "language": {"token_count": 3},
                "state": {"values": [1.0, 2.0]},
                "prefix_hidden_states": [{"layer": 0}],
                "expert_attention": {"state": {"mean_mass": 0.5}},
                "action_chunk": {"shape": [1, 2, 2]},
                "action_delta": {"baseline": None, "previous": None},
            },
            torch.ones(1, 2, 2),
        )

    def fake_preprocessor(observation):
        captured["preprocessor_input"] = observation
        assert observation["task"] == "pick cube"
        assert OBS_STATE in observation
        assert f"{OBS_IMAGES}.front" in observation
        return {
            f"{OBS_IMAGES}.front": observation[f"{OBS_IMAGES}.front"],
            OBS_STATE: observation[OBS_STATE],
            OBS_LANGUAGE_TOKENS: torch.ones(1, 3, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        }

    def fake_postprocessor(action):
        captured["postprocessor_action"] = action
        return action.cpu()

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_vlm_inspect.trace_smolvla_features",
        fake_trace,
    )

    observation_frame = {
        f"{OBS_IMAGES}.front": np.zeros((8, 8, 3), dtype=np.uint8),
        OBS_STATE: np.array([1.0, 2.0], dtype=np.float32),
    }

    trace, action_chunk, postprocessed_action = _run_feature_snapshot(
        policy=FakePolicy(),
        preprocessor=fake_preprocessor,
        postprocessor=fake_postprocessor,
        observation_frame=observation_frame,
        task="pick cube",
        device="cpu",
        robot_type="mock_robot",
        baseline_action=None,
        previous_action=None,
    )

    assert trace["task"] == "pick cube"
    assert trace["image_tokens"][0]["key"] == f"{OBS_IMAGES}.front"
    assert OBS_STATE in captured["trace_batch"]
    torch.testing.assert_close(captured["trace_batch"][OBS_STATE], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(action_chunk, torch.ones(1, 2, 2))
    torch.testing.assert_close(postprocessed_action, torch.ones(1, 2, 2))


def test_mock_trace_schema_contains_required_feature_sections(monkeypatch):
    def fake_trace(policy, batch, *, baseline_action=None, previous_action=None):
        return (
            {
                "schema_version": 1,
                "image_tokens": [{"key": f"{OBS_IMAGES}.front", "connector_tokens": {"shape": [1, 4, 8]}}],
                "language": {"token_count": 4},
                "state": {"values": [0.1, 0.2]},
                "state_embedding": {"shape": [1, 1, 8]},
                "prefix_hidden_states": [{"layer": 0, "shape": [1, 5, 8]}],
                "expert_attention": {"language": {"mean_mass": 0.25}},
                "action_chunk": {"shape": [1, 2, 3]},
                "action_delta": {"baseline": {"available": True}, "previous": None},
            },
            torch.zeros(1, 2, 3),
        )

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_vlm_inspect.trace_smolvla_features",
        fake_trace,
    )

    observation_frame = {
        f"{OBS_IMAGES}.front": np.zeros((8, 8, 3), dtype=np.uint8),
        OBS_STATE: np.zeros(2, dtype=np.float32),
    }

    def fake_preprocessor(observation):
        return {
            f"{OBS_IMAGES}.front": torch.zeros(1, 3, 8, 8),
            OBS_STATE: torch.zeros(1, 2),
            OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
        }

    trace, _action_chunk, _postprocessed_action = _run_feature_snapshot(
        policy=SimpleNamespace(config=SimpleNamespace(use_state=True, use_amp=False)),
        preprocessor=fake_preprocessor,
        postprocessor=lambda action: action,
        observation_frame=observation_frame,
        task="inspect",
        device="cpu",
        robot_type="mock_robot",
        baseline_action=torch.zeros(1, 2, 3),
        previous_action=None,
    )

    for key in (
        "image_tokens",
        "language",
        "state",
        "state_embedding",
        "prefix_hidden_states",
        "expert_attention",
        "action_chunk",
        "action_delta",
    ):
        assert key in trace


class _FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class _FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        assert add_generation_prompt
        return messages[0]["content"][1]["text"]

    def __call__(self, *, text, images, return_tensors):
        assert text == "what is visible?"
        assert len(images) == 1
        assert return_tensors == "pt"
        return _FakeInputs({"input_ids": torch.ones(1, 2, dtype=torch.long)})

    def batch_decode(self, output_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return ["a red cube"]


class _FakeVLM:
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 64
        return torch.tensor([[1, 1, 9, 10]])


class _FakeVLMWithExpert:
    processor = _FakeProcessor()
    vlm = _FakeVLM()

    def __init__(self, loaded_layers=2):
        self._loaded_layers = loaded_layers

    def get_vlm_model(self):
        return SimpleNamespace(text_model=SimpleNamespace(layers=[object()] * self._loaded_layers))


def test_answer_mode_generates_when_full_vlm_available():
    policy = SimpleNamespace(model=SimpleNamespace(vlm_with_expert=_FakeVLMWithExpert()))

    result = answer_with_smolvlm(
        policy,
        np.zeros((8, 8, 3), dtype=np.uint8),
        "what is visible?",
        device="cpu",
    )

    assert result == {"available": True, "answer": "a red cube", "reason": None}


def test_answer_mode_unavailable_for_truncated_vlm():
    policy = SimpleNamespace(model=SimpleNamespace(vlm_with_expert=_FakeVLMWithExpert(loaded_layers=1)))

    result = answer_with_smolvlm(
        policy,
        np.zeros((8, 8, 3), dtype=np.uint8),
        "what is visible?",
        device="cpu",
    )

    assert not result["available"]
    assert result["answer"] is None
    assert "truncated" in result["reason"]
