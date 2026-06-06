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
    FeatureSnapshotReference,
    VLMInspectConfig,
    _representation_delta,
    _resolve_pretrained_name_or_path,
    _run_feature_snapshot,
    _tensor_delta,
    answer_with_smolvlm,
    ensure_required_state,
    ensure_smolvla_policy_config,
    trace_smolvla_features,
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


def test_tensor_delta_reports_required_metrics():
    current = torch.tensor([[1.0, 3.0]])
    reference = torch.tensor([[0.0, 1.0]])

    delta = _tensor_delta(current, reference)

    assert delta["available"]
    assert delta["shape_compatible"]
    assert delta["l2"] == pytest.approx(5**0.5)
    assert delta["mean_abs"] == pytest.approx(1.5)
    assert delta["max_abs"] == pytest.approx(2.0)
    assert "cosine_distance" in delta


def test_tensor_delta_marks_shape_mismatch_unavailable():
    delta = _tensor_delta(torch.zeros(1, 2), torch.zeros(1, 3))

    assert not delta["available"]
    assert not delta["shape_compatible"]
    assert "shape mismatch" in delta["reason"]


def test_representation_delta_handles_missing_references():
    current = {"prefix_output": torch.ones(1, 2)}
    reference = FeatureSnapshotReference(tensors={}, action_chunk=torch.zeros(1, 1, 1))

    assert _representation_delta(current, None) == {}
    delta = _representation_delta(current, reference)

    assert not delta["prefix_output"]["available"]
    assert delta["prefix_output"]["reason"] == "reference tensor is missing"


def test_feature_snapshot_sends_camera_task_and_real_state_to_preprocessor(monkeypatch):
    captured = {}

    class FakePolicy:
        config = SimpleNamespace(use_state=True, use_amp=False)

    def fake_trace(policy, batch, *, tensor_dir=None, baseline_reference=None, previous_reference=None):
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
            FeatureSnapshotReference(
                tensors={"prefix_output": torch.ones(1, 1, 2)},
                action_chunk=torch.ones(1, 2, 2),
            ),
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
        tensor_dir=None,
        baseline_reference=None,
        previous_reference=None,
    )

    assert trace["task"] == "pick cube"
    assert trace["image_tokens"][0]["key"] == f"{OBS_IMAGES}.front"
    assert OBS_STATE in captured["trace_batch"]
    torch.testing.assert_close(captured["trace_batch"][OBS_STATE], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(action_chunk.action_chunk, torch.ones(1, 2, 2))
    torch.testing.assert_close(postprocessed_action, torch.ones(1, 2, 2))


def test_mock_trace_schema_contains_required_feature_sections(monkeypatch):
    def fake_trace(policy, batch, *, tensor_dir=None, baseline_reference=None, previous_reference=None):
        return (
            {
                "schema_version": 1,
                "tensor_artifacts": {"prefix_output": {"path": "tensors/prefix_output.npz"}},
                "image_tokens": [{"key": f"{OBS_IMAGES}.front", "connector_tokens": {"shape": [1, 4, 8]}}],
                "language": {"token_count": 4},
                "state": {"values": [0.1, 0.2]},
                "state_embedding": {"shape": [1, 1, 8]},
                "prefix_hidden_states": [{"layer": 0, "shape": [1, 5, 8]}],
                "expert_attention": {"language": {"mean_mass": 0.25}},
                "action_chunk": {"shape": [1, 2, 3]},
                "action_delta": {"baseline": {"available": True}, "previous": None},
                "representation_delta": {
                    "baseline": {"prefix_output": {"available": True}},
                    "previous": {},
                },
            },
            FeatureSnapshotReference(
                tensors={"prefix_output": torch.zeros(1, 5, 8)},
                action_chunk=torch.zeros(1, 2, 3),
            ),
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
        tensor_dir=None,
        baseline_reference=FeatureSnapshotReference(
            tensors={"prefix_output": torch.zeros(1, 5, 8)},
            action_chunk=torch.zeros(1, 2, 3),
        ),
        previous_reference=None,
    )

    for key in (
        "tensor_artifacts",
        "image_tokens",
        "language",
        "state",
        "state_embedding",
        "prefix_hidden_states",
        "expert_attention",
        "action_chunk",
        "action_delta",
        "representation_delta",
    ):
        assert key in trace


class _FakeActionOutProj:
    def __call__(self, suffix_out):
        return suffix_out


class _FakeTraceVLMWithExpert:
    def embed_image(self, image):
        value = image.to(dtype=torch.float32).mean(dim=(1, 2, 3))
        return value[:, None, None].expand(image.shape[0], 2, 3).contiguous()

    def forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
        trace=None,
    ):
        if fill_kv_cache:
            prefix_embs = inputs_embeds[0]
            cache = {
                0: {
                    "key_states": prefix_embs[:, :, None, :],
                    "value_states": (prefix_embs + 1.0)[:, :, None, :],
                }
            }
            if trace is not None:
                trace.setdefault("prefix_hidden_states", []).append(
                    {"layer": 0, "shape": list(prefix_embs.shape)}
                )
            return [prefix_embs + 2.0, None], cache

        suffix_embs = inputs_embeds[1]
        return [None, suffix_embs + 0.5], past_key_values


class _FakeTraceFlow:
    def __init__(self, config):
        self.config = config
        self.vlm_with_expert = _FakeTraceVLMWithExpert()
        self.action_out_proj = _FakeActionOutProj()
        self.add_image_special_tokens = False
        self.rtc_processor = None
        self.sample_noise_calls = 0

    def _rtc_enabled(self):
        return False

    def sample_noise(self, shape, device):
        self.sample_noise_calls += 1
        return torch.full(shape, 0.25, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        image_embs = [self.vlm_with_expert.embed_image(image) for image in images]
        lang_embs = lang_tokens.to(dtype=torch.float32)[:, :, None].expand(-1, -1, 3)
        prefix_embs = torch.cat([*image_embs, lang_embs], dim=1)
        prefix_pad_masks = torch.ones(prefix_embs.shape[:2], dtype=torch.bool, device=prefix_embs.device)
        prefix_att_masks = torch.zeros(prefix_embs.shape[:2], dtype=torch.bool, device=prefix_embs.device)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def embed_suffix(self, noisy_actions, timestep):
        suffix_embs = noisy_actions + timestep[:, None, None]
        suffix_pad_masks = torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
        suffix_att_masks = torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
        return suffix_embs, suffix_pad_masks, suffix_att_masks


class _FakeTracePolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            use_state=False,
            image_features={f"{OBS_IMAGES}.front": SimpleNamespace()},
            chunk_size=2,
            max_action_dim=2,
            num_steps=2,
            use_cache=True,
            action_feature=SimpleNamespace(shape=(2,)),
            adapt_to_pi_aloha=False,
        )
        self.model = _FakeTraceFlow(self.config)

    def eval(self):
        return self

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        key = f"{OBS_IMAGES}.front"
        return [batch[key]], [torch.ones(batch[key].shape[0], dtype=torch.bool)]

    def prepare_state(self, batch):
        return None

    def predict_action_chunk(self, batch):
        raise AssertionError("trace_smolvla_features must not rerun predict_action_chunk")


def _fake_trace_batch(image_value=0.0):
    return {
        f"{OBS_IMAGES}.front": torch.full((1, 3, 4, 4), image_value),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }


def test_trace_writes_tensor_artifacts_and_uses_instrumented_inference(tmp_path):
    policy = _FakeTracePolicy()

    trace1, reference1 = trace_smolvla_features(
        policy,
        _fake_trace_batch(0.0),
        tensor_dir=tmp_path / "snapshot_0001" / "tensors",
    )
    trace2, _reference2 = trace_smolvla_features(
        policy,
        _fake_trace_batch(1.0),
        tensor_dir=tmp_path / "snapshot_0002" / "tensors",
        baseline_reference=reference1,
        previous_reference=reference1,
    )

    assert policy.model.sample_noise_calls == 2
    torch.testing.assert_close(
        reference1.action_chunk,
        torch.full((1, 2, 2), -0.8125),
    )
    for artifact_name in (
        "image_connector_tokens",
        "prefix_output",
        "prefix_kv_cache",
        "suffix_out_steps",
    ):
        artifact = trace1["tensor_artifacts"][artifact_name]
        assert artifact["path"] == f"tensors/{artifact_name}.npz"
        assert (tmp_path / "snapshot_0001" / artifact["path"]).exists()
        assert artifact["arrays"]

    suffix_artifact = trace1["tensor_artifacts"]["suffix_out_steps"]["arrays"]["suffix_out_steps"]
    assert suffix_artifact["shape"] == [2, 1, 2, 2]
    assert suffix_artifact["step_count"] == 2
    image_delta = trace2["representation_delta"]["baseline"][f"image_connector_tokens:{OBS_IMAGES}.front"]
    assert image_delta["available"]
    assert image_delta["mean_abs"] > 0
    assert trace2["representation_delta"]["previous"][f"image_connector_tokens:{OBS_IMAGES}.front"][
        "available"
    ]


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
