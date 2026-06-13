# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import contextlib
import dataclasses
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.processor import ProcessorStep

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.model_test_mode is False
    assert cfg.input_device == "keyboard"
    assert cfg.keyboard.next_episode == "right"
    assert cfg.keyboard.rerecord_episode == "left"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


def test_safe_push_to_hub_failure_is_non_fatal(caplog):
    from lerobot.rollout.strategies.core import safe_push_to_hub

    dataset = MagicMock()
    dataset.num_episodes = 1
    dataset.push_to_hub.side_effect = RuntimeError("forbidden")

    with caplog.at_level("ERROR"):
        assert safe_push_to_hub(dataset, tags=["test"], private=True) is False

    dataset.push_to_hub.assert_called_once_with(tags=["test"], private=True)
    assert "Push to hub failed: forbidden" in caplog.text


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


class _DropObservationStateStep(ProcessorStep):
    def __init__(self):
        self.reset_calls = 0

    def __call__(self, transition):
        from lerobot.types import TransitionKey
        from lerobot.utils.constants import OBS_STATE

        new_transition = transition.copy()
        observation = dict(new_transition[TransitionKey.OBSERVATION])
        observation.pop(OBS_STATE, None)
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def reset(self):
        self.reset_calls += 1

    def transform_features(self, features):
        return features


class _SyncStubPolicy:
    def __init__(
        self,
        *,
        action_chunk: torch.Tensor | None = None,
        select_action: torch.Tensor | None = None,
        n_action_steps: int | None = None,
    ):
        if action_chunk is None:
            action_chunk = torch.empty(1, 0, 2)
        if select_action is None:
            select_action = torch.zeros(1, action_chunk.shape[-1])
        self.action_chunk = action_chunk
        self.select_action_tensor = select_action
        self.config = SimpleNamespace(
            use_amp=False,
            n_action_steps=n_action_steps or action_chunk.shape[1],
            action_feature_names=None,
        )
        self.predict_action_chunk_calls = 0
        self.select_action_calls = 0
        self.reset_calls = 0
        self.last_observation = None

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation):
        self.predict_action_chunk_calls += 1
        self.last_observation = observation
        return self.action_chunk.clone()

    def select_action(self, observation):
        self.select_action_calls += 1
        self.last_observation = observation
        return self.select_action_tensor.clone()


def _make_test_sync_engine(policy, preprocessor, postprocessor):
    from lerobot.rollout import SyncInferenceEngine
    from lerobot.utils.constants import ACTION

    action_names = ["joint_0.pos", "joint_1.pos"]
    return SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features={ACTION: {"names": action_names}},
        ordered_action_keys=action_names,
        task="test task",
        device="cpu",
        robot_type="mock_robot",
    )


def _make_obs_frame(state):
    from lerobot.utils.constants import OBS_STATE

    return {OBS_STATE: np.asarray(state, dtype=np.float32)}


def test_sync_relative_actions_fifo_anchors_chunk_to_refill_state():
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    action_chunk = torch.tensor([[[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]]], dtype=torch.float32)
    policy = _SyncStubPolicy(action_chunk=action_chunk)
    relative_step = RelativeActionsProcessorStep(enabled=True)
    preprocessor = PolicyProcessorPipeline(steps=[relative_step, _DropObservationStateStep()])
    postprocessor = PolicyProcessorPipeline(
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    actions = [
        engine.get_action(_make_obs_frame([10.0, 20.0])),
        engine.get_action(_make_obs_frame([100.0, 200.0])),
        engine.get_action(_make_obs_frame([-1.0, -2.0])),
    ]

    expected = action_chunk.squeeze(0) + torch.tensor([10.0, 20.0])
    for action, expected_action in zip(actions, expected, strict=True):
        torch.testing.assert_close(action, expected_action)
    assert policy.predict_action_chunk_calls == 1
    assert policy.select_action_calls == 0


def test_sync_relative_actions_no_state_model_input_still_postprocesses_from_raw_state():
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import OBS_STATE

    action_chunk = torch.tensor([[[1.0, -1.5]]], dtype=torch.float32)
    policy = _SyncStubPolicy(action_chunk=action_chunk)
    relative_step = RelativeActionsProcessorStep(enabled=True)
    preprocessor = PolicyProcessorPipeline(steps=[relative_step, _DropObservationStateStep()])
    postprocessor = PolicyProcessorPipeline(
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    action = engine.get_action(_make_obs_frame([5.0, 6.0]))

    assert OBS_STATE not in policy.last_observation
    torch.testing.assert_close(action, torch.tensor([6.0, 4.5]))


def test_sync_non_relative_actions_keep_select_action_path():
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    policy = _SyncStubPolicy(
        action_chunk=torch.tensor([[[100.0, 200.0]]], dtype=torch.float32),
        select_action=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    preprocessor = PolicyProcessorPipeline(steps=[])
    postprocessor = PolicyProcessorPipeline(
        steps=[],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    action = engine.get_action(_make_obs_frame([10.0, 20.0]))

    torch.testing.assert_close(action, torch.tensor([1.0, 2.0]))
    assert policy.select_action_calls == 1
    assert policy.predict_action_chunk_calls == 0


def test_sync_relative_actions_intra_chunk_smoothing_applies_before_fifo():
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        IntraChunkSmoothingProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    t = torch.linspace(-1.0, 1.0, 8)
    alternating = torch.where(torch.arange(t.numel()) % 2 == 0, 1.0, -1.0)
    action_chunk = torch.stack([t + 0.1 * alternating, -0.5 * t - 0.2 * alternating], dim=-1).unsqueeze(
        0
    )
    policy = _SyncStubPolicy(action_chunk=action_chunk)
    relative_step = RelativeActionsProcessorStep(enabled=True)
    smoothing_step = IntraChunkSmoothingProcessorStep(enabled=True)
    preprocessor = PolicyProcessorPipeline(steps=[relative_step, _DropObservationStateStep()])
    postprocessor = PolicyProcessorPipeline(
        steps=[
            AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step),
            smoothing_step,
        ],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    actions = [engine.get_action(_make_obs_frame([10.0, 20.0])) for _ in range(action_chunk.shape[1])]

    absolute_chunk = action_chunk + torch.tensor([10.0, 20.0])
    expected = smoothing_step.action(absolute_chunk).squeeze(0)
    for action, expected_action in zip(actions, expected, strict=True):
        torch.testing.assert_close(action, expected_action)
    assert not torch.allclose(expected, absolute_chunk.squeeze(0))
    assert policy.predict_action_chunk_calls == 1
    assert policy.select_action_calls == 0


def test_sync_non_relative_select_action_ignores_intra_chunk_smoothing():
    from lerobot.processor import (
        IntraChunkSmoothingProcessorStep,
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    policy = _SyncStubPolicy(
        action_chunk=torch.tensor([[[100.0, 200.0], [300.0, 400.0]]], dtype=torch.float32),
        select_action=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )
    preprocessor = PolicyProcessorPipeline(steps=[])
    postprocessor = PolicyProcessorPipeline(
        steps=[IntraChunkSmoothingProcessorStep(enabled=True)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    action = engine.get_action(_make_obs_frame([10.0, 20.0]))

    torch.testing.assert_close(action, torch.tensor([1.0, 2.0]))
    assert policy.select_action_calls == 1
    assert policy.predict_action_chunk_calls == 0


def test_sync_relative_actions_reset_clears_fifo_and_cached_state():
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    action_chunk = torch.tensor([[[0.25, 0.5], [1.0, 1.5]]], dtype=torch.float32)
    policy = _SyncStubPolicy(action_chunk=action_chunk)
    relative_step = RelativeActionsProcessorStep(enabled=True)
    drop_state_step = _DropObservationStateStep()
    preprocessor = PolicyProcessorPipeline(steps=[relative_step, drop_state_step])
    postprocessor = PolicyProcessorPipeline(
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = _make_test_sync_engine(policy, preprocessor, postprocessor)

    torch.testing.assert_close(engine.get_action(_make_obs_frame([1.0, 2.0])), torch.tensor([1.25, 2.5]))
    assert relative_step.get_cached_state() is not None

    engine.reset()

    assert policy.reset_calls == 1
    assert drop_state_step.reset_calls == 1
    assert relative_step.get_cached_state() is None
    torch.testing.assert_close(
        engine.get_action(_make_obs_frame([10.0, 20.0])), torch.tensor([10.25, 20.5])
    )


def test_rollout_context_appends_runtime_intra_chunk_smoothing_step():
    from lerobot.processor import IntraChunkSmoothingProcessorStep, PolicyProcessorPipeline
    from lerobot.rollout.context import _append_intra_chunk_smoothing_step

    cfg = SimpleNamespace(intra_chunk_smoothing=True, intra_chunk_smoothing_degree=3)
    postprocessor = PolicyProcessorPipeline(steps=[])

    result = _append_intra_chunk_smoothing_step(postprocessor, cfg)

    assert result is postprocessor
    assert len(postprocessor.steps) == 1
    assert isinstance(postprocessor.steps[0], IntraChunkSmoothingProcessorStep)
    assert postprocessor.steps[0].get_config() == {"enabled": True, "degree": 3}


def test_rtc_postprocess_action_chunk_smooths_processed_but_keeps_original_raw():
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import (
        IntraChunkSmoothingProcessorStep,
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )
    from lerobot.rollout import RTCInferenceEngine

    t = torch.linspace(-1.0, 1.0, 8)
    alternating = torch.where(torch.arange(t.numel()) % 2 == 0, 1.0, -1.0)
    actions = torch.stack([t + 0.1 * alternating, t - 0.15 * alternating], dim=-1).unsqueeze(0)
    preprocessor = PolicyProcessorPipeline(steps=[])
    smoothing_step = IntraChunkSmoothingProcessorStep(enabled=True)
    postprocessor = PolicyProcessorPipeline(
        steps=[smoothing_step],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    engine = RTCInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace(action_feature_names=None), reset=lambda: None),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=SimpleNamespace(robot_type="mock_robot", action_features={}),
        rtc_config=RTCConfig(enabled=True),
        hw_features={},
        task="test task",
        fps=30.0,
        device="cpu",
    )

    original, processed = engine._postprocess_action_chunk(actions)

    torch.testing.assert_close(original, actions.squeeze(0))
    torch.testing.assert_close(processed, smoothing_step.action(actions).squeeze(0))
    assert not torch.allclose(processed, original)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


class _FakeContinuousDataset:
    def __init__(self, has_frames: bool = False):
        self._has_frames = has_frames
        self.add_frame_calls = 0
        self.save_episode_calls = 0
        self.clear_episode_buffer_calls = 0
        self.num_episodes = 0

    def has_pending_frames(self):
        return self._has_frames

    def add_frame(self, frame):
        self.add_frame_calls += 1
        self._has_frames = True
        self.last_frame = frame

    def save_episode(self):
        assert self._has_frames
        self.save_episode_calls += 1
        self.num_episodes += 1
        self._has_frames = False

    def clear_episode_buffer(self, delete_images: bool = True):
        self.clear_episode_buffer_calls += 1
        self._has_frames = False


def _make_dagger_continuous_context(dataset, shutdown_event: Event, interpolation_multiplier: int = 1):
    from lerobot.utils.constants import ACTION, OBS_STATE

    dataset_cfg = SimpleNamespace(single_task="test task", push_to_hub=False, tags=[], private=False)
    cfg = SimpleNamespace(
        fps=30,
        interpolation_multiplier=interpolation_multiplier,
        dataset=dataset_cfg,
        task="test task",
        play_sounds=False,
        duration=0,
        use_torch_compile=False,
        display_data=False,
        display_compressed_images=False,
    )
    robot = MagicMock()
    robot.send_action = MagicMock()
    teleop = MagicMock()
    teleop.feedback_features = {}
    processors = SimpleNamespace(
        robot_observation_processor=MagicMock(side_effect=lambda obs: obs),
        teleop_action_processor=MagicMock(side_effect=lambda transition: transition[0]),
        robot_action_processor=MagicMock(side_effect=lambda transition: transition[0]),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event),
        hardware=SimpleNamespace(robot_wrapper=robot, teleop=teleop),
        data=SimpleNamespace(
            dataset=dataset,
            dataset_features={
                OBS_STATE: {"dtype": "float32", "shape": (1,), "names": ["x"]},
                ACTION: {"dtype": "float32", "shape": (1,), "names": ["x"]},
            },
        ),
        processors=processors,
    )
    return ctx, robot


def _make_dagger_continuous_strategy(monkeypatch, dataset, shutdown_event, *, interpolation_multiplier=1):
    import lerobot.rollout.strategies.dagger as dagger_module
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy

    monkeypatch.setattr(dagger_module, "VideoEncodingManager", lambda _dataset: contextlib.nullcontext())
    monkeypatch.setattr(dagger_module, "precise_sleep", lambda _sleep_t: None)
    monkeypatch.setattr(dagger_module, "log_say", lambda *args, **kwargs: None)
    monkeypatch.setattr(dagger_module, "send_next_action", MagicMock(return_value={"x": 1.0}))

    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True))
    strategy._episode_duration_s = 999.0
    strategy._engine = MagicMock()
    strategy._interpolator = MagicMock()
    strategy._interpolator.get_control_interval.return_value = 1.0
    strategy._interpolator.needs_new_action.return_value = True
    ctx, robot = _make_dagger_continuous_context(dataset, shutdown_event, interpolation_multiplier)
    return strategy, ctx, robot


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_middle_mouse_click_requests_correction(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_module
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    middle_button = object()

    class FakeMouseListener:
        instances = []

        def __init__(self, on_click):
            self.on_click = on_click
            self.started = False
            self.stopped = False
            self.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    fake_mouse = SimpleNamespace(
        Button=SimpleNamespace(middle=middle_button),
        Listener=FakeMouseListener,
    )
    events = DAggerEvents()
    events.request_transition("pause_resume")
    assert events.consume_transition() == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    monkeypatch.setattr(dagger_module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(dagger_module, "mouse", fake_mouse)
    monkeypatch.setattr(dagger_module, "is_headless", lambda: False)

    listener = dagger_module._init_dagger_mouse(events)
    listener.on_click(0, 0, middle_button, True)

    assert listener.started
    assert events.consume_transition() == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)


def test_dagger_autonomous_to_paused_smooth_handover_preserves_prefixed_keys(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_module
    from lerobot.rollout.strategies import DAggerPhase, DAggerStrategy

    handovers = []
    teleop = SimpleNamespace(feedback_features={"left_motor.pos": float, "right_motor.pos": float})
    teleop.enable_torque = MagicMock()
    teleop.disable_torque = MagicMock()
    robot = MagicMock()
    ctx = SimpleNamespace(hardware=SimpleNamespace(teleop=teleop, robot_wrapper=robot))
    prev_action = {"left_motor.pos": 1.0, "right_motor.pos": 2.0}

    monkeypatch.setattr(
        dagger_module,
        "_teleop_smooth_move_to",
        lambda teleop_arg, target: handovers.append((teleop_arg, target.copy())),
    )

    DAggerStrategy._apply_transition(
        DAggerPhase.AUTONOMOUS,
        DAggerPhase.PAUSED,
        MagicMock(),
        MagicMock(),
        ctx,
        prev_action,
    )

    assert handovers == [(teleop, prev_action)]
    robot.send_action.assert_not_called()


def test_dagger_teleop_smooth_move_to_preserves_bimanual_feedback(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_module

    feedback_calls = []
    teleop = SimpleNamespace(
        get_action=MagicMock(return_value={"left_motor.pos": 0.0, "right_motor.pos": 10.0}),
        send_feedback=MagicMock(side_effect=lambda feedback: feedback_calls.append(feedback.copy())),
        enable_torque=MagicMock(),
    )

    monkeypatch.setattr(dagger_module.time, "sleep", lambda _seconds: None)

    dagger_module._teleop_smooth_move_to(
        teleop,
        {"left_motor.pos": 4.0, "right_motor.pos": 14.0},
        duration_s=0.0,
    )

    teleop.enable_torque.assert_called_once()
    assert feedback_calls[-1] == {"left_motor.pos": 4.0, "right_motor.pos": 14.0}


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.next_episode_requested.set()
    events.rerecord_requested.set()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert events.next_episode_requested.is_set()
    assert events.rerecord_requested.is_set()


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.next_episode_requested.set()
    events.rerecord_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()
    assert not events.next_episode_requested.is_set()
    assert not events.rerecord_requested.is_set()


def test_dagger_continuous_right_arrow_saves_and_resets_record_counter(monkeypatch):
    dataset = _FakeContinuousDataset()
    shutdown_event = Event()
    strategy, ctx, robot = _make_dagger_continuous_strategy(
        monkeypatch, dataset, shutdown_event, interpolation_multiplier=2
    )
    observations = 0

    def get_observation():
        nonlocal observations
        observations += 1
        if observations == 1:
            strategy._events.next_episode_requested.set()
        else:
            shutdown_event.set()
        return {"x": 0.0}

    robot.get_observation.side_effect = get_observation

    strategy._run_continuous(ctx)

    assert dataset.add_frame_calls == 2
    assert dataset.save_episode_calls == 2
    assert dataset.num_episodes == 2


def test_dagger_continuous_left_arrow_discards_in_progress_episode(monkeypatch):
    from lerobot.rollout.strategies import DAggerPhase

    dataset = _FakeContinuousDataset(has_frames=True)
    shutdown_event = Event()
    strategy, ctx, robot = _make_dagger_continuous_strategy(monkeypatch, dataset, shutdown_event)

    def request_rerecord():
        strategy._events.phase = DAggerPhase.PAUSED
        strategy._events.rerecord_requested.set()

    strategy._engine.resume.side_effect = request_rerecord
    robot.get_observation.side_effect = lambda: shutdown_event.set() or {"x": 0.0}

    strategy._run_continuous(ctx)

    assert dataset.clear_episode_buffer_calls == 1
    assert dataset.save_episode_calls == 0
    assert dataset.num_episodes == 0


def test_dagger_continuous_correction_to_paused_saves_episode(monkeypatch):
    from lerobot.rollout.strategies import DAggerPhase

    dataset = _FakeContinuousDataset(has_frames=True)
    shutdown_event = Event()
    strategy, ctx, robot = _make_dagger_continuous_strategy(monkeypatch, dataset, shutdown_event)

    def stop_correction():
        strategy._events.phase = DAggerPhase.CORRECTING
        strategy._events.request_transition("correction")

    strategy._engine.resume.side_effect = stop_correction
    robot.get_observation.side_effect = lambda: shutdown_event.set() or {"x": 0.0}

    strategy._run_continuous(ctx)

    assert dataset.save_episode_calls == 1
    assert dataset.num_episodes == 1


def test_dagger_continuous_next_episode_skips_empty_buffer(monkeypatch):
    from lerobot.rollout.strategies import DAggerPhase

    dataset = _FakeContinuousDataset(has_frames=False)
    shutdown_event = Event()
    strategy, ctx, robot = _make_dagger_continuous_strategy(monkeypatch, dataset, shutdown_event)

    def request_next_episode():
        strategy._events.phase = DAggerPhase.PAUSED
        strategy._events.next_episode_requested.set()

    strategy._engine.resume.side_effect = request_next_episode
    robot.get_observation.side_effect = lambda: shutdown_event.set() or {"x": 0.0}

    strategy._run_continuous(ctx)

    assert dataset.save_episode_calls == 0
    assert dataset.num_episodes == 0


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}


def test_dagger_events_clear_episode_requests():
    from lerobot.rollout.strategies import DAggerEvents

    events = DAggerEvents()
    events.upload_requested.set()
    events.save_episode_requested.set()
    events.discard_episode_requested.set()
    events.clear_pending_controls()
    assert not events.upload_requested.is_set()
    assert not events.save_episode_requested.is_set()
    assert not events.discard_episode_requested.is_set()


def _make_dagger_model_test_context(reset_time_s=0.0):
    dataset_cfg = SimpleNamespace(
        reset_time_s=reset_time_s,
        single_task="test task",
        tags=[],
        private=False,
        push_to_hub=False,
    )
    cfg = SimpleNamespace(
        fps=10.0,
        interpolation_multiplier=1,
        dataset=dataset_cfg,
        task="test task",
        play_sounds=False,
        display_data=False,
        display_compressed_images=False,
        use_torch_compile=False,
        duration=0.0,
    )
    robot = MagicMock()
    robot.get_observation.return_value = {"motor_1.pos": 0.0}
    teleop = MagicMock()
    teleop.feedback_features = {}
    teleop.get_action.return_value = {"motor_1.pos": 1.0}
    dataset = MagicMock()
    dataset.add_frame = MagicMock()
    processors = SimpleNamespace(
        robot_observation_processor=MagicMock(side_effect=lambda obs: obs),
        teleop_action_processor=MagicMock(side_effect=lambda args: args[0]),
        robot_action_processor=MagicMock(side_effect=lambda args: args[0]),
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=Event()),
        hardware=SimpleNamespace(
            robot_wrapper=robot,
            teleop=teleop,
            initial_position={"motor_1.pos": 0.0},
        ),
        processors=processors,
        data=SimpleNamespace(dataset=dataset, dataset_features={}, ordered_action_keys=[]),
    )
    return ctx, dataset


def test_dagger_continuous_episode_save_and_discard_guards(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_mod
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerStrategy

    monkeypatch.setattr(dagger_mod, "log_say", MagicMock())
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True, model_test_mode=True))
    dataset = MagicMock()
    dataset.num_episodes = 0
    dataset.has_pending_frames.return_value = True

    def save_episode():
        dataset.num_episodes += 1
        dataset.has_pending_frames.return_value = False

    dataset.save_episode.side_effect = save_episode
    assert (
        strategy._save_continuous_episode_if_pending(
            dataset, elapsed=1.0, play_sounds=False
        )
        is True
    )
    dataset.save_episode.assert_called_once()
    assert strategy._needs_push.is_set()

    assert (
        strategy._save_continuous_episode_if_pending(
            dataset, elapsed=1.0, play_sounds=False
        )
        is False
    )
    assert dataset.save_episode.call_count == 1

    dataset.has_pending_frames.return_value = True
    assert strategy._discard_continuous_episode_if_pending(dataset) is True
    dataset.clear_episode_buffer.assert_called_once()


def test_dagger_model_test_episode_reset_zero_reset_time(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_mod
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase, DAggerStrategy

    ctx, _ = _make_dagger_model_test_context(reset_time_s=0.0)
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True, model_test_mode=True))
    strategy._cached_obs_processed = {"stale": True}
    strategy._warmup_flushed = True
    engine = MagicMock()
    interpolator = MagicMock()
    events = DAggerEvents()
    events.phase = DAggerPhase.PAUSED
    events.upload_requested.set()
    events.save_episode_requested.set()

    return_mock = MagicMock()
    monkeypatch.setattr(DAggerStrategy, "_return_to_initial_position", return_mock)
    monkeypatch.setattr(dagger_mod, "log_say", MagicMock())

    strategy._run_model_test_episode_reset(ctx, engine, interpolator, events, control_interval=0.1)

    engine.pause.assert_called_once()
    engine.resume.assert_called_once()
    assert engine.reset.call_count == 2
    assert interpolator.reset.call_count == 2
    return_mock.assert_called_once_with(ctx.hardware)
    assert strategy._cached_obs_processed is None
    assert strategy._warmup_flushed is False
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()
    assert not events.save_episode_requested.is_set()


def test_dagger_model_test_reset_window_uses_teleop_without_recording(monkeypatch):
    import lerobot.rollout.strategies.dagger as dagger_mod
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerEvents, DAggerStrategy

    ctx, dataset = _make_dagger_model_test_context(reset_time_s=60.0)
    strategy = DAggerStrategy(DAggerStrategyConfig(record_autonomous=True, model_test_mode=True))
    events = DAggerEvents()

    def stop_after_first_action(_action):
        ctx.runtime.shutdown_event.set()

    ctx.hardware.robot_wrapper.send_action.side_effect = stop_after_first_action
    monkeypatch.setattr(dagger_mod, "precise_sleep", lambda _seconds: None)

    strategy._run_model_test_reset_window(ctx, events, control_interval=0.1)

    ctx.hardware.robot_wrapper.get_observation.assert_called()
    ctx.hardware.teleop.get_action.assert_called()
    ctx.hardware.robot_wrapper.send_action.assert_called_once_with({"motor_1.pos": 1.0})
    dataset.add_frame.assert_not_called()
