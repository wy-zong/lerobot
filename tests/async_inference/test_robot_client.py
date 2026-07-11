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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import logging
import pickle  # nosec
import sys
import threading
import time
from queue import Queue
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


def _make_keyboard_stop_client():
    return SimpleNamespace(logger=_FakeLogger(), shutdown_event=threading.Event())


def _make_fake_pynput_module():
    esc_key = object()

    class FakeListener:
        instances = []

        def __init__(self, on_press):
            self.on_press = on_press
            self.started = False
            self.stopped = False
            self.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

    fake_keyboard = SimpleNamespace(Key=SimpleNamespace(esc=esc_key), Listener=FakeListener)
    fake_pynput = ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard

    return fake_pynput, fake_keyboard, FakeListener


class _FakeDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0
        self.saved_episode_metadata = []
        self.finalized = False
        self.pushed = 0
        self.repo_id = "user/rollout_fake"

    @property
    def num_episodes(self):
        return self.saved

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, episode_metadata=None):
        self.saved += 1
        self.saved_episode_metadata.append(episode_metadata)
        self.frames.clear()

    def has_pending_frames(self):
        return bool(self.frames)

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, **_kwargs):
        self.pushed += 1


class _ActuatedTeleop:
    feedback_features = {"left_motor.pos": float, "right_motor.pos": float}

    def __init__(self):
        self.enable_calls = 0
        self.disable_calls = 0
        self.is_connected = True

    def enable_torque(self):
        self.enable_calls += 1

    def disable_torque(self):
        self.disable_calls += 1


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("inference_mode", ["async", "sync"])
def test_robot_client_config_accepts_supported_inference_modes(inference_mode: str):
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        inference_mode=inference_mode,
    )

    assert config.inference_mode == inference_mode


def test_robot_client_config_rejects_unknown_inference_mode():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="inference_mode"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            server_address="localhost:9999",
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            inference_mode="streaming",
        )


def test_robot_client_config_accepts_intra_chunk_smoothing():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        intra_chunk_smoothing=True,
    )

    assert config.intra_chunk_smoothing is True
    assert config.intra_chunk_smoothing_degree == 3


def test_robot_client_config_rejects_non_cubic_intra_chunk_smoothing_degree():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="degree=3"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            server_address="localhost:9999",
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
            intra_chunk_smoothing=True,
            intra_chunk_smoothing_degree=2,
        )


def test_robot_client_config_validates_rollout_recording_requirements():
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig, SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    base_kwargs = {
        "robot": MockRobotConfig(),
        "server_address": "localhost:9999",
        "policy_type": "test",
        "pretrained_name_or_path": "test",
        "actions_per_chunk": 20,
    }

    RobotClientConfig(**base_kwargs)

    with pytest.raises(ValueError, match="dataset.repo_id"):
        RobotClientConfig(**base_kwargs, strategy=SentryStrategyConfig())

    with pytest.raises(ValueError, match="teleop.type"):
        RobotClientConfig(
            **base_kwargs,
            strategy=DAggerStrategyConfig(),
            dataset=DatasetRecordConfig(repo_id="user/rollout_dagger"),
        )

    with pytest.raises(ValueError, match="dataset fps must match robot client fps"):
        RobotClientConfig(
            **base_kwargs,
            strategy=SentryStrategyConfig(),
            dataset=DatasetRecordConfig(repo_id="user/rollout_sentry", fps=60),
        )

    dataset = DatasetRecordConfig(repo_id="user/rollout_dagger", streaming_encoding=False)
    cfg = RobotClientConfig(
        **base_kwargs,
        strategy=DAggerStrategyConfig(record_autonomous=True),
        dataset=dataset,
        teleop=MockTeleopConfig(),
    )

    assert cfg.dataset.streaming_encoding is True


def test_remote_recorder_builds_features_and_frames(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    fake_dataset = _FakeDataset()
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)

    robot = MockRobot(MockRobotConfig(n_motors=3, random_values=False, static_values=[1, 2, 3]))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        task="pick",
        strategy=DAggerStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_features", video=False),
        teleop=MockTeleopConfig(),
    )

    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    frame = recorder.build_frame(
        {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
        {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0},
        intervention=True,
    )

    assert "intervention" in recorder.features
    assert frame["task"] == "pick"
    np.testing.assert_array_equal(frame["observation.state"], np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(frame["action"], np.array([4, 5, 6], dtype=np.float32))
    np.testing.assert_array_equal(frame["intervention"], np.array([True], dtype=bool))


def test_remote_sentry_recorder_rotates_and_queues_push(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    fake_dataset = _FakeDataset()
    push_calls = []
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)
    monkeypatch.setattr(RemoteRolloutRecorder, "background_push", lambda self: push_calls.append(self))

    robot = MockRobot(MockRobotConfig(n_motors=3))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=SentryStrategyConfig(upload_every_n_episodes=1),
        dataset=DatasetRecordConfig(repo_id="user/rollout_sentry", video=False),
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    recorder.episode_duration_s = 0.01
    recorder._episode_start = time.perf_counter() - 1.0

    recorder.record_sentry_action(
        {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
        {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0},
    )

    assert fake_dataset.saved == 1
    assert push_calls == [recorder]


@pytest.mark.parametrize(
    ("pressed_key", "expected_success"),
    [("s", True), ("S", True), ("f", False), ("F", False)],
)
def test_remote_recorder_labels_episode_success_with_single_key(
    monkeypatch, pressed_key, expected_success
):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    fake_pynput, _, fake_listener = _make_fake_pynput_module()
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: _FakeDataset())

    robot = MockRobot(MockRobotConfig(n_motors=3))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=SentryStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_sentry", video=False),
        label_episode_success=True,
        play_sounds=False,
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    recorder._setup_episode_label_keyboard(threading.Event())
    recorder._episode_label_requested.set()

    listener = fake_listener.instances[-1]
    listener.on_press(SimpleNamespace(char="x"))
    assert not recorder._episode_label_ready.is_set()

    listener.on_press(SimpleNamespace(char=pressed_key))

    assert recorder._episode_label_ready.is_set()
    assert recorder._episode_success is expected_success


def test_remote_recorder_persists_episode_success_metadata(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import SentryStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    fake_dataset = _FakeDataset()
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)
    monkeypatch.setattr(RemoteRolloutRecorder, "_request_episode_success_label", lambda _self: False)

    robot = MockRobot(MockRobotConfig(n_motors=3))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=SentryStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_sentry", video=False),
        label_episode_success=True,
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    recorder.add_frame(
        {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
        {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0},
    )

    assert recorder.save_episode_if_pending()
    assert fake_dataset.saved_episode_metadata == [{"episode_success": False}]


def test_highlight_success_key_does_not_request_another_episode(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import HighlightStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    fake_pynput, _, fake_listener = _make_fake_pynput_module()
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: _FakeDataset())

    robot = MockRobot(MockRobotConfig(n_motors=3))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=HighlightStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_highlight", video=False),
        label_episode_success=True,
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    recorder._setup_episode_label_keyboard(threading.Event())
    recorder._episode_label_requested.set()

    fake_listener.instances[-1].on_press(SimpleNamespace(char="s"))

    assert recorder._episode_success is True
    assert not recorder._save_requested.is_set()


def test_remote_highlight_recorder_buffers_saves_and_pushes(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import HighlightStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    fake_dataset = _FakeDataset()
    push_calls = []
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)
    monkeypatch.setattr(RemoteRolloutRecorder, "background_push", lambda self: push_calls.append(self))

    robot = MockRobot(MockRobotConfig(n_motors=3))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=HighlightStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_highlight", video=False),
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    obs = {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0}
    action = {"motor_1.pos": 4.0, "motor_2.pos": 5.0, "motor_3.pos": 6.0}

    recorder.record_highlight_action(obs, action)
    assert len(recorder._ring) == 1
    assert fake_dataset.frames == []

    recorder._save_requested.set()
    recorder.record_highlight_action(obs, action)
    assert len(fake_dataset.frames) == 2

    recorder._push_requested.set()
    recorder.record_highlight_action(obs, action)
    assert push_calls == [recorder]

    recorder._save_requested.set()
    recorder.record_highlight_action(obs, action)
    assert fake_dataset.saved == 1


def test_remote_dagger_corrections_only_records_interventions(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteDAggerController, RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    fake_dataset = _FakeDataset()
    clear_calls = []
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)

    robot = MockRobot(MockRobotConfig(n_motors=3, random_values=False, static_values=[0, 0, 0]))
    robot.connect()
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=DAggerStrategyConfig(num_episodes=1),
        dataset=DatasetRecordConfig(repo_id="user/rollout_dagger", video=False),
        teleop=MockTeleopConfig(random_values=False, static_values=[1, 2, 3]),
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    shutdown_event = threading.Event()
    controller = RemoteDAggerController(
        cfg,
        robot,
        recorder,
        logging.getLogger("test"),
        shutdown_event,
        lambda: clear_calls.append("clear"),
        lambda: None,
    )
    controller.start()

    controller.on_policy_action(
        {"motor_1.pos": 0.0, "motor_2.pos": 0.0, "motor_3.pos": 0.0},
        {"motor_1.pos": 0.0, "motor_2.pos": 0.0, "motor_3.pos": 0.0},
    )
    assert fake_dataset.frames == []

    controller.events.request_transition("pause_resume")
    controller.consume_controls()
    assert controller.phase == DAggerPhase.PAUSED

    controller.events.request_transition("correction")
    controller.consume_controls()
    observation, action = controller.hold_or_correct()
    assert observation is not None
    assert action == {"motor_1.pos": 1, "motor_2.pos": 2, "motor_3.pos": 3}
    np.testing.assert_array_equal(fake_dataset.frames[-1]["intervention"], np.array([True]))

    controller.events.request_transition("correction")
    controller.consume_controls()

    assert controller.phase == DAggerPhase.PAUSED
    assert fake_dataset.saved == 1
    assert controller.events.stop_recording.is_set()
    assert clear_calls

    controller.close()
    robot.disconnect()


def test_remote_dagger_record_autonomous_stops_at_target_episode(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteDAggerController, RemoteRolloutRecorder
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    fake_dataset = _FakeDataset()
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)

    robot = MockRobot(MockRobotConfig(n_motors=3, random_values=False, static_values=[0, 0, 0]))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=DAggerStrategyConfig(num_episodes=1, record_autonomous=True, model_test_mode=True),
        dataset=DatasetRecordConfig(
            repo_id="user/rollout_dagger",
            video=False,
            reset_time_s=0,
        ),
        teleop=MockTeleopConfig(),
    )
    recorder = RemoteRolloutRecorder(cfg, robot, logging.getLogger("test"))
    shutdown_event = threading.Event()
    reset_calls = []
    controller = RemoteDAggerController(
        cfg,
        robot,
        recorder,
        logging.getLogger("test"),
        shutdown_event,
        lambda: None,
        lambda: reset_calls.append("reset"),
    )

    controller.on_policy_action(
        {"motor_1.pos": 0.0, "motor_2.pos": 0.0, "motor_3.pos": 0.0},
        {"motor_1.pos": 1.0, "motor_2.pos": 2.0, "motor_3.pos": 3.0},
    )
    controller.events.save_episode_requested.set()
    controller.consume_controls()

    assert fake_dataset.saved == 1
    assert controller.events.stop_recording.is_set()
    assert shutdown_event.is_set()
    assert reset_calls == []


def test_robot_client_records_autonomous_wait_frames(monkeypatch):
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteRolloutRecorder
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase
    from tests.mocks.mock_robot import MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    fake_dataset = _FakeDataset()
    monkeypatch.setattr(RemoteRolloutRecorder, "_create_or_resume_dataset", lambda *_args: fake_dataset)

    cfg = RobotClientConfig(
        robot=MockRobotConfig(n_motors=3, random_values=False, static_values=[1, 2, 3]),
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=DAggerStrategyConfig(record_autonomous=True),
        dataset=DatasetRecordConfig(repo_id="user/rollout_dagger", video=False),
        teleop=MockTeleopConfig(),
    )
    client = RobotClient(cfg)

    try:
        assert client.dagger_controller is not None
        client.dagger_controller.events.phase = DAggerPhase.AUTONOMOUS
        client.dagger_controller.last_action = {
            "motor_1.pos": 4.0,
            "motor_2.pos": 5.0,
            "motor_3.pos": 6.0,
        }

        observation = client._record_autonomous_wait_frame()

        assert observation == {"motor_1.pos": 1, "motor_2.pos": 2, "motor_3.pos": 3}
        assert len(fake_dataset.frames) == 1
        frame = fake_dataset.frames[-1]
        np.testing.assert_array_equal(frame["observation.state"], np.array([1, 2, 3], dtype=np.float32))
        np.testing.assert_array_equal(frame["action"], np.array([4, 5, 6], dtype=np.float32))
        np.testing.assert_array_equal(frame["intervention"], np.array([False], dtype=bool))
    finally:
        client.stop()


def test_remote_dagger_actuated_handover_and_torque_transitions(monkeypatch):
    import lerobot.async_inference.recording as recording_module
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.recording import RemoteDAggerController
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.rollout import DAggerStrategyConfig
    from lerobot.rollout.strategies import DAggerPhase
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig
    from tests.mocks.mock_teleop import MockTeleopConfig

    clear_calls = []
    handovers = []
    last_action = {"left_motor.pos": 1.0, "right_motor.pos": 2.0}
    teleop = _ActuatedTeleop()

    monkeypatch.setattr(
        recording_module,
        "_teleop_smooth_move_to",
        lambda teleop_arg, target: handovers.append((teleop_arg, target.copy())),
    )

    robot = MockRobot(MockRobotConfig(n_motors=2))
    cfg = RobotClientConfig(
        robot=robot.config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        strategy=DAggerStrategyConfig(),
        dataset=DatasetRecordConfig(repo_id="user/rollout_dagger", video=False),
        teleop=MockTeleopConfig(),
    )
    controller = RemoteDAggerController(
        cfg,
        robot,
        SimpleNamespace(),
        logging.getLogger("test"),
        threading.Event(),
        lambda: clear_calls.append("clear"),
        lambda: None,
    )
    controller.teleop = teleop
    controller.last_action = last_action.copy()

    controller._apply_transition(DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)
    assert clear_calls == ["clear"]
    assert handovers == [(teleop, last_action)]
    assert controller.last_action == last_action

    controller._apply_transition(DAggerPhase.PAUSED, DAggerPhase.CORRECTING)
    assert clear_calls == ["clear", "clear"]
    assert teleop.disable_calls == 1

    controller._apply_transition(DAggerPhase.CORRECTING, DAggerPhase.PAUSED)
    assert teleop.enable_calls == 1

    controller._apply_transition(DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)
    assert clear_calls == ["clear", "clear", "clear"]
    assert controller.last_action is None
    assert teleop.disable_calls == 2


def test_robot_client_display_data_logs_policy_actions(monkeypatch, robot_client):
    import lerobot.async_inference.robot_client as robot_client_module

    logged = []

    def fake_log_rerun_data(**kwargs):
        logged.append(kwargs)

    monkeypatch.setattr(robot_client_module, "log_rerun_data", fake_log_rerun_data)
    robot_client.config.display_data = True
    robot_client.config.display_compressed_images = True

    observation = {"motor_1.pos": 1.0}
    action = {"motor_1.pos": 2.0}
    robot_client._record_policy_action(observation, action)

    assert logged == [
        {
            "observation": observation,
            "action": action,
            "compress_images": True,
        }
    ]


def test_keyboard_stop_listener_sets_shutdown_on_escape(monkeypatch):
    import lerobot.async_inference.robot_client as robot_client_module

    client = _make_keyboard_stop_client()
    fake_pynput, fake_keyboard, fake_listener_cls = _make_fake_pynput_module()
    monkeypatch.setattr(robot_client_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)

    listener = robot_client_module._start_keyboard_stop_listener(client)

    assert listener is fake_listener_cls.instances[0]
    assert listener.started is True
    assert listener.on_press(fake_keyboard.Key.esc) is False
    assert client.shutdown_event.is_set() is True


def test_keyboard_stop_listener_ignores_non_escape(monkeypatch):
    import lerobot.async_inference.robot_client as robot_client_module

    client = _make_keyboard_stop_client()
    fake_pynput, _fake_keyboard, fake_listener_cls = _make_fake_pynput_module()
    monkeypatch.setattr(robot_client_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)

    listener = robot_client_module._start_keyboard_stop_listener(client)

    assert listener is fake_listener_cls.instances[0]
    assert listener.on_press(object()) is None
    assert client.shutdown_event.is_set() is False


def test_keyboard_stop_listener_returns_none_in_headless_linux(monkeypatch):
    import lerobot.async_inference.robot_client as robot_client_module

    client = _make_keyboard_stop_client()
    monkeypatch.setattr(robot_client_module.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)

    listener = robot_client_module._start_keyboard_stop_listener(client)

    assert listener is None
    assert client.shutdown_event.is_set() is False
    assert client.logger.warnings == [
        "Headless environment detected. ESC keyboard shutdown is unavailable."
    ]


def test_keyboard_stop_listener_returns_none_without_pynput(monkeypatch):
    import lerobot.async_inference.robot_client as robot_client_module

    client = _make_keyboard_stop_client()
    fake_pynput = ModuleType("pynput")
    monkeypatch.setattr(robot_client_module.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)
    monkeypatch.delitem(sys.modules, "pynput.keyboard", raising=False)

    listener = robot_client_module._start_keyboard_stop_listener(client)

    assert listener is None
    assert client.shutdown_event.is_set() is False
    assert client.logger.warnings[0].startswith("Could not start ESC keyboard listener:")


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


def test_smooth_timed_actions_preserves_metadata_and_smooths_chunk(robot_client):
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.processor import IntraChunkSmoothingProcessorStep

    t = torch.linspace(-1.0, 1.0, 8)
    alternating = torch.where(torch.arange(t.numel()) % 2 == 0, 1.0, -1.0)
    action_chunk = torch.stack([t + 0.1 * alternating, -0.5 * t - 0.2 * alternating], dim=-1)
    timed_actions = [
        TimedAction(timestamp=100.0 + i, timestep=10 + i, action=action)
        for i, action in enumerate(action_chunk)
    ]
    robot_client._intra_chunk_smoothing_step = IntraChunkSmoothingProcessorStep(enabled=True)

    smoothed_actions = robot_client._smooth_timed_actions(timed_actions)

    expected = robot_client._intra_chunk_smoothing_step.action(action_chunk)
    for i, (smoothed_action, original_action) in enumerate(zip(smoothed_actions, timed_actions, strict=True)):
        assert smoothed_action.get_timestamp() == original_action.get_timestamp()
        assert smoothed_action.get_timestep() == original_action.get_timestep()
        torch.testing.assert_close(smoothed_action.get_action(), expected[i])

    assert not torch.allclose(torch.stack([action.get_action() for action in smoothed_actions]), action_chunk)


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_ready_to_send_observation_sync_requires_empty_queue_and_no_pending_chunk(robot_client):
    robot_client.inference_mode = "sync"
    robot_client.must_go.set()
    robot_client.awaiting_action_chunk.clear()

    robot_client.action_queue = Queue()
    robot_client.action_queue.put(_make_actions(start_ts=time.time(), start_t=0, count=1)[0])
    assert robot_client._ready_to_send_observation() is False

    robot_client.action_queue = Queue()
    assert robot_client._ready_to_send_observation() is True

    robot_client.awaiting_action_chunk.set()
    assert robot_client._ready_to_send_observation() is False

    robot_client.awaiting_action_chunk.clear()
    robot_client.must_go.clear()
    assert robot_client._ready_to_send_observation() is False


def test_sync_control_loop_observation_marks_pending_after_must_go_send(monkeypatch, robot_client):
    robot_client.inference_mode = "sync"
    robot_client.action_queue = Queue()
    robot_client.must_go.set()
    robot_client.awaiting_action_chunk.clear()

    sent_observations = []

    def fake_send_observation(observation):
        sent_observations.append(observation)
        return True

    monkeypatch.setattr(robot_client, "send_observation", fake_send_observation)

    robot_client.control_loop_observation(task="test task")

    assert len(sent_observations) == 1
    assert sent_observations[0].must_go is True
    assert robot_client.awaiting_action_chunk.is_set() is True
    assert robot_client.must_go.is_set() is False


def test_receive_actions_clears_sync_pending_chunk(monkeypatch, robot_client):
    from lerobot.transport import services_pb2

    robot_client.inference_mode = "sync"
    robot_client.awaiting_action_chunk.set()

    actions = _make_actions(start_ts=time.time(), start_t=0, count=2)

    class FakeStub:
        def GetActions(self, request):  # noqa: N802
            return services_pb2.Actions(data=pickle.dumps(actions))

    robot_client.stub = FakeStub()
    original_aggregate = robot_client._aggregate_action_queues

    def aggregate_and_stop(incoming_actions, aggregate_fn=None):
        try:
            return original_aggregate(incoming_actions, aggregate_fn)
        finally:
            robot_client.shutdown_event.set()

    monkeypatch.setattr(robot_client, "_aggregate_action_queues", aggregate_and_stop)

    action_thread = threading.Thread(target=robot_client.receive_actions)
    action_thread.start()
    robot_client.start_barrier.wait(timeout=1)
    action_thread.join(timeout=1)

    assert action_thread.is_alive() is False
    assert robot_client.awaiting_action_chunk.is_set() is False
    assert robot_client.action_queue.qsize() == 2


def test_sync_mode_waits_for_chunk_exhaustion_before_next_observation(monkeypatch, robot_client):
    robot_client.inference_mode = "sync"
    robot_client.action_queue = Queue()
    robot_client.latest_action = -1
    robot_client.must_go.set()
    robot_client.awaiting_action_chunk.clear()

    sent_observations = []

    def fake_send_observation(observation):
        sent_observations.append(observation)
        return True

    monkeypatch.setattr(robot_client, "send_observation", fake_send_observation)

    assert robot_client._ready_to_send_observation() is True
    robot_client.control_loop_observation(task="test task")
    assert len(sent_observations) == 1
    assert sent_observations[-1].get_timestep() == 0
    assert robot_client.awaiting_action_chunk.is_set() is True

    robot_client._aggregate_action_queues(_make_actions(start_ts=time.time(), start_t=0, count=3))
    robot_client.awaiting_action_chunk.clear()
    robot_client.must_go.set()

    assert robot_client._ready_to_send_observation() is False
    robot_client.control_loop_action()
    assert robot_client._ready_to_send_observation() is False
    robot_client.control_loop_action()
    assert robot_client._ready_to_send_observation() is False

    robot_client.control_loop_action()
    assert robot_client._ready_to_send_observation() is True
    robot_client.control_loop_observation(task="test task")
    assert len(sent_observations) == 2
    assert sent_observations[-1].get_timestep() == 3


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )
