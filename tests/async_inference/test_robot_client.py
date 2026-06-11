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
        self.finalized = False
        self.pushed = 0
        self.repo_id = "user/rollout_fake"

    @property
    def num_episodes(self):
        return self.saved

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1
        self.frames.clear()

    def has_pending_frames(self):
        return bool(self.frames)

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, **_kwargs):
        self.pushed += 1


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
