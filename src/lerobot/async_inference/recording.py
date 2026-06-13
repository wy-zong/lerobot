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

"""Execution-host recording helpers for async inference.

The async policy server intentionally remains stateless with respect to rollout
recording.  These helpers live on the robot client side and mirror the dataset
and interaction semantics of ``lerobot-rollout`` using observations and actions
available locally at execution time.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.rollout.configs import (
    BaseStrategyConfig,
    DAggerKeyboardConfig,
    DAggerPedalConfig,
    DAggerStrategyConfig,
    HighlightStrategyConfig,
    SentryStrategyConfig,
)
from lerobot.rollout.ring_buffer import RolloutRingBuffer
from lerobot.rollout.strategies import DAggerEvents, DAggerPhase, estimate_max_episode_seconds
from lerobot.rollout.strategies.core import safe_push_to_hub
from lerobot.rollout.strategies.dagger import (
    _init_dagger_mouse,
    _teleop_smooth_move_to,
    _teleop_supports_feedback,
)
from lerobot.teleoperators import Teleoperator, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.pedal import start_pedal_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_rerun_data

if TYPE_CHECKING:
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)


def remote_dataset_features(robot: Robot, dataset_cfg: DatasetRecordConfig, *, dagger: bool = False) -> dict:
    """Build dataset features from the execution-side robot hardware schema."""
    observation_features = hw_to_dataset_features(
        robot.observation_features,
        OBS_STR,
        use_video=dataset_cfg.video,
    )
    action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=dataset_cfg.video)
    features = combine_feature_dicts(observation_features, action_features)
    if dagger:
        features["intervention"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    return features


def _is_headless_linux() -> bool:
    return ("DISPLAY" not in os.environ) and ("linux" in sys.platform)


def _dataset_has_pending_frames(dataset) -> bool:
    has_pending = getattr(dataset, "has_pending_frames", None)
    if callable(has_pending):
        return bool(has_pending())
    return True


class RemoteRolloutRecorder:
    """Local dataset writer for rollout-style async inference recording."""

    def __init__(self, cfg: RobotClientConfig, robot: Robot, log: logging.Logger) -> None:
        self.cfg = cfg
        self.robot = robot
        self.logger = log
        self.strategy = cfg.strategy
        self.enabled = not isinstance(self.strategy, BaseStrategyConfig)

        self.dataset = None
        self.features: dict[str, dict] = {}
        self._video_manager: VideoEncodingManager | None = None
        self._started = False
        self._closed = False

        self._episode_lock = Lock()
        self._push_executor: ThreadPoolExecutor | None = None
        self._pending_push: Future | None = None
        self._needs_push = Event()
        self._episode_start = time.perf_counter()
        self._episodes_since_push = 0
        self.episode_duration_s = 0.0

        self._ring: RolloutRingBuffer | None = None
        self._highlight_listener = None
        self._save_requested = Event()
        self._recording_live = Event()
        self._push_requested = Event()

        if not self.enabled:
            return

        dataset_cfg = self._dataset_cfg
        self.features = remote_dataset_features(
            robot,
            dataset_cfg,
            dagger=isinstance(self.strategy, DAggerStrategyConfig),
        )
        self.dataset = self._create_or_resume_dataset(dataset_cfg)
        self._push_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"async-{self.strategy.type}-push",
        )

        if isinstance(self.strategy, (SentryStrategyConfig, DAggerStrategyConfig)):
            target_mb = self.strategy.target_video_file_size_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB
            self.episode_duration_s = estimate_max_episode_seconds(
                self.features,
                cfg.fps,
                target_size_mb=target_mb,
            )

        if isinstance(self.strategy, HighlightStrategyConfig):
            self._ring = RolloutRingBuffer(
                max_seconds=self.strategy.ring_buffer_seconds,
                max_memory_mb=self.strategy.ring_buffer_max_memory_mb,
                fps=cfg.fps,
            )

        self.logger.info("Remote rollout recorder ready (strategy=%s)", self.strategy.type)

    @property
    def _dataset_cfg(self) -> DatasetRecordConfig:
        if self.cfg.dataset is None:
            raise ValueError(f"{self.strategy.type} strategy requires dataset configuration")
        return self.cfg.dataset

    @property
    def task(self) -> str:
        dataset_cfg = self.cfg.dataset
        return dataset_cfg.single_task if dataset_cfg and dataset_cfg.single_task else self.cfg.task

    @property
    def episode_elapsed(self) -> float:
        return time.perf_counter() - self._episode_start

    def reset_episode_timer(self) -> None:
        self._episode_start = time.perf_counter()

    def start(self, shutdown_event: Event) -> None:
        if not self.enabled or self._started:
            return
        self._video_manager = VideoEncodingManager(self.dataset)
        self._video_manager.__enter__()
        self._started = True
        self.reset_episode_timer()
        if isinstance(self.strategy, HighlightStrategyConfig):
            self._setup_highlight_keyboard(shutdown_event)

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True

        with contextlib.suppress(Exception):
            if self._recording_live.is_set() or (
                isinstance(self.strategy, (SentryStrategyConfig, DAggerStrategyConfig))
                and _dataset_has_pending_frames(self.dataset)
            ):
                self.save_episode_if_pending(require_pending=True)

        if self._highlight_listener is not None:
            self._highlight_listener.stop()
            self._highlight_listener = None

        if self._push_executor is not None:
            self._push_executor.shutdown(wait=True)
            self._push_executor = None

        if self._video_manager is not None:
            try:
                self._video_manager.__exit__(None, None, None)
            finally:
                self._video_manager = None
        elif self.dataset is not None:
            self.dataset.finalize()

        dataset_cfg = self.cfg.dataset
        if (
            self.dataset is not None
            and dataset_cfg is not None
            and dataset_cfg.push_to_hub
            and self._needs_push.is_set()
        ):
            self.logger.info("Pushing final async rollout dataset to hub...")
            if safe_push_to_hub(self.dataset, tags=dataset_cfg.tags, private=dataset_cfg.private):
                self._needs_push.clear()
                log_say("Dataset uploaded to hub", self.cfg.play_sounds)

    def build_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        *,
        intervention: bool | None = None,
    ) -> dict[str, Any]:
        obs_frame = build_dataset_frame(self.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.features, action, prefix=ACTION)
        frame = {**obs_frame, **action_frame, "task": self.task}
        if intervention is not None:
            frame["intervention"] = np.array([intervention], dtype=bool)
        return frame

    def add_frame(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        *,
        intervention: bool | None = None,
    ) -> None:
        if not self.enabled or self.dataset is None or self._closed:
            return
        self.dataset.add_frame(self.build_frame(observation, action, intervention=intervention))

    def record_sentry_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self._closed:
            return
        self.add_frame(observation, action)
        self.maybe_rotate_episode(upload_every_n_episodes=self.strategy.upload_every_n_episodes)

    def record_highlight_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self.dataset is None or self._ring is None or self._closed:
            return

        frame = self.build_frame(observation, action)

        if self._save_requested.is_set():
            self._save_requested.clear()
            if not self._recording_live.is_set():
                self.logger.info("Flushing highlight ring buffer (%d frames)", len(self._ring))
                for buffered_frame in self._ring.drain():
                    self.dataset.add_frame(buffered_frame)
                self._recording_live.set()
            else:
                self.dataset.add_frame(frame)
                self.save_episode_if_pending(require_pending=False)
                self._recording_live.clear()
                return

        if self._push_requested.is_set():
            self._push_requested.clear()
            self.logger.info("Highlight push requested by user")
            self.background_push()

        if self._recording_live.is_set():
            self.dataset.add_frame(frame)
        else:
            self._ring.append(frame)

    def save_episode_if_pending(
        self,
        *,
        require_pending: bool = True,
        upload_every_n_episodes: int | None = None,
    ) -> bool:
        if self.dataset is None:
            return False
        if require_pending and not _dataset_has_pending_frames(self.dataset):
            return False
        with self._episode_lock:
            self.dataset.save_episode()
        self._needs_push.set()
        self._episodes_since_push += 1
        self.logger.info("Episode saved (total: %d)", self.dataset.num_episodes)
        log_say(f"Episode {self.dataset.num_episodes} saved", self.cfg.play_sounds)

        if upload_every_n_episodes is not None and self._episodes_since_push >= upload_every_n_episodes:
            self.background_push()
            self._episodes_since_push = 0
        return True

    def discard_episode_if_pending(self) -> bool:
        if self.dataset is None or not _dataset_has_pending_frames(self.dataset):
            return False
        clear_buffer = getattr(self.dataset, "clear_episode_buffer", None)
        if callable(clear_buffer):
            clear_buffer()
            self.reset_episode_timer()
            self.logger.info("Pending episode buffer discarded")
            return True
        return False

    def maybe_rotate_episode(self, *, upload_every_n_episodes: int) -> bool:
        if self.episode_duration_s <= 0:
            return False
        if self.episode_elapsed < self.episode_duration_s:
            return False
        saved = self.save_episode_if_pending(
            require_pending=True,
            upload_every_n_episodes=upload_every_n_episodes,
        )
        self.reset_episode_timer()
        return saved

    def background_push(self) -> None:
        if self._push_executor is None or self.dataset is None:
            return

        dataset_cfg = self.cfg.dataset
        if self._pending_push is not None and not self._pending_push.done():
            self.logger.info("Previous async rollout push still in progress; queueing next")

        def _push() -> None:
            try:
                with self._episode_lock:
                    if safe_push_to_hub(
                        self.dataset,
                        tags=dataset_cfg.tags if dataset_cfg else None,
                        private=dataset_cfg.private if dataset_cfg else False,
                    ):
                        self._needs_push.clear()
                        self.logger.info("Background async rollout push complete")
            except Exception as e:
                self.logger.error("Background async rollout push failed: %s", e)

        self._pending_push = self._push_executor.submit(_push)
        self.logger.info("Background async rollout push task submitted")

    def _create_or_resume_dataset(self, dataset_cfg: DatasetRecordConfig):
        camera_count = len(getattr(self.robot, "cameras", {}) or {})
        image_writer_threads = dataset_cfg.num_image_writer_threads_per_camera * camera_count

        if self.cfg.resume:
            return LeRobotDataset.resume(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                batch_encoding_size=dataset_cfg.video_encoding_batch_size,
                camera_encoder=dataset_cfg.camera_encoder,
                streaming_encoding=dataset_cfg.streaming_encoding,
                encoder_queue_maxsize=dataset_cfg.encoder_queue_maxsize,
                encoder_threads=dataset_cfg.encoder_threads,
                image_writer_processes=dataset_cfg.num_image_writer_processes,
                image_writer_threads=image_writer_threads,
            )

        repo_name = dataset_cfg.repo_id.split("/", 1)[-1]
        if not repo_name.startswith("rollout_"):
            raise ValueError(
                "Dataset names for rollout must start with 'rollout_'. "
                "Use --dataset.repo_id=<user>/rollout_<name> for policy deployment datasets."
            )

        dataset_cfg.stamp_repo_id()
        target_video_mb = getattr(self.strategy, "target_video_file_size_mb", None)
        return LeRobotDataset.create(
            dataset_cfg.repo_id,
            dataset_cfg.fps,
            root=dataset_cfg.root,
            robot_type=self.robot.name,
            features=self.features,
            use_videos=dataset_cfg.video,
            image_writer_processes=dataset_cfg.num_image_writer_processes,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=dataset_cfg.video_encoding_batch_size,
            camera_encoder=dataset_cfg.camera_encoder,
            streaming_encoding=dataset_cfg.streaming_encoding,
            encoder_queue_maxsize=dataset_cfg.encoder_queue_maxsize,
            encoder_threads=dataset_cfg.encoder_threads,
            video_files_size_in_mb=target_video_mb,
        )

    def _setup_highlight_keyboard(self, shutdown_event: Event) -> None:
        if _is_headless_linux():
            self.logger.warning("Headless environment; highlight keyboard controls are unavailable")
            return

        try:
            from pynput import keyboard
        except Exception as e:
            self.logger.warning("Could not start highlight keyboard listener: %s", e)
            return

        strategy = self.strategy
        assert isinstance(strategy, HighlightStrategyConfig)

        def on_press(key):
            with contextlib.suppress(Exception):
                if hasattr(key, "char") and key.char == strategy.save_key:
                    self._save_requested.set()
                elif hasattr(key, "char") and key.char == strategy.push_key:
                    self._push_requested.set()
                elif key == keyboard.Key.esc:
                    self._save_requested.clear()
                    shutdown_event.set()

        self._highlight_listener = keyboard.Listener(on_press=on_press)
        self._highlight_listener.start()
        self.logger.info(
            "Highlight keyboard listener started (save='%s', push='%s', ESC=stop)",
            strategy.save_key,
            strategy.push_key,
        )


class RemoteDAggerController:
    """DAgger phase/input controller for the async robot client."""

    def __init__(
        self,
        cfg: RobotClientConfig,
        robot: Robot,
        recorder: RemoteRolloutRecorder,
        log: logging.Logger,
        shutdown_event: Event,
        clear_remote_policy_state: Callable[[], None],
        return_to_initial_position: Callable[[], None],
    ) -> None:
        self.cfg = cfg
        self.robot = robot
        self.strategy = cfg.strategy
        if not isinstance(self.strategy, DAggerStrategyConfig):
            raise TypeError("RemoteDAggerController requires DAggerStrategyConfig")
        self.recorder = recorder
        self.logger = log
        self.shutdown_event = shutdown_event
        self.clear_remote_policy_state = clear_remote_policy_state
        self.return_to_initial_position = return_to_initial_position

        self.events = DAggerEvents()
        self.teleop: Teleoperator | None = None
        self._listener = None
        self._mouse_listener = None
        self._pedal_thread = None
        self.last_action: dict[str, Any] | None = None
        self.recorded_corrections = 0

    @property
    def phase(self) -> DAggerPhase:
        return self.events.phase

    @property
    def is_autonomous(self) -> bool:
        return self.phase == DAggerPhase.AUTONOMOUS

    def start(self) -> None:
        self.teleop = make_teleoperator_from_config(self.cfg.teleop)
        self.teleop.connect()
        if self.strategy.input_device == "keyboard":
            self._listener = self._init_keyboard(self.events, self.strategy.keyboard)
        else:
            self._pedal_thread = self._init_pedal(self.events, self.strategy.pedal)
        self._mouse_listener = _init_dagger_mouse(self.events)
        self.logger.info("Remote DAgger controller started (input=%s)", self.strategy.input_device)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self.teleop is not None and self.teleop.is_connected:
            self.teleop.disconnect()

    def consume_controls(self) -> None:
        if self.events.stop_recording.is_set():
            self.shutdown_event.set()
            return

        transition = self.events.consume_transition()
        if transition is not None:
            old_phase, new_phase = transition
            self._apply_transition(old_phase, new_phase)
            if old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
                self._handle_correction_finished()

        if self.events.upload_requested.is_set():
            self.events.upload_requested.clear()
            if self.phase == DAggerPhase.CORRECTING:
                self.logger.info("Skipping upload while correction is in progress")
            else:
                self.recorder.background_push()

        if self.events.save_episode_requested.is_set():
            self.events.save_episode_requested.clear()
            target_reached = self._save_continuous_episode()
            if target_reached:
                return
            if self.strategy.model_test_mode and self.strategy.record_autonomous:
                self._run_model_test_episode_reset()

        if self.events.discard_episode_requested.is_set():
            self.events.discard_episode_requested.clear()
            self.recorder.discard_episode_if_pending()
            if self.strategy.model_test_mode and self.strategy.record_autonomous:
                self._run_model_test_episode_reset()

        if self.strategy.record_autonomous and self.phase != DAggerPhase.CORRECTING:
            rotated = self.recorder.maybe_rotate_episode(
                upload_every_n_episodes=self.strategy.upload_every_n_episodes
            )
            if rotated:
                if self._stop_if_target_reached():
                    return
                if self.strategy.model_test_mode:
                    self._run_model_test_episode_reset()

    def hold_or_correct(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.phase == DAggerPhase.PAUSED:
            if self.last_action is not None:
                self.robot.send_action(self.last_action)
            return None, self.last_action

        if self.phase != DAggerPhase.CORRECTING:
            return None, None

        if self.teleop is None:
            raise RuntimeError("DAgger correction requested before teleoperator was started")

        observation = self.robot.get_observation()
        teleop_action = self.teleop.get_action()
        performed_action = self.robot.send_action(teleop_action)
        action_for_recording = performed_action or teleop_action
        self.last_action = action_for_recording
        self._log_telemetry(observation, action_for_recording)
        self.recorder.add_frame(observation, action_for_recording, intervention=True)
        return observation, action_for_recording

    def on_policy_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        self.last_action = action
        if self.strategy.record_autonomous:
            self.recorder.add_frame(observation, action, intervention=False)

    def _log_telemetry(self, observation: dict[str, Any] | None, action: dict[str, Any] | None) -> None:
        if not self.cfg.display_data:
            return
        log_rerun_data(
            observation=observation,
            action=action,
            compress_images=self.cfg.display_compressed_images,
        )

    def _handle_correction_finished(self) -> None:
        if self.strategy.record_autonomous:
            if self.strategy.model_test_mode:
                target_reached = self._save_continuous_episode()
                if target_reached:
                    return
                self._run_model_test_episode_reset()
            return

        if self.recorder.save_episode_if_pending(require_pending=True):
            self.recorded_corrections += 1
            self.logger.info(
                "Correction %d/%d saved",
                self.recorded_corrections,
                self.strategy.num_episodes,
            )
            if self.recorded_corrections >= self.strategy.num_episodes:
                self.events.stop_recording.set()

    def _save_continuous_episode(self) -> bool:
        if self.recorder.save_episode_if_pending(
            require_pending=True,
            upload_every_n_episodes=self.strategy.upload_every_n_episodes,
        ):
            self.recorder.reset_episode_timer()
            return self._stop_if_target_reached()
        return False

    def _stop_if_target_reached(self) -> bool:
        target = self.strategy.num_episodes
        dataset = self.recorder.dataset
        if target is None or dataset is None or dataset.num_episodes < target:
            return False
        self.logger.info(
            "DAgger target episode count reached (%d/%d); stopping recording",
            dataset.num_episodes,
            target,
        )
        self.events.stop_recording.set()
        self.shutdown_event.set()
        return True

    def _apply_transition(self, old_phase: DAggerPhase, new_phase: DAggerPhase) -> None:
        self.logger.info("DAgger phase transition: %s -> %s", old_phase.value, new_phase.value)
        teleop_supports_feedback = self.teleop is not None and _teleop_supports_feedback(self.teleop)

        if old_phase == DAggerPhase.AUTONOMOUS and new_phase == DAggerPhase.PAUSED:
            self.clear_remote_policy_state()
            if teleop_supports_feedback and self.last_action is not None:
                self.logger.info("Smooth handover: moving leader arm to follower position")
                _teleop_smooth_move_to(self.teleop, self.last_action)
        elif old_phase == DAggerPhase.PAUSED and new_phase == DAggerPhase.CORRECTING:
            self.clear_remote_policy_state()
            if teleop_supports_feedback:
                self.teleop.disable_torque()
        elif old_phase == DAggerPhase.CORRECTING and new_phase == DAggerPhase.PAUSED:
            if teleop_supports_feedback:
                self.teleop.enable_torque()
        elif new_phase == DAggerPhase.AUTONOMOUS:
            self.clear_remote_policy_state()
            self.last_action = None
            if teleop_supports_feedback:
                self.teleop.disable_torque()

    def _run_model_test_episode_reset(self) -> None:
        if not self.strategy.record_autonomous:
            return

        self.clear_remote_policy_state()
        self.return_to_initial_position()
        reset_time_s = self.cfg.dataset.reset_time_s if self.cfg.dataset is not None else 0.0
        if reset_time_s > 0 and self.teleop is not None:
            reset_start = time.perf_counter()
            control_interval = 1 / self.cfg.fps
            while (
                time.perf_counter() - reset_start < reset_time_s
                and not self.events.stop_recording.is_set()
                and not self.shutdown_event.is_set()
            ):
                loop_start = time.perf_counter()
                teleop_action = self.teleop.get_action()
                self.robot.send_action(teleop_action)
                dt = time.perf_counter() - loop_start
                if (sleep_t := control_interval - dt) > 0:
                    precise_sleep(sleep_t)

        self.events.clear_pending_controls()
        self.events.phase = DAggerPhase.AUTONOMOUS
        self.clear_remote_policy_state()
        self.recorder.reset_episode_timer()

    def _init_keyboard(self, events: DAggerEvents, cfg: DAggerKeyboardConfig):
        if _is_headless_linux():
            self.logger.warning("Headless environment; DAgger keyboard controls are unavailable")
            return None

        try:
            from pynput import keyboard
        except Exception as e:
            self.logger.warning("Could not start DAgger keyboard listener: %s", e)
            return None

        special_keys = {
            "space": keyboard.Key.space,
            "tab": keyboard.Key.tab,
            "enter": keyboard.Key.enter,
            "left": keyboard.Key.left,
            "right": keyboard.Key.right,
        }

        def resolve_key(key) -> str | None:
            if key == keyboard.Key.esc:
                return "esc"
            for name, pynput_key in special_keys.items():
                if key == pynput_key:
                    return name
            if hasattr(key, "char") and key.char:
                return key.char
            return None

        key_to_event = {
            cfg.pause_resume: "pause_resume",
            cfg.correction: "correction",
        }

        def on_press(key):
            with contextlib.suppress(Exception):
                resolved = resolve_key(key)
                if resolved is None:
                    return
                if resolved == "esc":
                    events.stop_recording.set()
                elif resolved in key_to_event:
                    events.request_transition(key_to_event[resolved])
                elif resolved == cfg.upload:
                    events.upload_requested.set()
                elif resolved == cfg.next_episode:
                    events.save_episode_requested.set()
                elif resolved == cfg.rerecord_episode:
                    events.discard_episode_requested.set()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.logger.info(
            "DAgger keyboard listener started (pause='%s', correction='%s', upload='%s')",
            cfg.pause_resume,
            cfg.correction,
            cfg.upload,
        )
        return listener

    def _init_pedal(self, events: DAggerEvents, cfg: DAggerPedalConfig):
        code_to_event = {
            cfg.pause_resume: "pause_resume",
            cfg.correction: "correction",
        }

        def on_press(code: str) -> None:
            if code in code_to_event:
                events.request_transition(code_to_event[code])
            if code == cfg.upload:
                events.upload_requested.set()

        return start_pedal_listener(on_press, device_path=cfg.device_path)
