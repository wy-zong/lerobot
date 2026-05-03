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

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig


@dataclass
class _RobotCLIConfig:
    robot: RobotConfig


class _FakeSOFollower:
    instances = []

    def __init__(self, config):
        self.config = config
        self.cameras = {key: MagicMock(is_connected=True) for key in config.cameras}
        self.is_connected = True
        _FakeSOFollower.instances.append(self)

    @property
    def _motors_ft(self):
        return {"shoulder_pan.pos": float}

    @property
    def _cameras_ft(self):
        return {key: (cfg.height, cfg.width, 3) for key, cfg in self.config.cameras.items()}

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        obs = {"shoulder_pan.pos": 1.0}
        obs.update({key: f"{key}_image" for key in self.config.cameras})
        return obs

    def send_action(self, action):
        return action

    def calibrate(self):
        pass

    def configure(self):
        pass

    def setup_motors(self):
        pass


def _arm_config(cameras=None):
    return SOFollowerConfig(port="/dev/null", cameras=cameras or {})


def _opencv_camera(index):
    return OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30)


def _make_top_level_camera_mocks(*camera_names):
    return {
        name: MagicMock(is_connected=True, read_latest=MagicMock(return_value=f"{name}_image"))
        for name in camera_names
    }


def test_bi_so_follower_cli_parses_top_level_cameras():
    cameras = {
        "camera1": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
        },
    }

    config = draccus.parse(
        config_class=_RobotCLIConfig,
        args=[
            "--robot.type=bi_so_follower",
            "--robot.left_arm_config.port=/dev/ttyUSB0",
            "--robot.right_arm_config.port=/dev/ttyUSB1",
            f"--robot.cameras={json.dumps(cameras)}",
        ],
    )

    assert isinstance(config.robot, BiSOFollowerConfig)
    assert isinstance(config.robot.cameras["camera1"], OpenCVCameraConfig)


def test_bi_so_follower_top_level_cameras_use_unprefixed_features_and_observations():
    _FakeSOFollower.instances = []
    camera_mocks = _make_top_level_camera_mocks("camera1", "camera3")
    config = BiSOFollowerConfig(
        left_arm_config=_arm_config(cameras={"legacy_left": _opencv_camera(10)}),
        right_arm_config=_arm_config(cameras={"legacy_right": _opencv_camera(11)}),
        cameras={"camera1": _opencv_camera(0), "camera3": _opencv_camera(1)},
    )

    with (
        patch("lerobot.robots.bi_so_follower.bi_so_follower.SOFollower", _FakeSOFollower),
        patch(
            "lerobot.robots.bi_so_follower.bi_so_follower.make_cameras_from_configs",
            return_value=camera_mocks,
        ),
    ):
        robot = BiSOFollower(config)

    assert _FakeSOFollower.instances[0].config.cameras == {}
    assert _FakeSOFollower.instances[1].config.cameras == {}
    assert "camera1" in robot.observation_features
    assert "camera3" in robot.observation_features
    assert "left_legacy_left" not in robot.observation_features
    assert "right_legacy_right" not in robot.observation_features

    obs = robot.get_observation()

    assert obs["camera1"] == "camera1_image"
    assert obs["camera3"] == "camera3_image"
    assert "left_camera1" not in obs
    assert "right_camera3" not in obs


def test_bi_so_follower_nested_cameras_keep_prefixed_fallback_behavior():
    _FakeSOFollower.instances = []
    config = BiSOFollowerConfig(
        left_arm_config=_arm_config(cameras={"camera1": _opencv_camera(0)}),
        right_arm_config=_arm_config(cameras={"camera2": _opencv_camera(1)}),
    )

    with patch("lerobot.robots.bi_so_follower.bi_so_follower.SOFollower", _FakeSOFollower):
        robot = BiSOFollower(config)

    assert _FakeSOFollower.instances[0].config.cameras == config.left_arm_config.cameras
    assert _FakeSOFollower.instances[1].config.cameras == config.right_arm_config.cameras
    assert "left_camera1" in robot.observation_features
    assert "right_camera2" in robot.observation_features
    assert "camera1" not in robot.observation_features
    assert "camera2" not in robot.observation_features

    obs = robot.get_observation()

    assert obs["left_camera1"] == "camera1_image"
    assert obs["right_camera2"] == "camera2_image"
    assert "camera1" not in obs
    assert "camera2" not in obs
