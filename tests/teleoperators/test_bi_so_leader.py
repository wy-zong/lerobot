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

from __future__ import annotations

import pytest

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")


class _FakeSOLeader:
    def __init__(self, config):
        self.config = config
        self.sent_feedback = []
        self.enable_calls = 0
        self.disable_calls = 0
        self._is_connected = True
        self._is_calibrated = True

    @property
    def action_features(self):
        return {"shoulder_pan.pos": float, "gripper.pos": float}

    @property
    def feedback_features(self):
        return {"shoulder_pan.pos": float, "gripper.pos": float}

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def is_calibrated(self):
        return self._is_calibrated

    def connect(self, calibrate=True):
        self._is_connected = True

    def calibrate(self):
        self._is_calibrated = True

    def configure(self):
        pass

    def setup_motors(self):
        pass

    def enable_torque(self):
        self.enable_calls += 1

    def disable_torque(self):
        self.disable_calls += 1

    def get_action(self):
        return {"shoulder_pan.pos": 1.0, "gripper.pos": 2.0}

    def send_feedback(self, feedback):
        self.sent_feedback.append(feedback)

    def disconnect(self):
        self._is_connected = False


def test_bi_so_leader_actuated_feedback_interface(monkeypatch, tmp_path):
    import lerobot.teleoperators.bi_so_leader.bi_so_leader as bi_so_leader_module
    from lerobot.teleoperators.bi_so_leader.config_bi_so_leader import BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeaderConfig

    monkeypatch.setattr(bi_so_leader_module, "SOLeader", _FakeSOLeader)

    leader = bi_so_leader_module.BiSOLeader(
        BiSOLeaderConfig(
            id="test",
            calibration_dir=tmp_path,
            left_arm_config=SOLeaderConfig(port="left"),
            right_arm_config=SOLeaderConfig(port="right"),
        )
    )

    assert leader.feedback_features == {
        "left_shoulder_pan.pos": float,
        "left_gripper.pos": float,
        "right_shoulder_pan.pos": float,
        "right_gripper.pos": float,
    }

    leader.send_feedback(
        {
            "left_shoulder_pan.pos": 10.0,
            "left_gripper.pos": 11.0,
            "right_shoulder_pan.pos": 20.0,
            "right_gripper.pos": 21.0,
            "unprefixed.pos": 99.0,
        }
    )
    assert leader.left_arm.sent_feedback == [{"shoulder_pan.pos": 10.0, "gripper.pos": 11.0}]
    assert leader.right_arm.sent_feedback == [{"shoulder_pan.pos": 20.0, "gripper.pos": 21.0}]

    leader.enable_torque()
    leader.disable_torque()
    assert leader.left_arm.enable_calls == 1
    assert leader.right_arm.enable_calls == 1
    assert leader.left_arm.disable_calls == 1
    assert leader.right_arm.disable_calls == 1
