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

import pytest
import torch

from lerobot.scripts.lerobot_train_stage import install_stage_forward, validate_stage_dataset


class _FakeStagePolicy:
    def __init__(self):
        self.forward_stage_calls = []

    def forward_stage(self, batch, noise=None, time=None, reduction="mean", flow_loss_weight=10.0):
        self.forward_stage_calls.append(
            {
                "batch": batch,
                "noise": noise,
                "time": time,
                "reduction": reduction,
                "flow_loss_weight": flow_loss_weight,
            }
        )
        return torch.tensor(flow_loss_weight), {"loss": flow_loss_weight}


def test_validate_stage_dataset_accepts_subtask_annotations():
    dataset = SimpleNamespace(
        features={"subtask_index": object()},
        meta=SimpleNamespace(subtasks=["open gripper"]),
    )
    validate_stage_dataset(dataset, SimpleNamespace(subtask_required=True))


def test_validate_stage_dataset_reports_missing_annotations():
    dataset = SimpleNamespace(features={}, meta=SimpleNamespace(subtasks=None))

    with pytest.raises(ValueError, match="subtask_index.*meta/subtasks.parquet"):
        validate_stage_dataset(dataset, SimpleNamespace(subtask_required=True))


def test_validate_stage_dataset_can_be_disabled():
    dataset = SimpleNamespace(features={}, meta=SimpleNamespace(subtasks=None))
    validate_stage_dataset(dataset, SimpleNamespace(subtask_required=False))


def test_install_stage_forward_routes_to_nested_forward_stage():
    base_policy = _FakeStagePolicy()
    wrapper = SimpleNamespace(base_model=SimpleNamespace(model=base_policy))
    cfg = SimpleNamespace(flow_loss_weight=7.0)

    install_stage_forward(wrapper, cfg)
    loss, metrics = wrapper.forward({"sample": True}, reduction="mean")

    torch.testing.assert_close(loss, torch.tensor(7.0))
    assert metrics["loss"] == 7.0
    assert base_policy.forward_stage_calls[0]["flow_loss_weight"] == 7.0
    assert base_policy.forward_stage_calls[0]["batch"] == {"sample": True}
