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

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset

from lerobot.datasets.intervention import (
    apply_intervention_only_data,
    compute_intervention_only_data,
    selected_episode_boundaries,
    validate_intervention_only_config,
)
from lerobot.datasets.io_utils import load_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.streaming_dataset import Backtrackable, StreamingLeRobotDataset


def _meta(intervention_feature=None):
    features = {
        "observation.state": {"dtype": "float32", "shape": [2]},
        "action": {"dtype": "float32", "shape": [2]},
    }
    if intervention_feature is not None:
        features["intervention"] = intervention_feature
    return SimpleNamespace(features=features, stats={"visual": {"mean": np.array([0.5])}})


def test_validate_intervention_feature_schema_and_relative_actions():
    absolute = SimpleNamespace(use_relative_actions=False)
    relative = SimpleNamespace(use_relative_actions=True)

    with pytest.raises(ValueError, match="requires an 'intervention' feature"):
        validate_intervention_only_config(_meta(), absolute)
    with pytest.raises(ValueError, match=r"expected dtype bool and shape \[1\]"):
        validate_intervention_only_config(_meta({"dtype": "int64", "shape": [2]}), absolute)
    with pytest.raises(ValueError, match="does not support use_relative_actions=true"):
        validate_intervention_only_config(_meta({"dtype": "bool", "shape": [1]}), relative)


def test_true_only_stats_and_indices_exclude_false_rows():
    hf_dataset = Dataset.from_dict(
        {
            "intervention": [[True], [False], [True], [False]],
            "observation.state": [[1.0, 2.0], [10_000.0, 20_000.0], [3.0, 4.0], [-10_000.0, -20_000.0]],
            "action": [[5.0, 6.0], [30_000.0, 40_000.0], [7.0, 8.0], [-30_000.0, -40_000.0]],
        }
    )
    dataset = SimpleNamespace(
        hf_dataset=hf_dataset,
        meta=_meta({"dtype": "bool", "shape": [1]}),
    )

    data = compute_intervention_only_data(dataset)
    apply_intervention_only_data(dataset, data)

    assert data["total_frames"] == 4
    assert data["true_frames"] == 2
    assert data["true_indices"] == [0, 2]
    np.testing.assert_allclose(dataset.meta.stats["observation.state"]["mean"], [2.0, 3.0])
    np.testing.assert_allclose(dataset.meta.stats["observation.state"]["std"], [1.0, 1.0])
    np.testing.assert_allclose(dataset.meta.stats["action"]["mean"], [6.0, 7.0])
    assert "visual" in dataset.meta.stats


def test_true_only_scan_rejects_all_false():
    dataset = SimpleNamespace(
        hf_dataset=Dataset.from_dict(
            {
                "intervention": [[False], [False]],
                "observation.state": [[1.0], [2.0]],
                "action": [[3.0], [4.0]],
            }
        ),
        meta=_meta({"dtype": "bool", "shape": [1]}),
    )
    with pytest.raises(ValueError, match="no rows with intervention=true"):
        compute_intervention_only_data(dataset)


def test_mixed_dataset_keeps_demonstrations_and_corrections(tmp_path):
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    root = tmp_path / "mixed"
    writer = LeRobotDataset.create("test/mixed", fps=10, root=root, features=features, use_videos=False)
    patterns = [[True, True, True], [False, False, True, True]]
    value = 0
    for episode_index, pattern in enumerate(patterns):
        for intervention in pattern:
            writer.add_frame(
                {
                    "observation.state": np.array([value, value], dtype=np.float32),
                    "action": np.array([value, value], dtype=np.float32),
                    "intervention": np.array([intervention], dtype=bool),
                    "task": f"task-{episode_index}",
                }
            )
            value += 1
        writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset(
        "test/mixed",
        root=root,
        delta_timestamps={"action": [0.0, 0.1]},
        download_videos=False,
    )
    data = compute_intervention_only_data(dataset)
    disk_mean = load_stats(root)["action"]["mean"].copy()
    apply_intervention_only_data(dataset, data)
    starts, ends = selected_episode_boundaries(dataset)
    sampler = EpisodeAwareSampler(starts, ends, eligible_indices=data["true_indices"])

    assert sampler.indices == [0, 1, 2, 5, 6]
    np.testing.assert_allclose(dataset.meta.stats["action"]["mean"], [2.8, 2.8])
    np.testing.assert_array_equal(load_stats(root)["action"]["mean"], disk_mean)
    assert all(bool(dataset[index]["intervention"].item()) for index in sampler)
    np.testing.assert_array_equal(dataset[5]["action"], [[5.0, 5.0], [6.0, 6.0]])
    np.testing.assert_array_equal(dataset[6]["action_is_pad"], [False, True])

    streaming = StreamingLeRobotDataset(
        "test/mixed",
        root=root,
        episodes=[1],
        delta_timestamps={"action": [0.0, 0.1]},
        buffer_size=2,
        max_num_shards=2,
        shuffle=False,
        intervention_only=True,
    )
    streamed_frames = list(streaming)
    assert {int(frame["index"]) for frame in streamed_frames} == {5, 6}
    assert all(bool(frame["intervention"]) for frame in streamed_frames)
    frame_by_index = {int(frame["index"]): frame for frame in streamed_frames}
    np.testing.assert_array_equal(frame_by_index[5]["action"], [[5.0, 5.0], [6.0, 6.0]])
    np.testing.assert_array_equal(frame_by_index[6]["action_is_pad"], [False, True])


def test_episode_selection_and_drop_last_intersect_true_indices(tmp_path):
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    root = tmp_path / "selected"
    writer = LeRobotDataset.create("test/selected", fps=10, root=root, features=features, use_videos=False)
    for episode_index in range(2):
        for frame_index, intervention in enumerate([False, False, True, True]):
            writer.add_frame(
                {
                    "observation.state": np.array([frame_index, frame_index], dtype=np.float32),
                    "action": np.array([frame_index, frame_index], dtype=np.float32),
                    "intervention": np.array([intervention], dtype=bool),
                    "task": f"task-{episode_index}",
                }
            )
        writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("test/selected", root=root, episodes=[1], download_videos=False)
    data = compute_intervention_only_data(dataset)
    starts, ends = selected_episode_boundaries(dataset)
    sampler = EpisodeAwareSampler(
        starts,
        ends,
        drop_n_last_frames=1,
        eligible_indices=data["true_indices"],
    )
    assert data["true_indices"] == [2, 3]
    assert sampler.indices == [2]


def test_streaming_false_frame_returns_before_torch_conversion(monkeypatch):
    streaming = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
    streaming.intervention_only = True
    source = Backtrackable(
        iter([{"intervention": [False], "observation.state": [1.0], "action": [2.0]}]),
        history=1,
        lookahead=1,
    )

    def fail_if_called(_item):
        raise AssertionError("false frames must be skipped before tensor conversion")

    monkeypatch.setattr("lerobot.datasets.streaming_dataset.item_to_torch", fail_if_called)
    assert list(streaming.make_frame(source)) == []


def test_streaming_true_frame_action_chunk_and_padding_are_unchanged():
    rows = [
        {
            "intervention": [True],
            "action": [1.0],
            "episode_index": 0,
            "index": 0,
            "task_index": 0,
        },
        {
            "intervention": [True],
            "action": [2.0],
            "episode_index": 0,
            "index": 1,
            "task_index": 0,
        },
    ]

    def make_first_frame(intervention_only):
        streaming = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
        streaming.intervention_only = intervention_only
        streaming.delta_indices = {"action": [0, 1, 2]}
        streaming.delta_timestamps = {"action": [0.0, 1.0, 2.0]}
        streaming.meta = SimpleNamespace(
            video_keys=[],
            camera_keys=[],
            fps=1,
            tasks=pd.DataFrame(index=["task"]),
        )
        streaming.image_transforms = None
        source = Backtrackable(rows, history=1, lookahead=3)
        return list(streaming.make_frame(source))[0]

    filtered = make_first_frame(True)
    unfiltered = make_first_frame(False)
    np.testing.assert_array_equal(filtered["action"], unfiltered["action"])
    np.testing.assert_array_equal(filtered["action_is_pad"], unfiltered["action_is_pad"])
