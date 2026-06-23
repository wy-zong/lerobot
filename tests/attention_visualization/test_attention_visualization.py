from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from attention_visualization.core import (
    aggregate_attention_trace,
    build_sample_frames,
    episode_camera_keys,
    forward_fill_records,
    load_jsonl_records,
    map_model_cameras,
    select_episode,
)
from attention_visualization.render import (
    ROW_HEIGHT,
    ROW_WIDTH,
    compose_multi_camera_frame,
    render_model_row,
    row_width,
)
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.policies.smolvla.smolvlm_with_expert import _record_attention_for_trace


def test_select_longest_episode_and_sampling(tmp_path: Path):
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [3, 232, 244],
                "length": [500, 10280, 9015],
                "dataset_from_index": [0, 500, 10780],
                "dataset_to_index": [500, 10780, 19795],
                "tasks": [["other"], ["flatten and fold the rag then place"], ["other"]],
            }
        ),
        episode_dir / "file-000.parquet",
    )

    episode = select_episode(tmp_path)
    assert episode.episode_index == 232
    assert episode.length == 10280
    frames = build_sample_frames(episode.length, source_fps=60, sample_fps=5)
    assert frames[:3] == [0, 12, 24]
    assert frames[-1] == 10272
    assert len(frames) == 857


def test_discrete_state_has_no_synthetic_state_attention_group():
    trace = {
        "token_groups": {
            "cam_a": [[0, 1]],
            "cam_b": [[1, 2]],
            "cam_c": [[2, 3]],
            "language": [[3, 6]],
        },
        "_denoise_step": 0,
    }
    probabilities = torch.tensor([[[[0.1, 0.1, 0.1, 0.2, 0.2, 0.3]]]])
    _record_attention_for_trace(
        trace,
        {"layer": 1, "kind": "expert_cross", "fill_kv_cache": False},
        probabilities,
        key_length=6,
    )
    result = aggregate_attention_trace(trace, ["cam_a", "cam_b", "cam_c"])
    assert "state" not in result["attention_percent"]
    assert result["attention_percent"]["language"] == pytest.approx(70.0)


class _FakeVLM:
    def embed_image(self, image):
        return torch.ones(image.shape[0], 2, 4)

    def embed_language_tokens(self, tokens):
        return torch.ones(tokens.shape[0], tokens.shape[1], 4)


def _fake_flow_model(discrete_state: bool = False):
    model = object.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.vlm_with_expert = _FakeVLM()
    model.add_image_special_tokens = False
    model.prefix_length = 0
    model.config = SimpleNamespace(use_state=True, discrete_state_in_language=discrete_state)
    model.state_proj = nn.Linear(3, 4)
    return model


def test_regular_prefix_token_ranges_are_exact():
    model = _fake_flow_model()
    images = [torch.zeros(1, 3, 4, 4) for _ in range(3)]
    masks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    trace = {"image_group_names": ["cam_a", "cam_b", "cam_c"]}
    model.embed_prefix(
        images,
        masks,
        torch.ones(1, 4, dtype=torch.long),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
        state=torch.zeros(1, 3),
        trace=trace,
    )
    assert trace["token_groups"] == {
        "cam_a": [[0, 2]],
        "cam_b": [[2, 4]],
        "cam_c": [[4, 6]],
        "language": [[6, 10]],
        "state": [[10, 11]],
    }
    assert trace["valid_prefix_tokens"] == [9]


def test_attention_multi_ranges_and_normalization_conserve_mass():
    trace = {
        "token_groups": {
            "cam_a": [[0, 1]],
            "cam_b": [[1, 2]],
            "cam_c": [[2, 3]],
            "state": [[4, 5]],
            "language": [[3, 4], [5, 6]],
        },
        "_denoise_step": 0,
    }
    probabilities = torch.tensor([[[[0.10, 0.20, 0.15, 0.05, 0.30, 0.20]]]])
    _record_attention_for_trace(
        trace,
        {"layer": 1, "kind": "expert_cross", "fill_kv_cache": False},
        probabilities,
        key_length=6,
    )
    result = aggregate_attention_trace(trace, ["cam_a", "cam_b", "cam_c"])
    assert result["raw_attention_mass"]["language"] == pytest.approx(0.25)
    assert sum(result["attention_percent"].values()) == pytest.approx(100.0)
    assert result["image_total_raw_attention_mass"] == pytest.approx(0.45)
    assert result["image_total_percent"] == pytest.approx(
        sum(result["attention_percent"][key] for key in ["cam_a", "cam_b", "cam_c"])
    )
    assert result["layer_step_details"][0]["denoise_step"] == 0


def test_camera_mapping_forward_fill_and_interrupted_jsonl(tmp_path: Path):
    mapping = map_model_cameras(
        [
            "observation.images.left_camera1",
            "observation.images.left_camera4",
            "observation.images.right_camera2",
        ],
        [
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ],
    )
    assert list(mapping.values()) == [
        "observation.images.left_camera1",
        "observation.images.right_camera2",
        "observation.images.left_camera4",
    ]

    records = [{"frame_index": 0, "value": 1}, {"frame_index": 12, "value": 2}]
    filled = forward_fill_records(records, 15)
    assert [filled[index]["value"] for index in [0, 11, 12, 14]] == [1, 1, 2, 2]

    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + '\n{"frame_index":')
    assert load_jsonl_records(path) == records


def test_episode_camera_keys_recovers_retained_camera():
    info = {
        "features": {
            "observation.images.left_camera1": {"dtype": "video"},
            "observation.images.left_camera4": {"dtype": "video"},
            "observation.images.right_camera2": {"dtype": "video"},
        }
    }
    episode = SimpleNamespace(
        metadata={
            "videos/observation.images.left_camera1/chunk_index": 0,
            "videos/observation.images.left_camera3/chunk_index": 0,
            "videos/observation.images.left_camera4/chunk_index": 0,
            "videos/observation.images.right_camera2/chunk_index": 0,
        }
    )
    assert episode_camera_keys(info, episode) == [
        "observation.images.left_camera1",
        "observation.images.left_camera4",
        "observation.images.right_camera2",
        "observation.images.left_camera3",
    ]


def test_single_model_row_renderer():
    cameras = ["cam_a", "cam_b", "cam_c"]
    frames = {
        camera: np.full((48, 64, 3), fill_value=index * 80, dtype=np.uint8)
        for index, camera in enumerate(cameras)
    }
    record = {
        "attention_percent": {
            "cam_a": 10.0,
            "cam_b": 20.0,
            "cam_c": 30.0,
            "state": 15.0,
            "language": 25.0,
        },
        "image_total_percent": 60.0,
        "state_attention_available": True,
        "task": "flatten and fold the rag then place",
        "state_values": list(range(12)),
    }
    image = render_model_row(frames, cameras, record, "test/model", frame_index=120, source_fps=60)
    assert image.size == (ROW_WIDTH, ROW_HEIGHT)
    assert np.asarray(image).std() > 0

    discrete_record = dict(record)
    discrete_record["attention_percent"] = dict(record["attention_percent"])
    discrete_record["attention_percent"].pop("state")
    discrete_record["attention_percent"]["language"] = 40.0
    discrete_record["state_attention_available"] = False
    discrete_image = render_model_row(
        frames, cameras, discrete_record, "test/discrete", frame_index=120, source_fps=60
    )
    assert discrete_image.size == (ROW_WIDTH, ROW_HEIGHT)


def test_three_vs_four_camera_renderer():
    cameras = ["cam_a", "cam_b", "cam_c", "cam_d"]
    frames = {
        camera: np.full((48, 64, 3), fill_value=index * 60, dtype=np.uint8)
        for index, camera in enumerate(cameras)
    }
    base = {
        "attention_percent": {
            "cam_a": 15.0,
            "cam_b": 15.0,
            "cam_c": 15.0,
            "cam_d": 15.0,
            "state": 15.0,
            "language": 25.0,
        },
        "image_total_percent": 60.0,
        "state_attention_available": True,
        "task": "task",
        "state_values": [0.0] * 12,
    }
    image = compose_multi_camera_frame(
        frames,
        [cameras[:3], cameras],
        [base, base],
        ["three", "four"],
        frame_index=0,
        source_fps=60,
    )
    assert image.size == (row_width(4), ROW_HEIGHT * 2)
    assert np.asarray(image).std() > 0
