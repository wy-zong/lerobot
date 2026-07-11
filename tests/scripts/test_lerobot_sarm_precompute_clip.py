#!/usr/bin/env python

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

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.scripts import lerobot_sarm_precompute_clip as precompute


class _Episodes:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def to_pandas(self) -> pd.DataFrame:
        return self.dataframe.copy()


class _Metadata:
    repo_id = "test/dataset"
    revision = "test-revision"
    root = Path("/dataset")
    video_keys = ["observation.images.top"]
    total_episodes = 2
    total_frames = 4
    episodes = _Episodes(
        pd.DataFrame(
            {
                "episode_index": [0, 1],
                "dataset_from_index": [0, 2],
                "length": [2, 2],
                "videos/observation.images.top/chunk_index": [0, 0],
                "videos/observation.images.top/file_index": [0, 1],
                "videos/observation.images.top/from_timestamp": [0.0, 0.2],
            }
        )
    )

    def get_video_file_path(self, episode_index: int, camera_key: str) -> Path:
        del camera_key
        return Path(f"video-{episode_index}.mp4")


class _ImageProcessor:
    crop_size = {"height": 1, "width": 1}
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]

    @classmethod
    def from_pretrained(cls, model_id: str):
        del model_id
        return cls()


class _Vision:
    def __call__(self, pixel_values: torch.Tensor):
        return SimpleNamespace(pooler_output=torch.ones((len(pixel_values), 2), device=pixel_values.device))


class _Projection:
    def __call__(self, pooled: torch.Tensor):
        return pooled


class _Model:
    vision_model = _Vision()
    visual_projection = _Projection()
    config = SimpleNamespace(projection_dim=2)

    @classmethod
    def from_pretrained(cls, model_id: str):
        del model_id
        return cls()

    def to(self, device: torch.device):
        del device
        return self

    def eval(self):
        return self


class _Ffmpeg:
    def __init__(self, raw_frames: bytes, return_code: int = 0, stderr: bytes = b""):
        self.stdout = io.BytesIO(raw_frames)
        self.stderr = io.BytesIO(stderr)
        self.return_code = return_code
        self.killed = False

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return self.return_code

    def kill(self) -> None:
        self.killed = True


def _run(
    output: Path,
    popen,
    *,
    resume: bool = False,
    overwrite: bool = False,
    metadata: _Metadata | None = None,
) -> None:
    with (
        patch.object(precompute, "require_package"),
        patch.object(precompute, "LeRobotDatasetMetadata", return_value=metadata or _Metadata()),
        patch.object(precompute, "CLIPImageProcessor", _ImageProcessor),
        patch.object(precompute, "CLIPModel", _Model),
        patch.object(precompute, "_probe_video_size", return_value=(1, 1)),
        patch.object(precompute.subprocess, "Popen", side_effect=popen),
    ):
        precompute.precompute_for_camera(
            repo_id="test/dataset",
            camera_key="observation.images.top",
            output=output,
            device="cpu",
            chunk_frames=2,
            resume=resume,
            overwrite=overwrite,
        )


def test_interrupted_file_is_recomputed_when_resuming(tmp_path: Path) -> None:
    output = tmp_path / "clip.npy"
    calls = []

    def interrupted_popen(command, **kwargs):
        del kwargs
        video_path = Path(command[command.index("-i") + 1])
        calls.append(video_path.name)
        n_frames = 2 if video_path.name == "video-0.mp4" else 1
        return _Ffmpeg(bytes([1, 2, 3]) * n_frames)

    with pytest.raises(RuntimeError, match="Frame count mismatch"):
        _run(output, interrupted_popen)

    progress_path = Path(f"{output}.progress.json")
    progress = json.loads(progress_path.read_text())
    assert progress["completed_files"] == [[0, 0]]
    assert calls == ["video-0.mp4", "video-1.mp4"]

    resume_calls = []

    def resumed_popen(command, **kwargs):
        del kwargs
        video_path = Path(command[command.index("-i") + 1])
        resume_calls.append(video_path.name)
        return _Ffmpeg(bytes([4, 5, 6]) * 2)

    _run(output, resumed_popen, resume=True)

    assert resume_calls == ["video-1.mp4"]
    assert np.array_equal(np.load(output), np.ones((4, 2), dtype=np.float32))
    progress = json.loads(progress_path.read_text())
    assert progress["completed_files"] == [[0, 0], [0, 1]]

    def unexpected_popen(*args, **kwargs):
        raise AssertionError(f"ffmpeg should not run: {args}, {kwargs}")

    _run(output, unexpected_popen, resume=True)


def test_resume_scans_legacy_output_by_whole_video(tmp_path: Path) -> None:
    output = tmp_path / "legacy.npy"
    legacy = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(4, 2))
    legacy[0:2] = 1
    legacy[2] = 2
    legacy.flush()

    calls = []

    def popen(command, **kwargs):
        del kwargs
        video_path = Path(command[command.index("-i") + 1])
        calls.append(video_path.name)
        return _Ffmpeg(bytes([7, 8, 9]) * 2)

    _run(output, popen, resume=True)

    assert calls == ["video-1.mp4"]
    progress = json.loads(Path(f"{output}.progress.json").read_text())
    assert progress["completed_files"] == [[0, 0], [0, 1]]
    assert "could not be fully verified" in progress["warnings"][0]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("clip_model_id", "different/model"),
        ("dataset.camera_key", "observation.images.other"),
        ("dataset.manifest_fingerprint", "different-fingerprint"),
        ("output", {"shape": [99, 2], "dtype": "float32"}),
    ],
)
def test_checkpoint_identity_mismatch_is_rejected(tmp_path: Path, field: str, replacement: object) -> None:
    output = tmp_path / "clip.npy"
    _run(output, lambda *args, **kwargs: _Ffmpeg(bytes([1, 2, 3]) * 2))

    progress_path = Path(f"{output}.progress.json")
    progress = json.loads(progress_path.read_text())
    if field.startswith("dataset."):
        progress["dataset"][field.removeprefix("dataset.")] = replacement
    else:
        progress[field] = replacement
    progress_path.write_text(json.dumps(progress))

    with pytest.raises(ValueError, match="incompatible"):
        _run(output, lambda *args, **kwargs: pytest.fail("ffmpeg should not run"), resume=True)


def test_existing_output_requires_explicit_mode(tmp_path: Path) -> None:
    output = tmp_path / "clip.npy"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--resume"):
        precompute.precompute_for_camera(
            repo_id="test/dataset",
            camera_key="observation.images.top",
            output=output,
        )


def test_resume_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        precompute.precompute_for_camera(
            repo_id="test/dataset",
            camera_key="observation.images.top",
            output=tmp_path / "clip.npy",
            resume=True,
            overwrite=True,
        )


def test_nonzero_ffmpeg_exit_does_not_complete_file(tmp_path: Path) -> None:
    output = tmp_path / "clip.npy"

    with pytest.raises(RuntimeError, match="exit code 1"):
        _run(output, lambda *args, **kwargs: _Ffmpeg(bytes([1, 2, 3]) * 2, 1, b"decode failed"))

    progress = json.loads(Path(f"{output}.progress.json").read_text())
    assert progress["completed_files"] == []


def test_ffmpeg_permission_error_does_not_complete_file(tmp_path: Path) -> None:
    output = tmp_path / "clip.npy"

    with pytest.raises(PermissionError):
        _run(output, lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")))

    progress = json.loads(Path(f"{output}.progress.json").read_text())
    assert progress["completed_files"] == []


@pytest.mark.parametrize(
    "contents",
    [
        np.ones((4, 3), dtype=np.float32),
        np.ones((4, 2), dtype=np.float64),
    ],
)
def test_existing_output_shape_and_dtype_are_validated(tmp_path: Path, contents: np.ndarray) -> None:
    output = tmp_path / "clip.npy"
    np.save(output, contents)

    with pytest.raises(ValueError, match="incompatible"):
        _run(output, lambda *args, **kwargs: pytest.fail("ffmpeg should not run"), resume=True)


def test_legacy_scan_requires_every_row_in_a_video_to_be_valid(tmp_path: Path) -> None:
    output = tmp_path / "legacy.npy"
    legacy = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(6, 2))
    legacy[0:2] = 1
    legacy[4] = 1
    legacy.flush()
    video_files = [
        precompute._VideoFile(
            key=(0, file_index),
            path=Path(f"/video-{file_index}.mp4"),
            relative_path=f"video-{file_index}.mp4",
            global_indices=np.arange(file_index * 2, file_index * 2 + 2),
            manifest_episodes=[],
        )
        for file_index in range(3)
    ]

    assert precompute._scan_legacy_completed(legacy, video_files) == {(0, 0)}


def test_overwrite_replaces_output_and_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "clip.npy"
    _run(output, lambda *args, **kwargs: _Ffmpeg(bytes([1, 2, 3]) * 2))
    np.load(output, mmap_mode="r+")[0] = np.nan
    Path(f"{output}.progress.json").write_text("invalid")

    calls = []

    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        return _Ffmpeg(bytes([1, 2, 3]) * 2)

    _run(output, popen, overwrite=True)

    assert len(calls) == 2
    assert np.isfinite(np.load(output)).all()
    progress = json.loads(Path(f"{output}.progress.json").read_text())
    assert progress["completed_files"] == [[0, 0], [0, 1]]
