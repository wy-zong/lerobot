from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ActionChunk:
    episode_index: int
    chunk_index: int
    frame_indices: np.ndarray
    timestamps: np.ndarray
    action: np.ndarray
    intervention: np.ndarray
    is_partial: bool


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    fps: float
    action_names: list[str]
    total_frames: int
    total_episodes: int


def _scalar(value: object) -> object:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
        return value[0]
    return value


def load_action_frames(dataset_root: Path) -> tuple[DatasetInfo, pd.DataFrame]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    info_json = json.loads(info_path.read_text(encoding="utf-8"))
    action_feature = info_json.get("features", {}).get("action")
    if not action_feature:
        raise ValueError("dataset does not declare an action feature")
    files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found below {dataset_root / 'data'}")
    schema_names = set(pq.read_schema(files[0]).names)
    columns = ["action", "episode_index", "frame_index", "timestamp"]
    has_intervention = "intervention" in schema_names
    if has_intervention:
        columns.append("intervention")
    missing = set(columns) - schema_names
    if missing:
        raise ValueError(f"missing required parquet columns: {sorted(missing)}")
    frames = pd.concat([pq.read_table(path, columns=columns).to_pandas() for path in files], ignore_index=True)
    frames["action"] = frames["action"].map(lambda value: np.asarray(value, dtype=np.float64))
    for column in ("episode_index", "frame_index", "timestamp"):
        frames[column] = frames[column].map(_scalar)
    frames["intervention"] = frames["intervention"].map(_scalar).astype(bool) if has_intervention else False
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    names = action_feature.get("names") or [f"action_{i}" for i in range(action_feature["shape"][0])]
    expected_dim = len(names)
    invalid = frames["action"].map(lambda value: value.shape != (expected_dim,))
    if invalid.any():
        raise ValueError(f"found {int(invalid.sum())} actions with an invalid shape")
    if not np.isfinite(np.stack(frames["action"].to_numpy())).all():
        raise ValueError("action data contains NaN or Inf")
    info = DatasetInfo(dataset_root, float(info_json["fps"]), list(names), len(frames), int(frames["episode_index"].nunique()))
    return info, frames


def validate_frames(frames: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for episode_index, episode in frames.groupby("episode_index", sort=True):
        indices = episode["frame_index"].to_numpy(dtype=np.int64)
        timestamps = episode["timestamp"].to_numpy(dtype=np.float64)
        if len(indices) > 1 and np.any(np.diff(indices) != 1):
            warnings.append(f"episode {episode_index}: frame_index is not contiguous")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            warnings.append(f"episode {episode_index}: timestamp is not strictly increasing")
    return warnings


def make_chunks(frames: pd.DataFrame, chunk_length: int) -> list[ActionChunk]:
    if chunk_length < 2:
        raise ValueError("chunk_length must be at least 2")
    chunks: list[ActionChunk] = []
    for episode_index, episode in frames.groupby("episode_index", sort=True):
        episode = episode.reset_index(drop=True)
        for start in range(0, len(episode), chunk_length):
            window = episode.iloc[start : start + chunk_length]
            chunks.append(ActionChunk(
                int(episode_index), start // chunk_length,
                window["frame_index"].to_numpy(dtype=np.int64),
                window["timestamp"].to_numpy(dtype=np.float64),
                np.stack(window["action"].to_numpy()),
                window["intervention"].to_numpy(dtype=bool),
                len(window) < chunk_length,
            ))
    return chunks

