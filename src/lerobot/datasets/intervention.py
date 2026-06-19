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

"""Expert-only dataset validation, projected reads, and runtime statistics."""

from __future__ import annotations

import copy
from typing import Any

import datasets
import numpy as np

from lerobot.utils.constants import ACTION, OBS_STATE

from .compute_stats import get_feature_stats
from .lerobot_dataset import LeRobotDataset

INTERVENTION = "intervention"
_PROJECTED_FEATURES = (INTERVENTION, OBS_STATE, ACTION)


def validate_intervention_only_config(meta, trainable_config) -> None:
    """Validate requirements that can be checked from metadata, before policy creation."""
    if getattr(trainable_config, "use_relative_actions", False):
        raise ValueError(
            "dataset.intervention_only=true does not support use_relative_actions=true. "
            "Expert-only training currently requires absolute actions."
        )

    feature = meta.features.get(INTERVENTION)
    if feature is None:
        raise ValueError(
            "dataset.intervention_only=true requires an 'intervention' feature with dtype bool and shape [1]."
        )
    if feature.get("dtype") != "bool" or tuple(feature.get("shape", ())) != (1,):
        raise ValueError(
            "Invalid 'intervention' feature schema: expected dtype bool and shape [1], "
            f"got dtype {feature.get('dtype')!r} and shape {feature.get('shape')!r}."
        )

    missing = [key for key in (OBS_STATE, ACTION) if key not in meta.features]
    if missing:
        raise ValueError(
            "dataset.intervention_only=true requires state and action features for true-only "
            f"normalization stats; missing: {missing}."
        )


def _iter_projected_batches(dataset, batch_size: int = 8192):
    projected = dataset.hf_dataset.select_columns(list(_PROJECTED_FEATURES))
    if isinstance(dataset, LeRobotDataset):
        projected = projected.with_format(None)
    yield from projected.iter(batch_size=batch_size)


def _as_feature_batch(values: Any, feature_key: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] == 0:
        raise ValueError(f"Projected feature {feature_key!r} produced an empty batch.")
    return array


def compute_intervention_only_data(dataset) -> dict[str, Any]:
    """Scan only intervention/state/action columns and compute true-row indices and stats."""
    total_frames = 0
    true_frames = 0
    true_indices: list[int] = []
    true_features: dict[str, list[np.ndarray]] = {OBS_STATE: [], ACTION: []}
    collect_indices = not isinstance(dataset.hf_dataset, datasets.IterableDataset)

    for batch in _iter_projected_batches(dataset):
        interventions = _as_feature_batch(batch[INTERVENTION], INTERVENTION)
        if interventions.shape[1:] != (1,):
            raise ValueError(
                "Invalid 'intervention' row shape while reading parquet: expected [1], "
                f"got {list(interventions.shape[1:])}."
            )
        mask = interventions[:, 0].astype(bool, copy=False)
        batch_count = len(mask)
        batch_true = np.flatnonzero(mask)
        true_frames += len(batch_true)
        if collect_indices:
            true_indices.extend((total_frames + batch_true).tolist())

        for key in (OBS_STATE, ACTION):
            values = _as_feature_batch(batch[key], key)
            if len(values) != batch_count:
                raise ValueError(f"Projected feature {key!r} has {len(values)} rows, expected {batch_count}.")
            if batch_true.size:
                true_features[key].append(values[batch_true])
        total_frames += batch_count

    if true_frames == 0:
        raise ValueError(
            "dataset.intervention_only=true found no rows with intervention=true in the selected episodes."
        )

    stats = {}
    for key, chunks in true_features.items():
        values = np.concatenate(chunks, axis=0)
        stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)

    return {
        "total_frames": total_frames,
        "true_frames": true_frames,
        "true_indices": true_indices,
        "stats": stats,
    }


def apply_intervention_only_data(dataset, data: dict[str, Any]) -> None:
    """Install true-only runtime stats without modifying the dataset's stats.json."""
    runtime_stats = copy.deepcopy(dataset.meta.stats) if dataset.meta.stats is not None else {}
    runtime_stats.update(data["stats"])
    dataset.meta.stats = runtime_stats
    dataset.intervention_indices = data["true_indices"]
    dataset.intervention_total_frames = data["total_frames"]
    dataset.intervention_true_frames = data["true_frames"]


def selected_episode_boundaries(dataset: LeRobotDataset) -> tuple[list[int], list[int]]:
    """Return dataset-relative episode boundaries, including when episodes were selected."""
    episode_indices = np.asarray(dataset.hf_dataset.with_format(None)["episode_index"]).reshape(-1)
    if episode_indices.size == 0:
        return [], []
    starts = np.concatenate(([0], np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(episode_indices)]))
    return starts.tolist(), ends.tolist()
