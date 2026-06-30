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

"""Training-time language conditioning derived from SARM progress."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.rewards.sarm.rabc import resolve_hf_path
from lerobot.utils.import_utils import _pandas_available
from lerobot.utils.sample_weighting import resolve_sarm_progress_path

if TYPE_CHECKING or _pandas_available:
    import pandas as pd
else:
    pd = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


@dataclass
class SARMTaskConditioningConfig:
    """Configuration for binary task suffixes based on SARM progress."""

    type: str = "sarm_binary"
    progress_path: str | None = None
    head_mode: str = "sparse"
    threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.type != "sarm_binary":
            raise ValueError(f"Unknown task conditioning type: {self.type!r}. Supported type: 'sarm_binary'.")
        if self.head_mode not in {"sparse", "dense"}:
            raise ValueError(
                f"task_conditioning.head_mode must be 'sparse' or 'dense', got {self.head_mode!r}."
            )
        if not math.isfinite(self.threshold):
            raise ValueError(f"task_conditioning.threshold must be finite, got {self.threshold!r}.")


def _as_flat_batch(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        if tuple(array.shape[1:]) != (1,):
            raise ValueError(
                f"SARM binary task conditioning expected scalar {name!r} values, "
                f"got trailing shape {list(array.shape[1:])}."
            )
        array = array.reshape(-1)
    return array


def _as_batch_indices(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    indices = _as_flat_batch(values, "index")
    result = []
    for value in indices:
        index = int(value)
        if float(value) != index:
            raise ValueError(f"SARM binary task conditioning received non-integral index {value!r}.")
        result.append(index)
    return result


class SARMTaskConditioner:
    """Append positive/negative language labels to tasks without filtering samples."""

    def __init__(
        self,
        *,
        dataset: Any,
        progress_path: str | Path,
        chunk_size: int,
        head_mode: str = "sparse",
        threshold: float = 0.01,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"SARM binary task conditioning requires chunk_size > 0, got {chunk_size}.")
        if head_mode not in {"sparse", "dense"}:
            raise ValueError(f"head_mode must be 'sparse' or 'dense', got {head_mode!r}.")
        if not math.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold!r}.")

        self.progress_path = resolve_hf_path(progress_path)
        self.chunk_size = chunk_size
        self.head_mode = head_mode
        self.threshold = threshold
        self.progress_column = f"progress_{head_mode}"

        dataset_rows = self._load_dataset_rows(dataset)
        progress_rows = self._load_progress_rows()
        self.label_lookup = self._build_label_lookup(dataset_rows, progress_rows)

    @staticmethod
    def _load_dataset_rows(dataset: Any) -> dict[int, tuple[int, bool]]:
        features = getattr(getattr(dataset, "meta", None), "features", {})
        if "intervention" not in features:
            raise ValueError(
                "SARM binary task conditioning requires an 'intervention' feature in the dataset."
            )

        projected = dataset.hf_dataset.select_columns(["index", "episode_index", "intervention"])
        if hasattr(projected, "with_format"):
            projected = projected.with_format(None)

        rows: dict[int, tuple[int, bool]] = {}
        for batch in projected.iter(batch_size=8192):
            indices = _as_flat_batch(batch["index"], "index")
            episode_indices = _as_flat_batch(batch["episode_index"], "episode_index")
            interventions = _as_flat_batch(batch["intervention"], "intervention")
            if interventions.dtype != np.bool_:
                raise ValueError(
                    "SARM binary task conditioning requires boolean 'intervention' values, "
                    f"got dtype {interventions.dtype}."
                )
            if not (len(indices) == len(episode_indices) == len(interventions)):
                raise ValueError(
                    "Dataset index, episode_index, and intervention columns have different lengths."
                )

            for raw_index, raw_episode, raw_intervention in zip(
                indices, episode_indices, interventions, strict=True
            ):
                index = int(raw_index)
                episode = int(raw_episode)
                if float(raw_index) != index or float(raw_episode) != episode:
                    raise ValueError("Dataset index and episode_index values must be integral.")
                if index in rows:
                    raise ValueError(f"Duplicate dataset index {index} in selected training data.")
                rows[index] = (episode, bool(raw_intervention))

        if not rows:
            raise ValueError("SARM binary task conditioning cannot label an empty dataset.")
        return rows

    def _load_progress_rows(self) -> dict[int, tuple[int, float | None]]:
        progress_df = pd.read_parquet(self.progress_path)
        required_columns = {"index", "episode_index", self.progress_column}
        missing = sorted(required_columns.difference(progress_df.columns))
        if missing:
            available_progress = sorted(
                column for column in progress_df.columns if column.startswith("progress")
            )
            raise ValueError(
                f"SARM progress parquet is missing required columns {missing}. "
                f"Available progress columns: {available_progress}."
            )

        rows: dict[int, tuple[int, float | None]] = {}
        for row in progress_df.loc[:, ["index", "episode_index", self.progress_column]].itertuples(
            index=False, name=None
        ):
            raw_index, raw_episode, raw_progress = row
            if pd.isna(raw_index) or pd.isna(raw_episode):
                raise ValueError("SARM progress parquet contains a null index or episode_index.")
            index = int(raw_index)
            episode = int(raw_episode)
            if float(raw_index) != index or float(raw_episode) != episode:
                raise ValueError("SARM progress index and episode_index values must be integral.")
            if index in rows:
                raise ValueError(f"Duplicate index {index} in SARM progress parquet.")
            progress = None if pd.isna(raw_progress) else float(raw_progress)
            if progress is not None and not math.isfinite(progress):
                raise ValueError(f"SARM progress at index {index} must be finite, got {progress!r}.")
            rows[index] = (episode, progress)
        return rows

    def _build_label_lookup(
        self,
        dataset_rows: dict[int, tuple[int, bool]],
        progress_rows: dict[int, tuple[int, float | None]],
    ) -> dict[int, str]:
        episode_indices: dict[int, list[int]] = {}
        for index, (episode, _) in dataset_rows.items():
            episode_indices.setdefault(episode, []).append(index)

            progress_row = progress_rows.get(index)
            if progress_row is None:
                raise ValueError(f"Dataset index {index} is missing from the SARM progress parquet.")
            if progress_row[0] != episode:
                raise ValueError(
                    f"Episode mismatch at index {index}: dataset has {episode}, "
                    f"SARM progress has {progress_row[0]}."
                )

        episode_ends: dict[int, int] = {}
        next_intervention: dict[int, int | None] = {}
        for episode, indices in episode_indices.items():
            indices.sort()
            if indices[-1] - indices[0] + 1 != len(indices):
                raise ValueError(
                    f"Selected dataset episode {episode} has non-contiguous global frame indices."
                )
            episode_ends[episode] = indices[-1] + 1
            next_index = None
            for index in reversed(indices):
                if dataset_rows[index][1]:
                    next_index = index
                next_intervention[index] = next_index

        labels: dict[int, str] = {}
        for index, (episode, _) in dataset_rows.items():
            episode_end = episode_ends[episode]
            action_end = min(index + self.chunk_size, episode_end)
            next_intervention_index = next_intervention[index]
            if next_intervention_index is not None and next_intervention_index < action_end:
                labels[index] = "positive"
                continue

            future_index = min(index + self.chunk_size, episode_end - 1)
            current_progress = progress_rows[index][1]
            future_progress = progress_rows[future_index][1]
            if current_progress is None:
                raise ValueError(
                    f"Autonomous chunk at index {index} has missing {self.progress_column} progress."
                )
            if future_progress is None:
                raise ValueError(
                    f"Autonomous chunk at index {index} has missing endpoint "
                    f"{self.progress_column} progress at index {future_index}."
                )
            labels[index] = "positive" if future_progress - current_progress > self.threshold else "negative"
        return labels

    def condition_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Mutate ``batch['task']`` by appending the precomputed binary label."""
        if "index" not in batch:
            raise ValueError("SARM binary task conditioning requires batch['index'].")
        if "task" not in batch:
            raise ValueError("SARM binary task conditioning requires batch['task'].")

        indices = _as_batch_indices(batch["index"])
        tasks = batch["task"]
        tasks = [tasks] if isinstance(tasks, str) else list(tasks)
        if len(tasks) != len(indices):
            raise ValueError(f"Batch task count ({len(tasks)}) does not match index count ({len(indices)}).")

        conditioned_tasks = []
        for task, index in zip(tasks, indices, strict=True):
            if not isinstance(task, str):
                raise ValueError(f"Expected task text to be a string, got {type(task).__name__}.")
            try:
                label = self.label_lookup[index]
            except KeyError as error:
                raise ValueError(
                    f"Batch index {index} was not present when SARM task labels were built."
                ) from error
            conditioned_tasks.append(f"{task.rstrip()} task:{label}")
        batch["task"] = conditioned_tasks
        return batch


def make_task_conditioner(
    config: SARMTaskConditioningConfig | None,
    *,
    dataset: Any,
    policy: PreTrainedPolicy,
    dataset_root: str | Path | None = None,
    dataset_repo_id: str | None = None,
) -> SARMTaskConditioner | None:
    """Create the configured training-time task conditioner."""
    if config is None:
        return None
    if config.type != "sarm_binary":
        raise ValueError(f"Unknown task conditioning type: {config.type!r}. Supported type: 'sarm_binary'.")

    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError("SARM binary task conditioning requires a policy with 'chunk_size' in its config.")
    progress_path = resolve_sarm_progress_path(
        config.progress_path,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
        feature_name="SARM binary task conditioning",
    )
    return SARMTaskConditioner(
        dataset=dataset,
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        threshold=config.threshold,
    )
