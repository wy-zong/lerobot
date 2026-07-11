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

"""Annotate every frame in a LeRobot dataset with RECAP-SARM values and advantages."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.configs.rewards import RewardModelConfig
from lerobot.datasets import LeRobotDataset

from ..sarm.sarm_utils import pad_state_to_max_dim
from .configuration_recap_sarm import RECAPSARMConfig
from .modeling_recap_sarm import RECAPSARMRewardModel
from .processor_recap_sarm import SUCCESS_COLUMNS


class RECAPSARMAnnotationPreprocessor:
    """Inference-only encoder without training targets or augmentation dependencies."""

    def __init__(
        self,
        config: RECAPSARMConfig,
        *,
        dataset_meta: Any,
        dataset_stats: dict[str, Any],
        clip_model_path: str,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.config = config
        self.dataset_meta = dataset_meta
        self.dataset_stats = dataset_stats
        self.device = torch.device(config.device)
        local_clip = Path(clip_model_path).exists()
        self.clip_model = (
            CLIPModel.from_pretrained(clip_model_path, local_files_only=local_clip)
            .to(self.device)
            .eval()
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_model_path,
            use_fast=True,
            local_files_only=local_clip,
        )
        self._text_cache: dict[str, torch.Tensor] = {}
        self.precomputed_image_features = (
            np.load(config.precomputed_image_features_path, mmap_mode="r")
            if config.precomputed_image_features_path is not None
            else None
        )
        self.zero_image_feature = self._encode_images(np.zeros((1, 1, 3, 224, 224), dtype=np.uint8))[0, 0]

    @torch.inference_mode()
    def _encode_images(self, images: np.ndarray) -> torch.Tensor:
        from PIL import Image

        batch_size, seq_len = images.shape[:2]
        flat_images = images.reshape(batch_size * seq_len, *images.shape[2:])
        pil_images = []
        for image in flat_images:
            if image.shape[0] in (1, 3):
                image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            pil_images.append(Image.fromarray(image))

        embeddings = []
        for start in range(0, len(pil_images), self.config.clip_batch_size):
            inputs = self.clip_processor(
                images=pil_images[start : start + self.config.clip_batch_size],
                return_tensors="pt",
            )
            output = self.clip_model.get_image_features(
                **{key: value.to(self.device) for key, value in inputs.items()}
            )
            if not isinstance(output, torch.Tensor):
                output = output.pooler_output
                if output is None:
                    raise ValueError("CLIP image encoder did not return pooled features")
            embeddings.append(output.detach().cpu())
        return torch.cat(embeddings).reshape(batch_size, seq_len, -1)

    @torch.inference_mode()
    def _encode_text(self, text: str, batch_size: int) -> torch.Tensor:
        if text not in self._text_cache:
            inputs = self.clip_processor.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            output = self.clip_model.get_text_features(
                **{key: value.to(self.device) for key, value in inputs.items()}
            )
            if not isinstance(output, torch.Tensor):
                output = output.pooler_output
                if output is None:
                    raise ValueError("CLIP text encoder did not return pooled features")
            self._text_cache[text] = output.detach().cpu()
        return self._text_cache[text].expand(batch_size, -1)

    def _precomputed_video_features(
        self,
        frame_indices: np.ndarray,
        episode_indices: np.ndarray,
    ) -> torch.Tensor:
        absolute_indices = np.empty(
            (len(frame_indices), len(self.config.observation_delta_indices)),
            dtype=np.int64,
        )
        for batch_idx, (frame_index, episode_index) in enumerate(
            zip(frame_indices, episode_indices, strict=True)
        ):
            episode = self.dataset_meta.episodes[int(episode_index)]
            episode_start = int(episode["dataset_from_index"])
            episode_end = int(episode["dataset_to_index"])
            for time_idx, delta in enumerate(self.config.observation_delta_indices):
                absolute_indices[batch_idx, time_idx] = np.clip(
                    int(frame_index) + delta,
                    episode_start,
                    episode_end - 1,
                )

        features = np.asarray(self.precomputed_image_features[absolute_indices]).copy()
        features[:, 1 + self.config.n_obs_steps :] = self.zero_image_feature.numpy()
        return torch.from_numpy(features).float()

    def _normalize_state(self, state: Any) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        stats = self.dataset_stats.get(self.config.state_key)
        if stats is not None:
            mean = torch.as_tensor(stats["mean"], dtype=state_tensor.dtype)
            std = torch.as_tensor(stats["std"], dtype=state_tensor.dtype)
            state_tensor = (state_tensor - mean) / (std + 1e-8)
        state_tensor[:, 1 + self.config.n_obs_steps :] = 0
        return pad_state_to_max_dim(state_tensor, self.config.max_state_dim)

    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        frame_indices = torch.as_tensor(batch["index"]).cpu().numpy()
        episode_indices = torch.as_tensor(batch["episode_index"]).cpu().numpy()
        batch_size = len(frame_indices)

        if self.precomputed_image_features is not None:
            video_features = self._precomputed_video_features(frame_indices, episode_indices)
        else:
            images = torch.as_tensor(batch[self.config.image_key]).cpu().numpy()
            images[:, 1 + self.config.n_obs_steps :] = 0
            video_features = self._encode_images(images)

        state_features = None
        if self.config.state_key in batch:
            state_features = self._normalize_state(batch[self.config.state_key])

        return {
            "video_features": video_features,
            "text_features": self._encode_text(str(batch["task"]), batch_size),
            "state_features": state_features,
            "lengths": torch.full(
                (batch_size,),
                1 + self.config.n_obs_steps,
                dtype=torch.int32,
            ),
        }


def _stack_sample_values(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values)
    if isinstance(first, np.ndarray):
        return np.stack(values)
    return values


def _iter_chunks(items: list[int], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _episode_task(episode: Any) -> str:
    task = episode.get("task", episode.get("tasks", ""))
    if isinstance(task, (list, tuple, np.ndarray)):
        return str(task[0]) if len(task) else ""
    return str(task)


def _episode_success(episode: Any) -> bool | None:
    for key in SUCCESS_COLUMNS:
        value = episode.get(key)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"success", "successful", "true", "1", "yes"}:
                return True
            if normalized in {"failure", "failed", "false", "0", "no"}:
                return False
            return None
        return bool(value)
    return None


def compute_episode_advantages(
    values: np.ndarray,
    *,
    success: bool | None,
    normalization_length: int,
    failure_penalty: float,
    lookahead: int,
) -> np.ndarray:
    """Compute normalized n-step RECAP advantages for one complete episode."""

    values = np.asarray(values, dtype=np.float32)
    denominator = max(1, normalization_length)
    rewards = np.full(len(values), -1.0 / denominator, dtype=np.float32)
    if success is False and len(rewards):
        rewards[-1] -= failure_penalty / denominator

    advantages = np.full(len(values), np.nan, dtype=np.float32)
    for frame_idx in range(len(values)):
        bootstrap_idx = min(frame_idx + lookahead, len(values))
        if success is None and bootstrap_idx == len(values):
            continue
        bootstrap = values[bootstrap_idx] if bootstrap_idx < len(values) else 0.0
        advantages[frame_idx] = rewards[frame_idx:bootstrap_idx].sum() + bootstrap - values[frame_idx]
    return advantages


def _build_task_max_lengths(episodes: list[Any]) -> dict[str, int]:
    task_max_lengths: dict[str, int] = {}
    for episode in episodes:
        task = _episode_task(episode)
        length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        task_max_lengths[task] = max(task_max_lengths.get(task, 0), length)
    return task_max_lengths


def _normalization_length(
    episode: Any,
    config: RECAPSARMConfig,
    task_max_lengths: dict[str, int],
) -> int:
    episode_length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
    if not config.normalize_per_task or config.task_max_episode_length_source == "episode_length":
        return max(1, episode_length)

    for key in ("task_max_episode_length", "max_episode_length"):
        value = episode.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return max(1, int(value))
    return max(1, task_max_lengths.get(_episode_task(episode), episode_length))


def _build_batch(
    dataset: LeRobotDataset,
    query_indices: list[int],
    *,
    episode_index: int,
    task: str,
    image_key: str,
    state_key: str,
    include_image: bool,
) -> dict[str, Any]:
    samples = [dataset[index] for index in query_indices]
    batch: dict[str, Any] = {
        "task": task,
        "index": torch.tensor(query_indices, dtype=torch.long),
        "episode_index": torch.full((len(query_indices),), episode_index, dtype=torch.long),
    }
    if include_image:
        batch[image_key] = _stack_sample_values([sample[image_key] for sample in samples])
    if state_key in samples[0]:
        batch[state_key] = _stack_sample_values([sample[state_key] for sample in samples])
    return batch


def _interpolate(computed_indices: np.ndarray, values: np.ndarray, all_indices: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.full(len(all_indices), np.nan, dtype=np.float32)
    if valid.sum() == 1:
        return np.full(len(all_indices), values[valid][0], dtype=np.float32)
    return np.interp(all_indices, computed_indices[valid], values[valid]).astype(np.float32)


def _load_checkpoint_dataset_stats(
    reward_model_path: str,
    *,
    state_key: str,
    fallback_stats: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(reward_model_path)
    if not checkpoint_path.is_dir():
        return fallback_stats

    stats_path = checkpoint_path / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if not stats_path.is_file():
        logging.warning("Checkpoint normalization stats not found; using target dataset stats.")
        return fallback_stats

    from safetensors.torch import load_file

    flat_stats = load_file(stats_path)
    prefix = f"{state_key}."
    state_stats = {
        key.removeprefix(prefix): value
        for key, value in flat_stats.items()
        if key.startswith(prefix)
    }
    if "mean" not in state_stats or "std" not in state_stats:
        logging.warning("Checkpoint state mean/std not found; using target dataset stats.")
        return fallback_stats
    return {state_key: state_stats}


def load_annotation_resources(
    *,
    dataset_repo_id: str,
    dataset_root: str | Path | None,
    reward_model_path: str,
    device: str,
    image_key_override: str | None,
    precomputed_image_features_path: str | None,
    clip_model_path: str,
) -> tuple[LeRobotDataset, RECAPSARMRewardModel, Any]:
    config = RewardModelConfig.from_pretrained(reward_model_path)
    if not isinstance(config, RECAPSARMConfig):
        raise TypeError(f"Expected a recap_sarm checkpoint, got {config.type!r}")

    config.device = device
    if image_key_override:
        config.image_key = image_key_override
    if precomputed_image_features_path is not None:
        config.precomputed_image_features_path = precomputed_image_features_path

    model = RECAPSARMRewardModel.from_pretrained(
        reward_model_path,
        config=config,
        local_files_only=Path(reward_model_path).exists(),
    )
    model.to(device).eval()

    use_precomputed_images = config.precomputed_image_features_path is not None
    metadata_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        download_videos=False,
        skip_video_decode=True,
    )
    delta_timestamps = {
        config.state_key: [index / metadata_dataset.fps for index in config.observation_delta_indices]
    }
    if not use_precomputed_images:
        delta_timestamps[config.image_key] = [
            index / metadata_dataset.fps for index in config.observation_delta_indices
        ]

    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=not use_precomputed_images,
        skip_video_decode=use_precomputed_images,
    )
    preprocessor = RECAPSARMAnnotationPreprocessor(
        config=config,
        dataset_stats=_load_checkpoint_dataset_stats(
            reward_model_path,
            state_key=config.state_key,
            fallback_stats=dataset.meta.stats,
        ),
        dataset_meta=dataset.meta,
        clip_model_path=clip_model_path,
    )
    return dataset, model, preprocessor


def annotate_dataset(
    *,
    dataset_repo_id: str,
    reward_model_path: str,
    dataset_root: str | Path | None = None,
    output_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int = 32,
    stride: int = 1,
    image_key_override: str | None = None,
    precomputed_image_features_path: str | None = None,
    compute_advantage: bool = True,
    lookahead: int | None = None,
    episode_indices: list[int] | None = None,
    clip_model_path: str = "openai/clip-vit-base-patch32",
    missing_success: str = "nan",
    task_max_episode_length: int | None = None,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if missing_success not in {"nan", "failure", "success", "error"}:
        raise ValueError(f"Unsupported missing_success mode: {missing_success}")
    if task_max_episode_length is not None and task_max_episode_length < 1:
        raise ValueError("task_max_episode_length must be at least 1")

    dataset, model, preprocessor = load_annotation_resources(
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        reward_model_path=reward_model_path,
        device=device,
        image_key_override=image_key_override,
        precomputed_image_features_path=precomputed_image_features_path,
        clip_model_path=clip_model_path,
    )
    config = model.config
    lookahead = config.advantage_lookahead if lookahead is None else lookahead
    if lookahead < 1:
        raise ValueError("lookahead must be at least 1")

    include_image = config.precomputed_image_features_path is None
    target_index = config.n_obs_steps // 2 if config.temporal_window_mode == "bidirectional" else 0
    all_episodes = [dataset.meta.episodes[index] for index in range(dataset.num_episodes)]
    task_max_lengths = _build_task_max_lengths(all_episodes)
    if episode_indices is None:
        episode_indices = list(range(dataset.num_episodes))
    invalid_episodes = [index for index in episode_indices if not 0 <= index < dataset.num_episodes]
    if invalid_episodes:
        raise ValueError(f"Episode indices out of range: {invalid_episodes}")

    table_data: dict[str, list[Any]] = {
        "index": [],
        "episode_index": [],
        "frame_index": [],
        "value": [],
        "value_bin": [],
        "value_entropy": [],
    }
    all_advantages: list[float] = []
    missing_success_episodes: list[int] = []

    for episode_index in tqdm(episode_indices, desc="Episodes"):
        episode = all_episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        all_indices = np.arange(episode_start, episode_end, dtype=np.int64)
        query_indices = all_indices[::stride].tolist()
        if query_indices[-1] != episode_end - 1:
            query_indices.append(episode_end - 1)

        predicted: dict[int, tuple[float, int, float]] = {}
        for query_batch in _iter_chunks(query_indices, batch_size):
            batch = _build_batch(
                dataset,
                query_batch,
                episode_index=episode_index,
                task=_episode_task(episode),
                image_key=config.image_key,
                state_key=config.state_key,
                include_image=include_image,
            )
            with torch.inference_mode():
                output = model.predict_value_distribution(preprocessor(batch))
                values = output["value_expectation"][:, target_index]
                probabilities = output["value_probs"][:, target_index]
                bins = probabilities.argmax(dim=-1)
                entropies = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

            for offset, index in enumerate(query_batch):
                predicted[index] = (
                    float(values[offset].cpu()),
                    int(bins[offset].cpu()),
                    float(entropies[offset].cpu()),
                )

        computed_indices = np.array(sorted(predicted), dtype=np.int64)
        computed_values = np.array([predicted[index][0] for index in computed_indices])
        computed_bins = np.array([predicted[index][1] for index in computed_indices])
        computed_entropies = np.array([predicted[index][2] for index in computed_indices])

        episode_values = _interpolate(computed_indices, computed_values, all_indices)
        episode_bins = np.rint(_interpolate(computed_indices, computed_bins, all_indices)).astype(np.int64)
        episode_entropies = _interpolate(computed_indices, computed_entropies, all_indices)

        table_data["index"].extend(all_indices.tolist())
        table_data["episode_index"].extend([episode_index] * len(all_indices))
        table_data["frame_index"].extend(range(len(all_indices)))
        table_data["value"].extend(episode_values.tolist())
        table_data["value_bin"].extend(episode_bins.tolist())
        table_data["value_entropy"].extend(episode_entropies.tolist())

        success = _episode_success(episode)
        if compute_advantage and success is None:
            missing_success_episodes.append(episode_index)
            if missing_success == "error":
                raise ValueError(f"Episode {episode_index} is missing success metadata")
            if missing_success != "nan":
                success = missing_success == "success"
        if compute_advantage:
            all_advantages.extend(
                compute_episode_advantages(
                    episode_values,
                    success=success,
                    normalization_length=(
                        task_max_episode_length
                        if task_max_episode_length is not None
                        else _normalization_length(episode, config, task_max_lengths)
                    ),
                    failure_penalty=config.failure_penalty,
                    lookahead=lookahead,
                ).tolist()
            )

    if compute_advantage:
        table_data["advantage"] = all_advantages
    if missing_success_episodes:
        logging.warning(
            "Success metadata is missing for episodes %s; using mode %r.",
            missing_success_episodes,
            missing_success,
        )

    output_columns: dict[str, pa.Array] = {
        "index": pa.array(table_data["index"], type=pa.int64()),
        "episode_index": pa.array(table_data["episode_index"], type=pa.int64()),
        "frame_index": pa.array(table_data["frame_index"], type=pa.int64()),
        "value": pa.array(table_data["value"], type=pa.float32()),
        "value_bin": pa.array(table_data["value_bin"], type=pa.int64()),
        "value_entropy": pa.array(table_data["value_entropy"], type=pa.float32()),
    }
    if compute_advantage:
        output_columns["advantage"] = pa.array(table_data["advantage"], type=pa.float32())
    output = pa.table(output_columns)
    output = output.replace_schema_metadata(
        {
            b"reward_model_path": reward_model_path.encode(),
            b"advantage_lookahead": str(lookahead).encode(),
            b"task_max_episode_length": str(
                task_max_episode_length
                if task_max_episode_length is not None
                else max(task_max_lengths.values())
            ).encode(),
        }
    )

    result_path = (
        Path(dataset.root) / "recap_sarm_values.parquet" if output_path is None else Path(output_path)
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output, result_path)
    logging.info("Saved %d annotations to %s", len(output), result_path)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a LeRobot dataset with RECAP-SARM values.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--reward-model-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-key")
    parser.add_argument("--precomputed-image-features-path")
    parser.add_argument("--clip-model-path", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--missing-success",
        choices=["nan", "failure", "success", "error"],
        default="nan",
    )
    parser.add_argument("--task-max-episode-length", type=int)
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument(
        "--compute-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result_path = annotate_dataset(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        reward_model_path=args.reward_model_path,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        stride=args.stride,
        image_key_override=args.image_key,
        precomputed_image_features_path=args.precomputed_image_features_path,
        compute_advantage=args.compute_advantage,
        lookahead=args.lookahead,
        episode_indices=args.episodes,
        clip_model_path=args.clip_model_path,
        missing_success=args.missing_success,
        task_max_episode_length=args.task_max_episode_length,
    )
    print(f"RECAP-SARM annotations saved to: {result_path}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(result_path),
            path_in_repo="recap_sarm_values.parquet",
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded recap_sarm_values.parquet to {args.dataset_repo_id}")


if __name__ == "__main__":
    main()
