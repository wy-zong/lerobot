#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Compute SARM progress values for RA-BC (Reward-Aware Behavior Cloning) weighting.

This script processes all frames in a dataset with SARM to compute progress values [0, 1].
The results are saved as a parquet file that can be loaded during training for RA-BC weighting.

Uses multi-output extraction: each SARM query returns progress for 9 frames, so we only
need ~num_frames/30 queries instead of one per frame (~30x speedup).

Usage:
    # Full RA-BC computation with visualizations
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Faster computation with stride (compute every 5 frames, interpolate the rest)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --batch-size 16 \\
        --stride 5

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 5

The output is saved to the dataset's local cache directory as 'sarm_progress.parquet'.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel
from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors
from lerobot.rewards.sarm.sarm_utils import normalize_stage_tau


def get_reward_model_path_from_parquet(parquet_path: Path) -> str | None:
    """Read reward_model_path from parquet metadata if available."""
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path).schema.to_arrow_schema().metadata
        if metadata and b"reward_model_path" in metadata:
            return metadata[b"reward_model_path"].decode()
    except Exception:  # nosec B110
        return None
    return None


def load_sarm_resources(
    dataset_repo_id: str,
    reward_model_path: str,
    device: str = "cuda",
    image_key_override: str | None = None,
    precomputed_image_features_path: str | None = None,
    dataset_root: str | Path | None = None,
) -> tuple[LeRobotDataset, SARMRewardModel, any]:
    """
    Load SARM model, dataset, and preprocessor.

    Returns:
        Tuple of (dataset, reward_model, preprocessor)
    """
    logging.info(f"Loading model: {reward_model_path}")
    reward_model = SARMRewardModel.from_pretrained(reward_model_path)
    reward_model.config.device = device
    reward_model.to(device).eval()

    if image_key_override:
        logging.info(f"Overriding image_key from {reward_model.config.image_key} to {image_key_override}")
        reward_model.config.image_key = image_key_override

    if precomputed_image_features_path:
        logging.info(f"Using precomputed image features: {precomputed_image_features_path}")
        reward_model.config.precomputed_image_features_path = precomputed_image_features_path

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    delta_indices = reward_model.config.observation_delta_indices
    use_precomputed_images = reward_model.config.precomputed_image_features_path is not None

    logging.info(f"Loading dataset: {dataset_repo_id}")
    temp_dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        download_videos=not use_precomputed_images,
        skip_video_decode=use_precomputed_images,
    )
    fps = temp_dataset.fps

    delta_timestamps = {
        state_key: [idx / fps for idx in delta_indices],
    }
    if not use_precomputed_images:
        delta_timestamps[image_key] = [idx / fps for idx in delta_indices]

    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=not use_precomputed_images,
        skip_video_decode=use_precomputed_images,
    )
    logging.info(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    preprocess, _ = make_sarm_pre_post_processors(
        config=reward_model.config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )

    return dataset, reward_model, preprocess


def to_numpy_image(img) -> np.ndarray:
    """Convert image tensor to numpy uint8 (H, W, C)."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim == 4:
        # Take center frame for bidirectional sampling
        img = img[img.shape[0] // 2]
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        # Handle normalized images (may have negative values or values > 1)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        img = (img * 255).astype(np.uint8)
    return img


def visualize_episode(
    frames, progress_preds, stage_preds, title, output_path, stage_labels, gt_progress=None, gt_stages=None, display_indices=None
):
    """Create visualization with progress plot, stage probabilities, and sample frames.

    Same as sarm_inference_visualization.py
    """
    num_stages = stage_preds.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_stages))
    frame_indices = np.arange(len(progress_preds))

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    ax_progress, ax_stages, ax_frames = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # Progress plot
    ax_progress.plot(frame_indices, progress_preds, linewidth=2, color="#2E86AB", label="Predicted")
    ax_progress.fill_between(frame_indices, 0, progress_preds, alpha=0.3, color="#2E86AB")
    if gt_progress is not None:
        ax_progress.plot(
            frame_indices, gt_progress, linewidth=2, color="#28A745", linestyle="--", label="Ground Truth"
        )
    ax_progress.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax_progress.set_ylabel("Progress")
    ax_progress.set_title(f'Task: "{title}"', fontweight="bold")
    ax_progress.set_ylim(-0.05, 1.1)
    ax_progress.legend(loc="upper left")
    ax_progress.grid(True, alpha=0.3)

    # Stage predictions
    ax_stages.stackplot(
        frame_indices,
        *[stage_preds[:, i] for i in range(num_stages)],
        colors=colors,
        alpha=0.8,
        labels=stage_labels,
    )
    if gt_stages is not None:
        for change_idx in np.where(np.diff(gt_stages) != 0)[0] + 1:
            ax_stages.axvline(x=change_idx, color="black", linestyle="-", alpha=0.7, linewidth=1.5)
    ax_stages.set_xlabel("Frame")
    ax_stages.set_ylabel("Stage Probability")
    ax_stages.set_ylim(0, 1)
    ax_stages.legend(loc="upper left", ncol=min(num_stages, 5), fontsize=8)
    ax_stages.grid(True, alpha=0.3)

    # Sample frames
    ax_frames.axis("off")
    num_sample = len(frames)
    h, w = frames[0].shape[:2]
    combined = np.zeros((h, w * num_sample, 3), dtype=np.uint8)
    for i in range(num_sample):
        frame = frames[i]
        real_idx = display_indices[i] if display_indices is not None else int(i * (len(progress_preds) - 1) / max(1, num_sample - 1))

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        combined[:, i * w : (i + 1) * w] = frame
        stage_name = stage_labels[np.argmax(stage_preds[real_idx])][:15]
        ax_frames.text(
            i * w + w / 2,
            -10,
            f"Frame {real_idx}\n{progress_preds[real_idx]:.2f}\n{stage_name}",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_frames.imshow(combined)
    ax_frames.set_title("Sample Frames", pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_sarm_predictions(
    dataset: LeRobotDataset,
    reward_model: SARMRewardModel,
    preprocess,
    episode_indices: list[int],
    head_mode: str,
    output_dir: Path,
    num_display_frames: int = 5,
    stride: int = 1,
):
    """
    Visualize SARM predictions for multiple episodes.

    Computes predictions for every frame by default. With stride > 1, computes predictions
    every N frames and interpolates (progress + stage probabilities) for visualization.

    Args:
        dataset: LeRobotDataset with delta_timestamps configured
        reward_model: Loaded SARM model
        preprocess: Preprocessor from make_sarm_pre_post_processors
        episode_indices: List of episode indices to visualize
        head_mode: "sparse", "dense", or "both"
        output_dir: Directory to save visualizations
        num_display_frames: Number of frames to display in thumbnail strip (default: 5)
        stride: Compute predictions every N frames, interpolate the rest (default: 1)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    dual_mode = reward_model.config.uses_dual_heads
    device = reward_model.device

    # Center frame index for bidirectional sampling
    target_idx = reward_model.config.n_obs_steps // 2

    # Determine which heads to visualize
    schemes_to_viz = []
    if head_mode in ("sparse", "both") or not dual_mode:
        schemes_to_viz.append("sparse")
    if head_mode in ("dense", "both") and dual_mode:
        schemes_to_viz.append("dense")

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    for episode_idx in episode_indices:
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        task = dataset[ep_start].get("task", "perform the task")
        num_frames = ep_end - ep_start

        # Select frames for display thumbnails (evenly sampled from begin to end)
        display_indices = set(
            [
                ep_start + int(i * (num_frames - 1) / (num_display_frames - 1))
                for i in range(num_display_frames)
            ]
            if num_frames >= num_display_frames
            else list(range(ep_start, ep_end))
        )
        viz_frames = {}

        # Load display frames up-front (stride mode might skip them otherwise).
        for frame_idx in display_indices:
            sample = dataset[frame_idx]
            viz_frames[frame_idx] = to_numpy_image(sample[image_key])

        # Initialize storage for each scheme
        scheme_data = {}
        for scheme in schemes_to_viz:
            num_stages = getattr(reward_model.config, f"num_{scheme}_stages")
            scheme_data[scheme] = {
                "viz_progress": np.full(num_frames, np.nan),
                "viz_stages": np.full((num_frames, num_stages), np.nan),
                "viz_gt_progress": np.full(num_frames, np.nan),
                "viz_gt_stages": np.full(num_frames, np.nan),
                "target_key": f"{scheme}_targets",
                "num_stages": num_stages,
                "temporal_props": getattr(reward_model.config, f"{scheme}_temporal_proportions"),
                "subtask_names": getattr(reward_model.config, f"{scheme}_subtask_names"),
            }

        if stride > 1:
            logging.info(f"Visualization stride={stride}: inferring every {stride} frames and interpolating")

        # Process frames one at a time to avoid memory buildup
        frame_indices = list(range(ep_start, ep_end, stride))
        if (ep_end - 1) not in frame_indices:
            frame_indices.append(ep_end - 1)
        frame_indices = sorted(set(frame_indices))

        for frame_idx in tqdm(frame_indices, desc=f"Episode {episode_idx}", leave=False):
            local_idx = frame_idx - ep_start
            sample = dataset[frame_idx]

            batch = {
                image_key: sample[image_key],
                "task": task,
                "index": frame_idx,
                "episode_index": episode_idx,
            }
            if state_key in sample:
                batch[state_key] = sample[state_key]

            with torch.no_grad():
                processed = preprocess(batch)
                video_features = processed["video_features"].to(device)
                text_features = processed["text_features"].to(device)
                state_features = processed.get("state_features")
                if state_features is not None:
                    state_features = state_features.to(device)
                lengths = processed.get("lengths")

                for scheme in schemes_to_viz:
                    sd = scheme_data[scheme]

                    # Ground truth
                    # In stride visualization mode, ground-truth plots can be misleading
                    # (only sparse points are available), so we skip GT.
                    if stride == 1 and sd["target_key"] in processed:
                        gt_target = processed[sd["target_key"]][0, target_idx].cpu().item()
                        sd["viz_gt_stages"][local_idx] = int(gt_target)
                        sd["viz_gt_progress"][local_idx] = normalize_stage_tau(
                            gt_target,
                            num_stages=sd["num_stages"],
                            temporal_proportions=sd["temporal_props"],
                            subtask_names=sd["subtask_names"],
                        )

                    # Predictions
                    reward, stage_probs = reward_model.calculate_rewards(
                        text_embeddings=text_features,
                        video_embeddings=video_features,
                        state_features=state_features,
                        lengths=lengths,
                        return_all_frames=True,
                        return_stages=True,
                        head_mode=scheme,
                    )

                    # Handle both tensor and numpy outputs
                    if isinstance(reward, torch.Tensor):
                        reward = reward.cpu().numpy()
                        stage_probs = stage_probs.cpu().numpy()

                    if reward.ndim == 2:
                        sd["viz_progress"][local_idx] = reward[0, target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[0, target_idx, :]
                    else:
                        sd["viz_progress"][local_idx] = reward[target_idx]
                        sd["viz_stages"][local_idx] = stage_probs[target_idx, :]

                # Clear GPU memory after each frame
                del processed, video_features, text_features
                if state_features is not None:
                    del state_features

            torch.cuda.empty_cache()

        # Interpolate predictions back to per-frame arrays for smooth visualization.
        if stride > 1:
            all_local = np.arange(num_frames)
            for scheme in schemes_to_viz:
                sd = scheme_data[scheme]

                valid = np.isfinite(sd["viz_progress"])
                valid_idx = np.where(valid)[0]
                if valid_idx.size >= 1:
                    sd["viz_progress"] = interpolate_progress(
                        valid_idx, sd["viz_progress"][valid_idx], all_local
                    )

                    stage_interp = np.zeros_like(sd["viz_stages"], dtype=np.float32)
                    for s in range(sd["num_stages"]):
                        stage_interp[:, s] = interpolate_progress(
                            valid_idx, sd["viz_stages"][valid_idx, s], all_local
                        )

                    stage_interp = np.clip(stage_interp, 0.0, 1.0)
                    row_sums = stage_interp.sum(axis=1, keepdims=True)
                    nz = row_sums.squeeze(-1) > 0
                    stage_interp[nz] = stage_interp[nz] / row_sums[nz]
                    sd["viz_stages"] = stage_interp
                else:
                    # No valid points: keep NaNs/zeros; visualization will be empty.
                    sd["viz_stages"] = np.nan_to_num(sd["viz_stages"], nan=0.0)

        # Generate visualization for each head
        ordered_display_indices = sorted(display_indices)
        ordered_viz_frames = [viz_frames[idx] for idx in ordered_display_indices]
        # Make indices relative to the episode start for stage_preds/progress_preds array indexing
        relative_display_indices = [idx - ep_start for idx in ordered_display_indices]
        for scheme in schemes_to_viz:
            sd = scheme_data[scheme]
            stage_labels = sd["subtask_names"] or [f"Stage {i + 1}" for i in range(sd["num_stages"])]
            viz_path = output_dir / f"sarm_prediction_ep{episode_idx}_{scheme}.png"

            visualize_episode(
                frames=np.array(ordered_viz_frames),
                progress_preds=sd["viz_progress"],
                stage_preds=sd["viz_stages"],
                title=f"{task} (Episode {episode_idx})",
                output_path=viz_path,
                stage_labels=stage_labels,
                gt_progress=sd["viz_gt_progress"] if not np.all(np.isnan(sd["viz_gt_progress"])) else None,
                gt_stages=sd["viz_gt_stages"] if not np.all(np.isnan(sd["viz_gt_stages"])) else None,
                display_indices=relative_display_indices,
            )

        # Clear memory between episodes
        torch.cuda.empty_cache()

    logging.info(f"Visualizations saved to: {output_dir.absolute()}")


def generate_all_frame_indices(ep_start: int, ep_end: int, frame_gap: int = 30) -> list[int]:
    """Generate all frame indices, ordered by offset for cache-friendly access.

    Orders frames as: [0, 30, 60...], [1, 31, 61...], ..., [29, 59, 89...]
    This groups frames that share similar temporal windows together.
    """
    num_frames = ep_end - ep_start
    indices = []
    for offset in range(frame_gap):
        for frame_rel in range(offset, num_frames, frame_gap):
            indices.append(ep_start + frame_rel)
    return indices


def interpolate_progress(
    computed_indices: np.ndarray,
    computed_values: np.ndarray,
    all_indices: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate values to fill in gaps (robust to NaNs / edge cases)."""
    computed_indices = np.asarray(computed_indices)
    computed_values = np.asarray(computed_values)
    all_indices = np.asarray(all_indices)

    mask = np.isfinite(computed_values)
    if mask.sum() == 0:
        return np.full(all_indices.shape, np.nan, dtype=np.float32)
    if mask.sum() == 1:
        return np.full(all_indices.shape, float(computed_values[mask][0]), dtype=np.float32)

    out = np.interp(all_indices, computed_indices[mask], computed_values[mask])
    return out.astype(np.float32)


def iter_chunks(items: list[int], chunk_size: int):
    """Yield fixed-size chunks from a list."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _stack_sample_values(values: list):
    """Stack per-frame dataset values into a batch while preserving tensor/array type."""
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    return values


def build_sarm_batch(
    dataset: LeRobotDataset,
    query_indices: list[int],
    image_key: str,
    state_key: str,
    task: str,
    episode_idx: int,
    include_image: bool = True,
) -> dict:
    """Load dataset samples and collate them into a SARM preprocessing batch."""
    samples = [dataset[idx] for idx in query_indices]
    batch = {
        "task": task,
        "index": torch.tensor(query_indices, dtype=torch.long),
        "episode_index": torch.full((len(query_indices),), episode_idx, dtype=torch.long),
    }
    if include_image:
        batch[image_key] = _stack_sample_values([sample[image_key] for sample in samples])
    if state_key in samples[0]:
        batch[state_key] = _stack_sample_values([sample[state_key] for sample in samples])
    return batch


def infer_sarm_progress_batch(
    batch: dict,
    query_indices: list[int],
    reward_model: SARMRewardModel,
    preprocess,
    device: str,
    center_idx: int,
    compute_sparse: bool,
    compute_dense: bool,
) -> dict[int, tuple[float, float]]:
    """Run SARM progress inference for a batch of query frames."""
    with torch.no_grad():
        processed = preprocess(batch)
        video_features = processed["video_features"].to(device)
        text_features = processed["text_features"].to(device)
        state_features = processed.get("state_features")
        if state_features is not None:
            state_features = state_features.to(device)
        lengths = processed.get("lengths")

        sparse_vals = np.full(len(query_indices), np.nan, dtype=np.float32)
        dense_vals = np.full(len(query_indices), np.nan, dtype=np.float32)

        if compute_sparse:
            sparse_progress = reward_model.calculate_rewards(
                text_embeddings=text_features,
                video_embeddings=video_features,
                state_features=state_features,
                lengths=lengths,
                return_all_frames=True,
                head_mode="sparse",
            )
            sparse_progress = np.asarray(sparse_progress)
            if sparse_progress.ndim == 1:
                sparse_vals[0] = sparse_progress[center_idx]
            else:
                sparse_vals = sparse_progress[:, center_idx].astype(np.float32)

        if compute_dense:
            dense_progress = reward_model.calculate_rewards(
                text_embeddings=text_features,
                video_embeddings=video_features,
                state_features=state_features,
                lengths=lengths,
                return_all_frames=True,
                head_mode="dense",
            )
            dense_progress = np.asarray(dense_progress)
            if dense_progress.ndim == 1:
                dense_vals[0] = dense_progress[center_idx]
            else:
                dense_vals = dense_progress[:, center_idx].astype(np.float32)

        del processed, video_features, text_features
        if state_features is not None:
            del state_features

    return {
        query_idx: (float(sparse_vals[i]), float(dense_vals[i]))
        for i, query_idx in enumerate(query_indices)
    }


def compute_sarm_progress(
    dataset_repo_id: str,
    reward_model_path: str,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str = "cuda",
    num_visualizations: int = 5,
    output_dir: str = "./sarm_viz",
    stride: int = 1,
    image_key_override: str | None = None,
    batch_size: int = 1,
    precomputed_image_features_path: str | None = None,
    dataset_root: str | Path | None = None,
):
    """
    Compute SARM progress predictions for all frames in a dataset.

    Args:
        dataset_repo_id: HuggingFace dataset repo ID or local path
        reward_model_path: Path to pretrained SARM model
        dataset_root: Optional local dataset root
        output_path: Path to save results. If None, saves to dataset's cache directory
        head_mode: SARM head to use ("sparse", "dense", or "both")
        device: Device to use for inference
        num_visualizations: Number of episodes to visualize (0 to skip)
        output_dir: Directory to save visualizations
        stride: Compute progress every N frames, interpolate the rest (default: 1 = every frame)
        image_key_override: Override the image key from the model config
        batch_size: Number of query frames to process per model/preprocessor call
        precomputed_image_features_path: Path to precomputed CLIP image features memmap
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    dataset, reward_model, preprocess = load_sarm_resources(
        dataset_repo_id,
        reward_model_path,
        device,
        image_key_override,
        precomputed_image_features_path,
        dataset_root,
    )

    # Set preprocessor to eval mode to disable augmentations
    if hasattr(preprocess, "eval"):
        preprocess.eval()
    for step in preprocess.steps:
        if hasattr(step, "eval"):
            step.eval()

    image_key = reward_model.config.image_key
    state_key = reward_model.config.state_key
    frame_gap = reward_model.config.frame_gap
    include_image = reward_model.config.precomputed_image_features_path is None
    if not include_image and num_visualizations > 0:
        logging.warning(
            "Skipping visualizations because precomputed image features are enabled and images are not loaded. "
            "Run visualization separately without --precomputed-image-features-path if needed."
        )
        num_visualizations = 0
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    logging.info(f"Processing {total_frames} frames across {num_episodes} episodes")

    # Determine which heads to compute
    dual_mode = reward_model.config.uses_dual_heads
    compute_sparse = head_mode in ("sparse", "both") or not dual_mode
    compute_dense = head_mode in ("dense", "both") and dual_mode

    # Storage arrays
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_progress_sparse = [] if compute_sparse else None
    all_progress_dense = [] if compute_dense else None

    if stride > 1:
        logging.info(f"Using stride={stride}: computing every {stride} frames, interpolating the rest")

    # Process all episodes
    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep = dataset.meta.episodes[episode_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Get task description
        task = dataset[ep_start].get("task", "perform the task")

        # Generate frames to compute (with stride applied)
        all_ep_indices = generate_all_frame_indices(ep_start, ep_end, frame_gap)
        if stride > 1:
            # Only compute every stride-th frame (relative to episode start)
            compute_indices = [idx for idx in all_ep_indices if (idx - ep_start) % stride == 0]
            # Always include last frame for better interpolation at episode end
            last_frame = ep_end - 1
            if last_frame not in compute_indices:
                compute_indices.append(last_frame)
            compute_indices = sorted(set(compute_indices))
        else:
            compute_indices = all_ep_indices

        center_idx = reward_model.config.n_obs_steps // 2  # Center of bidirectional window

        # Dictionary to collect results
        frame_results = {}

        for query_batch in tqdm(
            list(iter_chunks(compute_indices, batch_size)),
            desc=f"  Ep {episode_idx}",
            leave=False,
        ):
            try:
                batch = build_sarm_batch(
                    dataset, query_batch, image_key, state_key, task, episode_idx, include_image
                )
                frame_results.update(
                    infer_sarm_progress_batch(
                        batch,
                        query_batch,
                        reward_model,
                        preprocess,
                        device,
                        center_idx,
                        compute_sparse,
                        compute_dense,
                    )
                )
            except Exception as e:
                if len(query_batch) == 1:
                    logging.warning(f"Failed to process frame {query_batch[0]}: {e}")
                    continue

                logging.warning(
                    f"Failed to process batch {query_batch[0]}-{query_batch[-1]} "
                    f"(size={len(query_batch)}), retrying one frame at a time: {e}"
                )
                for query_idx in query_batch:
                    try:
                        batch = build_sarm_batch(
                            dataset, [query_idx], image_key, state_key, task, episode_idx, include_image
                        )
                        frame_results.update(
                            infer_sarm_progress_batch(
                                batch,
                                [query_idx],
                                reward_model,
                                preprocess,
                                device,
                                center_idx,
                                compute_sparse,
                                compute_dense,
                            )
                        )
                    except Exception as frame_error:
                        logging.warning(f"Failed to process frame {query_idx}: {frame_error}")

        # Interpolate to get values for all frames
        computed_indices = np.array(sorted(frame_results.keys()))
        computed_sparse = (
            np.array([frame_results[i][0] for i in computed_indices]) if compute_sparse else None
        )
        computed_dense = np.array([frame_results[i][1] for i in computed_indices]) if compute_dense else None

        # All frame indices for this episode
        all_frame_idx_array = np.arange(ep_start, ep_end)

        if stride > 1 and len(computed_indices) > 1:
            # Interpolate progress values
            if compute_sparse:
                interp_sparse = interpolate_progress(computed_indices, computed_sparse, all_frame_idx_array)
            if compute_dense:
                interp_dense = interpolate_progress(computed_indices, computed_dense, all_frame_idx_array)
        else:
            # No interpolation needed
            interp_sparse = computed_sparse if compute_sparse else None
            interp_dense = computed_dense if compute_dense else None

        # Store results for all frames
        for i, frame_idx in enumerate(all_frame_idx_array):
            local_idx = frame_idx - ep_start
            all_indices.append(frame_idx)
            all_episode_indices.append(episode_idx)
            all_frame_indices.append(local_idx)
            if compute_sparse:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_sparse.append(float(interp_sparse[i]))
                elif frame_idx in frame_results:
                    all_progress_sparse.append(frame_results[frame_idx][0])
                else:
                    all_progress_sparse.append(np.nan)
            if compute_dense:
                if stride > 1 and len(computed_indices) > 1:
                    all_progress_dense.append(float(interp_dense[i]))
                elif frame_idx in frame_results:
                    all_progress_dense.append(frame_results[frame_idx][1])
                else:
                    all_progress_dense.append(np.nan)

    # Create output table
    table_data = {
        "index": np.array(all_indices, dtype=np.int64),
        "episode_index": np.array(all_episode_indices, dtype=np.int64),
        "frame_index": np.array(all_frame_indices, dtype=np.int64),
    }
    if compute_sparse:
        table_data["progress_sparse"] = np.array(all_progress_sparse, dtype=np.float32)
    if compute_dense:
        table_data["progress_dense"] = np.array(all_progress_dense, dtype=np.float32)

    # Sort by index
    df = pa.table(table_data).to_pandas()
    df = df.sort_values("index").reset_index(drop=True)
    final_table = pa.Table.from_pandas(df, preserve_index=False)

    # Add metadata with reward model path
    metadata = {b"reward_model_path": reward_model_path.encode()}
    final_table = final_table.replace_schema_metadata(metadata)

    # Determine output path
    output_path = Path(dataset.root) / "sarm_progress.parquet" if output_path is None else Path(output_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(final_table, output_path)
    logging.info(f"Saved {len(final_table)} frame progress values to {output_path}")

    # Print statistics
    if "progress_sparse" in df.columns:
        valid = df["progress_sparse"].dropna()
        logging.info(
            f"Sparse progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    if "progress_dense" in df.columns:
        valid = df["progress_dense"].dropna()
        logging.info(
            f"Dense progress: mean={valid.mean():.4f}, std={valid.std():.4f}, "
            f"min={valid.min():.4f}, max={valid.max():.4f}"
        )

    # Visualize episodes after processing
    if num_visualizations > 0:
        viz_episodes = list(range(min(num_visualizations, num_episodes)))
        logging.info(f"Generating {len(viz_episodes)} visualizations...")
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=head_mode,
            output_dir=Path(output_dir),
            stride=stride,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute SARM progress values for RA-BC weighting or visualize SARM predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full RA-BC computation with visualizations
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4

    # Visualize predictions only (no RA-BC computation)
    python src/lerobot/rewards/sarm/compute_rabc_weights.py \\
        --dataset-repo-id lerobot/aloha_sim_insertion_human \\
        --reward-model-path <USER>/sarm_single_uni4 \\
        --visualize-only \\
        --num-visualizations 10
        """,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repo ID or local path",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional local dataset root (avoids downloading or resolving the dataset from the Hub)",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to pretrained SARM model (reads from existing parquet metadata if not provided)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for parquet. If not set, saves to dataset's cache directory",
    )
    parser.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="SARM head to use (default: sparse)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    # Visualization options
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize SARM predictions (no RA-BC computation)",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of episodes to visualize (default: 5, set to 0 to skip)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sarm_viz",
        help="Output directory for visualizations (default: ./sarm_viz)",
    )
    parser.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        help="Upload progress file to the dataset repo on HuggingFace Hub (use --no-push-to-hub to skip)",
        default=True,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute progress every N frames, interpolate the rest (default: 1 = every frame)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of query frames to process per inference batch (default: 1)",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Override the image key from the model config (useful if dataset uses a different camera name)",
    )
    parser.add_argument(
        "--precomputed-image-features-path",
        type=str,
        default=None,
        help="Path to .npy CLIP image features from lerobot-sarm-precompute-clip",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Try to get reward_model_path from parquet metadata if not provided
    reward_model_path = args.reward_model_path
    if reward_model_path is None:
        # Load dataset to find parquet path
        temp_dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=args.dataset_root,
            download_videos=False,
        )
        parquet_path = Path(temp_dataset.root) / "sarm_progress.parquet"
        reward_model_path = get_reward_model_path_from_parquet(parquet_path)
        if reward_model_path:
            logging.info(f"Using reward model from parquet metadata: {reward_model_path}")
        else:
            raise ValueError(
                "--reward-model-path is required (no existing parquet with model metadata found)"
            )

    # Handle visualize-only mode
    if args.visualize_only:
        dataset, reward_model, preprocess = load_sarm_resources(
            args.dataset_repo_id,
            reward_model_path,
            args.device,
            args.image_key,
            args.precomputed_image_features_path,
            args.dataset_root,
        )
        logging.info(f"Visualization-only mode: visualizing {args.num_visualizations} episodes")
        viz_episodes = list(range(min(args.num_visualizations, dataset.num_episodes)))
        visualize_sarm_predictions(
            dataset=dataset,
            reward_model=reward_model,
            preprocess=preprocess,
            episode_indices=viz_episodes,
            head_mode=args.head_mode,
            output_dir=Path(args.output_dir),
            stride=args.stride,
        )
        print(f"\nVisualizations saved to: {Path(args.output_dir).absolute()}")
        return

    # Full RABC computation (compute_sarm_progress loads model/dataset itself)
    output_path = compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=reward_model_path,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        head_mode=args.head_mode,
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.output_dir,
        stride=args.stride,
        image_key_override=args.image_key,
        batch_size=args.batch_size,
        precomputed_image_features_path=args.precomputed_image_features_path,
    )

    print(f"\nSARM progress values saved to: {output_path}")

    # Upload to Hub if requested
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        hub_path = "sarm_progress.parquet"

        print(f"\nUploading to Hub: {args.dataset_repo_id}/{hub_path}")
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=hub_path,
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
        )
        print(
            f"Successfully uploaded to: https://huggingface.co/datasets/{args.dataset_repo_id}/blob/main/{hub_path}"
        )

        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: hf://datasets/{args.dataset_repo_id}/{hub_path}")
        print("  rabc_head_mode: sparse  # or dense")
    else:
        print("\nTo use in training, add to your config:")
        print("  use_rabc: true")
        print(f"  rabc_progress_path: {output_path}")
        print("  rabc_head_mode: sparse  # or dense")


if __name__ == "__main__":
    main()
