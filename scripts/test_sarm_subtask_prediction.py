#!/usr/bin/env python

"""Run repeated sampled SARM dense-subtask prediction over a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pa_dataset
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel

DEFAULT_CHECKPOINT = Path(
    "/home/wy/outputs/train/sarm_current_only_full_dataset/checkpoints/100000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path(
    "/home/wy/.cache/huggingface/lerobot/wuc1/bi_so101_ffp_20260603_200349_subtask"
)
DEFAULT_OUTPUT_DIR = Path("/home/wy/outputs/sarm_subtask_prediction_eval")
DEFAULT_IMAGE_KEY = "observation.images.left_camera1"
DEFAULT_CLIP_FEATURES = Path("meta/clip_features_left_camera1.npy")
STATE_KEY = "observation.state"
REQUIRED_OUTPUT_COLUMNS = [
    "index",
    "episode_index",
    "frame_index",
    "run_id",
    "gt_subtask_index",
    "gt_subtask_name",
    "pred_subtask_index",
    "pred_subtask_name",
    "pred_confidence",
    "progress_dense",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense SARM subtask prediction on every Nth global frame, repeat the pass, "
            "and write per-frame predictions plus summary/consistency metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-features", type=Path, default=DEFAULT_CLIP_FEATURES)
    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--state-key", default=STATE_KEY)
    parser.add_argument("--head-mode", default="dense", choices=["sparse", "dense"])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda or cpu.")
    parser.add_argument("--max-sampled-rows", type=int, default=None, help="Limit sampled rows for smoke tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--clip-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load openai/clip-vit-base-patch32 only from the local transformers cache.",
    )
    return parser.parse_args()


def resolve_clip_features_path(dataset_root: Path, clip_features: Path) -> Path:
    return clip_features if clip_features.is_absolute() else dataset_root / clip_features


def validate_inputs(args: argparse.Namespace) -> Path:
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.runs <= 0:
        raise ValueError(f"--runs must be positive, got {args.runs}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint config.json not found in: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not (args.dataset_root / "data").is_dir():
        raise FileNotFoundError(f"Dataset data directory not found: {args.dataset_root / 'data'}")
    clip_features_path = resolve_clip_features_path(args.dataset_root, args.clip_features)
    if not clip_features_path.is_file():
        raise FileNotFoundError(f"CLIP feature memmap not found: {clip_features_path}")
    return clip_features_path


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_frame_table(dataset_root: Path, state_key: str) -> pd.DataFrame:
    columns = ["index", "episode_index", "frame_index", "task_index", "subtask_index", state_key]
    dataset = pa_dataset.dataset(dataset_root / "data", format="parquet")
    table = dataset.to_table(columns=columns)
    frame_table = table.to_pandas()
    frame_table = frame_table.sort_values("index", kind="stable").reset_index(drop=True)
    return frame_table


def load_indexed_names(path: Path, id_column: str) -> dict[int, str]:
    df = pd.read_parquet(path)
    return {int(row[id_column]): str(index) for index, row in df.iterrows()}


def stack_state(values: Iterable[Any]) -> np.ndarray:
    stacked = np.stack([np.asarray(value, dtype=np.float32) for value in values])
    if stacked.ndim != 2:
        raise ValueError(f"Expected state array with shape (N, D), got {stacked.shape}")
    return stacked


def normalize_state(state: np.ndarray, checkpoint: Path, state_key: str, eps: float = 1e-8) -> np.ndarray:
    stats_path = checkpoint / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalizer stats not found: {stats_path}")
    stats = load_file(stats_path, device="cpu")
    mean_key = f"{state_key}.mean"
    std_key = f"{state_key}.std"
    if mean_key not in stats or std_key not in stats:
        raise KeyError(f"Missing {mean_key} or {std_key} in {stats_path}")
    mean = stats[mean_key].numpy().astype(np.float32)
    std = stats[std_key].numpy().astype(np.float32)
    if state.shape[1] != mean.shape[0]:
        raise ValueError(f"State dim {state.shape[1]} does not match normalizer dim {mean.shape[0]}")
    return (state - mean) / (std + eps)


@torch.no_grad()
def load_text_embeddings(
    task_names: dict[int, str],
    *,
    local_files_only: bool,
) -> dict[int, np.ndarray]:
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", local_files_only=local_files_only
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=local_files_only,
        use_fast=True,
    )
    clip_model.eval()

    embeddings: dict[int, np.ndarray] = {}
    for task_index, task_name in task_names.items():
        inputs = clip_processor.tokenizer([task_name], return_tensors="pt", padding=True, truncation=True)
        output = clip_model.get_text_features(**inputs)
        if not isinstance(output, torch.Tensor):
            output = output.pooler_output
            if output is None:
                raise ValueError("CLIP text feature output does not contain pooler_output.")
        embeddings[task_index] = output.detach().cpu().numpy().astype(np.float32)[0]
    return embeddings


def load_sarm_model(checkpoint: Path, device: str | None) -> SARMRewardModel:
    cfg = RewardModelConfig.from_pretrained(checkpoint)
    if device is not None:
        cfg.device = device
    model = SARMRewardModel.from_pretrained(checkpoint, config=cfg, strict=True)
    model.eval()
    return model


def batch_task_embeddings(task_indices: np.ndarray, embeddings: dict[int, np.ndarray]) -> np.ndarray:
    try:
        return np.stack([embeddings[int(task_index)] for task_index in task_indices]).astype(np.float32)
    except KeyError as exc:
        raise KeyError(f"No task text embedding found for task_index={exc.args[0]}") from exc


def predict_run(
    *,
    model: SARMRewardModel,
    sampled: pd.DataFrame,
    normalized_state: np.ndarray,
    clip_features: np.ndarray,
    task_embeddings: dict[int, np.ndarray],
    subtask_names: dict[int, str],
    batch_size: int,
    run_id: int,
    head_mode: str,
) -> pd.DataFrame:
    if head_mode == "dense":
        pred_names = dict(enumerate(model.config.dense_subtask_names or []))
    else:
        pred_names = dict(enumerate(model.config.sparse_subtask_names or []))

    rows: list[pd.DataFrame] = []
    frame_index = 0 if model.config.temporal_window_mode == "current_only" else model.config.n_obs_steps
    row_positions = sampled["_row_position"].to_numpy(dtype=np.int64)
    global_indices = sampled["index"].to_numpy(dtype=np.int64)
    task_indices = sampled["task_index"].to_numpy(dtype=np.int64)

    for start in tqdm(range(0, len(sampled), batch_size), desc=f"run_{run_id}", dynamic_ncols=True):
        end = min(start + batch_size, len(sampled))
        batch_positions = row_positions[start:end]
        batch_indices = global_indices[start:end]

        video = np.asarray(clip_features[batch_indices], dtype=np.float32)[:, None, :]
        state = normalized_state[batch_positions][:, None, :]
        text = batch_task_embeddings(task_indices[start:end], task_embeddings)
        lengths = np.ones(end - start, dtype=np.int32)

        progress, stage_probs, confidence = model.calculate_rewards(
            text,
            video,
            state,
            lengths=lengths,
            return_all_frames=False,
            return_stages=True,
            return_confidence=True,
            head_mode=head_mode,
            frame_index=frame_index,
        )
        progress = np.asarray(progress, dtype=np.float32)
        probs_at_frame = np.asarray(stage_probs, dtype=np.float32)[:, frame_index, :]
        confidence_at_frame = np.asarray(confidence, dtype=np.float32)[:, frame_index]
        pred_indices = probs_at_frame.argmax(axis=-1).astype(np.int64)

        batch = sampled.iloc[start:end]
        result = pd.DataFrame(
            {
                "index": batch["index"].to_numpy(dtype=np.int64),
                "episode_index": batch["episode_index"].to_numpy(dtype=np.int64),
                "frame_index": batch["frame_index"].to_numpy(dtype=np.int64),
                "run_id": np.full(end - start, run_id, dtype=np.int64),
                "gt_subtask_index": batch["subtask_index"].to_numpy(dtype=np.int64),
                "gt_subtask_name": [
                    subtask_names.get(int(idx), "") for idx in batch["subtask_index"].to_numpy()
                ],
                "pred_subtask_index": pred_indices,
                "pred_subtask_name": [pred_names.get(int(idx), "") for idx in pred_indices],
                "pred_confidence": confidence_at_frame,
                "progress_dense": progress,
            }
        )
        rows.append(result)

    if not rows:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    np.add.at(matrix, (gt[valid], pred[valid]), 1)
    return matrix.tolist()


def compute_run_metrics(predictions: pd.DataFrame, *, num_classes: int, expected_rows: int) -> dict[str, Any]:
    gt = predictions["gt_subtask_index"].to_numpy(dtype=np.int64)
    pred = predictions["pred_subtask_index"].to_numpy(dtype=np.int64)
    confidence = predictions["pred_confidence"].to_numpy(dtype=np.float32)
    progress = predictions["progress_dense"].to_numpy(dtype=np.float32)

    valid_gt = (gt >= 0) & (gt < num_classes)
    valid_pred = (pred >= 0) & (pred < num_classes)
    finite = np.isfinite(confidence) & np.isfinite(progress)
    valid = valid_gt & valid_pred & finite
    overall_accuracy = float((gt[valid] == pred[valid]).mean()) if valid.any() else float("nan")

    per_subtask_accuracy: dict[str, float] = {}
    present_accuracies: list[float] = []
    for subtask_idx in range(num_classes):
        mask = valid & (gt == subtask_idx)
        accuracy = float((pred[mask] == subtask_idx).mean()) if mask.any() else float("nan")
        per_subtask_accuracy[f"accuracy_subtask_{subtask_idx}"] = accuracy
        if not np.isnan(accuracy):
            present_accuracies.append(accuracy)

    invalid_count = int((~(valid_pred & finite)).sum())
    return {
        "run_id": int(predictions["run_id"].iloc[0]) if len(predictions) else None,
        "sampled_rows": int(len(predictions)),
        "expected_rows": int(expected_rows),
        "overall_accuracy": overall_accuracy,
        "macro_accuracy": float(np.mean(present_accuracies)) if present_accuracies else float("nan"),
        "mean_confidence": float(np.nanmean(confidence)) if len(confidence) else float("nan"),
        "invalid_prediction_count": invalid_count,
        **per_subtask_accuracy,
        "confusion_matrix": json.dumps(confusion_matrix(gt, pred, num_classes)),
    }


def compute_consistency(run_predictions: list[pd.DataFrame], num_classes: int) -> dict[str, Any]:
    if not run_predictions:
        return {}

    reference_indices = run_predictions[0]["index"].to_numpy(dtype=np.int64)
    same_index_sets = []
    for predictions in run_predictions:
        same_index_sets.append(bool(np.array_equal(reference_indices, predictions["index"].to_numpy(dtype=np.int64))))

    pred_matrix = np.stack(
        [predictions["pred_subtask_index"].to_numpy(dtype=np.int64) for predictions in run_predictions],
        axis=1,
    )
    progress_matrix = np.stack(
        [predictions["progress_dense"].to_numpy(dtype=np.float32) for predictions in run_predictions],
        axis=1,
    )
    agreement_mask = np.all(pred_matrix == pred_matrix[:, :1], axis=1)
    gt = run_predictions[0]["gt_subtask_index"].to_numpy(dtype=np.int64)

    per_subtask_agreement: dict[str, float] = {}
    for subtask_idx in range(num_classes):
        mask = gt == subtask_idx
        per_subtask_agreement[str(subtask_idx)] = (
            float(agreement_mask[mask].mean()) if mask.any() else None
        )

    differing_rows = np.where(~agreement_mask)[0]
    differing_frames = []
    for row_idx in differing_rows:
        differing_frames.append(
            {
                "index": int(reference_indices[row_idx]),
                "episode_index": int(run_predictions[0]["episode_index"].iloc[row_idx]),
                "frame_index": int(run_predictions[0]["frame_index"].iloc[row_idx]),
                "gt_subtask_index": int(gt[row_idx]),
                "pred_subtask_index_by_run": [
                    int(pred_matrix[row_idx, run_idx]) for run_idx in range(pred_matrix.shape[1])
                ],
                "progress_dense_by_run": [
                    float(progress_matrix[row_idx, run_idx]) for run_idx in range(progress_matrix.shape[1])
                ],
            }
        )

    progress_diffs = progress_matrix.max(axis=1) - progress_matrix.min(axis=1)
    finite_progress_diffs = progress_diffs[np.isfinite(progress_diffs)]

    return {
        "run_count": len(run_predictions),
        "same_sampled_indices_all_runs": bool(all(same_index_sets)),
        "sampled_rows": int(len(reference_indices)),
        "sampled_frame_agreement_rate": float(agreement_mask.mean()) if len(agreement_mask) else None,
        "per_subtask_agreement_rate": per_subtask_agreement,
        "max_abs_progress_dense_difference": (
            float(finite_progress_diffs.max()) if finite_progress_diffs.size else None
        ),
        "differing_frame_count": int(len(differing_rows)),
        "differing_frames": differing_frames,
    }


def write_plots(
    *,
    run_predictions: list[pd.DataFrame],
    output_dir: Path,
    num_classes: int,
    subtask_names: dict[int, str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib is not installed; skipping plots.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for predictions in run_predictions:
        run_id = int(predictions["run_id"].iloc[0])
        gt = predictions["gt_subtask_index"].to_numpy(dtype=np.int64)
        pred = predictions["pred_subtask_index"].to_numpy(dtype=np.int64)
        matrix = np.asarray(confusion_matrix(gt, pred, num_classes), dtype=np.int64)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Run {run_id} dense subtask confusion")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        labels = [subtask_names.get(idx, str(idx)) for idx in range(num_classes)]
        ax.set_xticks(np.arange(num_classes), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(num_classes), labels=labels)
        for row in range(num_classes):
            for col in range(num_classes):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / f"run_{run_id}_confusion.png", dpi=150)
        plt.close(fig)

    first = run_predictions[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(first["index"].to_numpy(), first["progress_dense"].to_numpy(), linewidth=0.8)
    ax.set_title("Run 1 dense progress over sampled global frames")
    ax.set_xlabel("Global frame index")
    ax.set_ylabel("progress_dense")
    fig.tight_layout()
    fig.savefig(plots_dir / "run_1_progress_dense.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clip_features_path = validate_inputs(args)

    info = read_json(args.dataset_root / "meta/info.json")
    logging.info(
        "Dataset frames=%s episodes=%s fps=%s",
        info.get("total_frames"),
        info.get("total_episodes"),
        info.get("fps"),
    )

    frame_table = load_frame_table(args.dataset_root, args.state_key)
    frame_table["_row_position"] = np.arange(len(frame_table), dtype=np.int64)
    sampled = frame_table[frame_table["index"] % args.stride == 0].copy()
    if args.max_sampled_rows is not None:
        sampled = sampled.iloc[: args.max_sampled_rows].copy()
    sampled = sampled.reset_index(drop=True)
    expected_rows = int(np.ceil(len(frame_table) / args.stride))
    if args.max_sampled_rows is not None:
        expected_rows = min(expected_rows, args.max_sampled_rows)
    logging.info("Sampled %s rows with stride %s.", len(sampled), args.stride)

    clip_features = np.load(clip_features_path, mmap_mode="r")
    if clip_features.shape[0] != len(frame_table):
        raise ValueError(
            f"CLIP feature rows ({clip_features.shape[0]}) do not match dataset rows ({len(frame_table)})"
        )

    state = stack_state(frame_table[args.state_key].to_numpy())
    normalized_state = normalize_state(state, args.checkpoint, args.state_key)
    task_names = load_indexed_names(args.dataset_root / "meta/tasks.parquet", "task_index")
    subtask_names = load_indexed_names(args.dataset_root / "meta/subtasks.parquet", "subtask_index")
    task_embeddings = load_text_embeddings(task_names, local_files_only=args.clip_local_files_only)

    model = load_sarm_model(args.checkpoint, args.device)
    if args.head_mode == "dense":
        num_classes = int(model.config.num_dense_stages or 0)
    else:
        num_classes = int(model.config.num_sparse_stages)
    if num_classes <= 0:
        raise ValueError(f"Could not determine number of classes for head_mode={args.head_mode}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for run_id in range(1, args.runs + 1):
        np.random.seed(args.seed + run_id)
        torch.manual_seed(args.seed + run_id)
        predictions = predict_run(
            model=model,
            sampled=sampled,
            normalized_state=normalized_state,
            clip_features=clip_features,
            task_embeddings=task_embeddings,
            subtask_names=subtask_names,
            batch_size=args.batch_size,
            run_id=run_id,
            head_mode=args.head_mode,
        )
        missing_columns = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in predictions.columns]
        if missing_columns:
            raise ValueError(f"Predictions are missing required columns: {missing_columns}")
        invalid_pred_values = sorted(set(predictions["pred_subtask_index"]) - set(range(num_classes)))
        if invalid_pred_values:
            raise ValueError(f"Prediction indices outside 0..{num_classes - 1}: {invalid_pred_values}")

        run_dir = args.output_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions[REQUIRED_OUTPUT_COLUMNS].to_parquet(run_dir / "predictions.parquet", index=False)

        metrics = compute_run_metrics(predictions, num_classes=num_classes, expected_rows=expected_rows)
        summary_rows.append(metrics)
        run_predictions.append(predictions)
        logging.info(
            "run_%s rows=%s accuracy=%.6f macro=%.6f invalid=%s",
            run_id,
            metrics["sampled_rows"],
            metrics["overall_accuracy"],
            metrics["macro_accuracy"],
            metrics["invalid_prediction_count"],
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    consistency = compute_consistency(run_predictions, num_classes)
    with open(args.output_dir / "consistency.json", "w") as f:
        json.dump(consistency, f, indent=2, allow_nan=False)

    if not args.skip_plots:
        write_plots(
            run_predictions=run_predictions,
            output_dir=args.output_dir,
            num_classes=num_classes,
            subtask_names=subtask_names,
        )

    logging.info("Wrote predictions and metrics under %s", args.output_dir)


if __name__ == "__main__":
    main()
