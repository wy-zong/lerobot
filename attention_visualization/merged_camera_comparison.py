"""Compare the cached three- and four-camera revisions of the merged policy."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from .__main__ import _assert_complete, _model_metadata, _verify_video  # noqa: E402
from .core import (  # noqa: E402
    DEFAULT_DATASET,
    build_sample_frames,
    dataset_camera_keys,
    episode_camera_keys,
    load_dataset_info,
    load_jsonl_records,
    map_model_cameras,
    resolve_local_dataset,
    select_episode,
    snapshot_revision,
    write_json_atomic,
)
from .inference import sample_model_attention  # noqa: E402
from .render import render_multi_camera_comparison_video  # noqa: E402

MODEL_ID = "wuc1/bi_so101_ffp_0615-14-12_merged"
THREE_CAMERA_REVISION = "743efd6ac86f0ab8cb0ad289dd07243906e52321"
FOUR_CAMERA_REVISION = "b2e0537f6e1746a7306965d496460d81cb13262c"
FOURTH_CAMERA = "observation.images.left_camera3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline three-camera versus four-camera attention comparison for the merged policy."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episode", default="auto-longest")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output-fps", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_visualization"))
    parser.add_argument("--phase", choices=("all", "sample", "render"), default="all")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force-four-camera",
        action="store_true",
        help="Replace only the four-camera samples; the existing three-camera result is preserved.",
    )
    parser.add_argument("--overwrite-video", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def _ensure_link(path: Path, target: Path) -> None:
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Staging path already exists with a different target: {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def _augmented_dataset_root(
    source_root: Path,
    info: dict[str, Any],
    four_camera_model: dict[str, Any],
    staging_root: Path,
) -> Path:
    """Expose the retained fourth-camera videos through a patched metadata view."""

    config = json.loads((Path(four_camera_model["path"]) / "config.json").read_text())
    model_shape = config["input_features"][FOURTH_CAMERA]["shape"]
    channels, height, width = (int(value) for value in model_shape)
    if channels != 3:
        raise ValueError(f"Expected RGB fourth camera, got shape {model_shape}")

    patched = copy.deepcopy(info)
    template = copy.deepcopy(patched["features"]["observation.images.left_camera1"])
    template["shape"] = [height, width, channels]
    video_info = template.get("info", {})
    video_info["video.height"] = height
    video_info["video.width"] = width
    video_info["video.channels"] = channels
    patched["features"][FOURTH_CAMERA] = template

    staging_root.mkdir(parents=True, exist_ok=True)
    meta_root = staging_root / "meta"
    meta_root.mkdir(exist_ok=True)
    _ensure_link(staging_root / "data", source_root / "data")
    _ensure_link(staging_root / "videos", source_root / "videos")
    for name in ("episodes", "tasks.parquet", "stats.json"):
        source = source_root / "meta" / name
        if source.exists():
            _ensure_link(meta_root / name, source)
    write_json_atomic(meta_root / "info.json", patched)
    return staging_root


def _sample_if_needed(
    *,
    model: dict[str, Any],
    dataset_root: Path,
    dataset_camera_names: list[str],
    sample_frames: list[int],
    episode_index: int,
    source_fps: float,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    completed = {int(record["frame_index"]) for record in load_jsonl_records(output_path)}
    if all(frame in completed for frame in sample_frames):
        model["completed_samples"] = len(completed)
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=dataset_root,
        episodes=[episode_index],
        video_backend=args.video_backend,
    )
    result = sample_model_attention(
        model_id=model["id"],
        model_path=Path(model["path"]),
        model_revision=model["revision"],
        vlm_path=Path(model["vlm_path"]),
        dataset=dataset,
        dataset_camera_keys=dataset_camera_names,
        sample_frames=sample_frames,
        episode_index=episode_index,
        source_fps=source_fps,
        output_path=output_path,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
        max_samples=args.max_samples,
    )
    model["camera_mapping"] = result["camera_mapping"]
    model["completed_samples"] = len(load_jsonl_records(output_path))


def main() -> None:
    args = _parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    dataset_root = resolve_local_dataset(args.dataset)
    info = load_dataset_info(dataset_root)
    source_fps = float(info["fps"])
    selector: str | int = args.episode if args.episode == "auto-longest" else int(args.episode)
    episode = select_episode(dataset_root, selector)
    source_episode_length = episode.length
    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            raise ValueError("--duration-seconds must be positive")
        length = min(episode.length, round(args.duration_seconds * source_fps))
        episode = replace(episode, length=length, dataset_to_index=episode.dataset_from_index + length)

    samples = build_sample_frames(episode.length, source_fps, args.sample_fps)
    three_cameras = dataset_camera_keys(info)
    retained_cameras = episode_camera_keys(info, episode)
    if len(three_cameras) != 3 or FOURTH_CAMERA not in retained_cameras:
        raise ValueError(
            f"Expected three visible cameras plus retained {FOURTH_CAMERA}; "
            f"visible={three_cameras}, retained={retained_cameras}"
        )

    three_model = _model_metadata(MODEL_ID, THREE_CAMERA_REVISION)
    four_model = _model_metadata(MODEL_ID, FOUR_CAMERA_REVISION)
    if len(three_model["image_keys"]) != 3 or len(four_model["image_keys"]) != 4:
        raise ValueError(
            f"Unexpected checkpoint cameras: 3-camera={three_model['image_keys']}, "
            f"4-camera={four_model['image_keys']}"
        )
    three_model["variant"] = "three_camera"
    four_model["variant"] = "four_camera"
    three_model["camera_mapping"] = map_model_cameras(three_cameras, three_model["image_keys"])
    four_model["camera_mapping"] = map_model_cameras(retained_cameras, four_model["image_keys"])

    episode_dir = args.output_dir / f"episode_{episode.episode_index:06d}"
    three_path = episode_dir / "bi_so101_ffp_0615-14-12_merged.attention.jsonl"
    four_path = episode_dir / "bi_so101_ffp_0615-14-12_merged.four_camera.attention.jsonl"
    video_path = episode_dir / "merged_three_vs_four_camera.mp4"
    manifest_path = episode_dir / "merged_camera_comparison_manifest.json"
    three_model["attention_jsonl"] = str(three_path)
    four_model["attention_jsonl"] = str(four_path)

    if args.force_four_camera:
        four_path.unlink(missing_ok=True)
    if args.overwrite_video:
        video_path.unlink(missing_ok=True)

    staging_root = Path("/tmp/lerobot_attention_four_camera") / f"episode_{episode.episode_index:06d}"
    augmented_root = _augmented_dataset_root(dataset_root, info, four_model, staging_root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "offline": True,
        "dataset": {
            "id": args.dataset,
            "path": str(dataset_root),
            "revision": snapshot_revision(dataset_root),
            "fps": source_fps,
            "visible_camera_keys": three_cameras,
            "retained_camera_keys": retained_cameras,
        },
        "episode": {
            **episode.manifest_dict(),
            "source_length_frames": source_episode_length,
            "rendered_length_frames": episode.length,
            "rendered_duration_seconds": episode.length / source_fps,
        },
        "models": [three_model, four_model],
        "sampling": {
            "sample_fps": args.sample_fps,
            "sample_count": len(samples),
            "output_fps": args.output_fps,
            "forward_fill": True,
            "seed": args.seed,
        },
        "comparison_note": (
            "The three-camera row uses current revision 743efd6; the four-camera row uses the "
            "cached historical checkpoint b2e0537 that was trained with left_camera3."
        ),
        "output_video": str(video_path),
    }
    write_json_atomic(manifest_path, manifest)

    if args.phase in {"all", "sample"}:
        _sample_if_needed(
            model=three_model,
            dataset_root=dataset_root,
            dataset_camera_names=three_cameras,
            sample_frames=samples,
            episode_index=episode.episode_index,
            source_fps=source_fps,
            output_path=three_path,
            args=args,
        )
        write_json_atomic(manifest_path, manifest)
        _sample_if_needed(
            model=four_model,
            dataset_root=augmented_root,
            dataset_camera_names=retained_cameras,
            sample_frames=samples,
            episode_index=episode.episode_index,
            source_fps=source_fps,
            output_path=four_path,
            args=args,
        )
        write_json_atomic(manifest_path, manifest)

    if args.phase in {"all", "render"}:
        _assert_complete(three_path, samples)
        _assert_complete(four_path, samples)
        render_multi_camera_comparison_video(
            dataset_root=dataset_root,
            episode=episode,
            camera_orders=[three_cameras, four_model["image_keys"]],
            attention_paths=[three_path, four_path],
            model_labels=[
                f"{MODEL_ID} [3 cameras · {THREE_CAMERA_REVISION[:7]}]",
                f"{MODEL_ID} [4 cameras · {FOUR_CAMERA_REVISION[:7]}]",
            ],
            output_path=video_path,
            output_fps=args.output_fps,
            overwrite=args.overwrite_video,
        )
        manifest["video_verification"] = _verify_video(video_path, args.output_fps, episode.length)
        write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
