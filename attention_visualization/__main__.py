"""CLI for the offline three-model SmolVLA attention comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

# Enforce offline behavior before importing any Hugging Face component.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from .core import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MODELS,
    build_sample_frames,
    dataset_camera_keys,
    load_dataset_info,
    load_jsonl_records,
    map_model_cameras,
    resolve_local_dataset,
    resolve_local_snapshot,
    select_episode,
    snapshot_revision,
    write_json_atomic,
)
from .inference import sample_model_attention  # noqa: E402
from .render import render_comparison_video  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline three-model SmolVLA attention comparison video."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episode", default="auto-longest")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--output-fps", type=float, default=60.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Limit sampling and rendering to this many seconds from the start of the episode.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_visualization"))
    parser.add_argument("--phase", choices=("all", "sample", "render"), default="all")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Replace existing samples and video.")
    parser.add_argument("--overwrite-video", action="store_true")
    parser.add_argument("--max-samples", type=int, help="Integration/debug limit per model.")
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def _model_metadata(model_id: str, revision: str | None = None) -> dict[str, Any]:
    model_path = resolve_local_snapshot(
        model_id,
        repo_type="model",
        revision=revision,
        required_files=(
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ),
    )
    config = json.loads((model_path / "config.json").read_text())
    vlm_id = config["vlm_model_name"]
    vlm_path = resolve_local_snapshot(
        vlm_id,
        repo_type="model",
        required_files=("config.json", "model.safetensors", "tokenizer.json"),
    )
    image_keys = [key for key, feature in config["input_features"].items() if feature["type"] == "VISUAL"]
    return {
        "id": model_id,
        "path": str(model_path),
        "revision": snapshot_revision(model_path),
        "vlm_id": vlm_id,
        "vlm_path": str(vlm_path),
        "vlm_revision": snapshot_revision(vlm_path),
        "image_keys": image_keys,
        "discrete_state_in_language": bool(config.get("discrete_state_in_language", False)),
        "original_compile_model": bool(config.get("compile_model", False)),
    }


def _attention_filename(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace("/", "_") + ".attention.jsonl"


def _assert_complete(path: Path, sample_frames: list[int]) -> None:
    completed = {int(record["frame_index"]) for record in load_jsonl_records(path)}
    missing = [frame for frame in sample_frames if frame not in completed]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        raise RuntimeError(f"{path} is missing {len(missing)} sampled frames (first: {preview})")


def _verify_video(path: Path, expected_fps: float, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    actual_fps = numerator / denominator
    actual_frames = int(stream["nb_read_frames"])
    if abs(actual_fps - expected_fps) > 1e-6 or actual_frames != expected_frames:
        raise RuntimeError(
            f"Video verification failed: fps={actual_fps}, frames={actual_frames}; "
            f"expected fps={expected_fps}, frames={expected_frames}"
        )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    return {
        "fps": actual_fps,
        "frame_count": actual_frames,
        "duration": float(stream["duration"]),
        "fully_decoded": True,
    }


def main() -> None:
    args = _parse_args()
    if len(args.models) != 3:
        raise ValueError(f"Exactly three models are required, got {len(args.models)}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    dataset_root = resolve_local_dataset(args.dataset)
    info = load_dataset_info(dataset_root)
    source_fps = float(info["fps"])
    episode_selector: str | int = args.episode if args.episode == "auto-longest" else int(args.episode)
    episode = select_episode(dataset_root, episode_selector)
    source_episode_length = episode.length
    if args.duration_seconds is not None:
        if args.duration_seconds <= 0:
            raise ValueError("--duration-seconds must be positive")
        limited_frames = min(episode.length, round(args.duration_seconds * source_fps))
        episode = replace(
            episode,
            length=limited_frames,
            dataset_to_index=episode.dataset_from_index + limited_frames,
        )
    samples = build_sample_frames(episode.length, source_fps, args.sample_fps)
    cameras = dataset_camera_keys(info)
    if len(cameras) != 3:
        raise ValueError(f"Expected exactly three dataset cameras, got {cameras}")

    models = [_model_metadata(model_id) for model_id in args.models]
    episode_dir = args.output_dir / f"episode_{episode.episode_index:06d}"
    attention_paths = [episode_dir / _attention_filename(model["id"]) for model in models]
    video_path = episode_dir / "comparison.mp4"
    manifest_path = episode_dir / "run_manifest.json"
    for model, path in zip(models, attention_paths, strict=True):
        model["camera_mapping"] = map_model_cameras(cameras, model["image_keys"])
        model["attention_jsonl"] = str(path)

    if args.force:
        for path in [*episode_dir.glob("*.attention.jsonl"), video_path]:
            path.unlink(missing_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "offline": True,
        "dataset": {
            "id": args.dataset,
            "path": str(dataset_root),
            "revision": snapshot_revision(dataset_root),
            "fps": source_fps,
            "camera_keys": cameras,
        },
        "episode": {
            **episode.manifest_dict(),
            "source_length_frames": source_episode_length,
            "rendered_length_frames": episode.length,
            "rendered_duration_seconds": episode.length / source_fps,
        },
        "models": models,
        "sampling": {
            "sample_fps": args.sample_fps,
            "sample_interval_frames": samples[1] if len(samples) > 1 else 1,
            "sample_count": len(samples),
            "output_fps": args.output_fps,
            "forward_fill": True,
            "seed": args.seed,
        },
        "aggregation": {
            "attention_kind": "expert_cross",
            "layers": "all expert cross-attention layers",
            "heads": "mean",
            "action_queries": "mean",
            "denoise_steps": "mean over all 10 steps",
            "display_normalization": "grouped camera/state/language mass sums to 100%",
            "raw_mass_retained": True,
            "discrete_state": (
                "The complete prompt, including `State: ...`, is attributed to language; "
                "no independent state attention exists."
            ),
        },
        "compile_model_forced_off": True,
        "output_video": str(video_path),
    }
    write_json_atomic(manifest_path, manifest)
    print(
        f"dataset={args.dataset} episode={episode.episode_index} frames={episode.length} "
        f"fps={source_fps:g} samples={len(samples)}",
        flush=True,
    )

    if args.phase in {"all", "sample"}:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            repo_id=args.dataset,
            root=dataset_root,
            episodes=[episode.episode_index],
            video_backend=args.video_backend,
        )
        if len(dataset) < episode.length:
            raise RuntimeError(
                f"Selected dataset length is {len(dataset)}, shorter than requested {episode.length}"
            )
        for model, output_path in zip(models, attention_paths, strict=True):
            result = sample_model_attention(
                model_id=model["id"],
                model_path=Path(model["path"]),
                model_revision=model["revision"],
                vlm_path=Path(model["vlm_path"]),
                dataset=dataset,
                dataset_camera_keys=cameras,
                sample_frames=samples,
                episode_index=episode.episode_index,
                source_fps=source_fps,
                output_path=output_path,
                device=args.device,
                seed=args.seed,
                resume=args.resume,
                max_samples=args.max_samples,
            )
            if result.get("camera_mapping"):
                model["camera_mapping"] = result["camera_mapping"]
            model["completed_samples"] = len(load_jsonl_records(output_path))
            write_json_atomic(manifest_path, manifest)

    if args.phase in {"all", "render"}:
        for path in attention_paths:
            _assert_complete(path, samples)
        render_comparison_video(
            dataset_root=dataset_root,
            episode=episode,
            camera_order=cameras,
            attention_paths=attention_paths,
            model_labels=[model["id"] for model in models],
            output_path=video_path,
            output_fps=args.output_fps,
            overwrite=args.overwrite_video or args.force,
        )
        manifest["video_verification"] = _verify_video(video_path, args.output_fps, episode.length)
        write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
