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

"""Precompute CLIP image features for one camera of a LeRobot dataset.

The output is a float32 ``.npy`` memmap with shape ``(N_total_frames, projection_dim)``.
Rows are indexed by absolute global frame index and can be passed to SARM via
``SARMConfig.precomputed_image_features_path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.utils import init_logging

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPImageProcessor, CLIPModel
else:
    CLIPImageProcessor = None  # type: ignore[assignment, misc]
    CLIPModel = None  # type: ignore[assignment, misc]

_PROGRESS_VERSION = 1
_OUTPUT_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True)
class _VideoFile:
    key: tuple[int, int]
    path: Path
    relative_path: str
    global_indices: np.ndarray
    manifest_episodes: list[dict[str, int | float]]

    @property
    def n_frames(self) -> int:
        return len(self.global_indices)


@torch.no_grad()
def _preprocess_on_device(
    raw_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, crop_size: int
) -> torch.Tensor:
    """GPU-batched equivalent of HF CLIPImageProcessor preprocessing."""
    x = raw_uint8.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    _, _, height, width = x.shape
    if height < width:
        new_h, new_w = crop_size, int(round(width * crop_size / height))
    else:
        new_h, new_w = int(round(height * crop_size / width)), crop_size

    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic", antialias=True, align_corners=False)
    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2
    x = x[:, :, top : top + crop_size, left : left + crop_size]
    return (x - mean) / std


def _square_crop_size(image_processor: CLIPImageProcessor) -> int:
    crop_size_value = image_processor.crop_size
    try:
        height, width = crop_size_value["height"], crop_size_value["width"]
        if height != width:
            raise ValueError(f"Non-square CLIP crop size: {crop_size_value}")
        return int(height)
    except (TypeError, KeyError, AttributeError):
        return int(crop_size_value)


def _probe_video_size(video_path: Path) -> tuple[int, int]:
    probe = subprocess.run(  # nosec B603,B607
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = map(int, probe.stdout.strip().split(","))
    return width, height


def _progress_path(output: Path) -> Path:
    return Path(f"{output}.progress.json")


def _build_video_files(meta: LeRobotDatasetMetadata, camera_key: str) -> list[_VideoFile]:
    chunk_column = f"videos/{camera_key}/chunk_index"
    file_column = f"videos/{camera_key}/file_index"
    timestamp_column = f"videos/{camera_key}/from_timestamp"
    episodes_df = (
        meta.episodes.to_pandas()
        .sort_values([chunk_column, file_column, timestamp_column])
        .reset_index(drop=True)
    )
    video_files = []
    for (chunk_index, file_index), unordered_group in episodes_df.groupby(
        [chunk_column, file_column], sort=True
    ):
        group = unordered_group.sort_values(timestamp_column).reset_index(drop=True)
        global_indices_parts = []
        manifest_episodes = []
        for _, episode in group.iterrows():
            length = int(episode["length"])
            global_start = int(episode["dataset_from_index"])
            global_indices_parts.append(np.arange(global_start, global_start + length, dtype=np.int64))
            manifest_episodes.append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "dataset_from_index": global_start,
                    "length": length,
                    "from_timestamp": float(episode[timestamp_column]),
                }
            )

        episode_index = int(group.iloc[0]["episode_index"])
        path = meta.root / meta.get_video_file_path(episode_index, camera_key)
        video_files.append(
            _VideoFile(
                key=(int(chunk_index), int(file_index)),
                path=path,
                relative_path=path.relative_to(meta.root).as_posix(),
                global_indices=np.concatenate(global_indices_parts),
                manifest_episodes=manifest_episodes,
            )
        )
    dataset_ranges = sorted(
        (int(episode["dataset_from_index"]), int(episode["length"]))
        for video_file in video_files
        for episode in video_file.manifest_episodes
    )
    expected_start = 0
    manifest_is_contiguous = True
    for dataset_from_index, length in dataset_ranges:
        if dataset_from_index != expected_start or length <= 0:
            manifest_is_contiguous = False
            break
        expected_start += length
    if not manifest_is_contiguous or expected_start != meta.total_frames:
        raise ValueError(
            f"Video manifest for camera {camera_key!r} does not map exactly once to all "
            f"{meta.total_frames} dataset frames"
        )
    return video_files


def _manifest_fingerprint(video_files: list[_VideoFile], total_frames: int) -> str:
    manifest = {
        "total_frames": total_frames,
        "files": [
            {
                "chunk_index": video_file.key[0],
                "file_index": video_file.key[1],
                "path": video_file.relative_path,
                "episodes": video_file.manifest_episodes,
            }
            for video_file in video_files
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _new_progress(
    *,
    repo_id: str,
    revision: str,
    camera_key: str,
    clip_model_id: str,
    shape: tuple[int, int],
    manifest_fingerprint: str,
) -> dict:
    return {
        "version": _PROGRESS_VERSION,
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "camera_key": camera_key,
            "manifest_fingerprint": manifest_fingerprint,
        },
        "clip_model_id": clip_model_id,
        "output": {"shape": list(shape), "dtype": _OUTPUT_DTYPE.name},
        "completed_files": [],
        "warnings": [],
    }


def _write_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as temporary_file:
            json.dump(progress, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_progress(path: Path) -> dict:
    try:
        with path.open() as progress_file:
            progress = json.load(progress_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read checkpoint {path}: {error}. Keep the existing files and use --overwrite "
            "to start over."
        ) from error
    if not isinstance(progress, dict):
        raise ValueError(f"Invalid checkpoint {path}. Use --overwrite to start over.")
    return progress


def _validate_progress(progress: dict, expected: dict, path: Path) -> set[tuple[int, int]]:
    identity_keys = ("version", "dataset", "clip_model_id", "output")
    mismatches = [key for key in identity_keys if progress.get(key) != expected[key]]
    if mismatches:
        raise ValueError(
            f"Checkpoint {path} is incompatible with the current run ({', '.join(mismatches)} differ). "
            "The existing output was preserved; use --overwrite to start over."
        )

    completed_raw = progress.get("completed_files")
    if not isinstance(completed_raw, list) or any(
        not isinstance(key, list) or len(key) != 2 or any(type(value) is not int for value in key)
        for key in completed_raw
    ):
        raise ValueError(f"Invalid completed_files in checkpoint {path}. Use --overwrite to start over.")
    return {tuple(key) for key in completed_raw}


def _open_existing_output(output: Path, expected_shape: tuple[int, int]) -> np.memmap:
    try:
        out = np.load(output, mmap_mode="r+")
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot open existing output {output}: {error}. It was preserved; use --overwrite to start over."
        ) from error
    if out.shape != expected_shape or out.dtype != _OUTPUT_DTYPE:
        actual = f"shape={out.shape}, dtype={out.dtype}"
        expected = f"shape={expected_shape}, dtype={_OUTPUT_DTYPE.name}"
        raise ValueError(
            f"Existing output {output} is incompatible ({actual}; expected {expected}). "
            "It was preserved; use --overwrite to start over."
        )
    return out


def _scan_legacy_completed(out: np.memmap, video_files: list[_VideoFile]) -> set[tuple[int, int]]:
    completed = set()
    for video_file in video_files:
        file_is_complete = True
        for start in range(0, video_file.n_frames, 8192):
            rows = out[video_file.global_indices[start : start + 8192]]
            if not np.isfinite(rows).all() or not np.all(np.any(rows != 0, axis=1)):
                file_is_complete = False
                break
        if file_is_complete:
            completed.add(video_file.key)
    return completed


def _read_up_to(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wait_for_ffmpeg(ffmpeg: subprocess.Popen, video_path: Path) -> tuple[int, str]:
    try:
        return_code = ffmpeg.wait(timeout=5)
    except subprocess.TimeoutExpired as error:
        ffmpeg.kill()
        ffmpeg.wait()
        raise RuntimeError(f"ffmpeg did not exit after decoding {video_path}") from error
    stderr = ffmpeg.stderr.read().decode(errors="replace").strip() if ffmpeg.stderr is not None else ""
    return return_code, stderr


def precompute_for_camera(
    repo_id: str,
    camera_key: str,
    output: Path,
    root: Path | None = None,
    revision: str | None = None,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    chunk_frames: int = 512,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    """Precompute CLIP image features for one camera of a LeRobot dataset."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if chunk_frames <= 0:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")

    progress_path = _progress_path(output)
    if output.exists() and not (resume or overwrite):
        raise FileExistsError(
            f"Output {output} already exists. Use --resume to continue or --overwrite to start over."
        )
    if overwrite:
        output.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    require_package("transformers", extra="sarm")

    logging.info("Loading dataset metadata for %s", repo_id)
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    if camera_key not in meta.video_keys:
        raise ValueError(f"camera_key={camera_key!r} not found in dataset video keys: {meta.video_keys}")

    n_frames_total = meta.total_frames
    logging.info("episodes=%s total_frames=%s", meta.total_episodes, n_frames_total)

    output.parent.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)

    logging.info("Loading CLIP model and image processor: %s", clip_model_id)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
    crop_size = _square_crop_size(image_processor)
    mean = torch.tensor(image_processor.image_mean, device=torch_device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, device=torch_device).view(1, 3, 1, 1)
    logging.info(
        "crop_size=%s mean=%s std=%s", crop_size, image_processor.image_mean, image_processor.image_std
    )

    model = CLIPModel.from_pretrained(clip_model_id).to(torch_device).eval()
    vision = model.vision_model
    projection = model.visual_projection
    proj_dim = int(model.config.projection_dim)
    expected_shape = (n_frames_total, proj_dim)
    video_files = _build_video_files(meta, camera_key)
    fingerprint = _manifest_fingerprint(video_files, n_frames_total)
    expected_progress = _new_progress(
        repo_id=repo_id,
        revision=meta.revision,
        camera_key=camera_key,
        clip_model_id=clip_model_id,
        shape=expected_shape,
        manifest_fingerprint=fingerprint,
    )

    if output.exists():
        out = _open_existing_output(output, expected_shape)
        if progress_path.exists():
            progress = _load_progress(progress_path)
            completed_files = _validate_progress(progress, expected_progress, progress_path)
        else:
            completed_files = _scan_legacy_completed(out, video_files)
            progress = expected_progress
            warning = (
                "Resumed a legacy .npy without a checkpoint; its dataset, camera, and CLIP model identity "
                "could not be fully verified."
            )
            progress["warnings"].append(warning)
            progress["completed_files"] = [list(key) for key in sorted(completed_files)]
            _write_progress(progress_path, progress)
            logging.warning(warning)
    else:
        if progress_path.exists():
            logging.warning(
                "Ignoring orphan checkpoint %s because output %s does not exist", progress_path, output
            )
        out = np.lib.format.open_memmap(output, mode="w+", dtype=_OUTPUT_DTYPE, shape=expected_shape)
        progress = expected_progress
        completed_files = set()
        out.flush()
        _write_progress(progress_path, progress)

    known_file_keys = {video_file.key for video_file in video_files}
    unknown_completed = completed_files - known_file_keys
    if unknown_completed:
        raise ValueError(
            f"Checkpoint {progress_path} contains files absent from the current manifest: "
            f"{sorted(unknown_completed)}. Use --overwrite to start over."
        )
    logging.info("Output memmap: %s shape=%s dtype=%s", output, out.shape, out.dtype)

    t0 = time.time()
    n_preexisting = sum(
        video_file.n_frames for video_file in video_files if video_file.key in completed_files
    )
    n_session = 0
    n_files = len(video_files)

    with torch.no_grad():
        for file_idx, video_file in enumerate(video_files, start=1):
            if video_file.key in completed_files:
                continue

            mp4_path = video_file.path
            width, height = _probe_video_size(mp4_path)
            frame_bytes = height * width * 3
            n_frames_in_file = video_file.n_frames
            ffmpeg = subprocess.Popen(  # nosec B603,B607
                [
                    "ffmpeg",
                    "-i",
                    str(mp4_path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vsync",
                    "0",
                    "-xerror",
                    "-loglevel",
                    "error",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=128 * 1024 * 1024,
            )
            if ffmpeg.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe was not created")

            try:
                file_frame_idx = 0
                while file_frame_idx < n_frames_in_file:
                    n_want = min(chunk_frames, n_frames_in_file - file_frame_idx)
                    buf = _read_up_to(ffmpeg.stdout, n_want * frame_bytes)
                    if not buf:
                        break
                    if len(buf) % frame_bytes:
                        raise RuntimeError(
                            f"ffmpeg returned a partial frame for {mp4_path}: {len(buf)} bytes "
                            f"is not divisible by {frame_bytes}"
                        )
                    n_got = len(buf) // frame_bytes

                    raw_np = np.frombuffer(buf, dtype=np.uint8).reshape(n_got, height, width, 3)
                    raw = torch.from_numpy(raw_np.copy()).to(torch_device, non_blocking=True)
                    pixel_values = _preprocess_on_device(raw, mean, std, crop_size)
                    vision_out = vision(pixel_values=pixel_values)
                    pooled = (
                        vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out[1]
                    )
                    embeddings = projection(pooled).float().cpu().numpy()
                    if embeddings.shape != (n_got, proj_dim):
                        raise RuntimeError(
                            f"Invalid embedding shape for {mp4_path}: got {embeddings.shape}, "
                            f"expected {(n_got, proj_dim)}"
                        )
                    if not np.isfinite(embeddings).all() or not np.all(np.any(embeddings != 0, axis=1)):
                        raise RuntimeError(f"CLIP produced non-finite or all-zero embeddings for {mp4_path}")
                    global_indices = video_file.global_indices[file_frame_idx : file_frame_idx + n_got]
                    out[global_indices] = embeddings
                    file_frame_idx += n_got
                extra = _read_up_to(ffmpeg.stdout, frame_bytes)
            except BaseException:
                ffmpeg.stdout.close()
                ffmpeg.kill()
                ffmpeg.wait()
                raise
            ffmpeg.stdout.close()
            return_code, stderr = _wait_for_ffmpeg(ffmpeg, mp4_path)
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg failed for {mp4_path} with exit code {return_code}"
                    + (f": {stderr}" if stderr else "")
                )
            if file_frame_idx != n_frames_in_file or extra:
                actual = file_frame_idx + (len(extra) // frame_bytes)
                suffix = " (plus a partial frame)" if len(extra) % frame_bytes else ""
                raise RuntimeError(
                    f"Frame count mismatch for {mp4_path}: expected {n_frames_in_file}, got {actual}{suffix}"
                )

            out.flush()
            completed_files.add(video_file.key)
            progress["completed_files"] = [list(key) for key in sorted(completed_files)]
            _write_progress(progress_path, progress)
            n_session += n_frames_in_file

            elapsed = time.time() - t0
            rate = n_session / max(elapsed, 1e-6)
            n_completed = n_preexisting + n_session
            eta = (n_frames_total - n_completed) / max(rate, 1e-6)
            logging.info(
                "[file %s/%s] completed=%s+%s/%s (%.1f%%) session_rate=%.0f fps eta=%.0fs",
                file_idx,
                n_files,
                n_preexisting,
                n_session,
                n_frames_total,
                100 * n_completed / n_frames_total,
                rate,
                eta,
            )

    out.flush()
    elapsed = time.time() - t0
    if n_session:
        logging.info("Done in %.1fs (%.0f session fps). Output: %s", elapsed, n_session / elapsed, output)
    else:
        logging.info("Done: all %s frames were already complete. Output: %s", n_preexisting, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLIP image features for a LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo ID, e.g. lerobot/pusht.")
    parser.add_argument(
        "--camera-key", type=str, required=True, help="Video feature key, e.g. observation.images.top."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path for the feature memmap.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root.")
    parser.add_argument("--revision", type=str, default=None, help="Dataset revision.")
    parser.add_argument(
        "--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID."
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda or cpu.")
    parser.add_argument("--chunk-frames", type=int, default=512, help="Batch size per CLIP forward.")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume", action="store_true", help="Resume from an existing output and checkpoint."
    )
    output_mode.add_argument(
        "--overwrite", action="store_true", help="Delete existing output/checkpoint and start over."
    )
    args = parser.parse_args()

    init_logging()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    precompute_for_camera(
        repo_id=args.repo_id,
        camera_key=args.camera_key,
        output=args.output,
        root=args.root,
        revision=args.revision,
        clip_model_id=args.clip_model,
        device=device,
        chunk_frames=args.chunk_frames,
        resume=args.resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
