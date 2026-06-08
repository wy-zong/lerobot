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
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

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


def precompute_for_camera(
    repo_id: str,
    camera_key: str,
    output: Path,
    root: Path | None = None,
    revision: str | None = None,
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    chunk_frames: int = 512,
) -> None:
    """Precompute CLIP image features for one camera of a LeRobot dataset."""
    require_package("transformers", extra="sarm")

    logging.info("Loading dataset metadata for %s", repo_id)
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    if camera_key not in meta.video_keys:
        raise ValueError(f"camera_key={camera_key!r} not found in dataset video keys: {meta.video_keys}")

    n_frames_total = meta.total_frames
    logging.info("episodes=%s total_frames=%s", meta.total_episodes, n_frames_total)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

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

    out = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(n_frames_total, proj_dim))
    logging.info("Output memmap: %s shape=%s dtype=%s", output, out.shape, out.dtype)

    episodes_df = (
        meta.episodes.to_pandas()
        .sort_values(
            [
                f"videos/{camera_key}/chunk_index",
                f"videos/{camera_key}/file_index",
                f"videos/{camera_key}/from_timestamp",
            ]
        )
        .reset_index(drop=True)
    )
    file_groups = episodes_df.groupby(
        [f"videos/{camera_key}/chunk_index", f"videos/{camera_key}/file_index"],
        sort=True,
    )

    sample_ep_idx = int(episodes_df.iloc[0]["episode_index"])
    sample_path = meta.root / meta.get_video_file_path(sample_ep_idx, camera_key)
    width, height = _probe_video_size(sample_path)
    frame_bytes = height * width * 3
    logging.info("video=%sx%s frame_bytes=%s", width, height, frame_bytes)

    t0 = time.time()
    n_done = 0
    n_files = len(file_groups)

    with torch.no_grad():
        for file_idx, ((_, file_i), group) in enumerate(file_groups, start=1):
            mp4_ep_idx = int(group.iloc[0]["episode_index"])
            mp4_path = meta.root / meta.get_video_file_path(mp4_ep_idx, camera_key)
            group = group.sort_values(f"videos/{camera_key}/from_timestamp").reset_index(drop=True)

            n_frames_in_file = int(group["length"].sum())
            mp4_frame_to_global = np.empty(n_frames_in_file, dtype=np.int64)
            cursor = 0
            for _, ep in group.iterrows():
                length = int(ep["length"])
                global_start = int(ep["dataset_from_index"])
                mp4_frame_to_global[cursor : cursor + length] = np.arange(global_start, global_start + length)
                cursor += length

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
                    "-loglevel",
                    "error",
                    "-",
                ],
                stdout=subprocess.PIPE,
                bufsize=128 * 1024 * 1024,
            )
            if ffmpeg.stdout is None:
                raise RuntimeError("ffmpeg stdout pipe was not created")

            try:
                file_frame_idx = 0
                while True:
                    n_want = min(chunk_frames, n_frames_in_file - file_frame_idx)
                    if n_want <= 0:
                        break

                    buf = ffmpeg.stdout.read(n_want * frame_bytes)
                    if not buf:
                        break

                    n_got = len(buf) // frame_bytes
                    if n_got == 0:
                        break

                    raw_np = np.frombuffer(buf[: n_got * frame_bytes], dtype=np.uint8).reshape(
                        n_got, height, width, 3
                    )
                    raw = torch.from_numpy(raw_np.copy()).to(torch_device, non_blocking=True)
                    pixel_values = _preprocess_on_device(raw, mean, std, crop_size)
                    vision_out = vision(pixel_values=pixel_values)
                    pooled = (
                        vision_out.pooler_output if hasattr(vision_out, "pooler_output") else vision_out[1]
                    )
                    embeddings = projection(pooled).float().cpu().numpy()
                    global_indices = mp4_frame_to_global[file_frame_idx : file_frame_idx + n_got]
                    out[global_indices] = embeddings
                    file_frame_idx += n_got
                    n_done += n_got

                if file_frame_idx != n_frames_in_file:
                    logging.warning(
                        "file-%03d: expected %s frames, got %s",
                        file_i,
                        n_frames_in_file,
                        file_frame_idx,
                    )
            finally:
                ffmpeg.stdout.close()
                ffmpeg.wait(timeout=5)

            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n_frames_total - n_done) / max(rate, 1e-6)
            logging.info(
                "[file %s/%s] %s/%s (%.1f%%) rate=%.0f fps eta=%.0fs",
                file_idx,
                n_files,
                n_done,
                n_frames_total,
                100 * n_done / n_frames_total,
                rate,
                eta,
            )

    out.flush()
    elapsed = time.time() - t0
    logging.info("Done in %.1fs (%.0f fps). Output: %s", elapsed, n_frames_total / elapsed, output)


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
    )


if __name__ == "__main__":
    main()
