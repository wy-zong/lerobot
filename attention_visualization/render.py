"""Frame compositor and streaming FFmpeg renderer."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import EpisodeInfo, forward_fill_records, load_dataset_info, load_jsonl_records

CAMERA_WIDTH = 300
CAMERA_HEIGHT = 225
PANEL_WIDTH = 660
ROW_WIDTH = CAMERA_WIDTH * 3 + PANEL_WIDTH
ROW_HEIGHT = 292


def row_width(camera_slots: int) -> int:
    if camera_slots <= 0:
        raise ValueError("camera_slots must be positive")
    return CAMERA_WIDTH * camera_slots + PANEL_WIDTH


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _short_name(key: str) -> str:
    return key.removeprefix("observation.images.")


def _bar(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + 9), radius=4, fill=(48, 52, 62))
    fill_width = int(max(0.0, min(100.0, value)) * width / 100.0)
    if fill_width:
        draw.rounded_rectangle((x, y, x + fill_width, y + 9), radius=4, fill=color)


def render_model_row(
    camera_frames: dict[str, np.ndarray],
    camera_order: list[str],
    record: dict[str, Any],
    model_label: str,
    *,
    frame_index: int,
    source_fps: float,
    camera_slots: int | None = None,
) -> Image.Image:
    """Render one model row. Kept separate so it can be unit tested without FFmpeg."""

    camera_slots = len(camera_order) if camera_slots is None else camera_slots
    if not camera_order or len(camera_order) > camera_slots:
        raise ValueError(f"Cannot place {len(camera_order)} cameras in {camera_slots} slots")
    canvas = Image.new("RGB", (row_width(camera_slots), ROW_HEIGHT), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(17, bold=True)
    body_font = _font(15)
    small_font = _font(13)
    muted = (180, 187, 200)
    accent = (91, 173, 255)

    draw.text((10, 5), model_label, font=title_font, fill=(244, 246, 250))
    percentages = record["attention_percent"]
    image_y = 31
    for index, camera in enumerate(camera_order):
        array = camera_frames[camera]
        image = Image.fromarray(array.astype(np.uint8), mode="RGB").resize(
            (CAMERA_WIDTH, CAMERA_HEIGHT), Image.Resampling.LANCZOS
        )
        x = index * CAMERA_WIDTH
        canvas.paste(image, (x, image_y))
        overlay_y = image_y + CAMERA_HEIGHT - 27
        draw.rectangle((x, overlay_y, x + CAMERA_WIDTH, image_y + CAMERA_HEIGHT), fill=(0, 0, 0))
        value = float(percentages.get(camera, 0.0))
        label = f"{_short_name(camera)}  {value:5.1f}%"
        draw.text((x + 8, overlay_y + 5), label, font=small_font, fill=(255, 255, 255))
        _bar(draw, (x + 8, image_y + CAMERA_HEIGHT + 10), CAMERA_WIDTH - 16, value, accent)

    panel_x = CAMERA_WIDTH * camera_slots + 18
    y = 12
    draw.text(
        (panel_x, y),
        f"Image total  {record['image_total_percent']:5.1f}%",
        font=title_font,
        fill=accent,
    )
    y += 31
    language_percent = float(percentages.get("language", 0.0))
    if record.get("state_attention_available", True):
        state_percent = float(percentages["state"])
        draw.text(
            (panel_x, y),
            f"State       {state_percent:5.1f}%",
            font=body_font,
            fill=(255, 184, 99),
        )
        _bar(draw, (panel_x + 180, y + 5), 270, state_percent, (255, 184, 99))
    else:
        draw.text(
            (panel_x, y),
            "State       N/A (inside language)",
            font=body_font,
            fill=muted,
        )
    y += 27
    draw.text(
        (panel_x, y),
        f"Language    {language_percent:5.1f}%",
        font=body_font,
        fill=(153, 221, 159),
    )
    _bar(draw, (panel_x + 180, y + 5), 270, language_percent, (153, 221, 159))
    y += 34

    task = str(record.get("task", ""))
    draw.text((panel_x, y), "Task:", font=body_font, fill=muted)
    y += 21
    for line in textwrap.wrap(task, width=61)[:2]:
        draw.text((panel_x, y), line, font=body_font, fill=(235, 238, 244))
        y += 20
    y += 5

    state_values = record.get("state_values", [])
    state_text = " ".join(f"{float(value):.2f}" for value in state_values)
    draw.text((panel_x, y), "State values:", font=body_font, fill=muted)
    y += 21
    for line in textwrap.wrap(state_text, width=66)[:3]:
        draw.text((panel_x, y), line, font=small_font, fill=(235, 238, 244))
        y += 18

    time_seconds = frame_index / source_fps
    footer = f"t={time_seconds:7.2f}s    frame={frame_index:05d}"
    draw.text((panel_x, ROW_HEIGHT - 25), footer, font=body_font, fill=muted)
    return canvas


def compose_comparison_frame(
    camera_frames: dict[str, np.ndarray],
    camera_order: list[str],
    model_records: list[dict[str, Any]],
    model_labels: list[str],
    *,
    frame_index: int,
    source_fps: float,
) -> Image.Image:
    if len(model_records) != 3 or len(model_labels) != 3:
        raise ValueError("The comparison video requires exactly three model records and labels")
    canvas = Image.new("RGB", (ROW_WIDTH, ROW_HEIGHT * 3), (18, 20, 26))
    for index, (record, label) in enumerate(zip(model_records, model_labels, strict=True)):
        row = render_model_row(
            camera_frames,
            camera_order,
            record,
            label,
            frame_index=frame_index,
            source_fps=source_fps,
        )
        canvas.paste(row, (0, index * ROW_HEIGHT))
    return canvas


def compose_multi_camera_frame(
    camera_frames: dict[str, np.ndarray],
    camera_orders: list[list[str]],
    model_records: list[dict[str, Any]],
    model_labels: list[str],
    *,
    frame_index: int,
    source_fps: float,
) -> Image.Image:
    """Compose arbitrary rows while aligning their right-hand panels."""

    if not model_records or len(model_records) != len(model_labels) or len(model_records) != len(camera_orders):
        raise ValueError("Records, labels, and camera orders must have the same non-zero length")
    camera_slots = max(len(order) for order in camera_orders)
    canvas = Image.new(
        "RGB", (row_width(camera_slots), ROW_HEIGHT * len(model_records)), (18, 20, 26)
    )
    for index, (record, label, order) in enumerate(
        zip(model_records, model_labels, camera_orders, strict=True)
    ):
        row = render_model_row(
            camera_frames,
            order,
            record,
            label,
            frame_index=frame_index,
            source_fps=source_fps,
            camera_slots=camera_slots,
        )
        canvas.paste(row, (0, index * ROW_HEIGHT))
    return canvas


def _read_exact(stream, byte_count: int) -> bytes:
    chunks = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) != byte_count:
        raise EOFError(f"FFmpeg returned {len(value)} bytes; expected {byte_count}")
    return value


class RawVideoReader:
    def __init__(self, path: Path, start_seconds: float, frame_count: int):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.9f}",
            "-i",
            str(path),
            "-an",
            "-vf",
            f"scale={CAMERA_WIDTH}:{CAMERA_HEIGHT}:flags=lanczos",
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read(self) -> np.ndarray:
        assert self.process.stdout is not None
        data = _read_exact(self.process.stdout, CAMERA_WIDTH * CAMERA_HEIGHT * 3)
        return np.frombuffer(data, dtype=np.uint8).reshape(CAMERA_HEIGHT, CAMERA_WIDTH, 3)

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        return_code = self.process.wait()
        if return_code:
            stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"Video decoder failed with exit code {return_code}: {stderr}")

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait()


def _video_path_and_start(
    dataset_root: Path,
    info: dict[str, Any],
    episode: EpisodeInfo,
    camera: str,
) -> tuple[Path, float]:
    row = episode.metadata
    chunk = int(row[f"videos/{camera}/chunk_index"])
    file_index = int(row[f"videos/{camera}/file_index"])
    relative = info["video_path"].format(video_key=camera, chunk_index=chunk, file_index=file_index)
    start = float(row[f"videos/{camera}/from_timestamp"])
    path = dataset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing local source video: {path}")
    return path, start


def render_comparison_video(
    *,
    dataset_root: Path,
    episode: EpisodeInfo,
    camera_order: list[str],
    attention_paths: list[Path],
    model_labels: list[str],
    output_path: Path,
    output_fps: float,
    overwrite: bool = False,
) -> None:
    render_multi_camera_comparison_video(
        dataset_root=dataset_root,
        episode=episode,
        camera_orders=[camera_order for _ in attention_paths],
        attention_paths=attention_paths,
        model_labels=model_labels,
        output_path=output_path,
        output_fps=output_fps,
        overwrite=overwrite,
    )


def render_multi_camera_comparison_video(
    *,
    dataset_root: Path,
    episode: EpisodeInfo,
    camera_orders: list[list[str]],
    attention_paths: list[Path],
    model_labels: list[str],
    output_path: Path,
    output_fps: float,
    overwrite: bool = False,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists: {output_path}; pass --overwrite-video")
    if not camera_orders or len(camera_orders) != len(attention_paths) or len(camera_orders) != len(model_labels):
        raise ValueError("Camera orders, attention paths, and model labels must have equal non-zero length")
    info = load_dataset_info(dataset_root)
    source_fps = float(info["fps"])
    filled_records = [
        forward_fill_records(load_jsonl_records(path), episode.length) for path in attention_paths
    ]

    source_cameras = list(dict.fromkeys(camera for order in camera_orders for camera in order))
    readers = []
    for camera in source_cameras:
        path, start = _video_path_and_start(dataset_root, info, episode, camera)
        readers.append(RawVideoReader(path, start, episode.length))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_slots = max(len(order) for order in camera_orders)
    width = row_width(camera_slots)
    height = ROW_HEIGHT * len(attention_paths)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{output_fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert encoder.stdin is not None
        for frame_index in range(episode.length):
            frames = {
                camera: reader.read() for camera, reader in zip(source_cameras, readers, strict=True)
            }
            image = compose_multi_camera_frame(
                frames,
                camera_orders,
                [records[frame_index] for records in filled_records],
                model_labels,
                frame_index=frame_index,
                source_fps=source_fps,
            )
            encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
            if frame_index % 600 == 0 or frame_index + 1 == episode.length:
                print(f"rendered {frame_index + 1}/{episode.length} frames", flush=True)
        encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            stderr = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
            raise RuntimeError(f"Video encoder failed with exit code {return_code}: {stderr}")
        for reader in readers:
            reader.close()
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        for reader in readers:
            reader.terminate()
        raise
