"""Pure helpers shared by attention inference, resume handling, and rendering."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_DATASET = "wuc1/bi_so101_ffp_0615-14-12-dagger_merged"
DEFAULT_MODELS = (
    "wuc1/bi_so101_ffp_0615-14-12-dagger_merged3cam_discrete_state",
    "wuc1/bi_so101_ffp_0615-14-12_merged",
    "wuc1/bi_so101_ffp_0615-14-12-dagger_merged3cam",
)


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def task(self) -> str:
        return self.tasks[0] if self.tasks else ""

    def manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("metadata")
        return value


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def hf_hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", _hf_home() / "hub"))


def hf_lerobot_home() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", _hf_home() / "lerobot"))


def _repo_cache_name(repo_id: str, repo_type: str) -> str:
    prefix = "models" if repo_type == "model" else "datasets"
    return f"{prefix}--{repo_id.replace('/', '--')}"


def snapshot_revision(path: Path) -> str | None:
    path = path.resolve()
    return path.name if path.parent.name == "snapshots" else None


def resolve_local_snapshot(
    repo_or_path: str | Path,
    *,
    repo_type: str,
    revision: str | None = None,
    required_files: Iterable[str] = (),
) -> Path:
    """Resolve a local path or Hub cache entry without making a network call."""

    candidate = Path(repo_or_path).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
    else:
        repo_id = str(repo_or_path)
        cache_root = hf_hub_cache() / _repo_cache_name(repo_id, repo_type)
        snapshots = cache_root / "snapshots"
        if not snapshots.is_dir():
            raise FileNotFoundError(f"No local {repo_type} cache found for {repo_id}: {snapshots}")

        selected_revision = revision
        main_ref = cache_root / "refs" / "main"
        if selected_revision is None and main_ref.is_file():
            selected_revision = main_ref.read_text().strip()
        if selected_revision and (snapshots / selected_revision).is_dir():
            resolved = (snapshots / selected_revision).resolve()
        elif revision is not None:
            available = sorted(path.name for path in snapshots.iterdir() if path.is_dir())
            names = ", ".join(available) or "none"
            raise FileNotFoundError(
                f"Cached revision {revision} is unavailable for {repo_id}; cached snapshots: {names}"
            )
        else:
            available = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if len(available) != 1:
                names = ", ".join(path.name for path in available) or "none"
                raise FileNotFoundError(
                    f"Cannot choose a cached revision for {repo_id}; cached snapshots: {names}"
                )
            resolved = available[0].resolve()

    missing = [name for name in required_files if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Local cache at {resolved} is incomplete; missing: {', '.join(missing)}")
    return resolved


def resolve_local_dataset(repo_or_path: str | Path) -> Path:
    """Resolve a complete LeRobot dataset, preferring the legacy local dataset root."""

    candidate = Path(repo_or_path).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
    else:
        legacy = hf_lerobot_home() / str(repo_or_path)
        if (legacy / "meta" / "info.json").is_file():
            resolved = legacy.resolve()
        else:
            lerobot_hub = hf_lerobot_home() / "hub"
            cache_name = _repo_cache_name(str(repo_or_path), "dataset")
            original_hub_cache = os.environ.get("HF_HUB_CACHE")
            try:
                os.environ["HF_HUB_CACHE"] = str(lerobot_hub)
                resolved = resolve_local_snapshot(
                    repo_or_path,
                    repo_type="dataset",
                    required_files=("meta/info.json", "meta/tasks.parquet"),
                )
            finally:
                if original_hub_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = original_hub_cache
            if resolved.parent.parent.name != cache_name and not (resolved / "meta" / "info.json").is_file():
                raise FileNotFoundError(f"No complete local dataset found for {repo_or_path}")

    required = ("meta/info.json", "meta/tasks.parquet", "data", "videos")
    missing = [name for name in required if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(f"Local dataset at {resolved} is incomplete; missing: {', '.join(missing)}")
    return resolved


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "meta" / "info.json").read_text())


def load_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found under {dataset_root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def select_episode(dataset_root: Path, episode: str | int = "auto-longest") -> EpisodeInfo:
    rows = load_episode_rows(dataset_root)
    if episode == "auto-longest":
        row = max(rows, key=lambda value: (int(value["length"]), -int(value["episode_index"])))
    else:
        episode_index = int(episode)
        try:
            row = next(value for value in rows if int(value["episode_index"]) == episode_index)
        except StopIteration as error:
            raise ValueError(f"Episode {episode_index} is not present in {dataset_root}") from error

    return EpisodeInfo(
        episode_index=int(row["episode_index"]),
        length=int(row["length"]),
        dataset_from_index=int(row["dataset_from_index"]),
        dataset_to_index=int(row["dataset_to_index"]),
        tasks=tuple(row.get("tasks") or ()),
        metadata=row,
    )


def dataset_camera_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") in {"video", "image"}]


def episode_camera_keys(info: dict[str, Any], episode: EpisodeInfo) -> list[str]:
    """Return cameras backed by both info.json and legacy episode metadata.

    Some locally cached dataset revisions removed a camera from ``info.json``
    while retaining its video files and per-episode location columns. The
    four-camera SmolVLA revision needs that historical camera.
    """

    visible = dataset_camera_keys(info)
    prefix = "videos/"
    suffix = "/chunk_index"
    hidden = sorted(
        key[len(prefix) : -len(suffix)]
        for key in episode.metadata
        if key.startswith(prefix) and key.endswith(suffix)
        and key[len(prefix) : -len(suffix)] not in visible
    )
    return [*visible, *hidden]


def build_sample_frames(frame_count: int, source_fps: float, sample_fps: float) -> list[int]:
    if frame_count <= 0 or source_fps <= 0 or sample_fps <= 0:
        raise ValueError("frame_count, source_fps, and sample_fps must be positive")
    interval = source_fps / sample_fps
    rounded_interval = round(interval)
    if rounded_interval < 1 or abs(interval - rounded_interval) > 1e-9:
        raise ValueError(
            f"sample_fps={sample_fps:g} must divide source_fps={source_fps:g} into an integer interval"
        )
    return list(range(0, frame_count, rounded_interval))


def _camera_number(key: str) -> str | None:
    match = re.search(r"camera(\d+)$", key)
    return match.group(1) if match else None


def map_model_cameras(dataset_keys: list[str], model_keys: list[str]) -> dict[str, str]:
    """Map each model camera key to a distinct source camera deterministically."""

    if len(dataset_keys) < len(model_keys):
        raise ValueError(f"Dataset has {len(dataset_keys)} cameras but model expects {len(model_keys)}")

    available = list(dataset_keys)
    result: dict[str, str] = {}
    for model_key in model_keys:
        if model_key in available:
            source = model_key
        else:
            number = _camera_number(model_key)
            numbered = [key for key in available if _camera_number(key) == number]
            source = numbered[0] if numbered else available[0]
        result[model_key] = source
        available.remove(source)
    return result


def aggregate_attention_trace(trace: dict[str, Any], camera_names: list[str]) -> dict[str, Any]:
    """Aggregate expert cross-attention over layer, head, action query, and denoise step."""

    details = [
        entry
        for entry in trace.get("attention", [])
        if entry.get("kind") == "expert_cross" and entry.get("fill_kv_cache") is False
    ]
    if not details:
        raise ValueError("Trace contains no expert cross-attention entries")

    traced_groups = trace.get("token_groups", {})
    group_names = [*camera_names]
    if "state" in traced_groups:
        group_names.append("state")
    if "language" in traced_groups:
        group_names.append("language")
    raw = {
        name: sum(float(entry.get("group_mass", {}).get(name, 0.0)) for entry in details) / len(details)
        for name in group_names
    }
    grouped_total = sum(raw.values())
    if grouped_total <= 0:
        raise ValueError("Grouped expert cross-attention mass is zero")
    percentages = {name: value * 100.0 / grouped_total for name, value in raw.items()}
    image_raw = sum(raw[name] for name in camera_names)
    image_percent = sum(percentages[name] for name in camera_names)

    return {
        "raw_attention_mass": raw,
        "raw_grouped_total": grouped_total,
        "attention_percent": percentages,
        "image_total_raw_attention_mass": image_raw,
        "image_total_percent": image_percent,
        "layer_step_details": details,
    }


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number == len(lines):
                break  # Ignore a partial final line left by interruption.
            raise
    return records


def forward_fill_records(records: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("Cannot forward-fill an empty attention record list")
    by_frame = {int(record["frame_index"]): record for record in records}
    if 0 not in by_frame:
        raise ValueError("Attention records must include frame 0 before rendering")

    result = []
    current = by_frame[0]
    for frame_index in range(frame_count):
        current = by_frame.get(frame_index, current)
        result.append(current)
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
