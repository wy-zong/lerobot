from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ActionChunk
from .methods import SmoothingMethod

SCORE_WEIGHTS = {
    "smoothness": 0.60,
    "high_frequency": 0.20,
    "runtime": 0.20,
}


@dataclass
class BenchmarkResult:
    per_chunk: pd.DataFrame
    summary: pd.DataFrame
    outputs: dict[tuple[int, int, str], np.ndarray]
    errors: list[dict[str, object]]


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95)) if values.size else 0.0


def _derivative(values: np.ndarray, order: int, fps: float) -> np.ndarray:
    return np.diff(values, n=order) * fps**order if len(values) > order else np.array([], dtype=np.float64)


def _frequency_ratios(values: np.ndarray, fps: float, threshold_hz: float = 10.0) -> tuple[float, float]:
    if len(values) < 4:
        return 0.0, 0.0
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / fps)
    non_dc = frequencies > 0.0
    total = float(power[non_dc].sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0, 0.0
    low = float(power[(frequencies > 0.0) & (frequencies <= threshold_hz)].sum() / total)
    high = float(power[frequencies > threshold_hz].sum() / total)
    return low, high


def _high_frequency_ratio(values: np.ndarray, fps: float, threshold_hz: float = 10.0) -> float:
    return _frequency_ratios(values, fps, threshold_hz)[1]


def _low_frequency_ratio(values: np.ndarray, fps: float, threshold_hz: float = 10.0) -> float:
    return _frequency_ratios(values, fps, threshold_hz)[0]


def _minmax_cost(series: pd.Series) -> pd.Series:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan)
    fill = float(values.max()) if values.notna().any() else 0.0
    values = values.fillna(fill)
    low, high = float(values.min()), float(values.max())
    if high - low <= np.finfo(np.float64).eps:
        return pd.Series(0.0, index=series.index)
    return (values - low) / (high - low)


def run_benchmark(chunks: list[ActionChunk], methods: list[SmoothingMethod], action_names: list[str], fps: float) -> BenchmarkResult:
    rows: list[dict[str, object]] = []
    outputs: dict[tuple[int, int, str], np.ndarray] = {}
    errors: list[dict[str, object]] = []

    for chunk in chunks:
        intervention_fraction = float(np.mean(chunk.intervention))
        segment = "intervention" if intervention_fraction == 1.0 else "autonomous" if intervention_fraction == 0.0 else "mixed"
        for method in methods:
            output, runtime_ms, error = method.apply(chunk.action, fps)
            key = (chunk.episode_index, chunk.chunk_index, method.method_id)
            outputs[key] = output
            if error:
                errors.append({"episode_index": chunk.episode_index, "chunk_index": chunk.chunk_index, "method_id": method.method_id, "error": error})
            for dimension, action_name in enumerate(action_names):
                raw = chunk.action[:, dimension]
                smooth = output[:, dimension]
                residual = smooth - raw
                velocity = _derivative(smooth, 1, fps)
                acceleration = _derivative(smooth, 2, fps)
                jerk = _derivative(smooth, 3, fps)
                low_frequency_ratio, high_frequency_ratio = _frequency_ratios(smooth, fps)
                low, high = float(raw.min()), float(raw.max())
                tolerance = max((high - low) * 1e-9, 1e-12)
                rows.append({
                    "episode_index": chunk.episode_index,
                    "chunk_index": chunk.chunk_index,
                    "first_frame_index": int(chunk.frame_indices[0]),
                    "sample_count": len(chunk.action),
                    "is_partial": chunk.is_partial,
                    "segment": segment,
                    "intervention_fraction": intervention_fraction,
                    "method_id": method.method_id,
                    "family": method.family,
                    "causal": method.causal,
                    "parameters": json.dumps(method.parameters, sort_keys=True),
                    "runtime_ms": runtime_ms,
                    "fallback_error": error or "",
                    "action_index": dimension,
                    "action_name": action_name,
                    "rmse": _rms(residual),
                    "mae": float(np.mean(np.abs(residual))),
                    "max_abs_error": float(np.max(np.abs(residual))),
                    "start_abs_error": abs(float(residual[0])),
                    "end_abs_error": abs(float(residual[-1])),
                    "velocity_rms": _rms(velocity),
                    "velocity_p95": _p95(velocity),
                    "acceleration_rms": _rms(acceleration),
                    "acceleration_p95": _p95(acceleration),
                    "jerk_rms": _rms(jerk),
                    "jerk_p95": _p95(jerk),
                    "low_frequency_ratio": low_frequency_ratio,
                    "high_frequency_ratio": high_frequency_ratio,
                    "range_violation_fraction": float(np.mean((smooth < low - tolerance) | (smooth > high + tolerance))),
                })

    per_chunk = pd.DataFrame(rows)
    summary = per_chunk.groupby(["method_id", "family", "causal", "parameters"], as_index=False, dropna=False).agg(
        chunk_count=("chunk_index", "count"),
        fallback_count=("fallback_error", lambda values: int((values != "").sum() / len(action_names))),
        runtime_mean_ms=("runtime_ms", "mean"),
        runtime_p95_ms=("runtime_ms", lambda values: float(np.percentile(values, 95))),
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        max_abs_error=("max_abs_error", "max"),
        endpoint_error=("start_abs_error", "mean"),
        acceleration_rms=("acceleration_rms", "mean"),
        jerk_rms=("jerk_rms", "mean"),
        low_frequency_ratio=("low_frequency_ratio", "mean"),
        high_frequency_ratio=("high_frequency_ratio", "mean"),
        range_violation_fraction=("range_violation_fraction", "mean"),
    )
    summary["chunk_count"] = len(chunks)
    raw = summary.loc[summary["method_id"] == "raw"].iloc[0]
    for metric in ("acceleration_rms", "jerk_rms", "high_frequency_ratio"):
        baseline = float(raw[metric])
        summary[f"{metric}_reduction_pct"] = 100.0 * (baseline - summary[metric]) / baseline if baseline else 0.0

    smoothness_cost = (_minmax_cost(summary["acceleration_rms"]) + _minmax_cost(summary["jerk_rms"])) / 2
    frequency_cost = _minmax_cost(summary["high_frequency_ratio"])
    runtime_cost = _minmax_cost(summary["runtime_p95_ms"])
    summary["deployment_score"] = 100.0 * (1.0 - (
        SCORE_WEIGHTS["smoothness"] * smoothness_cost
        + SCORE_WEIGHTS["high_frequency"] * frequency_cost
        + SCORE_WEIGHTS["runtime"] * runtime_cost
    ))
    summary = summary.sort_values(["deployment_score", "method_id"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return BenchmarkResult(per_chunk, summary, outputs, errors)
