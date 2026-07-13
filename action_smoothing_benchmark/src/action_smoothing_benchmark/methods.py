from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfilt, sosfilt_zi, sosfiltfilt

Array = np.ndarray


@dataclass(frozen=True)
class SmoothingMethod:
    method_id: str
    family: str
    causal: bool
    parameters: dict[str, float | int | str]
    transform: Callable[[Array, float], Array]

    def apply(self, action: Array, fps: float) -> tuple[Array, float, str | None]:
        start_ns = perf_counter_ns()
        try:
            result = np.asarray(self.transform(action, fps), dtype=np.float64)
            if result.shape != action.shape:
                raise ValueError(f"shape changed from {action.shape} to {result.shape}")
            if not np.isfinite(result).all():
                raise ValueError("result contains NaN or Inf")
            error = None
        except (ValueError, np.linalg.LinAlgError) as exc:
            result = action.copy()
            error = str(exc)
        runtime_ms = (perf_counter_ns() - start_ns) / 1_000_000
        return result, runtime_ms, error


def _raw(action: Array, _: float) -> Array:
    return action.copy()


def _polynomial(action: Array, degree: int) -> Array:
    if len(action) < degree + 1:
        raise ValueError(f"degree {degree} requires at least {degree + 1} samples")
    time = np.linspace(-1.0, 1.0, len(action), dtype=np.float64)
    design = np.stack([time**power for power in range(degree + 1)], axis=-1)
    coefficients = np.linalg.lstsq(design, action, rcond=None)[0]
    return design @ coefficients


def _ema(action: Array, alpha: float) -> Array:
    result = np.empty_like(action)
    result[0] = action[0]
    for index in range(1, len(action)):
        result[index] = alpha * action[index] + (1.0 - alpha) * result[index - 1]
    return result


def _butterworth(action: Array, fps: float, order: int, cutoff_hz: float, zero_phase: bool) -> Array:
    if cutoff_hz >= fps / 2:
        raise ValueError("cutoff must be below the Nyquist frequency")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fps, output="sos")
    if zero_phase:
        return sosfiltfilt(sos, action, axis=0)
    result = np.empty_like(action)
    base_zi = sosfilt_zi(sos)
    for dimension in range(action.shape[1]):
        zi = base_zi * action[0, dimension]
        result[:, dimension], _ = sosfilt(sos, action[:, dimension], zi=zi)
    return result


def _savgol(action: Array, window: int, polyorder: int) -> Array:
    if len(action) < window:
        raise ValueError(f"window {window} requires at least {window} samples")
    return savgol_filter(action, window_length=window, polyorder=polyorder, axis=0, mode="interp")


def _spline(action: Array) -> Array:
    if len(action) < 5:
        raise ValueError("smoothing spline requires at least 5 samples")
    time = np.linspace(0.0, 1.0, len(action), dtype=np.float64)
    result = np.empty_like(action)
    for dimension in range(action.shape[1]):
        spline = make_smoothing_spline(time, action[:, dimension], lam=None)
        result[:, dimension] = spline(time)
    return result


def build_method_catalog() -> list[SmoothingMethod]:
    methods = [SmoothingMethod("raw", "raw", True, {}, _raw)]
    for degree in range(1, 6):
        methods.append(SmoothingMethod(f"polynomial_{degree}", "polynomial", False, {"degree": degree}, lambda action, _fps, degree=degree: _polynomial(action, degree)))
    for alpha in (0.1, 0.2, 0.4):
        methods.append(SmoothingMethod(f"ema_alpha_{alpha:g}", "ema", True, {"alpha": alpha}, lambda action, _fps, alpha=alpha: _ema(action, alpha)))
    for zero_phase, label in ((False, "causal"), (True, "zero_phase")):
        for order in (2, 4):
            for cutoff_hz in (3.0, 6.0, 10.0):
                methods.append(SmoothingMethod(
                    f"butterworth_{label}_o{order}_fc{cutoff_hz:g}", f"butterworth_{label}", not zero_phase,
                    {"order": order, "cutoff_hz": cutoff_hz},
                    lambda action, fps, order=order, cutoff_hz=cutoff_hz, zero_phase=zero_phase: _butterworth(action, fps, order, cutoff_hz, zero_phase),
                ))
    for window in (7, 11, 15):
        for polyorder in (2, 3):
            methods.append(SmoothingMethod(
                f"savgol_w{window}_p{polyorder}", "savitzky_golay", False,
                {"window": window, "polyorder": polyorder},
                lambda action, _fps, window=window, polyorder=polyorder: savgol_filter(action, window, polyorder, axis=0, mode="interp") if len(action) >= window else (_ for _ in ()).throw(ValueError(f"window {window} requires at least {window} samples")),
            ))
    for sigma in (1.0, 2.0, 3.0):
        methods.append(SmoothingMethod(f"gaussian_sigma_{sigma:g}", "gaussian", False, {"sigma_steps": sigma}, lambda action, _fps, sigma=sigma: gaussian_filter1d(action, sigma=sigma, axis=0, mode="nearest")))
    methods.append(SmoothingMethod("smoothing_spline_gcv", "smoothing_spline", False, {"lambda": "GCV"}, lambda action, _fps: _spline(action)))
    return methods

