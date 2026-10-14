"""A numpy reference for the fused features in pumps/features.py.

train_model.py uses it to build the training set, and record_run.py uses it to check that the
features Chalk computed for the demo fleet match. Windows are (now - w, now], like
`_.ts > _.chalk_window, _.ts <= _.chalk_now`.
"""

from __future__ import annotations

import numpy as np

from pumps.sensors import RATED_CURRENT_A, TAGS, PumpSpec, tag_series

MODEL_INPUTS = ["log_vib_per_load", "vib_trend", "temp_slope_c_per_h"]


def _window_mean(
    t_s: np.ndarray, v: np.ndarray, now_s: np.ndarray, window_s: int
) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(t_s, now_s, side="right")
    lo = np.searchsorted(t_s, now_s - window_s, side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)


def fused_features(spec: PumpSpec, now_s: np.ndarray) -> dict[str, np.ndarray]:
    series = {tag: tag_series(spec, tag) for tag in TAGS}
    vt, vv = series["vibration"]
    vib_rms_1h = np.sqrt(_window_mean(vt, vv * vv, now_s, 3600))
    vib_rms_6h = np.sqrt(_window_mean(vt, vv * vv, now_s, 6 * 3600))
    speed_1h = _window_mean(*series["shaft_speed"], now_s, 3600)
    current_1h = _window_mean(*series["motor_current"], now_s, 3600)
    temp_1h = _window_mean(*series["bearing_temp"], now_s, 3600)
    temp_6h = _window_mean(*series["bearing_temp"], now_s, 6 * 3600)
    load_frac = np.maximum(current_1h / RATED_CURRENT_A, 0.3)
    vib_per_load = vib_rms_1h / load_frac
    return {
        "vib_rms_1h": vib_rms_1h,
        "vib_rms_6h": vib_rms_6h,
        "speed_mean_1h": speed_1h,
        "current_mean_1h": current_1h,
        "bearing_temp_1h": temp_1h,
        "load_frac_1h": current_1h / RATED_CURRENT_A,
        "vib_per_load_1h": vib_per_load,
        "log_vib_per_load": np.log(np.maximum(vib_per_load, 0.05)),
        "vib_trend": vib_rms_1h / np.maximum(vib_rms_6h, 0.05),
        # Centers of the 1h and 6h windows are 2.5 h apart.
        "temp_slope_c_per_h": (temp_1h - temp_6h) / 2.5,
    }
