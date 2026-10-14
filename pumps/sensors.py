"""A seeded, fictional sensor historian for Northfield Water's pumping stations.

Every pump reports five tags at different rates, the way a real historian does:

| Tag            | Unit  | Every |
| -------------- | ----- | ----- |
| vibration      | mm/s  | 10 s  |
| motor_current  | A     | 15 s  |
| shaft_speed    | rpm   | 30 s  |
| discharge_psi  | bar   | 30 s  |
| bearing_temp   | °C    | 60 s  |

Demand follows a daily cycle, so speed, current, pressure and vibration all move together with load.
A bearing fault adds vibration energy, heat and a little current on top of that, growing until the
pump trips. The same generator produces the demo fleet (fixed below) and the training fleet used by
train_model.py, so the model never sees the demo pumps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

START = datetime(2026, 10, 10, tzinfo=timezone.utc)
HOURS = 48

TAGS: dict[str, int] = {
    "vibration": 10,
    "motor_current": 15,
    "shaft_speed": 30,
    "discharge_pressure": 30,
    "bearing_temp": 60,
}
RATED_CURRENT_A = 92.0
MAX_RPM = 1780.0


@dataclass(frozen=True)
class PumpSpec:
    id: str
    station: str
    load_offset: float
    vib_baseline: float
    seed: int
    fault_onset_h: float | None
    failure_h: float | None


DEMO_FLEET: list[PumpSpec] = [
    PumpSpec("P-101", "Riverside", 0.00, 1.10, 101, None, None),
    PumpSpec("P-102", "Riverside", -0.08, 0.95, 102, None, None),
    PumpSpec("P-103", "Hillcrest", 0.05, 1.20, 103, None, None),
    PumpSpec("P-104", "Hillcrest", 0.02, 1.05, 104, 18.0, 44.0),
    PumpSpec("P-105", "Old Mill", -0.12, 1.00, 105, None, None),
    PumpSpec("P-106", "Old Mill", 0.30, 1.15, 106, None, None),
    PumpSpec("P-107", "Lakeview", 0.08, 1.25, 107, None, None),
    PumpSpec("P-108", "Lakeview", -0.04, 0.90, 108, None, None),
]


def severity(spec: PumpSpec, t_h: np.ndarray) -> np.ndarray:
    """0 before the fault starts, rising convexly to 1 at failure."""
    if spec.fault_onset_h is None or spec.failure_h is None:
        return np.zeros_like(t_h)
    x = np.clip(
        (t_h - spec.fault_onset_h) / (spec.failure_h - spec.fault_onset_h), 0.0, 1.0
    )
    return x**2.2


def load(spec: PumpSpec, t_h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    daily = 0.55 + 0.28 * np.sin(2 * math.pi * (t_h - 6.0) / 24.0)
    # Slow random drift, the same for every tag of a pump at a given time.
    knots = rng.normal(0.0, 0.05, size=HOURS + 2)
    drift = np.interp(t_h, np.arange(HOURS + 2), knots)
    return np.clip(daily + spec.load_offset + drift, 0.15, 1.0)


def running(spec: PumpSpec, t_h: np.ndarray) -> np.ndarray:
    if spec.failure_h is None:
        return np.ones_like(t_h, dtype=bool)
    return t_h < spec.failure_h


def tag_series(spec: PumpSpec, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """(seconds since START, value) for one tag of one pump. Deterministic for a given spec."""
    step = TAGS[tag]
    t_s = np.arange(step, HOURS * 3600 + 1, step, dtype=np.int64)
    t_h = t_s / 3600.0
    # Load uses its own fixed stream so every tag sees the same demand curve.
    ld = load(spec, t_h, np.random.default_rng(spec.seed))
    rng = np.random.default_rng([spec.seed, list(TAGS).index(tag)])
    sev = severity(spec, t_h)
    on = running(spec, t_h)
    rpm = 1450.0 + 330.0 * ld
    if tag == "shaft_speed":
        v = rpm + rng.normal(0, 6, t_s.size)
        v = np.where(on, v, 0.0)
    elif tag == "motor_current":
        v = (
            RATED_CURRENT_A * (0.42 + 0.55 * ld)
            + 7.0 * sev
            + rng.normal(0, 1.2, t_s.size)
        )
        v = np.where(on, v, 0.0)
    elif tag == "discharge_pressure":
        v = 3.6 + 2.1 * ld + rng.normal(0, 0.06, t_s.size)
        v = np.where(on, v, 0.3 + rng.normal(0, 0.02, t_s.size))
    elif tag == "vibration":
        # Running vibration scales with hydraulic load, which is also what drives motor current.
        amp = spec.vib_baseline * 1.6 * (0.42 + 0.55 * ld) * (1.0 + 6.5 * sev)
        # Bearing defects show up as intermittent impacts on top of the running vibration.
        impacts = (rng.random(t_s.size) < 0.05 + 0.25 * sev) * rng.normal(
            0, 3.0 * sev + 1e-9, t_s.size
        )
        v = amp * math.sqrt(2) * np.sin(rng.uniform(0, 2 * math.pi, t_s.size)) + impacts
        v = v + rng.normal(0, 0.05, t_s.size)
        v = np.where(on, v, rng.normal(0, 0.04, t_s.size))
    elif tag == "bearing_temp":
        ambient = 22.0 + 3.0 * np.sin(2 * math.pi * (t_h - 9.0) / 24.0)
        target = np.where(on, ambient + 16.0 + 14.0 * ld + 24.0 * sev, ambient)
        # First-order thermal lag with a ~25 minute time constant.
        alpha = 1.0 - math.exp(-step / (25 * 60))
        temp = np.empty_like(target)
        cur = target[0]
        for i, tgt in enumerate(target):
            cur += alpha * (tgt - cur)
            temp[i] = cur
        v = temp + rng.normal(0, 0.15, t_s.size)
    else:
        raise ValueError(tag)
    return t_s, v.astype(np.float64)


def timestamp(seconds: np.ndarray) -> list[datetime]:
    return [START + timedelta(seconds=int(s)) for s in seconds]


def random_fleet(n: int, seed: int) -> list[PumpSpec]:
    """A training fleet: random load levels, and a bearing fault on about 60% of pumps."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        faulty = rng.random() < 0.6
        onset = float(rng.uniform(2, 34)) if faulty else None
        fail = (
            float(min(onset + rng.uniform(14, 34), HOURS + 30))
            if onset is not None
            else None
        )
        out.append(
            PumpSpec(
                id=f"T-{i:03d}",
                station="training",
                load_offset=float(rng.uniform(-0.2, 0.35)),
                vib_baseline=float(rng.uniform(0.85, 1.3)),
                seed=10_000 + i,
                fault_onset_h=onset,
                failure_h=fail,
            )
        )
    return out
