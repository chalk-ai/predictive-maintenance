"""Features for Northfield Water's pump fleet.

Every sensor sample is one `SensorReading` row: a tag, a timestamp and a value, at whatever rate
that tag reports. `Pump` fuses them. Each aggregation filters the pump's readings to one tag and a
window ending at `_.chalk_now`, so tags sampled every 10 s and every 60 s land on the same time grid,
and the same definitions serve a live query (now) and an offline query at any past instant.

The risk and remaining-useful-life models are scikit-learn fits from train_model.py, written out as
native Chalk expressions over the fused features.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import chalk.functions as F
from chalk import DataFrame, Windowed, windowed
from chalk.features import FeatureTime, _, features, has_many

from pumps.model_weights import (
    RISK_INTERCEPT,
    RISK_WEIGHTS,
    RUL_CAP_H,
    RUL_INTERCEPT,
    RUL_WEIGHTS,
)
from pumps.sensors import RATED_CURRENT_A

HOUR = timedelta(hours=1)
SIX_HOURS = timedelta(hours=6)


@features
class SensorReading:
    """One historian sample. In production this is a SQL resolver over the historian table."""

    id: str
    pump_id: str
    tag: str
    value: float
    value_sq: float = _.value * _.value
    ts: FeatureTime


@features
class Pump:
    id: str
    station: str
    rated_current_a: float
    installed_at: datetime

    readings: DataFrame[SensorReading] = has_many(
        lambda: SensorReading.pump_id == Pump.id
    )

    # Samples received per tag: the tags arrive at different rates, which the fused features hide.
    vibration_samples: Windowed[int] = windowed(
        "10m",
        "1h",
        expression=_.readings[
            _.tag == "vibration", _.ts > _.chalk_window, _.ts <= _.chalk_now
        ].count(),
        default=0,
    )
    bearing_temp_samples: Windowed[int] = windowed(
        "10m",
        "1h",
        expression=_.readings[
            _.tag == "bearing_temp", _.ts > _.chalk_window, _.ts <= _.chalk_now
        ].count(),
        default=0,
    )

    # Vibration energy: RMS is the square root of the mean square over the window.
    vib_ms_1h: float = _.readings[
        _.value_sq,
        _.tag == "vibration",
        _.ts > _.chalk_now - HOUR,
        _.ts <= _.chalk_now,
    ].mean()
    vib_ms_6h: float = _.readings[
        _.value_sq,
        _.tag == "vibration",
        _.ts > _.chalk_now - SIX_HOURS,
        _.ts <= _.chalk_now,
    ].mean()
    vib_rms_1h: float = F.sqrt(_.vib_ms_1h)
    vib_rms_6h: float = F.sqrt(_.vib_ms_6h)

    speed_mean_1h: float = _.readings[
        _.value, _.tag == "shaft_speed", _.ts > _.chalk_now - HOUR, _.ts <= _.chalk_now
    ].mean()
    current_mean_1h: float = _.readings[
        _.value,
        _.tag == "motor_current",
        _.ts > _.chalk_now - HOUR,
        _.ts <= _.chalk_now,
    ].mean()
    pressure_mean_1h: float = _.readings[
        _.value,
        _.tag == "discharge_pressure",
        _.ts > _.chalk_now - HOUR,
        _.ts <= _.chalk_now,
    ].mean()
    bearing_temp_1h: float = _.readings[
        _.value, _.tag == "bearing_temp", _.ts > _.chalk_now - HOUR, _.ts <= _.chalk_now
    ].mean()
    bearing_temp_6h: float = _.readings[
        _.value,
        _.tag == "bearing_temp",
        _.ts > _.chalk_now - SIX_HOURS,
        _.ts <= _.chalk_now,
    ].mean()

    # Cross-sensor fusion. Vibration rises with load on a healthy pump, so it is judged per unit
    # of load, measured independently by motor current.
    load_frac_1h: float = _.current_mean_1h / RATED_CURRENT_A
    vib_per_load_1h: float = _.vib_rms_1h / F.greatest(_.load_frac_1h, 0.3)
    log_vib_per_load: float = F.ln(F.greatest(_.vib_per_load_1h, 0.05))
    vib_trend: float = _.vib_rms_1h / F.greatest(_.vib_rms_6h, 0.05)
    # The 1 h and 6 h window centers are 2.5 h apart.
    temp_slope_c_per_h: float = (_.bearing_temp_1h - _.bearing_temp_6h) / 2.5

    running: bool = _.speed_mean_1h > 100.0

    failure_risk_12h: float = F.sigmoid(
        RISK_INTERCEPT
        + RISK_WEIGHTS[0] * _.log_vib_per_load
        + RISK_WEIGHTS[1] * _.vib_trend
        + RISK_WEIGHTS[2] * _.temp_slope_c_per_h
    )
    rul_hours: float = F.least(
        RUL_CAP_H,
        F.greatest(
            0.0,
            RUL_INTERCEPT
            + RUL_WEIGHTS[0] * _.log_vib_per_load
            + RUL_WEIGHTS[1] * _.vib_trend
            + RUL_WEIGHTS[2] * _.temp_slope_c_per_h,
        ),
    )
    as_of: datetime = _.chalk_now
