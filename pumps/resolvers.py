"""Resolvers over the fictional fleet and its seeded sensor historian.

`all_readings` returns every sample ever recorded, each stamped with when it was taken. It does no
time filtering: the `_.chalk_now` windows in features.py do that, so the same resolver serves the
present and any point in the past. In a real deployment it would be a SQL resolver over the
historian's table.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
from chalk import DataFrame, online
from chalk.features import Features

from pumps.features import Pump, SensorReading
from pumps.sensors import DEMO_FLEET, RATED_CURRENT_A, START, TAGS, tag_series

_SPECS = {spec.id: spec for spec in DEMO_FLEET}
_INSTALLED = datetime(2019, 4, 1, tzinfo=timezone.utc)


@online
def get_pump(
    id: Pump.id,
) -> Features[Pump.station, Pump.rated_current_a, Pump.installed_at]:
    return Pump(
        station=_SPECS[id].station,
        rated_current_a=RATED_CURRENT_A,
        installed_at=_INSTALLED,
    )


@online
def all_readings() -> DataFrame[
    SensorReading.id,
    SensorReading.pump_id,
    SensorReading.tag,
    SensorReading.value,
    SensorReading.ts,
]:
    ids: list[str] = []
    pumps: list[str] = []
    tags: list[str] = []
    values: list[np.ndarray] = []
    stamps: list[np.ndarray] = []
    start = np.datetime64(START.replace(tzinfo=None), "s")
    for spec in DEMO_FLEET:
        for tag in TAGS:
            t_s, v = tag_series(spec, tag)
            ids.extend(f"{spec.id}:{tag}:{int(s)}" for s in t_s)
            pumps.extend([spec.id] * t_s.size)
            tags.extend([tag] * t_s.size)
            values.append(v)
            stamps.append(start + t_s.astype("timedelta64[s]"))
    ts = np.concatenate(stamps).astype("datetime64[us]")
    table = pa.table(
        {
            str(SensorReading.id): pa.array(ids, pa.large_utf8()),
            str(SensorReading.pump_id): pa.array(pumps, pa.large_utf8()),
            str(SensorReading.tag): pa.array(tags, pa.large_utf8()),
            str(SensorReading.value): pa.array(np.concatenate(values), pa.float64()),
            str(SensorReading.ts): pa.array(ts, pa.timestamp("us", tz="UTC")),
        }
    )
    return DataFrame(table)
