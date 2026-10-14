#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy"]
# ///
"""Condense results/ and the sensor historian into the data behind the demo page.

./export_page_data.py > results/page_data.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pumps.model_weights import HOLDOUT_AUC, RISK_HORIZON_H
from pumps.sensors import DEMO_FLEET, HOURS, TAGS, tag_series

HERE = Path(__file__).resolve().parent
TRACE_BUCKET_S = 600
SERIES = [
    "failure_risk_12h",
    "rul_hours",
    "vib_rms_1h",
    "vib_per_load_1h",
    "vib_trend",
    "temp_slope_c_per_h",
    "load_frac_1h",
    "bearing_temp_1h",
]


def bucketed(spec, tag: str) -> list[float]:
    t_s, v = tag_series(spec, tag)
    idx = (t_s - 1) // TRACE_BUCKET_S
    n = HOURS * 3600 // TRACE_BUCKET_S
    if tag == "vibration":
        sums = np.bincount(idx, weights=v * v, minlength=n)
        out = np.sqrt(sums / np.bincount(idx, minlength=n))
    else:
        out = np.bincount(idx, weights=v, minlength=n) / np.bincount(idx, minlength=n)
    return [round(float(x), 3) for x in out[:n]]


def main() -> None:
    grid = json.loads((HERE / "results" / "fleet_grid.json").read_text())
    query = json.loads((HERE / "results" / "query.json").read_text())
    pumps = []
    for spec in DEMO_FLEET:
        rows = [r for r in grid if r["id"] == spec.id]
        rows.sort(key=lambda r: r["as_of"])
        pumps.append(
            {
                "id": spec.id,
                "station": spec.station,
                "failureHour": spec.failure_h,
                "series": {
                    k: [None if r[k] is None else round(float(r[k]), 4) for r in rows]
                    for k in SERIES
                },
                "running": [bool(r["running"]) for r in rows],
                "traces": {tag: bucketed(spec, tag) for tag in TAGS},
            }
        )
    out = {
        "stepMinutes": query["step_minutes"],
        "traceBucketSeconds": TRACE_BUCKET_S,
        "hours": HOURS,
        "riskHorizonHours": RISK_HORIZON_H,
        "holdoutAuc": HOLDOUT_AUC,
        "datasetId": query["dataset_id"],
        "rows": query["rows"],
        "maxAbsDiffVsReference": max(query["max_abs_diff_vs_reference"].values()),
        "sampleSeconds": TAGS,
        "pumps": pumps,
    }
    json.dump(out, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    main()
