#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkpy[runtime]==2.155.2", "numpy", "pandas"]
# ///
"""Record the fleet's fused features and risk on a 30-minute grid with one offline query.

Every (pump, instant) row is queried with `input_times` set to that instant, so each row sees only
the sensor samples taken before it: exactly what the model would have seen live at that moment.
The result is checked against fusion_reference.py and written to results/.

    ./record_run.py
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_reference import fused_features
from pumps.sensors import DEMO_FLEET, HOURS, START

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
STEP_MIN = 30
CHECKED = [
    "vib_rms_1h",
    "vib_per_load_1h",
    "vib_trend",
    "temp_slope_c_per_h",
    "load_frac_1h",
]


def chalk_client(branch: str):
    from chalk.client import ChalkClient

    return ChalkClient(branch=branch)


def main() -> None:
    from pumps.features import Pump

    ticks_min = list(range(STEP_MIN, HOURS * 60 + 1, STEP_MIN))
    pump_ids = [p.id for p in DEMO_FLEET for _ in ticks_min]
    times = [START + timedelta(minutes=m) for _ in DEMO_FLEET for m in ticks_min]
    outputs = [
        Pump.id,
        Pump.station,
        Pump.as_of,
        Pump.running,
        Pump.vibration_samples["10m"],
        Pump.vibration_samples["1h"],
        Pump.bearing_temp_samples["10m"],
        Pump.bearing_temp_samples["1h"],
        Pump.vib_rms_1h,
        Pump.vib_rms_6h,
        Pump.speed_mean_1h,
        Pump.current_mean_1h,
        Pump.pressure_mean_1h,
        Pump.bearing_temp_1h,
        Pump.load_frac_1h,
        Pump.vib_per_load_1h,
        Pump.vib_trend,
        Pump.temp_slope_c_per_h,
        Pump.failure_risk_12h,
        Pump.rul_hours,
    ]
    client = chalk_client('predictive-maintenance')
    ds = client.offline_query(
        input={Pump.id: pump_ids},
        input_times=times,
        output=outputs,
        recompute_features=True,
        run_asynchronously=True,
        wait=True,
        show_progress=False,
    )
    df = ds.to_pandas()
    df.columns = [
        c.split(".", 1)[1] if c.startswith("pump.") else c for c in df.columns
    ]
    df = df.sort_values(["id", "as_of"]).reset_index(drop=True)
    print(
        f"offline query {ds.dataset_id or ds.revisions[0].revision_id}: {len(df)} rows"
    )
    OUT.mkdir(exist_ok=True)
    df.to_json(OUT / "fleet_grid.json", orient="records", date_format="iso", indent=0)

    # Parity with the numpy reference the model was trained on.
    worst = {}
    for spec in DEMO_FLEET:
        rows = df[df["id"] == spec.id]
        now_s = (
            (rows["as_of"] - pd.Timestamp(START))
            .dt.total_seconds()
            .round()
            .astype(int)
            .to_numpy()
        )
        ref = fused_features(spec, now_s)
        for k in CHECKED:
            got, want = rows[k].to_numpy(dtype=float), ref[k]
            ok = np.isfinite(want)
            worst[k] = max(
                worst.get(k, 0.0), float(np.nanmax(np.abs(got[ok] - want[ok])))
            )
    print("max |chalk - reference|:", {k: f"{v:.2e}" for k, v in worst.items()})

    meta = {
        "dataset_id": str(ds.dataset_id) if ds.dataset_id else None,
        "revision_ids": [str(r.revision_id) for r in ds.revisions],
        "branch": 'predictive-maintenance',
        "rows": len(df),
        "step_minutes": STEP_MIN,
        "max_abs_diff_vs_reference": worst,
    }
    (OUT / "query.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
