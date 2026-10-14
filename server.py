"""The live API behind the predictive-maintenance demo page.

    GET /api/at?pump=P-104&minute=2160   fused features and failure risk as of that minute, computed
                                         by a Chalk online query with `now` pinned to that instant
    GET /api/raw?pump=P-104&minute=2160  the native-rate samples from the 10 minutes before it
    GET /healthz

The Chalk credentials are a service token that can only run online queries. Requests are capped per
client and in total, because the page is public.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import timedelta

import numpy as np
import uvicorn
from chalk.client import ChalkClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pumps.features import Pump
from pumps.sensors import DEMO_FLEET, HOURS, START, TAGS, tag_series

PUMP_IDS = {p.id for p in DEMO_FLEET}
SPECS = {p.id: p for p in DEMO_FLEET}
PER_CLIENT_PER_MIN = 20
MAX_CONCURRENT = 4
RAW_WINDOW_S = 600
# An idle branch server sleeps, and the first query after that takes ~30 s; one cheap query every
# few minutes keeps visitor queries at a few seconds.
KEEPALIVE_S = int(os.environ.get("PM_KEEPALIVE_S", "240"))

OUTPUTS = [
    Pump.as_of,
    Pump.running,
    Pump.vibration_samples["10m"],
    Pump.bearing_temp_samples["10m"],
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

client = ChalkClient(
    client_id=os.environ["PM_CHALK_CLIENT_ID"],
    client_secret=os.environ["PM_CHALK_CLIENT_SECRET"],
    api_server=os.environ["PM_CHALK_API_SERVER"],
    environment=os.environ["PM_CHALK_ENVIRONMENT_ID"],
    branch=os.environ["PM_CHALK_BRANCH"],
)
gate = asyncio.Semaphore(MAX_CONCURRENT)
recent: dict[str, deque[float]] = defaultdict(deque)


async def keep_branch_warm() -> None:
    while True:
        try:
            await asyncio.to_thread(
                client.query,
                input={Pump.id: "P-101"},
                output=[Pump.failure_risk_12h],
                now=START + timedelta(hours=HOURS),
                query_name="predictive-maintenance-keepalive",
            )
        except Exception as e:  # noqa: BLE001 - any failure must not end the keepalive loop
            print(f"keepalive failed: {e}", flush=True)
        await asyncio.sleep(KEEPALIVE_S)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(keep_branch_warm()) if KEEPALIVE_S > 0 else None
    yield
    if task is not None:
        task.cancel()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://(chalk\.ai|[a-z0-9-]+\.chalk\.dev|[a-z0-9-]+\.vercel\.app)|http://localhost:\d+",
    allow_methods=["GET"],
)


def _check(request: Request, pump: str, minute: int) -> None:
    if pump not in PUMP_IDS:
        raise HTTPException(400, "unknown pump")
    if not 1 <= minute <= HOURS * 60:
        raise HTTPException(400, f"minute must be in 1..{HOURS * 60}")
    ip = request.headers.get(
        "x-forwarded-for", request.client.host if request.client else "?"
    ).split(",")[0]
    now = time.monotonic()
    q = recent[ip]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= PER_CLIENT_PER_MIN:
        raise HTTPException(429, "slow down: 20 queries a minute per visitor")
    q.append(now)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/at")
async def at(request: Request, pump: str, minute: int) -> dict:
    _check(request, pump, minute)
    when = START + timedelta(minutes=minute)
    async with gate:
        t0 = time.perf_counter()
        result = await asyncio.to_thread(
            client.query,
            input={Pump.id: pump},
            output=OUTPUTS,
            now=when,
            query_name="predictive-maintenance-demo",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
    if result.errors:
        raise HTTPException(502, "; ".join(str(e.message) for e in result.errors)[:300])
    values = {
        k.split(".", 1)[1]: (v.isoformat() if hasattr(v, "isoformat") else v)
        for k, v in result.to_dict().items()
    }
    meta = result.meta
    return {
        "pump": pump,
        "minute": minute,
        "features": values,
        "latency_ms": round(elapsed_ms),
        "query_id": meta.query_id if meta else None,
    }


@app.get("/api/raw")
async def raw(request: Request, pump: str, minute: int) -> dict:
    _check(request, pump, minute)
    end = minute * 60
    out = {}
    for tag in TAGS:
        t_s, v = tag_series(SPECS[pump], tag)
        keep = (t_s > end - RAW_WINDOW_S) & (t_s <= end)
        out[tag] = {
            "t": (t_s[keep] - end).tolist(),
            "v": np.round(v[keep], 3).tolist(),
        }
    return {"pump": pump, "minute": minute, "window_s": RAW_WINDOW_S, "tags": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
