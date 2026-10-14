#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the demo's live API (server.py) as a ScalingGroup.

It needs a service token that can only run online queries, stored as a Chalk secret:

    chalk token create --name "predictive-maintenance demo" --permission query.online --output token.json
    jq -j .client_secret token.json | chalk secret set PREDICTIVE_MAINTENANCE_DEMO_CLIENT_SECRET
    ./deploy.py --client-id "$(jq -r .client_id token.json)"

Credentials for the deploy itself come from `chalk config` (or the CHALK_* env vars).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="predictive-maintenance")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--branch", default="predictive-maintenance")
    parser.add_argument(
        "--client-id", required=True, help="client id of the online-query-only token"
    )
    parser.add_argument("--secret", default="PREDICTIVE_MAINTENANCE_DEMO_CLIENT_SECRET")
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Secret

    image = (
        Image.debian_slim(python_version="3.12")
        .pip_install(
            [
                "chalkpy[runtime]==2.155.2",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
            ]
        )
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
    )
    for f in ("__init__.py", "features.py", "sensors.py", "model_weights.py"):
        image = image.add_local_file(
            str(HERE / "pumps" / f), f"/app/pumps/{f}", strategy="copy"
        )
    image = image.workdir("/app")

    sg = ScalingGroup(
        image=image,
        name=args.name,
        cpu="1",
        memory="2Gi",
        env={
            "PM_CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "PM_CHALK_ENVIRONMENT_ID": os.environ["CHALK_ENVIRONMENT_ID"],
            "PM_CHALK_BRANCH": args.branch,
            "PM_CHALK_CLIENT_ID": args.client_id,
            "PM_CHALK_CLIENT_SECRET": Secret.from_chalk_env(args.secret),
            "PORT": "8080",
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/healthz"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=1800.0, ready_timeout=900.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
