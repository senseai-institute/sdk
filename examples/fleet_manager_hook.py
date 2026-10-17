# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""What a fleet manager wires: the health endpoint, a telemetry upload hook, usage reports.

    uv run python examples/fleet_manager_hook.py

The upload hook is the only way telemetry leaves the robot. Here it just counts the spool
files it was offered and returns True (= delete after upload).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from _fixture import fixture_config
from senseai_sdk import SenseAI
from senseai_sdk.health import HealthServer

UPLOADED: list[str] = []


def upload(path: Path) -> bool:
    """Your fleet manager's upload. Return True to delete the spool file once it is safe."""
    UPLOADED.append(path.name)
    return True


def main() -> None:
    # the hook is named "module:function"; when this file runs as a script, make that name
    # resolve to this module instance so UPLOADED below is the list the hook appends to
    import sys

    sys.modules.setdefault("fleet_manager_hook", sys.modules[__name__])
    with fixture_config(telemetry={"enabled": True, "upload": "fleet_manager_hook:upload"}) as path:
        sai = SenseAI(path)
        server = HealthServer(sai.health, port=0)
        server.start()
        sai.start()  # threaded mode, replays at real time
        import time

        time.sleep(0.5)
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/health", timeout=5) as r:
            h = json.load(r)
        print("health:", h["state"], "ticks", h["ticks"], "reasons", h["reasons"][:1])
        sai.stop()
        server.stop()
        print("telemetry files offered to the upload hook:", UPLOADED)
        print("usage report:", json.dumps(sai.meter.reports[-1]["ticks"]))


if __name__ == "__main__":
    main()
