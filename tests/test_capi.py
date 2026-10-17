from __future__ import annotations

import re
import shutil
import struct
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import pytest

from senseai_sdk.capi import shim
from senseai_sdk.stages import FallbackCommand, Provenance

CAPI = Path(__file__).resolve().parents[1] / "src" / "senseai_sdk" / "capi"


def fallback(t: int = 5, action: str = "brake") -> FallbackCommand:
    return FallbackCommand(t, action, 0.25, -0.1, True, False, True, 1.5, 2.6, Provenance(reason="x"))  # type: ignore[arg-type]


def test_shm_layout_round_trip() -> None:
    w = shim.ShmWriter()
    try:
        reader = shared_memory.SharedMemory(name=w.name)
        assert reader.buf is not None
        magic, version = struct.unpack_from("<II", reader.buf, 0)
        assert magic == shim.MAGIC and version == 1
        assert struct.unpack_from("<Q", reader.buf, shim.OFF_SEQ)[0] == 0
        assert w.write_command(fallback()) == 1
        assert w.write_command(fallback(t=6)) == 2
        seq = struct.unpack_from("<Q", reader.buf, shim.OFF_SEQ)[0]
        assert seq == 4  # even: stable
        t, action, scale, lat, stop, hold, latched, degraded, risk, thr = struct.unpack_from(
            "<qiffBBBBff", reader.buf, shim.OFF_CMD
        )
        assert (t, action, stop, hold, latched, degraded) == (6, 2, 1, 0, 1, 1)
        assert scale == pytest.approx(0.25) and lat == pytest.approx(-0.1)
        assert risk == pytest.approx(1.5) and thr == pytest.approx(2.6)
        for k in range(70):
            w.write_event(k, "track.new", f"event {k}")
        assert struct.unpack_from("<Q", reader.buf, shim.OFF_EVT_HEAD)[0] == 70
        slot = shim.OFF_EVT_RING + (69 % shim.EVENT_SLOTS) * shim.EVENT_SLOT_BYTES
        t_ns, kind, text = struct.unpack_from("<q32s480s", reader.buf, slot)
        assert t_ns == 69 and kind.rstrip(b"\0") == b"track.new" and text.rstrip(b"\0") == b"event 69"
        reader.close()
    finally:
        w.close()


def test_shim_protocol_over_pipes(config_path: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "senseai_sdk.capi.shim", "--config", str(config_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    ready = proc.stdout.readline().split()
    assert ready[0] == "READY" and int(ready[2]) == shim.SHM_SIZE
    shm = shared_memory.SharedMemory(name=ready[1])
    assert shm.buf is not None

    def cmd(line: str) -> str:
        assert proc.stdin and proc.stdout
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        return proc.stdout.readline().strip()

    assert cmd("PROFILE panther 0") == "OK"
    assert cmd("STEP") == "OK 0"
    assert cmd("STEP") == "OK 1"
    assert struct.unpack_from("<Q", shm.buf, shim.OFF_SEQ)[0] == 4
    assert cmd("HEALTH").startswith('{"state":')
    assert cmd("BOGUS").startswith("ERROR unknown op")
    assert cmd("PROFILE hawk 0").startswith("ERROR")
    n = 2
    while True:
        r = cmd("STEP")
        if r == "END":
            break
        n += 1
    assert n == 50
    assert cmd("RUN") == "OK"
    assert cmd("STEP").startswith("ERROR running")
    assert cmd("QUIT") == "OK"
    shm.close()
    assert proc.wait(timeout=30) == 0


def test_shim_reports_config_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("endpoint_id: x\n")
    out = subprocess.run(
        [sys.executable, "-m", "senseai_sdk.capi.shim", "--config", str(bad)],
        input="",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 1 and out.stdout.startswith("ERROR ConfigError")


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_example_c_builds_and_runs(config_path: Path, tmp_path: Path) -> None:
    exe = tmp_path / "example"
    build = subprocess.run(
        ["cc", "-Wall", "-Wextra", "-o", str(exe), str(CAPI / "example.c"), str(CAPI / "senseai.c"), "-lrt"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    env = {
        "SENSEAI_PYTHON": sys.executable,
        "PATH": "/usr/bin:/bin",
        **{k: v for k, v in __import__("os").environ.items() if k.startswith("SENSEAI_")},
    }
    run = subprocess.run(
        [str(exe), str(config_path), "20"], capture_output=True, text=True, timeout=300, env=env
    )
    assert run.returncode == 0, run.stderr
    # the real reflex on a quiet fixture: no action, not degraded (it was the watchdog's brake before reflex
    # shipped). Later ticks may brake if a loaded machine starves the world stage, so only the count's shape
    # is checked there.
    assert "tick 1 action 0 speed_scale 1.00 stop 0 degraded 0" in run.stdout
    assert re.search(r"^20 steps, \d+ with stop requested$", run.stdout, re.M)
    assert 'health: {"state":' in run.stdout
