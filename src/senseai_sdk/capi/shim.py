# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The Python side of the C API: a worker process the C client drives over pipes.

Protocol (one line per message, stdin → stdout):

- on start: ``READY <shm_name> <size>`` or ``ERROR <message>``
- ``STEP`` → ``OK <seq>`` | ``END`` | ``ERROR …``    (single-loop tick)
- ``RUN`` → ``OK``                                    (threaded mode; STEP is refused after)
- ``PROFILE <name> <version>`` (0 = newest) → ``OK`` | ``ERROR …``
- ``HEALTH`` → one line of JSON
- ``QUIT`` → ``OK`` and exit

Shared memory (``SHM_LAYOUT``), written by this process, read by the C client:

====== ====== ============================================================
offset bytes  field
====== ====== ============================================================
0      4      magic ``0x53414931`` ("SAI1")
4      4      layout version (1)
8      8      seqlock: odd while a command is being written; ``seq // 2`` = tick count
16     48     command: int64 t_ns, int32 action, f32 speed_scale, f32 lateral, u8 stop,
              u8 hold, u8 latched, u8 degraded, f32 risk, f32 threshold, pad to 48
64     8      events head: total events written (uint64)
72     64×528 event ring: int64 t_ns, char[32] type, char[480] text, 8 pad
====== ====== ============================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import struct
import sys
from collections.abc import Sequence
from multiprocessing import shared_memory
from typing import Any

from senseai_sdk.errors import SenseAIError

__all__ = ["SHM_LAYOUT", "ShmWriter", "main", "pack_command"]

ACTIONS = {"none": 0, "slow": 1, "brake": 2, "swerve": 3, "retract": 4, "hold": 5}
MAGIC = 0x53414931
VERSION = 1
EVENT_SLOTS = 64
EVENT_SLOT_BYTES = 528
OFF_SEQ, OFF_CMD, OFF_EVT_HEAD, OFF_EVT_RING = 8, 16, 64, 72
SHM_SIZE = OFF_EVT_RING + EVENT_SLOTS * EVENT_SLOT_BYTES
SHM_LAYOUT = {
    "magic": 0,
    "version": 4,
    "seq": 8,
    "command": 16,
    "events_head": 64,
    "events_ring": 72,
    "size": SHM_SIZE,
}

_CMD = struct.Struct("<qiffBBBBff16x")  # 48 bytes
_EVT = struct.Struct("<q32s480s8x")  # 528 bytes


def pack_command(cmd: Any) -> bytes:
    """A reflex (or fallback) command as the 48-byte record."""
    return _CMD.pack(
        int(cmd.t_master_ns),
        ACTIONS.get(str(cmd.action), 0),
        float(cmd.speed_scale),
        float(cmd.lateral_offset_m),
        1 if cmd.stop else 0,
        1 if cmd.hold else 0,
        1 if getattr(cmd, "latched", False) else 0,
        1 if getattr(cmd, "degraded", False) else 0,
        float(getattr(cmd, "risk", 0.0)),
        float(getattr(cmd, "threshold", 0.0)),
    )


class ShmWriter:
    """Owns the segment; writes commands under the seqlock and events into the ring."""

    def __init__(self, name: str | None = None) -> None:
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=SHM_SIZE)
        assert self.shm.buf is not None
        self.buf: memoryview = self.shm.buf
        buf = self.buf
        buf[:SHM_SIZE] = bytes(SHM_SIZE)
        struct.pack_into("<II", buf, 0, MAGIC, VERSION)
        self._seq = 0
        self._events = 0

    @property
    def name(self) -> str:
        """The segment name (without the leading slash)."""
        return self.shm.name

    def write_command(self, cmd: Any) -> int:
        """Publish ``cmd``; returns the tick count."""
        buf = self.buf
        self._seq += 1  # odd: writing
        struct.pack_into("<Q", buf, OFF_SEQ, self._seq)
        buf[OFF_CMD : OFF_CMD + _CMD.size] = pack_command(cmd)
        self._seq += 1  # even: stable
        struct.pack_into("<Q", buf, OFF_SEQ, self._seq)
        return self._seq // 2

    def write_event(self, t_ns: int, kind: str, text: str) -> None:
        """Append to the ring (overwrites the oldest after 64)."""
        slot = OFF_EVT_RING + (self._events % EVENT_SLOTS) * EVENT_SLOT_BYTES
        self.buf[slot : slot + _EVT.size] = _EVT.pack(int(t_ns), kind.encode()[:31], text.encode()[:479])
        self._events += 1
        struct.pack_into("<Q", self.buf, OFF_EVT_HEAD, self._events)

    def close(self) -> None:
        """Unlink the segment."""
        self.buf.release()
        self.shm.close()
        with contextlib.suppress(FileNotFoundError):
            self.shm.unlink()


_proto: Any = None


def _reply(line: str) -> None:
    out = _proto or sys.stdout
    out.write(line + "\n")
    out.flush()


def _claim_stdout() -> None:
    """Keep fd 1 for the protocol only: everything else (library logs) goes to stderr.

    Upstream packages log to stdout (ingest's structlog printer, torch warnings); one stray
    line would corrupt the reply stream, so the protocol takes a private copy of fd 1 and
    fd 1 itself is pointed at stderr.
    """
    global _proto
    import os

    fd = os.dup(1)
    os.dup2(2, 1)
    _proto = os.fdopen(fd, "w", buffering=1)
    sys.stdout = sys.stderr


def serve(config_path: str, inp: Any = None, shm_name: str | None = None) -> int:
    """Run the worker loop until QUIT or EOF. Returns the exit code."""
    inp = inp or sys.stdin
    _claim_stdout()
    try:
        from senseai_sdk.api import SenseAI

        sai = SenseAI(config_path)
    except (SenseAIError, OSError) as e:
        _reply(f"ERROR {type(e).__name__}: {e}")
        return 1
    writer = ShmWriter(shm_name)

    def on_command(cmd: Any) -> None:
        writer.write_command(cmd)

    sai.on_command(on_command)
    sai.on_event(
        lambda ev: writer.write_event(
            int(getattr(ev, "t_master_ns", 0)), str(getattr(ev, "type", "")), str(getattr(ev, "text", ev))
        )
    )
    _reply(f"READY {writer.name} {SHM_SIZE}")
    running = False
    try:
        for raw in inp:
            line = raw.strip()
            if not line:
                continue
            op, _, rest = line.partition(" ")
            if op == "QUIT":
                _reply("OK")
                break
            if op == "STEP":
                if running:
                    _reply("ERROR running: STEP is not allowed after RUN")
                    continue
                try:
                    res = sai.step()
                except StopIteration:
                    _reply("END")
                    continue
                except SenseAIError as e:
                    _reply(f"ERROR {e}")
                    continue
                _reply(f"OK {res.tick_index}")
            elif op == "RUN":
                sai.start()
                running = True
                _reply("OK")
            elif op == "PROFILE":
                name, _, ver = rest.partition(" ")
                try:
                    sai.load_profile(name, int(ver) if ver and int(ver) > 0 else None)
                    _reply("OK")
                except SenseAIError as e:
                    _reply(f"ERROR {e}")
            elif op == "HEALTH":
                _reply(json.dumps(sai.health().to_json(), separators=(",", ":")))
            else:
                _reply(f"ERROR unknown op {op!r}")
    finally:
        sai.stop()
        writer.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m senseai_sdk.capi.shim --config senseai.yaml``."""
    ap = argparse.ArgumentParser(prog="senseai_sdk.capi.shim")
    ap.add_argument("--config", required=True)
    ap.add_argument("--shm-name", default=None)
    a = ap.parse_args(argv)
    return serve(a.config, shm_name=a.shm_name)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
