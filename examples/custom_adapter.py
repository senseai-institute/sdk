# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Bind your own sensor: any ``senseai_ingest`` adapter can be added or swapped at runtime.

    uv run python examples/custom_adapter.py

The adapter below replays a CSV-like list of IMU rows; the same shape works for a live
device (yield samples as they arrive, with ``t_master_ns`` = host arrival time).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from senseai_ingest.adapters.base import Adapter
from senseai_ingest.stream import ClockMode, Modality, Sample, SourceSpec

from _fixture import fixture_config
from senseai_sdk import SenseAI


class ListImu(Adapter):
    """Replays ``(t_ns, ax, ay, az, gx, gy, gz)`` rows as IMU samples."""

    def __init__(self, rows: Sequence[Sequence[float]], source_id: str = "imu0") -> None:
        self._rows, self._sid = rows, source_id

    def spec(self) -> SourceSpec:
        return SourceSpec(
            source_id=self._sid,
            modality=Modality.IMU,
            nominal_rate_hz=200.0,
            meta={"frame_id": "imu"},
            clock_mode=ClockMode.FIXED,
        )

    def open(self) -> None:
        pass

    def read(self) -> Iterator[Sample]:
        for t_ns, *vals in self._rows:
            v = np.asarray(vals, dtype=np.float32)
            yield Sample(self._sid, Modality.IMU, int(t_ns), int(t_ns), v, {})

    def close(self) -> None:
        pass


def main() -> None:
    rows = [(k * 5_000_000, 0.0, 0.0, 9.81, 0.0, 0.0, 0.01 * k) for k in range(400)]  # 2 s at 200 Hz
    with fixture_config() as path:
        sai = SenseAI(path)
        sai.bind_sensor("imu0", ListImu(rows))  # replaces the fixture's synthetic imu0
        for _ in range(30):
            sai.step()
        print(sai.health().summary())
        sai.stop()


if __name__ == "__main__":
    main()
