# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The adapter names a ``senseai.yaml`` may use, and how each maps to an ingest adapter.

This is a lookup table, not an abstraction: every entry names an existing ``senseai_ingest``
class and the keyword arguments its constructor takes. Config validation only needs the
names; :func:`build` imports the class when a pipeline is built, so ``import senseai_sdk``
stays light.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from senseai_ingest.adapters.base import Adapter


@dataclass(frozen=True, slots=True)
class AdapterEntry:
    """One row of the table.

    Attributes:
        name: The name used in ``sensors[].adapter``.
        module: Import path of the module holding the class or factory.
        attr: Class or factory name inside that module.
        kind: ``live`` opens hardware; ``replay`` reads a recording.
        config_cls: For live adapters that take a config dataclass: its attribute name in
            the same module. The SDK builds ``config_cls(source_id=..., **options)`` and
            passes it as the single positional argument.
        description: One line for ``senseai init`` and ``doctor``.
    """

    name: str
    module: str
    attr: str
    kind: str
    config_cls: str | None
    description: str


_ING = "senseai_ingest.adapters"

ADAPTERS: Mapping[str, AdapterEntry] = {
    e.name: e
    for e in (
        # -- live (hardware) -------------------------------------------------------------
        AdapterEntry(
            "prophesee",
            f"{_ING}.live.dvs",
            "DvsAdapter",
            "live",
            "DvsConfig",
            "Prophesee / iniVation event camera (options: sensor, serial, batch_us, bias_file, roi)",
        ),
        AdapterEntry(
            "v4l2",
            f"{_ING}.live.v4l2",
            "V4l2Adapter",
            "live",
            "V4l2Config",
            "USB / CSI frame camera via V4L2 (options: device, width, height, fps)",
        ),
        AdapterEntry(
            "respeaker",
            f"{_ING}.live.mic_array",
            "MicArrayAdapter",
            "live",
            "MicArrayConfig",
            "Microphone array via sounddevice (options: device, geometry, channels, sample_rate, block)",
        ),
        AdapterEntry(
            "serial",
            f"{_ING}.live.serial_imu",
            "SerialSensorHub",
            "live",
            "SerialConfig",
            "IMU / bumper / motor current over a serial line (options: port, baud, joints)",
        ),
        AdapterEntry(
            "can",
            f"{_ING}.live.can_bus",
            "CanAdapter",
            "live",
            "CanConfig",
            "CAN bus signals (options: channel, interface, bitrate, signals)",
        ),
        # -- replay (recordings) ---------------------------------------------------------
        AdapterEntry(
            "dsec",
            f"{_ING}.replay.dsec",
            "DsecReplay",
            "replay",
            None,
            "DSEC sequence directory (options: path, speed)",
        ),
        AdapterEntry(
            "m3ed",
            f"{_ING}.replay.m3ed",
            "M3edReplay",
            "replay",
            None,
            "M3ED HDF5 file (options: path, speed)",
        ),
        AdapterEntry(
            "mvsec",
            f"{_ING}.replay.mvsec",
            "MvsecReplay",
            "replay",
            None,
            "MVSEC HDF5 file (options: path, speed)",
        ),
        AdapterEntry(
            "rosbag",
            f"{_ING}.replay.rosbag",
            "RosbagReplay",
            "replay",
            None,
            "rosbag1 / rosbag2 (options: path, topic, speed)",
        ),
        AdapterEntry(
            "wav",
            f"{_ING}.replay.wav",
            "WavReplay",
            "replay",
            None,
            "Multichannel WAV file (options: path, block, speed)",
        ),
        AdapterEntry(
            "hdf5",
            f"{_ING}.replay.hdf5",
            "Hdf5Replay",
            "replay",
            None,
            "Any HDF5 recording with a layout YAML (options: path, layout, speed)",
        ),
        AdapterEntry(
            "synthetic",
            f"{_ING}.replay.synthetic",
            "SyntheticReplay",
            "replay",
            None,
            "Generated samples for tests and the quickstart (options: modality, rate_hz, duration_s, seed)",
        ),
    )
}


def names() -> list[str]:
    """Every adapter name, sorted."""
    return sorted(ADAPTERS)


def build(name: str, source_id: str, options: Mapping[str, Any]) -> Adapter:
    """Instantiate the ingest adapter ``name`` for ``source_id`` with the config's options.

    Raises:
        KeyError: Unknown name. Config validation rejects these first.
        TypeError: An option the adapter does not accept, re-raised with the adapter named.
    """
    import importlib

    entry = ADAPTERS[name]
    mod = importlib.import_module(entry.module)
    cls = getattr(mod, entry.attr)
    try:
        if entry.config_cls is not None:
            cfg = getattr(mod, entry.config_cls)(source_id=source_id, **options)
            adapter: Adapter = cls(cfg)
        elif hasattr(cls, "from_sequence"):
            # recording factories (DSEC, M3ED, MVSEC) open every source of a sequence; take ours
            opts = dict(options)
            path = opts.pop("path")
            found = cls.from_sequence(path, sources=[source_id], **opts)
            if len(found) != 1:
                raise TypeError(f"expected one source {source_id!r} in {path}, found {len(found)}")
            adapter = found[0]
        else:
            adapter = cls(source_id=source_id, **options)
    except (TypeError, KeyError) as e:
        raise TypeError(f"adapter {name!r} for {source_id!r}: {e}") from e
    return adapter
