# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``senseai doctor``: every check a customer would otherwise discover at 2 a.m.

Each check returns one :class:`Check` with a verdict, what was found, and what to do about
it. Checks run in dependency order (a bad config stops before licence, a bad licence before
artefacts …) so the first failure is the one to fix.
"""

from __future__ import annotations

import datetime as dt
import importlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from senseai_sdk import adapters as adapter_table
from senseai_sdk.artefacts import ArtefactStore, Keyring, profile_artefact_id, profile_artefact_prefix
from senseai_sdk.config import Config
from senseai_sdk.errors import SenseAIError
from senseai_sdk.licence import Licence

__all__ = ["Check", "run_checks"]

Verdict = Literal["ok", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class Check:
    """One line of doctor output."""

    name: str
    verdict: Verdict
    detail: str
    fix: str = ""

    def line(self) -> str:
        """``[ OK ] name: detail`` or ``[FAIL] name: detail -> fix``."""
        tag = {"ok": " OK ", "warn": "WARN", "fail": "FAIL"}[self.verdict]
        tail = f" -> {self.fix}" if self.fix else ""
        return f"[{tag}] {self.name}: {self.detail}{tail}"


def run_checks(
    config_path: Path,
    *,
    keyring: Keyring | None = None,
    today: dt.date | None = None,
    probe_seconds: float = 0.5,
) -> Iterator[Check]:
    """Yield checks in order; stops after a failure that later checks depend on."""
    keyring = keyring or Keyring.default()

    # 1. config
    try:
        cfg = Config.load(config_path)
    except SenseAIError as e:
        yield Check(
            "config", "fail", str(e), "fix the key named in the message; `senseai init` writes a template"
        )
        return
    yield Check("config", "ok", f"{config_path} ({len(cfg.sensors)} sensors, tick {cfg.tick_ms} ms)")

    # 2. licence
    try:
        lic = Licence.parse(cfg.licence_key, keyring)
        state, days = lic.state(today)
        lic.check_endpoint(cfg.endpoint_id)
        lic.require_profile(cfg.profile.name)
    except SenseAIError as e:
        yield Check(
            "licence", "fail", str(e), "run `senseai licence activate <key>` with the key SenseAI issued"
        )
        return
    if state == "expired":
        yield Check(
            "licence", "fail", f"{lic.licence_id} expired on {lic.expires}, grace over", "renew the licence"
        )
        return
    if state == "grace":
        yield Check(
            "licence",
            "warn",
            f"{lic.licence_id} expired on {lic.expires}; {days} days of grace left",
            "renew",
        )
    else:
        yield Check("licence", "ok", f"{lic.licence_id} for class {lic.endpoint_class}, {days} days left")

    # 3. artefacts
    store = ArtefactStore(cfg.artefacts.store, cfg.artefacts.cache, keyring)
    arts: dict[str, Any] = {}
    for kind, aid in (("world", cfg.artefacts.world), ("reflex", cfg.artefacts.reflex)):
        try:
            arts[kind] = store.fetch(aid)
            yield Check(f"artefact {kind}", "ok", f"{aid} verified ({len(arts[kind].files)} files)")
        except SenseAIError as e:
            yield Check(
                f"artefact {kind}",
                "fail",
                str(e),
                f"check artefacts.store ({cfg.artefacts.store}) and the id",
            )
    if cfg.profile.name != "null":
        v = cfg.profile.version
        try:
            if v is None:
                root = Path(str(cfg.artefacts.store)).expanduser()
                prefix = profile_artefact_prefix(cfg.profile.name)
                found = sorted(root.glob(f"{prefix}*"))
                if not found:
                    raise SenseAIError(f"no {prefix}* in {root}")
                aid = found[-1].name
            else:
                aid = profile_artefact_id(cfg.profile.name, v)
            store.fetch(aid)
            yield Check("artefact profile", "ok", f"{aid} verified")
        except SenseAIError as e:
            yield Check("artefact profile", "fail", str(e), "pin profile.version to a bundle in the store")

    # 4. runtime packages
    for pkg, effect in (
        ("senseai_reflex", "no reflex commands; the watchdog drives the bridge"),
        ("senseai_cortex", "no semantic events"),
    ):
        try:
            importlib.import_module(pkg)
            yield Check(f"package {pkg}", "ok", "installed")
        except ImportError:
            yield Check(
                f"package {pkg}",
                "warn",
                f"not installed: {effect}",
                f"pip install {pkg.replace('_', '-')}",
            )

    # 5. tick vs world artefact
    world = arts.get("world")
    if world is not None:
        trained = world.meta.get("tick_ms")
        if trained is None:
            yield Check(
                "tick", "warn", f"{world.id} does not record its tick; the pipeline checks the checkpoint"
            )
        elif int(trained) != cfg.tick_ms:
            yield Check(
                "tick",
                "fail",
                f"config {cfg.tick_ms} ms, {world.id} trained at {trained} ms",
                f"set tick_ms: {trained}",
            )
        else:
            yield Check("tick", "ok", f"{cfg.tick_ms} ms matches {world.id}")

    # 6. sensors: build, open, read for a moment
    for s in cfg.sensors:
        name = f"sensor {s.id}"
        try:
            ad = adapter_table.build(s.adapter, s.id, s.options)
        except TypeError as e:
            yield Check(name, "fail", str(e), f"check the options on sensors[{s.id}]")
            continue
        try:
            ad.open()
            n = 0
            t0 = time.monotonic()
            it = ad.read()
            while time.monotonic() - t0 < probe_seconds:
                try:
                    next(it)
                except StopIteration:
                    break
                n += 1
                if n >= 10:
                    break
            ad.close()
        except NotImplementedError as e:  # ingest's AdapterNotAvailable
            yield Check(name, "fail", str(e).splitlines()[0], "install the vendor driver named above")
            continue
        except Exception as e:  # report, do not crash the doctor
            yield Check(
                name, "fail", f"{type(e).__name__}: {e}", "check the device path / file and permissions"
            )
            continue
        mod = ad.spec().modality
        if not lic.allows_sense(str(mod)):
            yield Check(
                name,
                "fail",
                f"{s.adapter} ({mod}) is not licensed",
                "remove the sensor or extend the licence",
            )
        elif n == 0:
            yield Check(
                name, "warn", f"{s.adapter} ({mod}) opened but produced no samples in {probe_seconds}s"
            )
        else:
            yield Check(name, "ok", f"{s.adapter} ({mod}) produced {n} samples")
        # 7. calibration
        if s.calib is None:
            if mod in ("event", "frame", "audio"):
                yield Check(
                    f"calibration {s.id}",
                    "warn",
                    "none: sensor assumed at the body origin",
                    "run `senseai calibrate`",
                )
        elif not Path(s.calib).is_file():
            yield Check(f"calibration {s.id}", "fail", f"{s.calib} does not exist", "run `senseai calibrate`")
        else:
            yield Check(f"calibration {s.id}", "ok", str(s.calib))

    # 8. clock
    now = dt.datetime.now(dt.UTC)
    if now.year < 2026:
        yield Check(
            "clock",
            "fail",
            f"system time is {now:%Y-%m-%d}",
            "set the clock (NTP); licences are date-checked",
        )
    else:
        res = time.get_clock_info("monotonic").resolution
        yield Check(
            "clock", "ok", f"system {now:%Y-%m-%d %H:%M:%S} UTC, monotonic resolution {res * 1e9:.0f} ns"
        )
