# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``senseai``: init, doctor, run, bench, calibrate, licence, profiles, logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from senseai_sdk.errors import SenseAIError

__all__ = ["main"]


def _init(a: argparse.Namespace) -> int:
    from senseai_sdk.config import template

    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return 2
    out.write_text(template(replay=a.replay, endpoint_id=a.endpoint_id))
    print(f"wrote {out}; edit endpoint_id, profile and sensors, then run `senseai doctor --config {out}`")
    if a.replay is None:
        print("live adapters are listed in the file; `--replay fixture` runs without hardware")
    return 0


def _doctor(a: argparse.Namespace) -> int:
    from senseai_sdk.doctor import run_checks

    worst = "ok"
    for c in run_checks(Path(a.config), probe_seconds=a.probe_seconds):
        print(c.line())
        if c.verdict == "fail":
            worst = "fail"
        elif c.verdict == "warn" and worst == "ok":
            worst = "warn"
    print(
        {
            "ok": "all checks passed",
            "warn": "passed with warnings",
            "fail": "FAILED: fix the first [FAIL] line",
        }[worst]
    )
    return 1 if worst == "fail" else 0


def _run(a: argparse.Namespace) -> int:
    from senseai_sdk.api import SenseAI

    sai = SenseAI(a.config, device=a.device)
    server = None
    if a.health_port is not None:
        from senseai_sdk.health import HealthServer

        server = HealthServer(sai.health, a.health_port)
        server.start()
        print(f"health at http://127.0.0.1:{server.port}/health")
    n = [0]
    fires = [0]

    def on_cmd(cmd: Any) -> None:
        n[0] += 1
        if getattr(cmd, "action", "none") != "none":
            fires[0] += 1
        if a.verbose:
            print(
                f"{cmd.t_master_ns / 1e9:10.3f}s {cmd.action:<6} scale {cmd.speed_scale:.2f} stop {cmd.stop}"
            )

    sai.on_command(on_cmd)
    sai.on_event(lambda ev: print(getattr(ev, "text", ev)))
    try:
        if a.ticks is None:
            sai.run()
        else:
            for _ in range(a.ticks):
                try:
                    sai.step()
                except StopIteration:
                    break
            sai.stop()
    finally:
        if server is not None:
            server.stop()
    h = sai.health()
    print(h.summary())
    if a.stats:
        print(json.dumps(h.to_json(), indent=2))
    print(f"{n[0]} commands, {fires[0]} non-'none'")
    return 0


def _licence(a: argparse.Namespace) -> int:
    from senseai_sdk.artefacts import Keyring
    from senseai_sdk.licence import Licence

    if a.action == "activate":
        key = a.key
        if key is None:
            print("usage: senseai licence activate <key>", file=sys.stderr)
            return 2
        lic = Licence.parse(key, Keyring.default())
        dest = Path(a.store_to).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(key.strip() + "\n")
        os.chmod(dest, 0o600)
        print(
            f"licence {lic.licence_id} verified and stored at {dest}; set licence_key: ${{SENSEAI_LICENCE}} "
            f"and export SENSEAI_LICENCE=$(cat {dest})"
        )
        return 0
    from senseai_sdk.config import Config

    cfg = Config.load(a.config)
    lic = Licence.parse(cfg.licence_key, Keyring.default())
    if a.action == "show":
        print(json.dumps(lic.describe(), indent=2))
        return 0
    # report: sign and print the spooled usage reports
    spool = cfg.artefacts.cache / "usage"
    files = sorted(spool.glob("usage-*.json")) if spool.is_dir() else []
    reports = [json.loads(f.read_text()) for f in files]
    if a.out:
        Path(a.out).write_text(json.dumps(reports, indent=2))
        print(f"{len(reports)} usage reports written to {a.out}")
    else:
        print(json.dumps(reports, indent=2))
    return 0


def _profiles(a: argparse.Namespace) -> int:
    from senseai_sdk.artefacts import ArtefactStore, Keyring
    from senseai_sdk.config import Config
    from senseai_sdk.licence import Licence

    cfg = Config.load(a.config)
    lic = Licence.parse(cfg.licence_key, Keyring.default())
    store = ArtefactStore(cfg.artefacts.store, cfg.artefacts.cache, Keyring.default())
    if store.remote:
        print("profiles cannot be listed from a remote store; pin profile.version in config", file=sys.stderr)
        return 2
    root = Path(str(store.store)).expanduser()
    found: dict[str, list[int]] = {}
    for p in sorted(root.glob("profile-*-v*")):
        name, _, vs = p.name[len("profile-") :].rpartition("-v")
        if vs.isdigit():
            found.setdefault(name, []).append(int(vs))
    if a.action == "list":
        print(f"{'profile':<12} {'versions':<12} licensed")
        print(f"{'null':<12} {'-':<12} yes")
        for name, versions in found.items():
            vs = ", ".join(map(str, sorted(versions)))
            print(f"{name:<12} {vs:<12} {'yes' if lic.allows_profile(name) else 'no'}")
        return 0
    name = a.name
    if name not in found:
        print(f"no profile {name!r} in {root}; have {sorted(found)}", file=sys.stderr)
        return 1
    v: int = a.version if a.version is not None else max(found[name])
    art = store.fetch(f"profile-{name}-v{v}")
    report = art.root / "priors" / f"{name}.v{v}.md"
    print(f"{art.id}: {len(art.files)} files, verified")
    if report.is_file():
        print(report.read_text())
    return 0


def _bench(a: argparse.Namespace) -> int:
    from senseai_sdk.bench import main_json, run_bench
    from senseai_sdk.config import Config

    cfg = Config.load(a.config)
    res = run_bench(
        cfg,
        ticks_file=Path(a.ticks) if a.ticks else None,
        max_ticks=a.max_ticks,
        warmup=a.warmup,
        device=a.device,
    )
    print(main_json(res) if a.json else res.table())
    if a.out:
        Path(a.out).write_text(main_json(res))
    return 0


def _calibrate(a: argparse.Namespace) -> int:
    from senseai_sdk import calibration as cal
    from senseai_sdk.config import Config

    cfg = Config.load(a.config)
    out = Path(a.out)
    if a.imu_rest:
        import numpy as np

        samples = np.loadtxt(a.imu_rest, delimiter=",")
        entry = cal.imu_at_rest(samples)
        p = cal.write_calibration(out / f"{a.sensor}.yaml", a.sensor, entry)
        print(f"wrote {p} from {len(samples)} rest samples")
        return 0

    def ask(prompt: str, default: str) -> str:
        if a.yes:
            return default
        ans = input(f"{prompt} [{default}]: ").strip()
        return ans or default

    files = cal.guided(cfg, out, ask=ask, only=a.sensor)
    print(f"{len(files)} calibration files in {out}; run `senseai doctor` to confirm they are picked up")
    return 0


def _logs(a: argparse.Namespace) -> int:
    """``logs tail <file>`` follows a cortex/telemetry JSONL file; ``logs explain --at T`` needs reflex."""
    if a.action == "explain":
        print(
            "senseai logs explain: the reflex audit log lives in senseai-reflex, which is not installed yet",
            file=sys.stderr,
        )
        return 3
    path = Path(a.file)
    if not path.is_file():
        print(f"{path}: no such file", file=sys.stderr)
        return 1
    import time

    with path.open() as f:
        for line in f:
            print(line.rstrip())
        while a.follow:
            line = f.readline()
            if line:
                print(line.rstrip())
            else:
                time.sleep(0.2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argparse tree."""
    ap = argparse.ArgumentParser(prog="senseai", description="SenseAI perception layer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write a commented senseai.yaml")
    p.add_argument("--out", default="senseai.yaml")
    p.add_argument("--replay", default=None, help="'fixture' for generated sources, or a recording path")
    p.add_argument("--endpoint-id", default="amr-0042")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_init)

    p = sub.add_parser("doctor", help="check config, licence, artefacts, devices, calibration, tick, clock")
    p.add_argument("--config", default="senseai.yaml")
    p.add_argument("--probe-seconds", type=float, default=0.5)
    p.set_defaults(fn=_doctor)

    p = sub.add_parser("run", help="run the pipeline")
    p.add_argument("--config", default="senseai.yaml")
    p.add_argument(
        "--ticks", type=int, default=None, help="stop after N ticks (default: run to end / Ctrl-C)"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--health-port", type=int, default=None)
    p.add_argument("--stats", action="store_true", help="print the health JSON at the end")
    p.add_argument("--verbose", action="store_true", help="print every command")
    p.set_defaults(fn=_run)

    p = sub.add_parser("licence", help="show | activate <key> | report")
    p.add_argument("action", choices=["show", "activate", "report"])
    p.add_argument("key", nargs="?")
    p.add_argument("--config", default="senseai.yaml")
    p.add_argument("--store-to", default="~/.senseai/licence.key")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_licence)

    p = sub.add_parser("profiles", help="list | show <name>")
    p.add_argument("action", choices=["list", "show"])
    p.add_argument("name", nargs="?")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--config", default="senseai.yaml")
    p.set_defaults(fn=_profiles)

    p = sub.add_parser("bench", help="sensor-to-actuator latency on this machine")
    p.add_argument("--config", default="senseai.yaml")
    p.add_argument("--ticks", default=None, help="an ingest ticks.h5 to replay instead of the sensors")
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_bench)

    p = sub.add_parser("calibrate", help="guided per-sensor calibration files")
    p.add_argument("--config", default="senseai.yaml")
    p.add_argument("--out", default="calib")
    p.add_argument("--sensor", default=None, help="only this sensor id")
    p.add_argument("--yes", action="store_true", help="accept every default (non-interactive)")
    p.add_argument(
        "--imu-rest", default=None, help="CSV of ax,ay,az,gx,gy,gz rows recorded at rest (with --sensor)"
    )
    p.set_defaults(fn=_calibrate)

    p = sub.add_parser("logs", help="tail <events.jsonl> | explain --at T")
    p.add_argument("action", choices=["tail", "explain"])
    p.add_argument("file", nargs="?", default="events.jsonl")
    p.add_argument("--at", type=float, default=None)
    p.add_argument("--follow", "-f", action="store_true")
    p.set_defaults(fn=_logs)
    return ap


def _logs_to_stderr() -> None:
    """Keep stdout for the command's own output: upstream packages log through structlog."""
    try:
        import structlog
    except ImportError:  # pragma: no cover
        return

    class _Stderr:  # resolves sys.stderr at write time, so a redirected stream is honoured
        def write(self, s: str) -> int:
            return sys.stderr.write(s)

        def flush(self) -> None:
            sys.stderr.flush()

    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=_Stderr()))  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    _logs_to_stderr()
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
        return code
    except SenseAIError as e:
        print(f"senseai: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


_ = dt  # keep datetime available to future subcommands (bench/logs timestamps)
