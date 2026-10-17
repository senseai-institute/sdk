# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Run the whole SenseAI pipeline on a DSEC sequence: ingest → tokens → profiles → world → reflex.

    uv run python examples/replay_dsec.py                                  # ingest's synthetic DSEC fixture
    uv run python examples/replay_dsec.py /data/dsec/zurich_city_09_a --profile panther \\
        --world-ckpt runs/dsec_night/last.ckpt --tokens-yaml runs/dsec_night/tokens.yaml \\
        --audit-out panther_09a.h5

``--world-mode prior`` needs no checkpoint at all: the structured species prior (no training,
``--prior-library`` selects the library entry, default felidae) produces the surprise maps. This
is the "no training: first real-data run" mode. Without ``--world-ckpt`` in ``learned`` mode the
world model is the SDK's tiny *untrained* fixture, so the reflex decisions mean nothing; the run
proves the plumbing. ``fused`` combines a checkpoint with the prior. With your checkpoint (trained
on tokens.h5 files made with the same ``tokens.yaml``) it is the real pipeline. Artefacts are signed with a
throwaway key and a test licence is issued, both in a temporary directory: development only.

Prints one line per tick for the first ticks and every ``--every`` ticks after, then counts by
action, stage latencies and health. ``--audit-out`` writes the reflex audit log (every command
with its provenance); explain any moment with ``senseai logs explain <file> --at <seconds>``.
"""

from __future__ import annotations

import argparse
import collections
import os
import tempfile
from pathlib import Path

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk import SenseAI
from senseai_sdk.artefacts import Keyring
from senseai_sdk.stages import FallbackCommand
from senseai_sdk.testing import build_fixture_store, fixture_config, issue_test_licence

SOURCES = ("events_left", "rgb_left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", help="DSEC sequence directory (default: a synthetic one)")
    ap.add_argument("--profile", default="null", help="null, panther, ... (any profile in the profiles repo)")
    ap.add_argument("--world-ckpt", type=Path, help="your senseai-world checkpoint")
    ap.add_argument("--tokens-yaml", type=Path, help="the tokens.yaml the checkpoint was trained against")
    ap.add_argument("--reflex-yaml", type=Path, help="reflex config (default: Gate 1 defaults)")
    ap.add_argument(
        "--world-mode",
        choices=["learned", "prior", "fused"],
        default="learned",
        help="learned = checkpoint; prior = structured species prior, no training; fused = both",
    )
    ap.add_argument(
        "--prior-library", type=Path, help="species library YAML for the prior (default: felidae)"
    )
    ap.add_argument("--speed", type=float, default=1.5, help="vehicle speed fed to the reflex, m/s")
    ap.add_argument("--ticks", type=int, default=0, help="stop after this many ticks (0 = whole sequence)")
    ap.add_argument("--every", type=int, default=100, help="print every N-th tick after the first 5")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--audit-out", type=Path, help="write the reflex audit log (HDF5) here")
    a = ap.parse_args()

    key = Ed25519PrivateKey.generate()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        seq = Path(a.sequence) if a.sequence else _synthetic(root)
        store = build_fixture_store(
            root / "store",
            key,
            sources=SOURCES,
            world_checkpoint=a.world_ckpt,
            tokens_yaml=a.tokens_yaml,
            reflex_yaml=a.reflex_yaml,
            world_mode=a.world_mode,
            library=a.prior_library,
        )
        os.environ["SENSEAI_RELEASE_KEYS"] = Keyring.from_private(key).public_hex()[0]
        cfg = fixture_config(store, root / "cache", licence_key=issue_test_licence(key))
        cfg["profile"] = {"name": a.profile}
        cfg["world"] = {
            "mode": a.world_mode,
            **({"library": str(a.prior_library)} if a.prior_library else {}),
        }
        cfg["robot"] = {"speed_cap_mps": max(a.speed, 0.1)}
        cfg["sensors"] = [
            {"id": "events_left", "adapter": "dsec", "path": str(seq), "spec": "generic_event_camera"},
            {"id": "rgb_left", "adapter": "dsec", "path": str(seq), "spec": "generic_rgb_camera"},
        ]
        (root / "senseai.yaml").write_text(yaml.safe_dump(cfg))
        return _run(SenseAI(root / "senseai.yaml", device=a.device), a)


def _synthetic(root: Path) -> Path:
    from senseai_ingest.testing import make_dsec_fixture

    return make_dsec_fixture(root / "dsec_synthetic", duration_s=1.0, events_per_s=200_000)


def _run(sai: SenseAI, a: argparse.Namespace) -> int:
    sai.update_context(speed=a.speed)
    actions: collections.Counter[str] = collections.Counter()
    fallback = warming = n = 0
    try:
        while not a.ticks or n < a.ticks:
            r = sai.step()
            n += 1
            c = r.command
            fallback += isinstance(c, FallbackCommand)
            warming += bool(r.surprise is not None and r.surprise.warming_up)
            actions[c.action] += 1
            if n <= 5 or n % a.every == 0:
                lat = " ".join(f"{k} {v / 1e6:.2f}" for k, v in r.latency_ns.items() if k != "sdk")
                kind = "watchdog" if isinstance(c, FallbackCommand) else f"reflex θ {c.threshold:.2f}"
                head = f"tick {r.tick_index:6d} {c.action:<6} scale {c.speed_scale:.2f} risk {c.risk:5.2f}"
                print(f"{head} {kind:<16} ms: {lat}")
    except StopIteration:
        pass
    h = sai.health()
    if a.audit_out is not None:
        sai.save_audit(a.audit_out)
        print(f"audit log: {a.audit_out}")
    sai.stop()
    print(
        f"\n{n} ticks, profile {a.profile}, world {a.world_mode} ({h.artefacts.get('world_model')}): "
        + ", ".join(f"{k} {v}" for k, v in sorted(actions.items()))
    )
    print(f"SDK watchdog commands: {fallback}; world warming-up ticks: {warming}")
    print(h.summary())
    return 0 if n and not fallback else 1


if __name__ == "__main__":
    raise SystemExit(main())
