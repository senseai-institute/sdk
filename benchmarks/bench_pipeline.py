# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""End-to-end latency of the SDK pipeline on the test fixture (tiny world model, CPU).

    uv run python benchmarks/bench_pipeline.py [--ticks N] [--device cpu|cuda]

Builds the signed fixture store from tests/conftest.py, runs the single-loop pipeline and
prints the per-stage table. The SDK overhead row (``sdk``) is the number the SDK owns; the
other rows belong to the upstream packages and, on CPU with the fixture-size model, are not
the production numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tests import conftest

from senseai_sdk.artefacts import Keyring
from senseai_sdk.bench import run_bench
from senseai_sdk.config import Config


class _Factory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def mktemp(self, name: str) -> Path:
        d = self.root / name
        d.mkdir()
        return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    key = Ed25519PrivateKey.generate()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = conftest.store.__wrapped__(_Factory(root), key)  # type: ignore[attr-defined]
        os.environ["SENSEAI_RELEASE_KEYS"] = Keyring.from_private(key).public_hex()[0]
        lic = conftest.licence_key.__wrapped__(key)  # type: ignore[attr-defined]
        d = conftest.base_config(store, root / "cache", licence_key=lic)
        seconds = a.ticks * 0.01 + 1.0
        for s in d["sensors"]:
            s["duration_s"] = seconds
        cfg = Config.from_mapping(d, env={}, base=root)
        res = run_bench(cfg, max_ticks=a.ticks, device=a.device)
    if a.json:
        from senseai_sdk.bench import main_json

        print(main_json(res))
    else:
        print(res.table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
