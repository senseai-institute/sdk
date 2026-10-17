# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Builds the signed fixture store and a config for the examples (no hardware, no real artefacts)."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import conftest

from senseai_sdk.artefacts import Keyring


class _Factory:
    def __init__(self, root: Path) -> None:
        self.root = root

    def mktemp(self, name: str) -> Path:
        d = self.root / name
        d.mkdir()
        return d


@contextmanager
def fixture_config(seconds: float = 2.0, **over: Any) -> Iterator[Path]:
    """Yield the path of a senseai.yaml whose store, cache and licence live in a temp dir."""
    key = Ed25519PrivateKey.generate()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = conftest.store.__wrapped__(_Factory(root), key)  # type: ignore[attr-defined]
        os.environ["SENSEAI_RELEASE_KEYS"] = Keyring.from_private(key).public_hex()[0]
        lic = conftest.licence_key.__wrapped__(key)  # type: ignore[attr-defined]
        d = conftest.base_config(store, root / "cache", licence_key=lic, **over)
        for s in d["sensors"]:
            s["duration_s"] = seconds
        p = root / "senseai.yaml"
        p.write_text(yaml.safe_dump(d))
        yield p
