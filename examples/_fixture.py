# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Builds the signed fixture store and a config for the examples (no hardware, no real artefacts)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import Keyring
from senseai_sdk.testing import build_fixture_store, issue_test_licence
from senseai_sdk.testing import fixture_config as _cfg


@contextmanager
def fixture_config(seconds: float = 2.0, **over: Any) -> Iterator[Path]:
    """Yield the path of a senseai.yaml whose store, cache and licence live in a temp dir."""
    key = Ed25519PrivateKey.generate()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = build_fixture_store(root / "store", key)
        os.environ["SENSEAI_RELEASE_KEYS"] = Keyring.from_private(key).public_hex()[0]
        d = _cfg(store, root / "cache", licence_key=issue_test_licence(key), seconds=seconds, **over)
        p = root / "senseai.yaml"
        p.write_text(yaml.safe_dump(d))
        yield p
