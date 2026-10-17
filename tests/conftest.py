"""Test fixtures over the signed fixture store in ``senseai_sdk.testing``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import Keyring
from senseai_sdk.config import Config
from senseai_sdk.testing import build_fixture_store, fixture_config, issue_test_licence


def base_config(store: Path, cache: Path, **over: Any) -> dict[str, Any]:
    return fixture_config(store, cache, **over)


@pytest.fixture(scope="session")
def release_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory, release_key: Ed25519PrivateKey) -> Path:
    """A signed local artefact store with world, reflex and two profiles."""
    return build_fixture_store(tmp_path_factory.mktemp("store"), release_key)


@pytest.fixture(scope="session")
def licence_key(release_key: Ed25519PrivateKey) -> str:
    return issue_test_licence(release_key)


@pytest.fixture
def keyring(release_key: Ed25519PrivateKey) -> Keyring:
    return Keyring.from_private(release_key)


@pytest.fixture
def trust_release_key(monkeypatch: pytest.MonkeyPatch, keyring: Keyring) -> None:
    monkeypatch.setenv("SENSEAI_RELEASE_KEYS", keyring.public_hex()[0])


@pytest.fixture
def config(store: Path, tmp_path: Path, trust_release_key: None, licence_key: str) -> Config:
    return Config.from_mapping(
        base_config(store, tmp_path / "cache", licence_key=licence_key), env={}, base=tmp_path
    )


@pytest.fixture
def config_path(store: Path, tmp_path: Path, trust_release_key: None, licence_key: str) -> Iterator[Path]:
    p = tmp_path / "senseai.yaml"
    p.write_text(yaml.safe_dump(base_config(store, tmp_path / "cache", licence_key=licence_key)))
    yield p
