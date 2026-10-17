from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import Keyring
from senseai_sdk.licence import GRACE_DAYS, Licence, LicenceError, Meter, issue_licence, verify_report

TODAY = dt.date(2026, 10, 17)


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def issue(key: Ed25519PrivateKey, **kw: object) -> str:
    base: dict[str, object] = {
        "licence_id": "L-1",
        "endpoint_class": "amr",
        "expires": TODAY + dt.timedelta(days=100),
        "issued": TODAY,
    }
    base.update(kw)
    return issue_licence(key, **base)  # type: ignore[arg-type]


def test_round_trip_and_state(key: Ed25519PrivateKey) -> None:
    lic = Licence.parse(
        issue(key, profiles=["panther"], senses=["event", "frame"]), Keyring.from_private(key)
    )
    assert lic.licence_id == "L-1" and lic.endpoint_class == "amr"
    assert lic.state(TODAY) == ("valid", 100)
    assert lic.allows_profile("panther") and lic.allows_profile("null") and not lic.allows_profile("hawk")
    assert lic.allows_sense("EVENT") and not lic.allows_sense("audio")
    with pytest.raises(LicenceError, match="does not allow profile 'hawk'"):
        lic.require_profile("hawk")
    with pytest.raises(LicenceError, match=r"does not allow senses \['audio'\]"):
        lic.require_senses(["event", "audio"])
    lic.check_endpoint("anything")
    assert lic.describe(TODAY)["state"] == "valid"


def test_grace_and_expiry(key: Ed25519PrivateKey) -> None:
    lic = Licence.parse(issue(key, expires=TODAY - dt.timedelta(days=1)), Keyring.from_private(key))
    assert lic.state(TODAY) == ("grace", GRACE_DAYS - 1)
    assert lic.state(TODAY + dt.timedelta(days=GRACE_DAYS - 1)) == ("grace", 0)
    assert lic.state(TODAY + dt.timedelta(days=GRACE_DAYS)) == ("expired", 0)


def test_endpoint_restriction(key: Ed25519PrivateKey) -> None:
    lic = Licence.parse(issue(key, endpoint_ids=["amr-1", "amr-2"]), Keyring.from_private(key))
    lic.check_endpoint("amr-1")
    with pytest.raises(LicenceError, match="does not cover endpoint 'amr-9'"):
        lic.check_endpoint("amr-9")


def test_bad_keys(key: Ed25519PrivateKey) -> None:
    kr = Keyring.from_private(key)
    with pytest.raises(LicenceError, match="SAI1"):
        Licence.parse("TEST", kr)
    good = issue(key)
    other = Keyring.from_private(Ed25519PrivateKey.generate())
    with pytest.raises(LicenceError, match="signature is invalid"):
        Licence.parse(good, other)
    p, payload, sig = good.split(".")
    with pytest.raises(LicenceError, match="signature is invalid"):
        Licence.parse(f"{p}.{payload[:-2]}Zz.{sig}", kr)
    with pytest.raises(LicenceError, match="SAI1"):
        Licence.parse(f"{p}.{payload}", kr)


def test_meter_reports_are_signed_and_spooled(key: Ed25519PrivateKey, tmp_path: Path) -> None:
    lic = Licence.parse(issue(key), Keyring.from_private(key))
    m = Meter(lic, "amr-1", spool=tmp_path / "usage")
    for _ in range(3):
        m.tick("panther", ["event", "frame"])
    m.tick("null")
    assert m.counts() == {"ticks": {"panther": 3, "null": 1}, "senses": {"panther": {"event": 3, "frame": 3}}}
    rep = m.report()
    assert rep["ticks"]["panther"] == 3 and verify_report(rep, lic.raw)
    assert m.counts() == {"ticks": {}, "senses": {}}
    files = list((tmp_path / "usage").glob("usage-*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text())
    assert verify_report(on_disk, lic.raw)
    on_disk["ticks"]["panther"] = 300
    assert not verify_report(on_disk, lic.raw)
    assert not verify_report(rep, "SAI1.other.key")
