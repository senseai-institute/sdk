# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Per-endpoint licences: offline verification, grace period, usage metering, signed reports.

A licence key is ``SAI1.<payload>.<signature>``: a base64url JSON payload and an Ed25519
signature over it by SenseAI's release key (the same keyring that signs artefacts). The
payload names the endpoint class, the allowed endpoint ids, profiles and senses, and the
expiry. Verification is offline; the SDK never phones home to run.

After expiry there is a 30-day grace period in which the pipeline runs and health warns.
After that, construction refuses. Metering counts ticks and active senses per profile and
writes reports signed with an HMAC keyed by the licence itself (a secret SenseAI shares with
the endpoint), so a report cannot be forged or edited after the fact.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from senseai_sdk.artefacts import Keyring
from senseai_sdk.errors import LicenceError

__all__ = ["GRACE_DAYS", "Licence", "LicenceError", "Meter", "issue_licence", "verify_report"]

GRACE_DAYS = 30
PREFIX = "SAI1"
LicenceState = Literal["valid", "grace", "expired"]


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _date(s: str, what: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        raise LicenceError(f"licence {what} {s!r} is not a YYYY-MM-DD date") from None


@dataclass(frozen=True, slots=True)
class Licence:
    """A verified licence."""

    licence_id: str
    endpoint_class: str
    endpoint_ids: tuple[str, ...] | None  # None = any
    profiles: tuple[str, ...] | None  # None = any
    senses: tuple[str, ...] | None  # None = any (ingest modalities: event, frame, audio, …)
    issued: dt.date
    expires: dt.date
    raw: str = field(repr=False, compare=False, default="")

    @classmethod
    def parse(cls, key: str, keyring: Keyring | None = None) -> Licence:
        """Verify ``key``'s signature and structure. Does not check the date; see :meth:`state`.

        Raises:
            LicenceError: Malformed, unsigned by a trusted key, or missing fields.
        """
        keyring = keyring or Keyring.default()
        parts = key.strip().split(".")
        if len(parts) != 3 or parts[0] != PREFIX:
            raise LicenceError(
                "licence key is not in the SAI1.<payload>.<signature> form; "
                "run `senseai licence activate <key>` with the key SenseAI issued"
            )
        try:
            payload, sig = _b64d(parts[1]), _b64d(parts[2])
        except (ValueError, binascii.Error) as e:
            raise LicenceError(f"licence key is not valid base64: {e}") from None
        if not keyring.verify(payload, sig):
            raise LicenceError("licence signature is invalid for every trusted release key")
        try:
            d = json.loads(payload)
        except json.JSONDecodeError as e:
            raise LicenceError(f"licence payload is not JSON: {e}") from e
        for k in ("licence_id", "endpoint_class", "issued", "expires"):
            if k not in d:
                raise LicenceError(f"licence payload has no {k!r}")

        def star(v: Any) -> tuple[str, ...] | None:
            return None if v in (None, "*") else tuple(str(x) for x in v)

        return cls(
            licence_id=str(d["licence_id"]),
            endpoint_class=str(d["endpoint_class"]),
            endpoint_ids=star(d.get("endpoint_ids", "*")),
            profiles=star(d.get("profiles", "*")),
            senses=star(d.get("senses", "*")),
            issued=_date(str(d["issued"]), "issued"),
            expires=_date(str(d["expires"]), "expires"),
            raw=key.strip(),
        )

    def state(self, today: dt.date | None = None) -> tuple[LicenceState, int]:
        """``(state, days_left)``: days until expiry while valid, days of grace left after it."""
        today = today or dt.date.today()
        left = (self.expires - today).days
        if left >= 0:
            return "valid", left
        grace_left = GRACE_DAYS + left
        if grace_left >= 0:
            return "grace", grace_left
        return "expired", 0

    def check_endpoint(self, endpoint_id: str) -> None:
        """Raise unless this licence covers ``endpoint_id``."""
        if self.endpoint_ids is not None and endpoint_id not in self.endpoint_ids:
            raise LicenceError(
                f"licence {self.licence_id} does not cover endpoint {endpoint_id!r} "
                f"(covers {', '.join(self.endpoint_ids)})"
            )

    def allows_profile(self, name: str) -> bool:
        """``null`` is always allowed; others per the licence."""
        return name == "null" or self.profiles is None or name in self.profiles

    def allows_sense(self, modality: str) -> bool:
        """Whether tokens from this ingest modality may run."""
        return self.senses is None or modality.lower() in self.senses

    def require_profile(self, name: str) -> None:
        """Raise unless the profile is allowed."""
        if not self.allows_profile(name):
            allowed = ", ".join(self.profiles or ())
            raise LicenceError(f"licence {self.licence_id} does not allow profile {name!r} (allows {allowed})")

    def require_senses(self, modalities: Iterable[str]) -> None:
        """Raise unless every modality is allowed."""
        bad = sorted({m for m in modalities if not self.allows_sense(m)})
        if bad:
            allowed = ", ".join(self.senses or ())
            raise LicenceError(f"licence {self.licence_id} does not allow senses {bad} (allows {allowed})")

    def describe(self, today: dt.date | None = None) -> dict[str, Any]:
        """A JSON-ready summary for ``senseai licence show`` and health."""
        state, days = self.state(today)
        return {
            "licence_id": self.licence_id,
            "endpoint_class": self.endpoint_class,
            "endpoint_ids": list(self.endpoint_ids) if self.endpoint_ids else "*",
            "profiles": list(self.profiles) if self.profiles else "*",
            "senses": list(self.senses) if self.senses else "*",
            "issued": self.issued.isoformat(),
            "expires": self.expires.isoformat(),
            "state": state,
            "days_left": days,
        }


def issue_licence(
    key: Ed25519PrivateKey,
    *,
    licence_id: str,
    endpoint_class: str,
    expires: dt.date,
    issued: dt.date | None = None,
    endpoint_ids: Iterable[str] | None = None,
    profiles: Iterable[str] | None = None,
    senses: Iterable[str] | None = None,
) -> str:
    """Sign a licence key (release tooling and tests). ``None`` lists mean any."""
    payload = {
        "licence_id": licence_id,
        "endpoint_class": endpoint_class,
        "endpoint_ids": "*" if endpoint_ids is None else list(endpoint_ids),
        "profiles": "*" if profiles is None else list(profiles),
        "senses": "*" if senses is None else list(senses),
        "issued": (issued or dt.date.today()).isoformat(),
        "expires": expires.isoformat(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{PREFIX}.{_b64e(raw)}.{_b64e(key.sign(raw))}"


# -- metering ---------------------------------------------------------------------------------


def _report_key(licence_raw: str) -> bytes:
    return hashlib.sha256(("senseai-usage:" + licence_raw).encode()).digest()


def _sign_report(report: Mapping[str, Any], licence_raw: str) -> str:
    body = json.dumps(
        {k: v for k, v in report.items() if k != "signature"}, sort_keys=True, separators=(",", ":")
    )
    return hmac.new(_report_key(licence_raw), body.encode(), hashlib.sha256).hexdigest()


def verify_report(report: Mapping[str, Any], licence_raw: str) -> bool:
    """True if ``report`` was signed under this licence and not edited since."""
    return hmac.compare_digest(_sign_report(report, licence_raw), str(report.get("signature", "")))


class Meter:
    """Counts ticks and active senses per profile; writes signed usage reports.

    Args:
        licence: The verified licence.
        endpoint_id: From config.
        spool: Directory for ``usage-<timestamp>.json`` reports; ``None`` keeps them in memory.
    """

    def __init__(self, licence: Licence, endpoint_id: str, spool: Path | None = None) -> None:
        self.licence = licence
        self.endpoint_id = endpoint_id
        self.spool = spool
        self._ticks: dict[str, int] = {}
        self._senses: dict[str, dict[str, int]] = {}
        self._started = dt.datetime.now(dt.UTC)
        self.reports: list[dict[str, Any]] = []

    def tick(self, profile: str, active_senses: Iterable[str] = ()) -> None:
        """Record one tick under ``profile`` with these senses carrying data."""
        self._ticks[profile] = self._ticks.get(profile, 0) + 1
        if active_senses:
            d = self._senses.setdefault(profile, {})
            for s in active_senses:
                d[s] = d.get(s, 0) + 1

    def counts(self) -> dict[str, Any]:
        """The live counters."""
        return {"ticks": dict(self._ticks), "senses": {k: dict(v) for k, v in self._senses.items()}}

    def report(self, *, reset: bool = True) -> dict[str, Any]:
        """Build, sign and (if spooling) write a usage report; then reset the counters."""
        now = dt.datetime.now(dt.UTC)
        rep: dict[str, Any] = {
            "format": "senseai-usage/1",
            "licence_id": self.licence.licence_id,
            "endpoint_id": self.endpoint_id,
            "period_start": self._started.isoformat(timespec="seconds"),
            "period_end": now.isoformat(timespec="seconds"),
            **self.counts(),
        }
        rep["signature"] = _sign_report(rep, self.licence.raw)
        self.reports.append(rep)
        if self.spool is not None:
            self.spool.mkdir(parents=True, exist_ok=True)
            name = f"usage-{now.strftime('%Y%m%dT%H%M%S')}-{len(self.reports)}.json"
            (self.spool / name).write_text(json.dumps(rep, indent=2))
        if reset:
            self._ticks, self._senses, self._started = {}, {}, now
        return rep
