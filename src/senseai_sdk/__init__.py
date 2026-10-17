# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""senseai-sdk: the SenseAI perception layer in one package.

    from senseai_sdk import SenseAI

Importing the package pulls in only the config layer (numpy, yaml). The runtime packages
(torch among them) load when a :class:`SenseAI` is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from senseai_sdk.config import Config
from senseai_sdk.errors import ArtefactError, ConfigError, LicenceError, PipelineError, SenseAIError

if TYPE_CHECKING:
    from senseai_sdk.api import Health, SenseAI, StepResult

__version__ = "0.1.0"

__all__ = [
    "ArtefactError",
    "Config",
    "ConfigError",
    "Health",
    "LicenceError",
    "PipelineError",
    "SenseAI",
    "SenseAIError",
    "StepResult",
    "__version__",
]

_LAZY = {"SenseAI": "senseai_sdk.api", "Health": "senseai_sdk.api", "StepResult": "senseai_sdk.api"}


def __getattr__(name: str) -> Any:
    """Resolve the lazily imported public names."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'senseai_sdk' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod), name)
