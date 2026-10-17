# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The SDK's exception types. Every message says what is wrong and what to do about it."""

from __future__ import annotations


class SenseAIError(Exception):
    """Base class for every error this package raises."""


class ConfigError(SenseAIError, ValueError):
    """``senseai.yaml`` is invalid. The message is ``<key path>: <problem>``."""


class LicenceError(SenseAIError):
    """The licence key is missing, invalid, for another endpoint, or past its grace period."""


class ArtefactError(SenseAIError):
    """An artefact is missing, unsigned, tampered with, or does not match its pinned version."""


class PipelineError(SenseAIError):
    """The pipeline cannot be built or run (a missing stage, a device that will not open)."""
