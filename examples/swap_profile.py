# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""Hot-swap the species profile while the pipeline runs: null → panther → null.

uv run python examples/swap_profile.py
"""

from __future__ import annotations

from _fixture import fixture_config
from senseai_sdk import SenseAI


def main() -> None:
    with fixture_config() as path:
        sai = SenseAI(path)
        for i in range(60):
            if i == 20:
                sai.load_profile("panther")  # applied at the next tick boundary, < 1 ms
            if i == 40:
                sai.load_profile("null")
            sai.step()
            if i in (19, 25, 45):
                print(f"tick {i}: profile {sai.health().profile}")
        sai.stop()


if __name__ == "__main__":
    main()
