# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""The quickstart: load a config, get commands and events.

    uv run python examples/quickstart.py [senseai.yaml]

With no argument it builds the test fixture (signed artefacts, a licence) in a temporary
directory and runs single-loop mode over it, so it works with no hardware and no store.
"""

from __future__ import annotations

import sys

from senseai_sdk import SenseAI


def main(config_path: str) -> None:
    sai = SenseAI(config_path)
    fires = 0

    def on_command(cmd: object) -> None:
        nonlocal fires
        if getattr(cmd, "action", "none") != "none":
            fires += 1

    sai.on_command(on_command)  # every tick (~100 Hz), including action="none"
    sai.on_event(lambda ev: print(getattr(ev, "text", ev)))  # on change only
    try:
        while True:  # single-loop: deterministic, one tick per call; sai.run() is the threaded mode
            sai.step()
    except StopIteration:
        pass
    finally:
        sai.stop()
    print(sai.health().summary())
    print(f"{fires} ticks carried a reflex action")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        from _fixture import fixture_config

        with fixture_config() as path:
            main(str(path))
