# senseai-sdk

The one package a SenseAI customer installs. It connects the six runtime packages into a single
pipeline: sensors in, bounded reflex commands and semantic events out. It also verifies and
meters the proprietary artefacts it loads, and adds nothing clever of its own.

```
 your sensors / recordings                       senseai-sdk                                  your robot
 ┌──────────────┐  ┌─────────────────────────────────────────────────────────────────────┐  ┌──────────────┐
 │ dvs0 rgb0    │─▶│ ingest ─▶ tokens ─▶ world ─▶ reflex ───────── ReflexCommand ────────┼─▶│ controller   │
 │ mic0 imu0 …  │  │   ▲          ▲        │         │  (≤ 15 ms sensor→actuator p99)    │  │ (/cmd_vel)   │
 └──────────────┘  │   │      profiles    └──▶ cortex ───────────── Event ───────────────┼─▶│ planner, LLM │
                   │ config · artefacts (signed) · licence · health · telemetry (opt-in)  │  │ fleet manager│
                   └─────────────────────────────────────────────────────────────────────┘  └──────────────┘
```

**What's on the box:** load a profile, bind your sensors and actuator, and get reflexes and
events. A new engineer can do this in under an hour without reading the other six READMEs.

**Status: design (step 1).** This README and `src/senseai_sdk/api.py` are the contract under
review. Nothing below is implemented yet. See [Upstream status](#upstream-status) for what
the SDK can and cannot compose today.

Apache-2.0, copyright SenseAI Institute. The world-model checkpoint, reflex weights and
profile priors that the SDK loads are **not** Apache-2.0. They are signed, licensed
artefacts, and this package never contains them.

---

## Quickstart

```bash
pip install senseai-sdk                          # pulls ingest, tokens, profiles, world, reflex, cortex
export SENSEAI_LICENCE=<your key>                # or run: senseai licence activate <key>
senseai init                                     # writes a commented senseai.yaml in this directory
$EDITOR senseai.yaml                             # endpoint_id, profile, sensors, reflex bridge: ~10 lines
senseai doctor                                   # checks sensors, calibration, artefacts, licence, clock
senseai run --config senseai.yaml                # runs the pipeline; Ctrl-C to stop
```

`senseai init --replay <recording>` writes a config with replay sources (a DSEC, M3ED or MVSEC
sequence, a rosbag, or the built-in fixture). Use it to get commands and events from
recorded data before you touch hardware. The same pipeline runs from Python:

```python
from senseai_sdk import SenseAI

sai = SenseAI("senseai.yaml")
sai.on_command(lambda cmd: print(cmd.action, cmd.speed_scale))   # every tick, ~100 Hz
sai.on_event(lambda ev: print(ev.text))                          # on change only
sai.run()                                                        # blocking; or sai.start() / sai.stop()
```

`examples/quickstart.py` is that script, about 20 lines, running on the Panther fixture.

---

## `senseai.yaml` reference

```yaml
endpoint_id: amr-0042                    # this robot; appears in licence metering, telemetry and health
licence_key: ${SENSEAI_LICENCE}          # ${ENV} is expanded; never commit a literal key
profile: {name: panther, version: 1}     # version omitted = newest the licence allows
artefacts: {world: world-v1.2, reflex: reflex-v1.0, cache: ~/.senseai}
sensors:
  - {id: dvs0, adapter: prophesee, device: /dev/evk4, calib: calib/dvs0.yaml}
  - {id: rgb0, adapter: v4l2, device: /dev/video0, calib: calib/rgb0.yaml}
  - {id: mic0, adapter: respeaker, channels: 4, calib: calib/mic0.yaml}
  - {id: imu0, adapter: serial, port: /dev/ttyUSB0}
tick_ms: 10
reflex: {bridge: ros2_cmd_vel, planner_topic: /cmd_vel, out_topic: /senseai/cmd_vel, max_lateral_m: 0.3}
cortex: {sinks: [ros2, jsonl], jsonl_path: /var/log/senseai/events.jsonl}
telemetry: {enabled: true, raw_frames: false, upload: null}
```

| key | type | default | meaning |
|---|---|---|---|
| `endpoint_id` | str | required | Stable robot identifier. Must match the endpoint class the licence allows. |
| `licence_key` | str | required | Signed licence key. `${VAR}` expands from the environment. |
| `profile.name` | str | `null` | Profile to load at start. `null` is the no-op profile and is always allowed. |
| `profile.version` | int | newest allowed | Pins a profile version. A pinned version that is missing fails at start, never silently. |
| `artefacts.world` | str | required | World-model artefact id (`<name>-v<semver>`). Its bundle also carries the tokenizer config and weights the checkpoint was trained with, so you never configure the tokenizer yourself. |
| `artefacts.reflex` | str | required | Reflex artefact id. |
| `artefacts.cache` | path | `~/.senseai` | Local artefact cache. Artefacts are verified each time they load, not just when they are fetched. |
| `artefacts.store` | path or URL | `${cache}/store` | Where artefacts are fetched from: a local directory or an HTTPS base URL. |
| `sensors[]` | list | required | One entry per ingest source. `id` becomes the ingest `source_id`. `adapter` names a live or replay adapter (the template from `senseai init` lists them). Every other key except `calib` is passed to the adapter's constructor, so `device`, `port`, `channels` and `path` are adapter options, not SDK keys. |
| `sensors[].calib` | path | none | The source's entry in tokens' `calibration.yaml` (`T_body_sensor`, `intrinsics`, …). The SDK merges the per-sensor files into one body-frame calibration. A sensor without one sits at the body origin, and `doctor` warns about it. |
| `tick_ms` | int | `10` | Aligner tick. Must equal the tick the world artefact was trained on; `doctor` and start-up both check this. |
| `reflex.bridge` | str | `callback` | `callback` (Python, for sim and tests) or `ros2_cmd_vel`. |
| `reflex.planner_topic` / `out_topic` | str | `/cmd_vel`, `/senseai/cmd_vel` | `ros2_cmd_vel` only. Point your controller at `out_topic`. |
| `reflex.max_lateral_m` | float | envelope limit | Can only **lower** the reflex safety envelope's lateral limit. A value above the envelope is an error, not a clamp. |
| `cortex.sinks` | list | `[callback]` | Any of `callback`, `jsonl`, `ros2`, `mqtt`. |
| `cortex.jsonl_path` | path | none | Required when `jsonl` is a sink. |
| `telemetry.enabled` | bool | `false` | Opt-in. See [Telemetry](#telemetry-and-privacy). |
| `telemetry.raw_frames` | bool | `false` | Include raw sensor frames in telemetry. Off unless you turn it on. |
| `telemetry.upload` | str or null | `null` | Upload hook (`module:function`). `null` means spool locally and never upload. |

Config errors name the key path, the value and what was expected
(`sensors[2].adapter: 'respeker' is not a known adapter; did you mean 'respeaker'?`).

---

## Python API

`from senseai_sdk import SenseAI` is the whole API. Everything in the six upstream packages is
still reachable, but you don't need it.

| method | what it does |
|---|---|
| `SenseAI(config)` | Validates config, verifies artefacts and licence, and builds the pipeline. It does not open devices. |
| `load_profile(name, version=None)` | Hot-swaps the profile between ticks, in < 1 ms. |
| `bind_sensor(source_id, adapter)` | Adds or replaces an ingest source at runtime (any `senseai_ingest` `Adapter`). |
| `bind_actuator(bridge)` | Sets the `ControllerBridge` that applies reflex commands to your controller. |
| `on_command(cb)` | Calls `cb(ReflexCommand)` every tick, including `action="none"`. |
| `on_event(cb)` | Calls `cb(Event)` for each cortex event. Events are emitted on change, not every cycle. |
| `update_context(**ctx)` | Updates scene and motion context: `lux`, `lux_map`, `speed`, `yaw_rate`, `heading`, `speed_cap`, `mode`. Applied before the next tick. |
| `run()` / `start()` / `stop()` | Run blocking, or start and stop a background thread. |
| `step()` | Runs exactly one tick in single-loop mode, for sim and tests. Returns a `StepResult`. |
| `health()` | Per-stage p50/p99, drops, watchdog, licence and artefact state. |
| `explain(t_ns)` | One-sentence explanation of the reflex command at `t_ns`. |
| `save_state(path)` / `load_state(path)` | Checkpoints tokenizer, world, reflex and cortex state, so a restart skips the warm-up. |

The full docstrings are in `src/senseai_sdk/api.py`.

### Single-loop vs threaded

- **Single-loop** (`step()`, `senseai run --single-loop`): one tick goes through every stage
  in order. It is deterministic, so the command and event hashes equal running the packages by
  hand. Use it for sim, tests and replay.
- **Threaded** (default for `run()` / `start()`): one thread per stage, with bounded queues
  between them. A full queue drops its oldest item and counts the drop. Cortex runs on its
  ~100 ms cycle and never sits on the reflex path.

### Fail-safe

The SDK never silently stops publishing commands. If a stage raises an exception, or stalls
past its budget, these things happen:

- `health()` reports `degraded`, naming the stage.
- The reflex watchdog takes over through the bound bridge: `slow` (speed × 0.5) after 50 ms
  without a surprise map, then `brake` after 250 ms.
- Cortex emits `sensor.degraded`, and later `sensor.restored`.

The `ros2_cmd_vel` bridge stops publishing while unhealthy. Your own safety stack therefore
stays authoritative, and the reflex never holds a stale command on `/senseai/cmd_vel`.

Profile swaps and context updates are queued and applied between ticks, never mid-tick.

---

## C API

For controllers that aren't written in Python. `capi/senseai.h`:

```c
senseai_handle_t *h;
int rc = senseai_init("senseai.yaml", &h);               /* starts the worker process */
senseai_load_profile(h, "panther", 0);                   /* 0 = newest allowed */
for (;;) {
    senseai_step(h);                                     /* or let the worker free-run */
    senseai_command_t cmd;
    if (senseai_get_command(h, &cmd) == SENSEAI_OK) apply(cmd.speed_scale, cmd.lateral_offset_m, cmd.stop);
    senseai_event_t ev[16]; int n = senseai_poll_events(h, ev, 16);
}
senseai_shutdown(h);
/* every call returns an int error code; senseai_strerror(rc) describes it */
```

At Gate 3 this is a thin shim. A long-running Python worker process does the work, and
ticks and commands cross a `multiprocessing.shared_memory` ring. A native build is out of
scope. `capi/example.c` builds with `cc` and runs against the fixture.

## ROS 2

```bash
pip install senseai-sdk[ros2]                            # rclpy itself comes from your ROS install
ros2 launch senseai_sdk senseai.launch.py config:=senseai.yaml
```

The launch file brings up the whole pipeline from the same YAML, using the packages' own
nodes. The SDK adds no new message types. Re-point your controller from `/cmd_vel` to
`/senseai/cmd_vel`, and subscribe your planner to `/senseai/events`.

---

## Artefacts and licence

**Artefacts.** The world checkpoint (with its tokenizer config and weights), the reflex
weights and the profile `.npz` priors each ship with a manifest. The manifest lists sha256
hashes and carries an Ed25519 signature from SenseAI's release key. The SDK checks the
signature and every hash each time it loads an artefact. A tampered, unsigned or unpinned
artefact never loads, and the error says which file failed and why. Profile bundles also
keep their own `senseai-profiles` HMAC sidecar, which is checked as well.

**Licence.** The licence key is signed by SenseAI. It encodes:

- the endpoint class
- the allowed profiles and senses
- the expiry date

The SDK verifies the key offline. There is no phone-home requirement to run. After the
licence expires there is a 30-day grace period: `health()` and `doctor` warn, and the
pipeline keeps running. After the grace period, `SenseAI(...)` refuses to start.

**Metering.** The SDK counts ticks and active senses per profile and writes signed usage
reports to a local spool. `senseai licence report` exports them for upload.

## Telemetry and privacy

Telemetry is **off** unless `telemetry.enabled: true`. When it is on, this is what it does:

- **Samples** surprise maps, reflex commands, cortex events and latency histograms.
- **Excludes raw frames, audio and event streams** unless `raw_frames: true`.
- **Spools locally**, up to a size cap.
- **Uploads only through a hook you provide** (`telemetry.upload`). The default is no upload.

The SDK never sends data anywhere on its own.

## Health

`SenseAI.health()` and the optional `/health` HTTP endpoint (`senseai run --health-port 8080`)
return JSON with these fields:

- per-stage p50/p99 latency
- queue drops
- watchdog state
- `ok | degraded | failed`
- licence state and days left
- loaded artefact versions
- active profile

## CLI

| command | |
|---|---|
| `senseai init [--replay PATH]` | Write a commented `senseai.yaml`. |
| `senseai doctor` | Check devices, calibration, artefacts, licence, tick and clock. Every failure says what to do next. |
| `senseai run --config senseai.yaml` | Run the pipeline. |
| `senseai bench --config … --ticks t.h5` | Sensor-to-actuator p50/p99 on this machine, per stage and end to end. |
| `senseai calibrate` | Guided: checkerboard (cameras), mic sweep (arrays), IMU at rest. Writes `calib/*.yaml`. |
| `senseai licence show\|activate\|report` | Inspect, install or export licence usage. |
| `senseai profiles list\|show` | Profiles the licence allows, with versions and audit report. |
| `senseai logs tail\|explain --at T` | Follow events; explain the reflex command at time T. |

## Latency budget

`senseai bench` measures each row with the packages' own stopwatches. The SDK's
contribution is plumbing, and its budget is ≤ 1 ms.

| stage | budget p99 | measured by |
|---|---|---|
| ingest sync | < 1 ms | aligner stats |
| tokens | ≤ 2 ms | tokenizer stopwatch |
| world | ≤ 6 ms (CUDA) | `SurpriseMap.latency_ns` |
| reflex | < 1 ms | `ReflexCommand.provenance.latency_ns` |
| transport | ≤ 3 ms | bridge publish timestamp |
| SDK overhead | ≤ 1 ms | pipeline stopwatch minus the sum of stages |
| **sensor → actuator** | **≤ 15 ms** | first sample's `t_master_ns` → bridge publish |

## Troubleshooting

Run `senseai doctor` first. It checks, in order:

- config
- licence
- artefact signatures
- each device opens and produces samples
- calibration present and plausible
- tick matches the world artefact
- clock offset and drift

It prints one line per check with a fix (`rgb0: /dev/video0 busy, held by pid 4121 (cheese)`).
`senseai logs explain --at <t>` answers "why did it slow down?" for any moment in the log.

## What the SDK will never do

- **Plan.** It does not choose a path. A reflex only modifies your planner's command.
- **Replace your controller or safety stack.** It sits in front of the controller, and it
  goes quiet when it is unhealthy.
- **Send raw sensor data anywhere** unless you turn it on and supply the upload hook.
- **Load an artefact it cannot verify**, or run past licence grace.
- **Add algorithms.** All perception lives in the upstream packages.

---

## Upstream status

The SDK composes these packages through their public contracts only:

| package | SDK uses | status (2026-10-17) |
|---|---|---|
| `senseai-ingest` | `Adapter`, `Aligner`, `SourceSpec` | Gate 1: replay works; live adapters raise `AdapterNotAvailable`. |
| `senseai-tokens` | `Tokenizer`, `TokenBatch` | Gate 1 complete on CPU. |
| `senseai-profiles` | `Registry`, `Profile` | Gate 1; `panther.v0` and `null.v0` bundles. |
| `senseai-world` | `WorldModel`, `SurpriseMap`, `WorldState` | Implemented; CPU `step()` p99 77–112 ms (over budget), CUDA not yet measured. |
| `senseai-reflex` | `ReflexEngine`, `ReflexCommand`, `ControllerBridge`, hooks | **Not started** (empty repository). |
| `senseai-cortex` | `CortexEngine`, `Event`, sinks | **Not started** (empty repository). |

## Develop

```bash
uv sync --extra dev
make test lint type        # pytest (coverage ≥ 85 %), ruff, mypy --strict
make bench                 # senseai bench on the fixture
uv build
```

## License

Apache-2.0. Copyright SenseAI Institute. See `LICENSE`.
