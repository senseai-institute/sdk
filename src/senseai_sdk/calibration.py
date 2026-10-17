# Copyright SenseAI Institute. Licensed under the Apache License, Version 2.0.
"""``senseai calibrate``: produce the per-sensor ``calib/<id>.yaml`` files tokens' body frame needs.

Three helpers, one per sensor family, each writing the same ``calibration.yaml`` shape as
``senseai_tokens.frame.BodyFrame`` reads (``sources: {<id>: {T_body_sensor, intrinsics, …}}``):

- cameras: intrinsics from a checkerboard (OpenCV, when installed) or from the datasheet field
  of view and resolution (always available; good to a few percent);
- microphone arrays: channel positions from a known geometry or typed in;
- IMUs: gravity direction at rest → the sensor's roll/pitch relative to the body, and the
  gyro bias.

Every helper is a pure function over arrays so it is testable without hardware; the guided
flow asks questions through an injectable ``ask`` function.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from senseai_sdk.config import Config

__all__ = [
    "camera_from_checkerboard",
    "camera_from_fov",
    "guided",
    "imu_at_rest",
    "mic_array_from_geometry",
    "transform_from_xyz_rpy",
    "write_calibration",
]

CONVENTION = "x_fwd_y_left_z_up"

#: Microphone array geometries (metres, array frame: x forward, y left, z up).
MIC_GEOMETRIES: Mapping[str, list[list[float]]] = {
    "respeaker_4": [[0.0, 0.0323, 0.0], [-0.0323, 0.0, 0.0], [0.0, -0.0323, 0.0], [0.0323, 0.0, 0.0]],
    "respeaker_6": [
        [0.0463 * math.cos(math.radians(60 * k)), 0.0463 * math.sin(math.radians(60 * k)), 0.0]
        for k in range(6)
    ],
    "linear_4_5cm": [[0.0, 0.075 - 0.05 * k, 0.0] for k in range(4)],
}


def transform_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> list[list[float]]:
    """A 4×4 ``T_body_sensor`` from a translation and roll/pitch/yaw (rad), ZYX order."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return [[float(v) for v in row] for row in t]


def camera_from_fov(
    width: int, height: int, hfov_deg: float, vfov_deg: float | None = None
) -> dict[str, Any]:
    """Pinhole intrinsics from resolution and field of view (datasheet values)."""
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    fy = fx if vfov_deg is None else (height / 2) / math.tan(math.radians(vfov_deg) / 2)
    return {
        "intrinsics": [[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0],
        "width": width,
        "height": height,
        "method": f"fov({hfov_deg} deg)",
    }


def camera_from_checkerboard(
    frames: Sequence[np.ndarray[Any, Any]], pattern: tuple[int, int] = (9, 6), square_m: float = 0.025
) -> dict[str, Any]:
    """Intrinsics from checkerboard views (needs OpenCV: ``pip install opencv-python-headless``).

    Raises:
        RuntimeError: OpenCV is not installed, or fewer than 5 frames show the full pattern.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("checkerboard calibration needs opencv-python-headless; or use --hfov") from e
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    obj_pts, img_pts = [], []
    size = None
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
        size = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 5 or size is None:
        raise RuntimeError(f"only {len(obj_pts)} frames showed the full {pattern} pattern; need at least 5")
    _, k, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return {
        "intrinsics": k.tolist(),
        "distortion": [float(v) for v in dist.ravel()[:4]],
        "width": size[0],
        "height": size[1],
        "method": f"checkerboard({len(obj_pts)} views)",
    }


def mic_array_from_geometry(geometry: str | Sequence[Sequence[float]]) -> dict[str, Any]:
    """Channel positions from a named geometry or an explicit ``[[x, y, z], …]`` list."""
    if isinstance(geometry, str):
        if geometry not in MIC_GEOMETRIES:
            raise KeyError(f"unknown mic geometry {geometry!r}; known: {sorted(MIC_GEOMETRIES)}")
        pos = MIC_GEOMETRIES[geometry]
        method = f"geometry({geometry})"
    else:
        pos = [[float(v) for v in p] for p in geometry]
        method = "geometry(explicit)"
    return {"channel_positions": pos, "method": method}


def imu_at_rest(samples: np.ndarray[Any, Any], *, gravity: float = 9.80665) -> dict[str, Any]:
    """From ≥ 1 s of ``[N, 6]`` (ax, ay, az, gx, gy, gz) at rest: mounting roll/pitch and gyro bias.

    The sensor's z axis is assumed to point roughly up; the measured gravity vector gives the
    roll and pitch of the sensor relative to the body (yaw is unobservable at rest and left 0).
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 6 or len(s) < 10:
        raise ValueError("need an [N >= 10, 6] array of accel + gyro samples")
    acc = s[:, :3].mean(0)
    gyro_bias = s[:, 3:].mean(0)
    norm = float(np.linalg.norm(acc))
    if not 0.5 * gravity < norm < 1.5 * gravity:
        raise ValueError(f"mean |accel| = {norm:.2f} m/s²; the IMU is not at rest or the units are not m/s²")
    g = acc / norm
    roll = math.atan2(g[1], g[2])
    pitch = math.atan2(-g[0], math.hypot(g[1], g[2]))
    return {
        "T_body_sensor": transform_from_xyz_rpy((0.0, 0.0, 0.0), (roll, pitch, 0.0)),
        "gyro_bias": [float(v) for v in gyro_bias],
        "accel_scale": gravity / norm,
        "method": f"rest({len(s)} samples)",
    }


def write_calibration(path: Path, source_id: str, entry: Mapping[str, Any]) -> Path:
    """Write (or merge into) a tokens-format calibration file for one source."""
    doc: dict[str, Any] = {"frame": {"convention": CONVENTION, "origin": "base_link"}, "sources": {}}
    if path.is_file():
        doc = yaml.safe_load(path.read_text()) or doc
        doc.setdefault("sources", {})
    doc["sources"][source_id] = dict(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


Ask = Callable[[str, str], str]


def guided(
    cfg: Config, out_dir: Path, *, ask: Ask, say: Callable[[str], None] = print, only: str | None = None
) -> list[Path]:
    """Walk the configured sensors and write ``<out_dir>/<id>.yaml`` for each.

    ``ask(prompt, default)`` returns the answer (``input`` in the CLI, a scripted dict in
    tests). Returns the files written.
    """
    written: list[Path] = []
    for s in cfg.sensors:
        if only and s.id != only:
            continue
        mod = str(s.options.get("modality", "")) or _modality_of(s.adapter)
        say(f"-- {s.id} ({s.adapter}, {mod or 'unknown modality'})")
        entry: dict[str, Any] = {}
        xyz = [
            float(v)
            for v in ask(f"{s.id} position on the robot x,y,z (m, x forward, y left)", "0,0,0").split(",")
        ]
        rpy = [
            math.radians(float(v))
            for v in ask(f"{s.id} orientation roll,pitch,yaw (deg)", "0,0,0").split(",")
        ]
        if mod in ("event", "frame"):
            w = int(ask(f"{s.id} image width (px)", "640"))
            h = int(ask(f"{s.id} image height (px)", "480"))
            hfov = float(ask(f"{s.id} horizontal field of view (deg, datasheet)", "90"))
            entry = camera_from_fov(w, h, hfov)
            say("  intrinsics from field of view; install opencv for a checkerboard fit")
        elif mod == "audio":
            geo = ask(
                f"{s.id} array geometry ({', '.join(sorted(MIC_GEOMETRIES))} or x,y,z;x,y,z;…)", "respeaker_4"
            )
            if geo in MIC_GEOMETRIES:
                entry = mic_array_from_geometry(geo)
            else:
                entry = mic_array_from_geometry([[float(v) for v in p.split(",")] for p in geo.split(";")])
        elif mod == "imu":
            say("  keep the robot still; the IMU's gravity reading sets its mounting angle")
            entry = {"method": "rest(pending)"}  # filled by `calibrate --imu-rest <file>` with real samples
        entry["T_body_sensor"] = transform_from_xyz_rpy(xyz, rpy)
        p = write_calibration(out_dir / f"{s.id}.yaml", s.id, entry)
        say(f"  wrote {p}; reference it as calib: {p} on sensors[{s.id}]")
        written.append(p)
    return written


def _modality_of(adapter: str) -> str:
    return {"prophesee": "event", "v4l2": "frame", "respeaker": "audio", "serial": "imu", "wav": "audio"}.get(
        adapter, ""
    )
