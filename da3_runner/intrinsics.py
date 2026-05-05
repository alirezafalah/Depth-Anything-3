"""Build per-frame intrinsics + extrinsics aligned to the staged interleaved order.

Reads `<dataset_dir>/camera_params.json` (4DGT format). Schema we depend on::

    {
      "cameras": {
        "<cam>": {
          "K": [[3x3 row-major]],
          "raw_position_mm": [x_mm, y_mm, z_mm],     # camera origin in vehicle frame
          "raw_rotation_deg": [rx, ry, rz]           # XYZ Euler camera->vehicle, degrees
        },
        ...
      }
    }

Vehicle frame: X=fwd, Y=left, Z=up (rear-axle origin).
Camera frame:  +X=optical, +Y=rows, +Z=cols.

DA3's input_processor scales intrinsics internally during resize, so we pass K at
the original image resolution (1390x1160) — see `api.py::_preprocess_inputs`.

WARNING about extrinsics:
- These describe the four cameras as **rigidly mounted on the vehicle**.
- Replicating them per timestamp tells DA3 the ego car never moves between frames.
- This is a *correct prior* for the static parking scene, and *wrong* for any
  driving / dynamic scene. We do not currently have per-frame ego poses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Vehicle units in camera_params.json are millimetres; DA3 lives in meters.
_MM_TO_M = 1e-3


def load_camera_meta(dataset_dir: str | Path) -> dict:
    p = Path(dataset_dir).expanduser() / "camera_params.json"
    if not p.exists():
        raise FileNotFoundError(f"camera_params.json not found at {p}")
    with p.open() as f:
        return json.load(f)


def load_camera_K_dict(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    """Return {camera_name: K(3,3)} from camera_params.json."""
    meta = load_camera_meta(dataset_dir)
    return {cam: np.array(d["K"], dtype=np.float32) for cam, d in meta["cameras"].items()}


def _euler_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Build R = Rx * Ry * Rz (XYZ intrinsic) from degrees."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("XYZ", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix().astype(np.float32)


def build_intrinsics_array(
    dataset_dir: str | Path,
    interleaved_cams: list[str],
) -> np.ndarray:
    """Build the (N, 3, 3) array aligned to the staged frame order."""
    K_by_cam = load_camera_K_dict(dataset_dir)
    missing = [c for c in interleaved_cams if c not in K_by_cam]
    if missing:
        raise KeyError(f"camera_params.json missing entries for: {missing}")
    return np.stack([K_by_cam[c] for c in interleaved_cams], axis=0).astype(np.float32)


def load_camera_extrinsics_dict(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    """Return {cam: world_to_cam(4,4)} where 'world' = vehicle frame.

    The JSON gives camera→vehicle (rotation + translation in mm). We convert to
    meters and invert to world-to-cam so it matches DA3's `extrinsics` argument
    convention (`api.py:115`: B,N,4,4 world-to-cam).
    """
    meta = load_camera_meta(dataset_dir)
    out: dict[str, np.ndarray] = {}
    for cam, d in meta["cameras"].items():
        R_c2v = _euler_xyz_deg_to_R(*d["raw_rotation_deg"])  # cam-to-vehicle
        t_c2v = np.asarray(d["raw_position_mm"], dtype=np.float32) * _MM_TO_M
        c2v = np.eye(4, dtype=np.float32)
        c2v[:3, :3] = R_c2v
        c2v[:3, 3] = t_c2v
        w2c = np.linalg.inv(c2v).astype(np.float32)  # vehicle->cam == world->cam under static-ego
        out[cam] = w2c
    return out


def build_extrinsics_array(
    dataset_dir: str | Path,
    interleaved_cams: list[str],
) -> np.ndarray:
    """(N, 4, 4) world-to-camera for the staged frame order. Static-ego assumption."""
    E_by_cam = load_camera_extrinsics_dict(dataset_dir)
    missing = [c for c in interleaved_cams if c not in E_by_cam]
    if missing:
        raise KeyError(f"camera_params.json missing extrinsics for: {missing}")
    return np.stack([E_by_cam[c] for c in interleaved_cams], axis=0).astype(np.float32)
