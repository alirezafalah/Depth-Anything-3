"""Streaming-mode TSDF post-processing.

Two outputs, gated by RunConfig flags:

- Per-chunk TSDF (`tsdf_per_chunk_streaming`):
    For each chunk, TSDF-fuse its frames in chunk-local coordinates, extract a
    point cloud, then apply that chunk's Sim3 (s, R, t) to bring it into the
    global frame. Concatenate -> `combined_pcd_tsdf.ply`. Per-chunk artifacts
    land at `pcd/{chunk_idx}_pcd_tsdf.ply`.
    Caveat: no cross-chunk averaging. Each chunk fuses independently.

- Global TSDF (`tsdf_global_streaming`):
    One ScalableTSDFVolume that receives every frame from every chunk, with
    depths metrically rescaled by the chunk's Sim3 scale and extrinsics taken
    from the streaming-aligned `camera_poses.txt`. Writes `pointcloud_tsdf_global.ply`.
    THIS is the cross-chunk average that should look the cleanest.

Inputs (all written by streaming itself):
    <run_dir>/chunk_metadata.npz          (chunk_indices, sim3_s/R/t, overlap_*)
    <run_dir>/camera_poses.txt            (16-float c2w per frame, global frame)
    <run_dir>/results_output/frame_*.npz  (image, depth, conf, intrinsics, per frame)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_metadata(run_dir: Path) -> dict:
    p = run_dir / "chunk_metadata.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — re-run streaming with the patched da3_streaming.py "
            "so chunk_metadata.npz is written."
        )
    raw = np.load(p, allow_pickle=True)
    return {
        "chunk_indices": raw["chunk_indices"],     # (n_chunks, 2)
        "sim3_s": raw["sim3_s"],                   # (n_chunks,)
        "sim3_R": raw["sim3_R"],                   # (n_chunks, 3, 3)
        "sim3_t": raw["sim3_t"],                   # (n_chunks, 3)
        "overlap": int(raw["overlap"]),
        "overlap_s": int(raw["overlap_s"]),
        "overlap_e": int(raw["overlap_e"]),
        "n_total_frames": int(raw["n_total_frames"]),
    }


def _load_global_c2w(run_dir: Path, n_frames: int) -> np.ndarray:
    """Read camera_poses.txt -> (n_frames, 4, 4) c2w."""
    p = run_dir / "camera_poses.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing")
    rows = []
    with p.open() as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 16:
                continue
            rows.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    if len(rows) != n_frames:
        print(
            f"[postprocess_streaming] WARN camera_poses.txt has {len(rows)} rows, "
            f"chunk_metadata says {n_frames}"
        )
    return np.stack(rows, axis=0)


def _load_frame(run_dir: Path, global_idx: int) -> dict | None:
    p = run_dir / "results_output" / f"frame_{global_idx}.npz"
    if not p.exists():
        return None
    npz = np.load(p)
    return {
        "image": npz["image"],
        "depth": npz["depth"],
        "conf": npz["conf"] if "conf" in npz.files else None,
        "intrinsics": npz["intrinsics"],
    }


def _chunk_owns(meta: dict, chunk_idx: int) -> range:
    """Which global frame indices does chunk_idx contribute to camera_poses.txt?

    Mirrors the slicing logic in da3_streaming.py::save_camera_poses (lines 765-805).
    """
    n_chunks = len(meta["chunk_indices"])
    start, end = meta["chunk_indices"][chunk_idx]
    if chunk_idx == 0:
        first_end = end if n_chunks == 1 else end - meta["overlap_e"]
        return range(int(start), int(first_end))
    is_last = chunk_idx == n_chunks - 1
    chunk_end = end if is_last else end - meta["overlap_e"]
    return range(int(start) + meta["overlap_s"], int(chunk_end))


def _o3d():
    import open3d as o3d  # noqa: F401

    return o3d


def _build_tsdf(voxel: float, trunc: float):
    o3d = _o3d()
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate_frame(vol, frame: dict, w2c: np.ndarray, conf_pct: float) -> None:
    o3d = _o3d()
    depth = frame["depth"].astype(np.float32).copy()
    if frame["conf"] is not None and conf_pct > 0:
        depth[frame["conf"] < np.percentile(frame["conf"], conf_pct)] = 0.0
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(frame["image"].astype(np.uint8))),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=1e6,
        convert_rgb_to_intensity=False,
    )
    K = frame["intrinsics"]
    H, W = depth.shape
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    W2C_h = np.eye(4)
    W2C_h[:3, :] = w2c[:3, :]  # accept (3,4) or (4,4)
    vol.integrate(rgbd, intr, W2C_h)


# ---------------------------------------------------------------------------
# Per-chunk TSDF
# ---------------------------------------------------------------------------

def tsdf_per_chunk(
    run_dir: Path,
    voxel: float = 0.05,
    trunc: float = 0.20,
    conf_pct: float = 30.0,
) -> Path:
    """For each chunk, fuse its frames in chunk-local coords and bring into global frame.

    Returns the path to the merged combined_pcd_tsdf.ply.
    """
    o3d = _o3d()
    meta = _load_metadata(run_dir)
    n_chunks = len(meta["chunk_indices"])
    pcd_dir = run_dir / "pcd"
    pcd_dir.mkdir(exist_ok=True)
    merged = o3d.geometry.PointCloud()

    raw_meta = np.load(run_dir / "chunk_metadata.npz", allow_pickle=True)
    chunk_extr = raw_meta["chunk_extrinsics"]  # array of (N_chunk, 3, 4)

    for ci in range(n_chunks):
        start, end = meta["chunk_indices"][ci]
        extr = chunk_extr[ci]   # (N_chunk, 3, 4) chunk-local w2c
        vol = _build_tsdf(voxel, trunc)
        n_used = 0
        for local_i, global_idx in enumerate(range(int(start), int(end))):
            frame = _load_frame(run_dir, global_idx)
            if frame is None:
                continue
            _integrate_frame(vol, frame, extr[local_i], conf_pct)
            n_used += 1
        local_pcd = vol.extract_point_cloud()

        # Apply chunk's Sim3 to bring into global frame.
        s, R, t = float(meta["sim3_s"][ci]), meta["sim3_R"][ci], meta["sim3_t"][ci]
        if len(local_pcd.points) > 0:
            pts = np.asarray(local_pcd.points)
            pts = (s * (R @ pts.T)).T + t
            local_pcd.points = o3d.utility.Vector3dVector(pts)

        out = pcd_dir / f"{ci}_pcd_tsdf.ply"
        o3d.io.write_point_cloud(str(out), local_pcd)
        print(
            f"[tsdf_per_chunk] chunk {ci}: {n_used} frames → "
            f"{len(local_pcd.points):,} pts → {out.name}"
        )
        merged += local_pcd

    out_merged = run_dir / "combined_pcd_tsdf.ply"
    o3d.io.write_point_cloud(str(out_merged), merged)
    print(f"[tsdf_per_chunk] merged: {len(merged.points):,} pts → {out_merged.name}")
    return out_merged


# ---------------------------------------------------------------------------
# Global TSDF
# ---------------------------------------------------------------------------

def tsdf_global(
    run_dir: Path,
    voxel: float = 0.05,
    trunc: float = 0.20,
    conf_pct: float = 30.0,
) -> Path:
    """One TSDF volume integrating every frame, using globally-aligned w2c.

    Per-chunk Sim3 scale is applied to depth so all frames are in the same
    metric scale before integration.
    """
    o3d = _o3d()
    meta = _load_metadata(run_dir)
    n_frames = meta["n_total_frames"]
    c2w_global = _load_global_c2w(run_dir, n_frames)

    # Build a per-frame scale lookup from chunk ownership.
    scale_per_frame = np.ones(n_frames, dtype=np.float64)
    for ci in range(len(meta["chunk_indices"])):
        s = float(meta["sim3_s"][ci])
        for gi in _chunk_owns(meta, ci):
            scale_per_frame[gi] = s

    vol = _build_tsdf(voxel, trunc)
    n_used = 0
    for gi in range(n_frames):
        frame = _load_frame(run_dir, gi)
        if frame is None:
            continue
        # Rescale depth into global metric (same scale used to transform extrinsics).
        s = scale_per_frame[gi]
        scaled = dict(frame)
        scaled["depth"] = frame["depth"].astype(np.float32) * s
        # Convert global c2w -> w2c (3,4) for _integrate_frame.
        w2c = np.linalg.inv(c2w_global[gi])
        _integrate_frame(vol, scaled, w2c[:3, :], conf_pct)
        n_used += 1

    pcd = vol.extract_point_cloud()
    out = run_dir / "pointcloud_tsdf_global.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    print(
        f"[tsdf_global] integrated {n_used} frames → {len(pcd.points):,} pts → {out.name}"
    )
    return out
