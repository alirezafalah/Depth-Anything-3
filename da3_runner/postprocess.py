"""Post-processing: back-project DA3 depth maps + optional TSDF fusion.

Ported from /home/CW01/uik07077/4dgt/scripts/da3_multiview_pointcloud.py — the
reference pipeline that produces noticeably cleaner output than DA3's built-in
GLB exporter, because it:

1. Back-projects every depth pixel (with a configurable stride) into world space
   using the model's own (K, w2c, depth), then concatenates instead of percentile-
   thresholding+capping like the GLB path.
2. Optionally fuses all views through Open3D's ScalableTSDFVolume so per-view
   contradictions average out and free space gets carved away.

Per-view confidence percentile is applied here (not globally), which matters when
one camera is much darker/blurrier than the others (front bumper, low light).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAM_TINTS = {
    "front": (255, 80, 80),
    "rear":  (80, 255, 80),
    "left":  (80, 80, 255),
    "right": (255, 255, 80),
}


def _backproject_view(
    depth: np.ndarray,
    K: np.ndarray,
    W2C: np.ndarray,
    rgb: np.ndarray,
    downsample: int,
    conf: np.ndarray | None,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one (depth, K, w2c) into world points + matching RGB."""
    H, W = depth.shape
    valid = depth > 0
    if conf is not None and conf_percentile > 0:
        thresh = np.percentile(conf, conf_percentile)
        valid &= conf >= thresh

    v_grid, u_grid = np.meshgrid(
        np.arange(0, H, downsample), np.arange(0, W, downsample), indexing="ij"
    )
    sample = valid[v_grid, u_grid]
    u, v = u_grid[sample], v_grid[sample]
    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pix = np.stack([u, v, np.ones_like(u)], axis=0).astype(np.float32)
    z = depth[v, u].astype(np.float32)
    X_cam = (np.linalg.inv(K) @ pix) * z
    X_cam_h = np.vstack([X_cam, np.ones((1, len(u)), dtype=np.float32)])

    W2C_h = np.eye(4, dtype=np.float32)
    W2C_h[:3, :] = W2C.astype(np.float32)
    X_world = (np.linalg.inv(W2C_h) @ X_cam_h)[:3, :].T
    return X_world, rgb[v, u]


def _o3d():
    import open3d as o3d  # noqa: F401  — late import keeps it optional

    return o3d


def write_pointcloud(
    pred,
    cams_per_frame: list[str],
    out_path: Path,
    downsample: int = 2,
    conf_percentile: float = 30.0,
    final_voxel: float = 0.0,
) -> int:
    """Back-project all DA3 views into one merged ply. Returns point count."""
    o3d = _o3d()
    depth = np.squeeze(np.asarray(pred.depth))
    K = np.asarray(pred.intrinsics)
    W2C = np.asarray(pred.extrinsics)
    rgb = np.asarray(pred.processed_images)
    conf = np.squeeze(np.asarray(pred.conf)) if getattr(pred, "conf", None) is not None else None

    pts_all, col_all = [], []
    for i in range(depth.shape[0]):
        pts, col = _backproject_view(
            depth[i], K[i], W2C[i], rgb[i], downsample,
            conf[i] if conf is not None else None, conf_percentile,
        )
        pts_all.append(pts)
        col_all.append(col)
    pts = np.concatenate(pts_all, axis=0) if pts_all else np.zeros((0, 3))
    col = np.concatenate(col_all, axis=0) if col_all else np.zeros((0, 3), dtype=np.uint8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255.0)
    if final_voxel > 0 and len(pcd.points):
        pcd = pcd.voxel_down_sample(final_voxel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return len(pcd.points)


def write_pointcloud_by_cam(
    pred,
    cams_per_frame: list[str],
    out_path: Path,
    downsample: int = 2,
    conf_percentile: float = 30.0,
    final_voxel: float = 0.0,
) -> int:
    """Per-camera tinted ply. Useful for spotting cross-camera misalignment."""
    o3d = _o3d()
    depth = np.squeeze(np.asarray(pred.depth))
    K = np.asarray(pred.intrinsics)
    W2C = np.asarray(pred.extrinsics)
    rgb = np.asarray(pred.processed_images)
    conf = np.squeeze(np.asarray(pred.conf)) if getattr(pred, "conf", None) is not None else None

    diag = o3d.geometry.PointCloud()
    for i in range(depth.shape[0]):
        cam = cams_per_frame[i] if i < len(cams_per_frame) else "front"
        pts, _ = _backproject_view(
            depth[i], K[i], W2C[i], rgb[i], downsample,
            conf[i] if conf is not None else None, conf_percentile,
        )
        if len(pts) == 0:
            continue
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        tint = np.array(CAM_TINTS.get(cam, (255, 255, 255)), dtype=np.float64) / 255.0
        sub.paint_uniform_color(tint)
        diag += sub
    if final_voxel > 0 and len(diag.points):
        diag = diag.voxel_down_sample(final_voxel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), diag)
    return len(diag.points)


def write_tsdf(
    pred,
    out_path: Path,
    voxel_length: float = 0.05,
    sdf_trunc: float = 0.20,
    conf_percentile: float = 30.0,
) -> int:
    """Fuse all per-view depth maps into a TSDF volume, write extracted ply."""
    o3d = _o3d()
    depth = np.squeeze(np.asarray(pred.depth))
    K = np.asarray(pred.intrinsics)
    W2C = np.asarray(pred.extrinsics)
    rgb = np.asarray(pred.processed_images)
    conf = np.squeeze(np.asarray(pred.conf)) if getattr(pred, "conf", None) is not None else None

    H, W = depth.shape[1], depth.shape[2]
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(depth.shape[0]):
        d = depth[i].astype(np.float32).copy()
        if conf is not None and conf_percentile > 0:
            d[conf[i] < np.percentile(conf[i], conf_percentile)] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb[i].astype(np.uint8))),
            o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=1e6, convert_rgb_to_intensity=False,
        )
        Ki = K[i]
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2])
        W2C_h = np.eye(4)
        W2C_h[:3, :] = W2C[i]
        vol.integrate(rgbd, intr, W2C_h)
    pcd = vol.extract_point_cloud()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)
    return len(pcd.points)
