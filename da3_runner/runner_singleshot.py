"""Single-shot runner: all staged frames in one DepthAnything3.inference call."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .config import RunConfig
from .intrinsics import build_extrinsics_array, build_intrinsics_array
from .postprocess import write_pointcloud, write_pointcloud_by_cam, write_tsdf
from .prepare_inputs import stage_inputs


def _build_export_format(cfg: RunConfig) -> str:
    parts: list[str] = ["mini_npz"]
    if cfg.export_glb:
        parts.append("glb")
    if cfg.export_3dgs:
        parts.append("gs_ply")
    if cfg.export_3dgs_video:
        parts.append("gs_video")
    for extra in cfg.export_extras:
        if extra not in parts:
            parts.append(extra)
    return "-".join(parts)


def _write_pose_files(pred, run_dir: Path) -> None:
    """Match streaming's camera_poses.txt + intrinsic.txt format.

    camera_poses.txt: one row per frame, 16 floats of 4x4 cam-to-world.
    intrinsic.txt:   one row per frame, "fx fy cx cy".
    """
    if pred.extrinsics is None or pred.intrinsics is None:
        return
    extr = np.asarray(pred.extrinsics)            # (N, 3, 4) world-to-cam
    intr = np.asarray(pred.intrinsics)            # (N, 3, 3)
    n = extr.shape[0]
    poses_path = run_dir / "camera_poses.txt"
    with poses_path.open("w") as f:
        for i in range(n):
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :] = extr[i]
            c2w = np.linalg.inv(w2c)
            f.write(" ".join(str(x) for x in c2w.flatten()) + "\n")
    intr_path = run_dir / "intrinsic.txt"
    with intr_path.open("w") as f:
        for i in range(n):
            K = intr[i]
            f.write(f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    print(f"[singleshot] wrote {poses_path.name} ({n} rows) + {intr_path.name}")


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _gpu_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def write_manifest(cfg: RunConfig, run_dir: Path, extra: dict) -> None:
    manifest = {
        "run_name": cfg.run_name,
        "config": cfg.to_dict(),
        "git_sha": _git_sha(),
        "gpu": _gpu_name(),
        **extra,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_singleshot(cfg: RunConfig) -> Path:
    """Execute a single-shot run and return the output directory."""
    import torch

    from depth_anything_3.api import DepthAnything3

    if cfg.export_3dgs or cfg.export_3dgs_video:
        try:
            import gsplat  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "3DGS export requested but `gsplat` is not installed. "
                "Install with: uv pip install -e '.[gs]'  (from repo root)"
            ) from e

    staged, cams_per_frame, paths = stage_inputs(cfg)

    intrinsics: np.ndarray | None = None
    if cfg.use_known_intrinsics:
        intrinsics = build_intrinsics_array(cfg.dataset_dir, cams_per_frame)

    extrinsics: np.ndarray | None = None
    if cfg.use_known_extrinsics:
        extrinsics = build_extrinsics_array(cfg.dataset_dir, cams_per_frame)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[singleshot] loading model {cfg.model_dir} ...")
    model = DepthAnything3.from_pretrained(cfg.model_dir).to("cuda")

    print(
        f"[singleshot] inferring {len(paths)} frames "
        f"@ res={cfg.process_res} ({cfg.process_res_method}) "
        f"intrinsics={'KNOWN' if intrinsics is not None else 'estimated'} "
        f"extrinsics={'KNOWN' if extrinsics is not None else 'estimated'} "
        f"use_ray_pose={cfg.use_ray_pose} "
        f"infer_gs={cfg.export_3dgs or cfg.export_3dgs_video}"
    )

    pred = model.inference(
        image=[str(p) for p in paths],
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        use_ray_pose=cfg.use_ray_pose,
        process_res=cfg.process_res,
        process_res_method=cfg.process_res_method,
        infer_gs=cfg.export_3dgs or cfg.export_3dgs_video,
        export_dir=str(run_dir),
        export_format=_build_export_format(cfg),
        show_cameras=cfg.show_cameras,
        conf_thresh_percentile=cfg.conf_thresh_percentile,
        num_max_points=cfg.num_max_points,
        feat_vis_fps=cfg.feat_vis_fps,
    )

    _write_pose_files(pred, run_dir)

    if cfg.export_pointcloud:
        n = write_pointcloud(
            pred, cams_per_frame, run_dir / "pointcloud.ply",
            downsample=cfg.backproj_downsample,
            conf_percentile=cfg.backproj_conf_percentile,
            final_voxel=cfg.final_voxel,
        )
        print(f"[singleshot] wrote pointcloud.ply ({n:,} pts)")
    if cfg.export_pointcloud_by_cam:
        n = write_pointcloud_by_cam(
            pred, cams_per_frame, run_dir / "pointcloud_by_cam.ply",
            downsample=cfg.backproj_downsample,
            conf_percentile=cfg.backproj_conf_percentile,
            final_voxel=cfg.final_voxel,
        )
        print(f"[singleshot] wrote pointcloud_by_cam.ply ({n:,} pts)")
    if cfg.export_tsdf:
        n = write_tsdf(
            pred, run_dir / "pointcloud_tsdf.ply",
            voxel_length=cfg.tsdf_voxel,
            sdf_trunc=cfg.tsdf_trunc,
            conf_percentile=cfg.backproj_conf_percentile,
        )
        print(f"[singleshot] wrote pointcloud_tsdf.ply ({n:,} pts)")

    write_manifest(
        cfg,
        run_dir,
        extra={
            "n_frames": len(paths),
            "staged_dir": str(staged),
            "depth_shape": list(getattr(pred.depth, "shape", [])),
        },
    )
    print(f"[singleshot] done. outputs at {run_dir}")
    return run_dir
