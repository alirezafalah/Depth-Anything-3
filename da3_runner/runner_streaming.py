"""Streaming runner: shells out to da3_streaming/da3_streaming.py with a per-run YAML.

Builds the YAML by overlaying RunConfig values on the upstream `static_front_poc.yaml`
template, drops a known_intrinsics .npy next to the staged inputs, and invokes the
streaming script as a subprocess so its globals (CUDA context, model load) are
isolated from the orchestrator process.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from .config import RunConfig
from .intrinsics import build_intrinsics_array
from .prepare_inputs import stage_inputs
from .runner_singleshot import write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
STREAMING_DIR = REPO_ROOT / "da3_streaming"
TEMPLATE_YAML = STREAMING_DIR / "configs" / "static_front_poc.yaml"


def _build_streaming_yaml(cfg: RunConfig, run_dir: Path, kix_npy: Path | None) -> Path:
    with TEMPLATE_YAML.open() as f:
        merged = yaml.safe_load(f)

    # Weights — point at downloaded streaming weights (or symlinks to HF cache).
    weights_dir = (STREAMING_DIR / cfg.streaming_weights_dir).resolve()
    merged["Weights"]["DA3"] = str(weights_dir / "model.safetensors")
    merged["Weights"]["DA3_CONFIG"] = str(weights_dir / "config.json")

    m = merged["Model"]
    m["chunk_size"] = cfg.chunk_size
    m["overlap"] = cfg.overlap
    m["ref_view_strategy"] = cfg.ref_view_strategy
    m["ref_view_strategy_loop"] = cfg.ref_view_strategy_loop
    m["align_method"] = cfg.align_method
    m["align_lib"] = cfg.align_lib
    m["scale_compute_method"] = cfg.scale_compute_method
    m["loop_enable"] = cfg.loop_enable
    m["save_depth_conf_result"] = cfg.save_depth_conf_result
    m["save_debug_info"] = cfg.save_debug_info
    m["delete_temp_files"] = cfg.delete_temp_files
    m.setdefault("IRLS", {})
    m["IRLS"]["delta"] = cfg.irls_delta
    m["IRLS"]["max_iters"] = cfg.irls_max_iters
    m["IRLS"]["tol"] = cfg.irls_tol
    m.setdefault("Pointcloud_Save", {})
    m["Pointcloud_Save"]["sample_ratio"] = cfg.pointcloud_sample_ratio
    m["Pointcloud_Save"]["conf_threshold_coef"] = cfg.pointcloud_conf_coef
    if kix_npy is not None:
        m["known_intrinsics_npy"] = str(kix_npy)

    out_yaml = run_dir / "streaming_config.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return out_yaml


def run_streaming(cfg: RunConfig) -> Path:
    staged, cams_per_frame, paths = stage_inputs(cfg)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    kix_npy: Path | None = None
    if cfg.use_known_intrinsics:
        K = build_intrinsics_array(cfg.dataset_dir, cams_per_frame)
        kix_npy = staged / "known_intrinsics.npy"
        np.save(kix_npy, K)

    yaml_path = _build_streaming_yaml(cfg, run_dir, kix_npy)

    cmd = [
        "python3",
        "da3_streaming.py",
        "--image_dir",
        str(staged),
        "--config",
        str(yaml_path),
        "--output_dir",
        str(run_dir),
    ]
    print(f"[streaming] launching: {' '.join(cmd)} (cwd={STREAMING_DIR})")
    proc = subprocess.run(cmd, cwd=STREAMING_DIR)
    if proc.returncode != 0:
        raise RuntimeError(f"da3_streaming.py exited with code {proc.returncode}")

    # Normalize output: combined_pcd.ply -> pointcloud.ply (symlink).
    combined = run_dir / "pcd" / "combined_pcd.ply"
    if combined.exists():
        link = run_dir / "pointcloud.ply"
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(combined.resolve())
        except OSError:
            shutil.copy2(combined, link)

    # Optional TSDF post-processing (per-chunk and/or global).
    if cfg.tsdf_per_chunk_streaming or cfg.tsdf_global_streaming:
        if not cfg.save_depth_conf_result:
            print(
                "[streaming] WARN: TSDF requested but save_depth_conf_result=false; "
                "results_output/ has no per-frame npz to integrate. Skipping TSDF."
            )
        else:
            from .postprocess_streaming import tsdf_global, tsdf_per_chunk

            if cfg.tsdf_per_chunk_streaming:
                tsdf_per_chunk(
                    run_dir,
                    voxel=cfg.tsdf_voxel,
                    trunc=cfg.tsdf_trunc,
                    conf_pct=cfg.backproj_conf_percentile,
                )
            if cfg.tsdf_global_streaming:
                tsdf_global(
                    run_dir,
                    voxel=cfg.tsdf_voxel,
                    trunc=cfg.tsdf_trunc,
                    conf_pct=cfg.backproj_conf_percentile,
                )

    write_manifest(
        cfg,
        run_dir,
        extra={
            "n_frames": len(paths),
            "staged_dir": str(staged),
            "streaming_yaml": str(yaml_path),
        },
    )
    print(f"[streaming] done. outputs at {run_dir}")
    return run_dir
