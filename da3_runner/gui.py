"""Gradio GUI for da3_runner.

Exposes every RunConfig knob as a widget. The GUI is a config builder + launcher:
"Save Config" writes YAML; "Run" shells out to `python -m da3_runner.cli run <yaml>`
and tails stdout into a textbox. Same outputs whether driven from GUI or CLI.

Launch:
    python -m da3_runner.gui                   # local-only
    DA3_GUI_SHARE=1 python -m da3_runner.gui   # public share link (cloud)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import yaml

from .config import CAMERAS_ALL, RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "da3_runner" / "configs"
DATA_ROOT = Path("~/data/4DGT").expanduser()


def list_scenes() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted(str(p) for p in DATA_ROOT.glob("ORIGINAL_*") if p.is_dir())


def list_existing_yamls() -> list[str]:
    return sorted(str(p) for p in CFG_DIR.glob("*.yaml"))


def build_cfg(*args) -> RunConfig:
    keys = [
        "dataset_dir", "cameras", "fps_in", "fps_out", "duration_seconds", "start_seconds",
        "bumper_blackfill", "bumper_rows_start", "bumper_rows_end",
        "process_res", "process_res_method",
        "mode", "model_dir", "streaming_weights_dir",
        "chunk_size", "overlap",
        "align_method", "align_lib", "scale_compute_method",
        "ref_view_strategy", "ref_view_strategy_loop", "loop_enable",
        "irls_delta", "irls_max_iters", "irls_tol",
        "pointcloud_sample_ratio", "pointcloud_conf_coef",
        "delete_temp_files", "save_debug_info", "save_depth_conf_result",
        "use_known_intrinsics", "use_known_extrinsics", "use_ray_pose",
        "export_pointcloud", "export_glb", "export_3dgs", "export_3dgs_video", "export_extras",
        "show_cameras", "conf_thresh_percentile", "num_max_points", "feat_vis_fps",
        "backproj_downsample", "backproj_conf_percentile",
        "export_tsdf", "tsdf_voxel", "tsdf_trunc",
        "export_pointcloud_by_cam", "final_voxel",
        "run_name",
    ]
    d = dict(zip(keys, args))
    d["bumper_rows"] = [int(d.pop("bumper_rows_start")), int(d.pop("bumper_rows_end"))]
    if not d["duration_seconds"] or d["duration_seconds"] <= 0:
        d["duration_seconds"] = None
    if not d["run_name"]:
        d["run_name"] = ""
    return RunConfig.from_dict(d)


def save_config(*args) -> str:
    cfg = build_cfg(*args)
    name = cfg.run_name or "unnamed"
    path = CFG_DIR / f"{name}.yaml"
    cfg.to_yaml(path)
    return f"Saved → {path}"


def _stream_subprocess(cmd: list[str], cwd: Path):
    """Yield stdout lines from a subprocess."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.stdout.close()
    code = proc.wait()
    yield f"\n[exit code: {code}]"


def run_now(*args):
    cfg = build_cfg(*args)
    name = cfg.run_name or "unnamed"
    yaml_path = CFG_DIR / f"{name}.yaml"
    cfg.to_yaml(yaml_path)
    cmd = [sys.executable, "-m", "da3_runner.cli", "run", str(yaml_path)]
    log = f"$ {' '.join(cmd)}\n"
    yield log
    for line in _stream_subprocess(cmd, cwd=REPO_ROOT):
        log += line + "\n"
        yield log


def main():
    with gr.Blocks(title="DA3 Multi-Cam Runner") as demo:
        gr.Markdown("# Depth Anything 3 — multi-camera experiment runner")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Dataset")
                dataset_dir = gr.Dropdown(
                    label="dataset_dir", choices=list_scenes(), allow_custom_value=True,
                    value=(list_scenes() or [""])[0],
                )
                cameras = gr.CheckboxGroup(
                    label="cameras (order = interleave order)",
                    choices=list(CAMERAS_ALL),
                    value=list(CAMERAS_ALL),
                )
                with gr.Row():
                    fps_in = gr.Number(label="fps_in", value=30.0)
                    fps_out = gr.Number(label="fps_out", value=5.0)
                with gr.Row():
                    duration_seconds = gr.Number(
                        label="duration_seconds (0 = whole)", value=6.0
                    )
                    start_seconds = gr.Number(label="start_seconds", value=0.0)

                gr.Markdown("### Preprocessing")
                bumper_blackfill = gr.Checkbox(label="bumper_blackfill (front)", value=True)
                with gr.Row():
                    bumper_rows_start = gr.Number(label="bumper_rows[0]", value=1043, precision=0)
                    bumper_rows_end = gr.Number(label="bumper_rows[1]", value=1160, precision=0)
                with gr.Row():
                    process_res = gr.Dropdown(
                        label="process_res", choices=[336, 504, 700, 1024], value=504,
                        allow_custom_value=True,
                    )
                    process_res_method = gr.Dropdown(
                        label="process_res_method",
                        choices=["upper_bound_resize", "lower_bound_resize"],
                        value="upper_bound_resize",
                    )

            with gr.Column():
                gr.Markdown("### Mode + model")
                mode = gr.Radio(
                    label="mode", choices=["singleshot", "streaming"], value="singleshot",
                )
                model_dir = gr.Textbox(
                    label="model_dir (HF id, single-shot)",
                    value="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                )
                streaming_weights_dir = gr.Textbox(
                    label="streaming_weights_dir (relative to da3_streaming/)",
                    value="./weights",
                )

                gr.Markdown("### Streaming-only")
                with gr.Row():
                    chunk_size = gr.Number(label="chunk_size", value=80, precision=0)
                    overlap = gr.Number(label="overlap", value=40, precision=0)
                with gr.Row():
                    align_method = gr.Dropdown(
                        label="align_method", choices=["sim3", "se3", "scale+se3"], value="sim3",
                    )
                    align_lib = gr.Dropdown(
                        label="align_lib", choices=["torch", "triton", "numba", "numpy"],
                        value="torch",
                    )
                    scale_compute_method = gr.Dropdown(
                        label="scale_compute_method (only for scale+se3)",
                        choices=["auto", "ransac", "weighted"], value="auto",
                    )
                with gr.Row():
                    ref_view_strategy = gr.Dropdown(
                        label="ref_view_strategy",
                        choices=["saddle_balanced", "middle", "first", "saddle_sim_range"],
                        value="saddle_balanced",
                    )
                    ref_view_strategy_loop = gr.Dropdown(
                        label="ref_view_strategy_loop",
                        choices=["saddle_balanced", "middle", "first", "saddle_sim_range"],
                        value="saddle_balanced",
                    )
                loop_enable = gr.Checkbox(label="loop_enable (needs SALAD weights)", value=False)

                gr.Markdown("#### Sim3 IRLS (per-pair alignment fit)")
                with gr.Row():
                    irls_delta = gr.Number(label="irls_delta", value=0.1)
                    irls_max_iters = gr.Number(label="irls_max_iters", value=5, precision=0)
                    irls_tol = gr.Number(label="irls_tol", value=1e-9)

                gr.Markdown("#### Streaming pointcloud merge")
                with gr.Row():
                    pointcloud_sample_ratio = gr.Number(
                        label="pointcloud_sample_ratio", value=0.015,
                    )
                    pointcloud_conf_coef = gr.Number(
                        label="pointcloud_conf_coef", value=0.75,
                    )

                gr.Markdown("#### Streaming debug")
                with gr.Row():
                    delete_temp_files = gr.Checkbox(label="delete_temp_files", value=True)
                    save_debug_info = gr.Checkbox(label="save_debug_info (Sim3 in npz)", value=False)
                    save_depth_conf_result = gr.Checkbox(
                        label="save_depth_conf_result (per-frame npz)", value=True,
                    )

                gr.Markdown("### Intrinsics + Extrinsics A/B (camera_params.json)")
                with gr.Row():
                    use_known_intrinsics = gr.Checkbox(
                        label="use_known_intrinsics", value=False,
                    )
                    use_known_extrinsics = gr.Checkbox(
                        label="use_known_extrinsics (single-shot only; assumes static ego!)",
                        value=False,
                    )
                use_ray_pose = gr.Checkbox(
                    label="use_ray_pose (single-shot only; ray head vs camera decoder)",
                    value=False,
                )

                gr.Markdown("### Exports")
                export_pointcloud = gr.Checkbox(label="export_pointcloud (back-projected ply)", value=True)
                export_glb = gr.Checkbox(label="export_glb (DA3's scene.glb)", value=False)
                export_3dgs = gr.Checkbox(label="export_3dgs (needs gsplat)", value=False)
                export_3dgs_video = gr.Checkbox(label="export_3dgs_video (needs gsplat)", value=False)
                export_extras = gr.CheckboxGroup(
                    label="export_extras (extra DA3 formats)",
                    choices=["npz", "depth_vis", "feat_vis", "colmap"],
                    value=[],
                )
                show_cameras = gr.Checkbox(label="show_cameras (in glb)", value=True)
                with gr.Row():
                    conf_thresh_percentile = gr.Number(
                        label="conf_thresh_percentile (glb)", value=40.0,
                    )
                    num_max_points = gr.Number(
                        label="num_max_points (glb)", value=1000000, precision=0,
                    )
                    feat_vis_fps = gr.Number(
                        label="feat_vis_fps", value=15, precision=0,
                    )

                gr.Markdown("#### Back-projection (single-shot pointcloud.ply)")
                with gr.Row():
                    backproj_downsample = gr.Number(
                        label="backproj_downsample (px stride)", value=2, precision=0,
                    )
                    backproj_conf_percentile = gr.Number(
                        label="backproj_conf_percentile (drop bottom-X%)", value=30.0,
                    )

                gr.Markdown("#### TSDF fusion (single-shot pointcloud_tsdf.ply)")
                export_tsdf = gr.Checkbox(label="export_tsdf", value=False)
                with gr.Row():
                    tsdf_voxel = gr.Number(label="tsdf_voxel (m)", value=0.05)
                    tsdf_trunc = gr.Number(label="tsdf_trunc (m)", value=0.20)

                gr.Markdown("#### Diagnostics")
                export_pointcloud_by_cam = gr.Checkbox(
                    label="export_pointcloud_by_cam (per-cam tinted ply)", value=False,
                )
                final_voxel = gr.Number(
                    label="final_voxel (downsample merged ply, 0=off)", value=0.0,
                )

                gr.Markdown("### Bookkeeping")
                run_name = gr.Textbox(label="run_name (empty = auto)", value="")

        widgets = [
            dataset_dir, cameras, fps_in, fps_out, duration_seconds, start_seconds,
            bumper_blackfill, bumper_rows_start, bumper_rows_end,
            process_res, process_res_method,
            mode, model_dir, streaming_weights_dir,
            chunk_size, overlap,
            align_method, align_lib, scale_compute_method,
            ref_view_strategy, ref_view_strategy_loop, loop_enable,
            irls_delta, irls_max_iters, irls_tol,
            pointcloud_sample_ratio, pointcloud_conf_coef,
            delete_temp_files, save_debug_info, save_depth_conf_result,
            use_known_intrinsics, use_known_extrinsics, use_ray_pose,
            export_pointcloud, export_glb, export_3dgs, export_3dgs_video, export_extras,
            show_cameras, conf_thresh_percentile, num_max_points, feat_vis_fps,
            backproj_downsample, backproj_conf_percentile,
            export_tsdf, tsdf_voxel, tsdf_trunc,
            export_pointcloud_by_cam, final_voxel,
            run_name,
        ]

        with gr.Row():
            save_btn = gr.Button("Save config")
            run_btn = gr.Button("Run", variant="primary")
        status = gr.Textbox(label="status", lines=1)
        log_box = gr.Textbox(label="log", lines=25, max_lines=200, autoscroll=True)

        save_btn.click(save_config, inputs=widgets, outputs=status)
        run_btn.click(run_now, inputs=widgets, outputs=log_box)

    share = bool(int(os.environ.get("DA3_GUI_SHARE", "0")))
    demo.launch(server_name="0.0.0.0", share=share)


if __name__ == "__main__":
    main()
