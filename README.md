# DA3 Multi-Camera Experiment Runner

Fork of [Depth Anything 3](README.upstream.md) that adds a YAML-driven runner +
Gradio GUI for sweeping DA3 over a 4-camera car dataset.

## What's here

- `da3_runner/` — orchestrator (CLI + GUI). Stages frames, optionally injects
  known intrinsics, dispatches DA3 in single-shot or streaming mode, exports
  point clouds and (optionally) 3D Gaussian Splats.
- `da3_streaming/` — upstream chunked-inference pipeline, with **one targeted
  patch** to accept caller-provided intrinsics via `Model.known_intrinsics_npy`
  in the YAML.
- `src/depth_anything_3/` — upstream model code (untouched).
- `README.upstream.md` — original upstream README.

## Install

```bash
# from repo root
uv pip install -e .                # core
uv pip install -e '.[gs]'          # + 3DGS via gsplat (optional, needed for export_3dgs)
uv pip install -e '.[app]'         # + Gradio (needed for the GUI)
uv pip install pyyaml typer        # runner deps (already pulled in by core)
```

For streaming mode, also have weights symlinked at `da3_streaming/weights/`
(already done on this machine — points at the HF cache copy of
`DA3NESTED-GIANT-LARGE-1.1`).

## Configure a run

Pick a template:

```bash
cp da3_runner/configs/single_shot_mode.yaml  my_run.yaml   # or
cp da3_runner/configs/streaming_mode.yaml    my_run.yaml   # or
cp da3_runner/configs/default.yaml           my_run.yaml   # all knobs commented
```

Every knob is documented in `da3_runner/configs/default.yaml`. The interesting
ones:

| Knob | Purpose |
|---|---|
| `cameras` | Subset and order of `[front, right, left, rear]`. Order = per-timestamp interleave. |
| `fps_in`/`fps_out` | Subsample by `stride = round(fps_in/fps_out)`. |
| `duration_seconds` | `null` = whole sequence; otherwise N seconds at `fps_out`. |
| `bumper_blackfill` | Zero rows 1043–1160 on every front frame. |
| `process_res` | DA3 internal resize target. 504 (default) → ~440 frames/H100/giant; 700 ≈ ~220; 1024 ≈ ~110. |
| `mode` | `singleshot` (one DA3 call) vs. `streaming` (chunked + Sim3-stitched). |
| `chunk_size`/`overlap` | Streaming only. **Must both be multiples of `len(cameras)`** so each chunk holds equal counts of every camera. |
| `use_known_intrinsics` | Read fx/fy/cx/cy from `camera_params.json` and pass to DA3. |
| `use_known_extrinsics` | **Single-shot only.** Read raw_position_mm + raw_rotation_deg from `camera_params.json` and replicate per-timestamp. **Assumes ego car never moves** — correct for static scenes, wrong for any driving scene. Streaming ignores this flag. |
| `use_ray_pose` | Single-shot only. Use ray-based pose head instead of the camera decoder. |
| `export_3dgs`/`export_3dgs_video` | Requires `gsplat` installed and a giant/nested model. |
| `export_extras` | List of extra DA3 export formats: any of `npz`, `depth_vis`, `feat_vis`, `colmap`. |
| `pointcloud_sample_ratio` / `pointcloud_conf_coef` | Streaming-only. Per-frame pixel fraction and confidence cutoff (`mean(conf) * coef`) for the merged `combined_pcd.ply`. The big knobs for streaming PLY density. |
| `irls_*` | Sim3 fit between adjacent chunks (per-pair alignment). Bigger `irls_delta` = more outlier-tolerant. |
| `delete_temp_files: false` | Keep `_tmp_results_unaligned/` and `_tmp_results_aligned/` for chunk-alignment debugging. |
| `save_debug_info: true` | Bake Sim3 (s, R, T) into the per-frame `results_output/frame_*.npz`. |

### `camera_params.json` — required schema

For `use_known_intrinsics` or `use_known_extrinsics`, the dataset folder must
contain a `camera_params.json` like:

```json
{
  "cameras": {
    "front": {
      "K":  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "raw_position_mm":  [x_mm, y_mm, z_mm],
      "raw_rotation_deg": [rx, ry, rz]
    },
    "rear":  {...}, "left": {...}, "right": {...}
  }
}
```

Vehicle frame: X=fwd, Y=left, Z=up (rear-axle origin). Rotations are XYZ Euler
camera→vehicle in degrees.

### Frame-count math

`n_timestamps = duration_seconds × fps_out`
`n_frames     = n_timestamps × len(cameras)`

For single-shot, keep `n_frames` ≤ ~440 at `process_res=504` on an H100-80GB.
For streaming, no limit — pick `chunk_size = k × len(cameras)` to fit in memory.

## Run

```bash
# CLI
python -m da3_runner.cli run my_run.yaml
python -m da3_runner.cli run my_run.yaml -o duration_seconds=2 -o fps_out=1   # quick smoke
python -m da3_runner.cli run my_run.yaml --dry-run                            # print resolved config
python -m da3_runner.cli list-scenes                                          # auto-discover scenes
python -m da3_runner.cli print-default                                        # dump default YAML

# GUI (local-only)
python -m da3_runner.gui            # → http://localhost:7860

# GUI (cloud, public share link)
DA3_GUI_SHARE=1 python -m da3_runner.gui
```

## Outputs

By default each run writes to `<dataset_dir>/DA3_output/<run_name>/` (i.e.
co-located with `camera_params.json` and `pinhole/`). Set `output_root` in the
YAML to override. Layout:

```
<dataset_dir>/DA3_output/<run_name>/
├── manifest.json              # full RunConfig + git SHA + GPU + n_frames
├── camera_poses.txt           # one row per frame, 16 floats of 4x4 cam-to-world (BOTH modes)
├── intrinsic.txt              # one row per frame: fx fy cx cy (BOTH modes)
├── pointcloud.ply             # back-projected, full-detail (single-shot)
├── pointcloud_tsdf.ply        # if export_tsdf=true (single-shot) — TSDF-fused, smoothest
├── pointcloud_by_cam.ply      # if export_pointcloud_by_cam=true — per-cam tinted, diagnostic
├── scene.glb                  # if export_glb=true (single-shot) — DA3's built-in glTF
├── staged_images/             # interleaved input frames (front bumper black-filled, others symlinked)
├── gs_ply/, gs_video/         # if export_3dgs / export_3dgs_video
└── (streaming) pcd/combined_pcd.ply, pointcloud.ply (symlink), results_output/, streaming_config.yaml
```

Open `.ply` files in CloudCompare or MeshLab; `.glb` opens in any glTF viewer
(e.g. https://gltf-viewer.donmccurdy.com).

## Why three different `.ply` files?

The single-shot runner can emit up to three point clouds from the same DA3
inference, processed differently:

| File | How it's built | When to use |
|---|---|---|
| `pointcloud.ply` | Back-project every depth pixel (stride `backproj_downsample`, drop bottom `backproj_conf_percentile`% per view) into world space, concatenate all views. | Default. Most detail; some duplication where cameras overlap. |
| `pointcloud_tsdf.ply` | Open3D `ScalableTSDFVolume`: every per-view depth is integrated into a truncated signed distance field at `tsdf_voxel` resolution; surfaces from all 4 cameras agree, contradictions average out, free-space gets carved. | Cleanest, most surface-like. Best for visualization and downstream meshing. |
| `pointcloud_by_cam.ply` | Same back-projection, but every point is tinted by its source camera (red=front, green=rear, blue=left, yellow=right). | Diagnostic. Spot misalignment between cameras instantly. |

This is the same post-processing approach as the reference pipeline at
`~/4dgt/scripts/da3_multiview_pointcloud.py` — the call to `model.inference()`
is identical to ours; the quality difference vs. DA3's built-in `scene.glb` is
**all** in this post-proc step (TSDF fusion + per-view back-projection vs. global
percentile + 1M-point cap).

### Streaming-mode TSDF (optional)

Streaming has its own two TSDF outputs, both gated by config and both run after
the streaming pipeline finishes:

| File | Source | Notes |
|---|---|---|
| `pcd/{i}_pcd_tsdf.ply`, `combined_pcd_tsdf.ply` | `tsdf_per_chunk_streaming: true` | Per chunk: TSDF the chunk's frames in chunk-local coords, then apply that chunk's Sim3 to bring into the global frame. Cleaner *per-chunk* artifacts, **no cross-chunk averaging**. |
| `pointcloud_tsdf_global.ply` | `tsdf_global_streaming: true` | One TSDF volume that integrates **every** frame using the globally-aligned `camera_poses.txt` + per-chunk depth scaling. **Cross-chunk averaging** — usually the cleanest result. |

Both require `save_depth_conf_result: true` (default) so `results_output/frame_*.npz`
is on disk for the post-proc to integrate. They also depend on the patched
`da3_streaming.py` writing `chunk_metadata.npz` at the end of a run.

Caveat: the per-chunk TSDF only changes the *saved* per-chunk PLY artifacts.
Streaming's own Sim3 alignment between chunks runs on the dense per-pixel point
maps DA3 returns (`align_2pcds` at `da3_streaming.py:322`), not on the saved PLYs,
so per-chunk TSDF does **not** affect alignment quality — only what you can
inspect afterwards.

## Intrinsics A/B

Run twice with `use_known_intrinsics: false` then `true`, same everything else.
Compare `pointcloud.ply` side-by-side (or `manifest.json` and the printed
intrinsics in the streaming logs). The runner already wires the known-intrinsics
path through both modes.

## Notes

- **Bumper**: black-filled in place. The other three cameras are symlinked from
  the source pinhole frames (no copy). Disk usage stays small.
- **Interleaving**: filenames are `t000000_front.jpg`, `t000000_right.jpg`, …
  Streaming sorts alphabetically (`da3_streaming.py:702`), so per-timestamp
  blocks of `len(cameras)` are kept together — chunks never split a timestamp.
- **gsplat**: if missing, runs requesting `export_3dgs` raise a clear error with
  the install command. Non-3DGS runs work without it.
