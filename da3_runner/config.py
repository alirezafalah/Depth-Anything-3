"""RunConfig: single source of truth for a DA3 experiment run.

Round-trips through YAML. Used by CLI, GUI, and both runners.
"""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

CAMERAS_ALL = ("front", "right", "left", "rear")
DEFAULT_OUTPUT_ROOT = Path("~/data/4DGT/da3_streaming_outputs").expanduser()
DEFAULT_INPUT_ROOT = Path("~/data/4DGT/da3_streaming_inputs").expanduser()


@dataclass
class RunConfig:
    # --- dataset selection ---
    dataset_dir: str = ""
    cameras: list[str] = field(default_factory=lambda: list(CAMERAS_ALL))
    fps_in: float = 30.0
    fps_out: float = 30.0
    duration_seconds: Optional[float] = None
    start_seconds: float = 0.0

    # --- preprocessing ---
    bumper_blackfill: bool = True
    bumper_rows: tuple[int, int] = (1043, 1160)
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"

    # --- mode + model ---
    mode: Literal["singleshot", "streaming"] = "singleshot"
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    streaming_weights_dir: str = "./weights"  # relative to da3_streaming/

    # --- streaming-only ---
    chunk_size: int = 120
    overlap: int = 60
    align_method: Literal["sim3", "se3", "scale+se3"] = "sim3"
    align_lib: Literal["triton", "torch", "numba", "numpy"] = "torch"
    # only used when align_method == "scale+se3"
    scale_compute_method: Literal["auto", "ransac", "weighted"] = "auto"
    ref_view_strategy: Literal[
        "saddle_balanced", "middle", "first", "saddle_sim_range"
    ] = "saddle_balanced"
    ref_view_strategy_loop: Literal[
        "saddle_balanced", "middle", "first", "saddle_sim_range"
    ] = "saddle_balanced"
    loop_enable: bool = False

    # IRLS (Iteratively Reweighted Least Squares) for the per-pair Sim3 fit.
    # Bigger delta = more tolerant to outliers; more iters = slower but tighter fit.
    irls_delta: float = 0.1
    irls_max_iters: int = 5
    irls_tol: float = 1e-9

    # Streaming pointcloud merge (combined_pcd.ply).
    pointcloud_sample_ratio: float = 0.015     # fraction of pixels kept per frame
    pointcloud_conf_coef: float = 0.75         # conf cutoff = mean(conf) * coef

    # Debug.
    delete_temp_files: bool = True             # if False, keep _tmp_results_* for alignment debugging
    save_debug_info: bool = False              # add Sim3 (s, R, T) to per-frame npz files
    save_depth_conf_result: bool = True        # write per-frame depth/conf/intrinsic npz

    # --- intrinsics + extrinsics A/B ---
    # Both come from <dataset_dir>/camera_params.json. Format:
    #   { "cameras": { "<cam>": { "K": 3x3,
    #                              "raw_position_mm": [x,y,z],
    #                              "raw_rotation_deg": [rx,ry,rz] }, ... } }
    use_known_intrinsics: bool = False
    use_known_extrinsics: bool = False         # WARNING: these are static cam-to-vehicle.
    # Replicating them per timestamp tells DA3 the ego car never moves — correct for
    # the "static" parking scene, WRONG for any dynamic/driving scene.
    extrinsics_source: str = "camera_params.json"
    intrinsics_source: str = "camera_params.json"

    # --- single-shot extras ---
    use_ray_pose: bool = False                  # ray-based pose head vs camera decoder

    # --- exports ---
    export_pointcloud: bool = True             # back-projected pointcloud.ply (BOTH modes)
    export_glb: bool = False                   # DA3's built-in glTF (single-shot only; same model output as ply)
    export_3dgs: bool = False
    export_3dgs_video: bool = False
    # Extra DA3 export formats. Allowed: "npz", "depth_vis", "feat_vis", "colmap".
    # "mini_npz" + "glb" + (optional) "gs_ply"/"gs_video" are added automatically.
    export_extras: list[str] = field(default_factory=list)
    show_cameras: bool = True
    conf_thresh_percentile: float = 40.0       # GLB only
    num_max_points: int = 1_000_000            # GLB only
    feat_vis_fps: int = 15                     # only used if "feat_vis" in export_extras

    # --- back-projection (single-shot) — produces pointcloud.ply ---
    backproj_downsample: int = 2               # pixel stride for back-projection (1=full, 2=quarter, ...)
    backproj_conf_percentile: float = 30.0     # drop bottom-X% confidence per view (0=keep all)

    # --- TSDF fusion (single-shot) — produces pointcloud_tsdf.ply ---
    export_tsdf: bool = False
    tsdf_voxel: float = 0.05                   # metres; smaller = more detail, more RAM
    tsdf_trunc: float = 0.20                   # metres; surface bandwidth (~ 4×voxel)

    # --- diagnostics ---
    export_pointcloud_by_cam: bool = False     # tinted PLY (red=front, green=rear, blue=left, yellow=right)
    final_voxel: float = 0.0                   # optional voxel downsample of merged ply (0=off)

    # --- bookkeeping ---
    run_name: str = ""
    # Where to write outputs and stage inputs. Empty string = put both inside the
    # dataset_dir itself (preferred): outputs go to <dataset_dir>/DA3_output/<run_name>/
    # and staged frames go to <dataset_dir>/DA3_output/<run_name>/staged_images/.
    # Set to a path to override (e.g. when /ws/shared is read-only).
    output_root: str = ""
    input_root: str = ""

    def __post_init__(self) -> None:
        if not self.run_name:
            self.run_name = self._auto_run_name()

    def _auto_run_name(self) -> str:
        scene = Path(self.dataset_dir).name or "scene"
        # strip the long prefix
        scene_short = scene.replace("ORIGINAL_", "").split("_camera-")[0]
        cams_short = "".join(c[0] for c in self.cameras)  # 'frlb' style
        dur = "full" if self.duration_seconds is None else f"{self.duration_seconds:g}s"
        priors = "".join(
            t for t, on in [("i", self.use_known_intrinsics), ("e", self.use_known_extrinsics)] if on
        ) or "no"
        if self.mode == "streaming":
            tail = f"c{self.chunk_size}o{self.overlap}"
        else:
            tail = "ss"
        return (
            f"{scene_short}_{self.mode[:2]}_{cams_short}"
            f"_{self.fps_out:g}fps_{dur}_r{self.process_res}_{priors}_{tail}"
        )

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if not self.dataset_dir:
            raise ValueError("dataset_dir is required")
        ds = Path(self.dataset_dir).expanduser()
        if not ds.exists():
            raise FileNotFoundError(f"dataset_dir does not exist: {ds}")
        for cam in self.cameras:
            if cam not in CAMERAS_ALL:
                raise ValueError(f"unknown camera {cam!r}; must be one of {CAMERAS_ALL}")
            if not (ds / "pinhole" / cam).is_dir():
                raise FileNotFoundError(f"missing camera dir: {ds / 'pinhole' / cam}")
        if self.mode == "streaming":
            n = len(self.cameras)
            if self.chunk_size % n != 0:
                raise ValueError(
                    f"chunk_size ({self.chunk_size}) must be a multiple of "
                    f"len(cameras) ({n}) so each chunk holds equal counts of every camera"
                )
            if self.overlap % n != 0:
                raise ValueError(
                    f"overlap ({self.overlap}) must be a multiple of len(cameras) ({n})"
                )
            if self.overlap >= self.chunk_size:
                raise ValueError("overlap must be < chunk_size")
        if self.use_known_extrinsics and self.mode == "streaming":
            print(
                "[RunConfig] note: streaming mode does not currently consume extrinsics; "
                "the toggle has no effect in this mode."
            )
        allowed_extras = {"npz", "depth_vis", "feat_vis", "colmap"}
        bad = [e for e in self.export_extras if e not in allowed_extras]
        if bad:
            raise ValueError(f"unknown export_extras: {bad}; allowed: {sorted(allowed_extras)}")

    # ------------------------------------------------------------------ #
    # YAML I/O
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["bumper_rows"] = list(self.bumper_rows)
        return d

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        if "bumper_rows" in d and isinstance(d["bumper_rows"], list):
            d["bumper_rows"] = tuple(d["bumper_rows"])
        # Drop unknown keys with a warning (forward-compat).
        known = {f.name for f in dc.fields(cls)}
        unknown = set(d) - known
        for k in unknown:
            print(f"[RunConfig] warning: unknown config key {k!r}, ignoring")
            d.pop(k)
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with Path(path).expanduser().open() as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # ------------------------------------------------------------------ #
    # derived paths
    # ------------------------------------------------------------------ #
    @property
    def run_dir(self) -> Path:
        """Where outputs go. Defaults to <dataset_dir>/DA3_output/<run_name>/."""
        if self.output_root:
            return Path(self.output_root).expanduser() / self.run_name
        return Path(self.dataset_dir).expanduser() / "DA3_output" / self.run_name

    @property
    def staged_dir(self) -> Path:
        """Where staged frames go. Defaults to run_dir/staged_images/."""
        if self.input_root:
            return Path(self.input_root).expanduser() / self.run_name
        return self.run_dir / "staged_images"

    def stride(self) -> int:
        s = max(1, round(self.fps_in / max(self.fps_out, 1e-6)))
        return s
