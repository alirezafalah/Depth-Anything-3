"""Stage 4-camera inputs into a single interleaved directory for DA3.

For each timestamp `t`, we emit `t{idx:06d}_{cam}.jpg` so alphabetical sort
yields per-timestamp blocks of size len(cameras). This guarantees a streaming
chunk of size `chunk_size` (a multiple of len(cameras)) holds equal counts of
every camera, and a chunk boundary never lands inside a single timestamp's
camera block.

Bumper black-fill: front frames get rows [bumper_rows[0]:bumper_rows[1]] zeroed
and saved as JPG (can't symlink because we mutate). All other camera frames are
symlinked to save disk and IO.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .config import RunConfig

EXTS = (".jpg", ".jpeg", ".png")


def _list_frames(cam_dir: Path) -> list[Path]:
    files = []
    for ext in EXTS:
        files.extend(sorted(cam_dir.glob(f"*{ext}")))
    return sorted(files)


def _select_indices(n_total: int, cfg: RunConfig) -> list[int]:
    stride = cfg.stride()
    start = int(round(cfg.start_seconds * cfg.fps_in))
    if cfg.duration_seconds is None:
        end = n_total
    else:
        n_wanted = int(round(cfg.duration_seconds * cfg.fps_out))
        end = start + n_wanted * stride
    end = min(end, n_total)
    return list(range(start, end, stride))


def _apply_bumper(img_path: Path, dest: Path, bumper_rows: tuple[int, int]) -> None:
    img = np.array(Image.open(img_path).convert("RGB"))
    r0, r1 = bumper_rows
    r0 = max(0, min(r0, img.shape[0]))
    r1 = max(0, min(r1, img.shape[0]))
    img[r0:r1, :, :] = 0
    Image.fromarray(img).save(dest, quality=95)


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_inputs(cfg: RunConfig) -> tuple[Path, list[str], list[Path]]:
    """Materialise the staged input directory.

    Returns:
        staged_dir, interleaved_cams, interleaved_paths

    `interleaved_cams[i]` is the camera name for staged frame i (parallel to
    `interleaved_paths[i]`). Used downstream by intrinsics.py.
    """
    cfg.validate()
    ds = Path(cfg.dataset_dir).expanduser()

    # Collect per-camera frame lists.
    per_cam_all: dict[str, list[Path]] = {}
    for cam in cfg.cameras:
        per_cam_all[cam] = _list_frames(ds / "pinhole" / cam)
        if not per_cam_all[cam]:
            raise FileNotFoundError(f"no frames in {ds / 'pinhole' / cam}")

    n_total = min(len(v) for v in per_cam_all.values())
    indices = _select_indices(n_total, cfg)
    if not indices:
        raise ValueError("no frames selected; check fps/duration/start_seconds")

    staged = cfg.staged_dir
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)

    interleaved_cams: list[str] = []
    interleaved_paths: list[Path] = []
    for t_local, src_idx in enumerate(indices):
        for cam in cfg.cameras:
            src = per_cam_all[cam][src_idx]
            dest = staged / f"t{t_local:06d}_{cam}.jpg"
            if cam == "front" and cfg.bumper_blackfill:
                _apply_bumper(src, dest, cfg.bumper_rows)
            else:
                _link_or_copy(src.resolve(), dest)
            interleaved_cams.append(cam)
            interleaved_paths.append(dest)

    print(
        f"[stage_inputs] {len(interleaved_paths)} frames "
        f"({len(indices)} timestamps × {len(cfg.cameras)} cams) -> {staged}"
    )
    return staged, interleaved_cams, interleaved_paths
