"""Typer CLI for da3_runner.

Examples:
    python -m da3_runner.cli run da3_runner/configs/static_singleshot.yaml
    python -m da3_runner.cli run cfg.yaml --override duration_seconds=2 fps_out=1
    python -m da3_runner.cli list-scenes
    python -m da3_runner.cli print-default
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

from .config import DEFAULT_INPUT_ROOT, RunConfig

app = typer.Typer(add_completion=False)


def _coerce(val: str):
    """Parse 'true'/'false'/'None'/numbers/lists from CLI override strings."""
    s = val.strip()
    if s.lower() in ("none", "null", "~"):
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    if s.startswith("[") and s.endswith("]"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    for o in overrides or []:
        if "=" not in o:
            raise typer.BadParameter(f"override must be key=value, got {o!r}")
        k, v = o.split("=", 1)
        cfg_dict[k.strip()] = _coerce(v)
    return cfg_dict


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, readable=True),
    override: list[str] = typer.Option(
        None, "--override", "-o", help="key=value overrides; repeatable"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print resolved config and exit"),
):
    """Execute a run from a YAML config (with optional CLI overrides)."""
    with config_path.open() as f:
        d = yaml.safe_load(f) or {}
    d = _apply_overrides(d, override or [])
    cfg = RunConfig.from_dict(d)
    cfg.validate()

    print("=" * 60)
    print(f"run_name : {cfg.run_name}")
    print(f"mode     : {cfg.mode}")
    print(f"dataset  : {cfg.dataset_dir}")
    print(f"cameras  : {cfg.cameras}")
    print(f"fps_out  : {cfg.fps_out}  duration={cfg.duration_seconds}")
    print(f"intrins. : {'KNOWN' if cfg.use_known_intrinsics else 'estimated'}")
    print(f"3DGS     : {cfg.export_3dgs} (video={cfg.export_3dgs_video})")
    print(f"output   : {cfg.run_dir}")
    print("=" * 60)
    if dry_run:
        return

    if cfg.mode == "singleshot":
        from .runner_singleshot import run_singleshot

        run_singleshot(cfg)
    elif cfg.mode == "streaming":
        from .runner_streaming import run_streaming

        run_streaming(cfg)
    else:
        raise typer.BadParameter(f"unknown mode {cfg.mode!r}")


@app.command("list-scenes")
def list_scenes(
    root: Path = typer.Option(
        Path("~/data/4DGT").expanduser(), "--root", help="dataset root directory"
    ),
):
    """List available ORIGINAL_* scene directories."""
    if not root.exists():
        typer.echo(f"root not found: {root}", err=True)
        raise typer.Exit(1)
    for p in sorted(root.glob("ORIGINAL_*")):
        if p.is_dir():
            typer.echo(str(p))


@app.command("print-default")
def print_default():
    """Print a default RunConfig as YAML."""
    cfg = RunConfig(dataset_dir="<set me>")
    cfg.run_name = "<auto>"
    yaml.safe_dump(cfg.to_dict(), sys.stdout, sort_keys=False)


if __name__ == "__main__":
    app()
