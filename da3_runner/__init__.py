"""da3_runner: orchestrator + GUI for DA3 multi-camera experiments."""

import sys as _sys
from pathlib import Path as _Path

# Ensure the upstream `depth_anything_3` package (not pip-installed in this fork)
# resolves from the in-repo src/ directory.
_SRC = _Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
