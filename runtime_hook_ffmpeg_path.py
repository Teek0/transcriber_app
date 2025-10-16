# runtime_hook_ffmpeg_path.py
# Ensures the bundled ffmpeg folder is on PATH at runtime,
# and sets an on-disk models cache next to the executable.

import os, sys
from pathlib import Path

try:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        ffdir = exe_dir / "ffmpeg"
        # Prepend bundled ffmpeg to PATH so `ffmpeg`/`ffprobe` resolve
        os.environ["PATH"] = str(ffdir) + os.pathsep + os.environ.get("PATH", "")
        # Keep models cache next to exe (fast, portable, works offline if models are present)
        os.environ.setdefault("HF_HOME", str(exe_dir / "models_cache"))
except Exception:
    # Non-fatal if anything goes sideways; the app itself will surface errors.
    pass
