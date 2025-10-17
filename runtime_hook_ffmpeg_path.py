# runtime_hook_ffmpeg_path.py — robust PATH for bundled ffmpeg
import os, sys
from pathlib import Path

try:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        # Accept both layouts:
        #   dist/LumiTranscriber/ffmpeg/ffmpeg.exe
        #   dist/LumiTranscriber/ffmpeg/bin/ffmpeg.exe
        candidates = [exe_dir / "ffmpeg", exe_dir / "ffmpeg" / "bin"]
        existing = [p for p in candidates if p.exists()]
        if existing:
            os.environ["PATH"] = os.pathsep.join([*(str(p) for p in existing), os.environ.get("PATH","")])
        # Keep models cache next to exe
        os.environ.setdefault("HF_HOME", str(exe_dir / "models_cache"))
except Exception:
    pass
