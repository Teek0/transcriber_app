"""
Lumi Transcriber GUI — Tkinter + faster-whisper (refactor legible)

Objetivo de este refactor:
  • Separar responsabilidades (utilidades, modelo, transcripción, GUI).
  • Nombres consistentes y tipos explícitos.
  • Docstrings y comentarios breves.
  • Corregir detalle: uso coherente del parámetro `language` al llamar a `model.transcribe`.
  • Mantener comportamiento existente: progreso por tiempo y guardado .txt.

Requisitos:
  - Python 3.9+
  - pip install faster-whisper
  - (opcional) pip install torch
  - FFmpeg en PATH (ffmpeg/ffprobe)
"""
from __future__ import annotations

import os
import sys
import json
import queue
import shutil
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ==========================
#   Configuración global
# ==========================
# Evitar symlinks (Windows) y fijar caché de HF
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf_lumi"))

# Si es binario PyInstaller, usar caché junto al ejecutable
if getattr(sys, "frozen", False):
    exe_dir = Path(sys.executable).parent
    os.environ.setdefault("HF_HOME", str(exe_dir / "models_cache"))

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # type: ignore

# ==========================
#   Constantes y mapas
# ==========================
VALID_EXTENSIONS = {".mkv", ".mp4", ".mp3", ".wav", ".flac", ".webm", ".m4a"}
DEFAULT_MODEL = "medium"  # tiny, base, small, medium, large-v2 (si disponible)
LANG_MAP = {"Spanish": "es", "English": "en", "Portuguese": "pt", "French": "fr", "Italian": "it"}
LANGUAGES = ["auto"] + list(LANG_MAP.keys())

# ==========================
#   Utilidades de sistema
# ==========================

def which(program: str) -> Optional[str]:
    """Versión portable de `which`.
    Devuelve ruta ejecutable o None.
    """
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";") if os.name == "nt" else [""]
    for p in paths:
        full = Path(p) / program
        for ext in exts:
            candidate = Path(str(full) + ext)
            if candidate.exists():
                return str(candidate)
    return None


def check_ffmpeg() -> tuple[bool, str]:
    """Comprueba presencia de ffmpeg y ffprobe en PATH."""
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return False, "FFmpeg/ffprobe no encontrados en PATH."
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return True, out.decode(errors="ignore").splitlines()[0]
    except Exception as e:
        return False, f"No pude ejecutar ffmpeg: {e}"


def get_duration_seconds(path: Path) -> Optional[float]:
    """Obtiene duración (segundos) usando ffprobe. None si falla."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration",
            str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        return float(data["format"]["duration"])  # type: ignore[index]
    except Exception:
        return None


def write_txt(out_path: Path, text: str) -> None:
    out_path.write_text(text, encoding="utf-8")

# ==========================
#   Modelo y transcripción
# ==========================

@dataclass
class ModelSpec:
    name: str = DEFAULT_MODEL
    compute_type: str = "float32"  # puedes cambiar a int8/float16 según HW


class Transcriber:
    """Facade simple sobre faster-whisper para un único proceso de transcripción."""

    def __init__(self, spec: ModelSpec, log_q: "queue.Queue[str]") -> None:
        self.spec = spec
        self.log_q = log_q
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        if WhisperModel is None:
            raise RuntimeError("'faster-whisper' no está instalado. Ejecuta: pip install faster-whisper")

        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf_lumi"))
        model_cache = cache_root / "hub" / f"models--Systran--faster-whisper-{self.spec.name}"

        self._log(f"Cargando modelo: {self.spec.name}… (primera vez puede tardar)")
        for attempt in range(2):
            try:
                m = WhisperModel(self.spec.name, compute_type=self.spec.compute_type)
                self._log("✅ Modelo cargado correctamente.")
                return m
            except Exception as e:
                msg = str(e)
                if "WinError 1314" in msg and attempt == 0:
                    self._log("⚠️ Permisos insuficientes (WinError 1314). Limpiando caché y reintentando…")
                    try:
                        if model_cache.exists():
                            shutil.rmtree(model_cache)
                            self._log(f"🧹 Caché borrado: {model_cache}")
                    except Exception as ex:
                        self._log(f"❌ No pude borrar el caché: {ex}")
                    continue
                raise

    # ---- API pública ----
    def transcribe(
        self,
        media_path: Path,
        language_ui: str,
        on_progress: Callable[[float], None],
    ) -> None:
        """Transcribe `media_path` y guarda .txt al lado.
        `language_ui` puede ser 'auto' o una etiqueta UI (p.ej. 'Spanish').
        `on_progress` recibe 0..100.
        """
        self._log(f"🎧 Transcribiendo: {media_path.name}")
        dur = get_duration_seconds(media_path)
        if dur:
            self._log(f"🎬 Duración estimada: ~{int(round(dur/60))} min ({int(dur)} s)")
        else:
            self._log("⚠️ No pude estimar duración con ffprobe.")

        # Normalizar idioma para la API
        lang: Optional[str]
        if language_ui == "auto":
            lang = None
        else:
            lang = LANG_MAP.get(language_ui) or language_ui  # acepta código ya normalizado

        # Importante: usar `lang` para la llamada a faster-whisper
        segments, _info = self.model.transcribe(str(media_path), language=lang)

        parts: list[str] = []
        last_end = 0.0
        for seg in segments:
            text_piece = (seg.text or "").strip()
            if text_piece:
                parts.append(text_piece)
            # progreso por tiempo
            end = float(getattr(seg, "end", 0.0) or last_end)
            last_end = end
            if dur and dur > 0:
                pct = max(0.0, min(100.0, end / dur * 100.0))
                on_progress(pct)

        text = " ".join(parts).strip()

        try:
            detected_lang = (_info.language if language_ui == "auto" else None)
        except Exception:
            detected_lang = None

        if language_ui == "auto":
            lang_code = (detected_lang or "auto")
        else:
            lang_code = LANG_MAP.get(language_ui) or language_ui

        def _safe(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in s).strip()

        stem = _safe(media_path.stem)
        lang_code = _safe(lang_code.lower())
        model_name = _safe(self.spec.name.lower())

        out_name = f"{stem}_{lang_code}_{model_name}.txt"
        out_txt = media_path.parent / out_name

        write_txt(out_txt, text)
        self._log(f"📄 Guardado TXT: {out_txt.name}")
        self._log("✅ Transcripción completada.")

    # ---- util interno ----
    def _log(self, msg: str) -> None:
        self.log_q.put(msg)

# ==========================
#   GUI (Tkinter)
# ==========================

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lumi Transcriber")
        self.geometry("760x600")
        self.minsize(720, 560)

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()  # reservado por si se añade cancelación real

        self._build_widgets()
        self.after(100, self._drain_log)

        ok, msg = check_ffmpeg()
        self._log(f"✅ FFmpeg detectado: {msg}" if ok else f"⚠️ {msg}")
        self._update_start_enabled()

    # ---------- Infra de log ----------
    def _log(self, text: str) -> None:
        self.log_q.put(text)

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                self.txt_log.configure(state="normal")
                self.txt_log.insert("end", line + "\n")
                self.txt_log.see("end")
                self.txt_log.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    # ---------- UI ----------
    def _build_widgets(self) -> None:
        pad = {"padx": 10, "pady": 8}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        self.mode = tk.StringVar(value="file")
        ttk.Radiobutton(top, text="Transcribir archivo", variable=self.mode, value="file", command=self._update_start_enabled).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(top, text="Transcribir carpeta", variable=self.mode, value="folder", command=self._update_start_enabled).grid(row=0, column=1, sticky="w")

        self.path_var = tk.StringVar()
        self.path_var.trace_add("write", lambda *_: self._update_start_enabled())
        ttk.Entry(top, textvariable=self.path_var).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(top, text="Elegir…", command=self._browse_path).grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(4, 0))
        top.columnconfigure(2, weight=1)
        
        ttk.Button(top, text="Abrir carpeta", command=self._open_folder).grid(row=1, column=4, sticky="e", padx=(6, 0), pady=(4, 0))


        opts = ttk.LabelFrame(self, text="Opciones")
        opts.pack(fill="x", **pad)
        ttk.Label(opts, text="Modelo").grid(row=0, column=0, sticky="w")
        self.cmb_model = ttk.Combobox(opts, state="readonly", values=["tiny", "base", "small", "medium", "large"])
        self.cmb_model.set(DEFAULT_MODEL)
        self.cmb_model.grid(row=0, column=1, sticky="w")

        ttk.Label(opts, text="Idioma").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.cmb_lang = ttk.Combobox(opts, state="readonly", values=LANGUAGES)
        self.cmb_lang.set("Spanish")
        self.cmb_lang.grid(row=0, column=3, sticky="w")

        ttk.Label(
            opts,
            text="* Si seleccionas un idioma distinto al original, se traducirá automáticamente al idioma elegido.",
            foreground="#555",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        self.btn_start = ttk.Button(actions, text="Iniciar", command=self.start, state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(actions, text="Detener", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        logf = ttk.LabelFrame(self, text="Registro")
        logf.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(logf, wrap="word", height=16, state="disabled")
        self.txt_log.pack(fill="both", expand=True)

        status = ttk.Frame(self)
        status.pack(fill="x", padx=10, pady=(0, 10))
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.progress.pack(fill="x", side="left", expand=True)
        self.lbl_status = ttk.Label(status, text="Listo")
        self.lbl_status.pack(side="left", padx=(10, 0))

    # ---------- Helpers de UI ----------
    def _browse_path(self) -> None:
        if self.mode.get() == "file":
            path = filedialog.askopenfilename(
                title="Selecciona un archivo de audio/video",
                filetypes=[("Medios", "*.mkv *.mp4 *.mp3 *.wav *.flac *.webm *.m4a")],
            )
        else:
            path = filedialog.askdirectory(title="Selecciona una carpeta")
        if path:
            self.path_var.set(path)

    def _valid_selection(self) -> bool:
        path = self.path_var.get().strip()
        if not path:
            return False
        p = Path(path)
        if self.mode.get() == "file":
            return p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        return p.is_dir()

    def _update_start_enabled(self) -> None:
        enabled = self._valid_selection() and not self.worker
        self.btn_start.configure(state="normal" if enabled else "disabled")

    def _set_running(self, running: bool) -> None:
        self.btn_start.configure(state="disabled" if running else ("normal" if self._valid_selection() else "disabled"))
        self.btn_stop.configure(state="normal" if running else "disabled")

    # ---------- Progreso visual ----------
    def _progress_reset(self) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.lbl_status.configure(text="Listo")

    def _progress_set(self, value: float, text: str) -> None:
        self.progress.configure(mode="determinate")
        self.progress.stop()
        self.progress["value"] = max(0, min(100, value))
        self.lbl_status.configure(text=text)

    def _progress_busy(self, text: str = "Cargando…") -> None:
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.lbl_status.configure(text=text)

    def _progress_to_determinate(self, text: str = "Procesando…") -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.lbl_status.configure(text=text)

    # ---------- Acciones ----------
    def start(self) -> None:
        if not self._valid_selection():
            messagebox.showwarning("Falta ruta", "Elige un archivo compatible o una carpeta válida.")
            return

        model_name = self.cmb_model.get()
        language_ui = self.cmb_lang.get()
        path = Path(self.path_var.get().strip())

        self.stop_flag.clear()
        self._set_running(True)

        def worker() -> None:
            try:
                self.after(0, lambda: self._progress_busy("Cargando modelo…"))
                transcriber = Transcriber(ModelSpec(name=model_name), self.log_q)

                _ = transcriber.model

                if path.is_file():
                    self._log("Iniciando transcripción…\n")
                    switched = {"done": False}
                    def onp(v: float) -> None:
                        if not switched["done"]:
                            switched["done"] = True
                            self.after(0, lambda: self._progress_to_determinate("Procesando…"))
                        self.after(0, lambda: self._progress_set(v, f"Procesando {v:.0f}%…"))

                    transcriber.transcribe(path, language_ui, onp)
                    self._progress_set(100, "Completado")
                else:
                    files = sorted([x for x in path.iterdir() if x.is_file() and x.suffix.lower() in VALID_EXTENSIONS])
                    self.after(0, lambda: self._progress_busy("Preparando…"))
                    total = len(files)
                    if not files:
                        self._log("No hay medios compatibles en la carpeta.\n")
                    for i, media in enumerate(files, start=1):
                        if self.stop_flag.is_set():
                            self._log("⏹️ Proceso detenido por el usuario.\n")
                            break

                        self._log(f"[{i}/{total}] -> {media.name}\n")
                        self.after(0, lambda i=i, total=total: self._progress_busy(f"Preparando {i}/{total}…"))

                        switched = {"done": False}
                        def onp(v: float, i=i, total=total) -> None:
                            if not switched["done"]:
                                switched["done"] = True
                                self.after(0, lambda i=i, total=total: self._progress_to_determinate(f"Procesando {i}/{total}…"))
                            self.after(0, lambda v=v, i=i, total=total: self._progress_set(v, f"Procesando {i}/{total} — {v:.0f}%"))

                        transcriber.transcribe(media, language_ui, onp)
                        self._progress_set(100, f"Procesando {i}/{total} — 100%")
                        if i < total:
                            self.after(0, lambda i=i + 1, total=total: self._progress_busy(f"Preparando {i}/{total}…"))

                    if total:
                        self._progress_set(100, "Completado")
                self._log("✨ Trabajo terminado.\n*-------------------------------*\n")
            except Exception as e:
                self._log(f"💥 Error: {e}")
            finally:
                self._set_running(False)
                self.worker = None
                self.after(500, self._progress_reset)
                self.after(0, self._update_start_enabled)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self._update_start_enabled()

    def stop(self) -> None:
        self.stop_flag.set()
        self._log("Solicitando detener… espera a que finalice el archivo en curso.")

    def _open_folder(self) -> None:
        """Abre la carpeta del archivo o la carpeta seleccionada."""
        path_str = self.path_var.get().strip()
        if not path_str:
            messagebox.showinfo("Sin selección", "Primero elige un archivo o una carpeta.")
            return

        p = Path(path_str)
        folder = p.parent if p.is_file() else p
        if not folder.exists():
            messagebox.showerror("Error", f"La carpeta no existe:\n{folder}")
            return

        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", folder])
            else:
                subprocess.run(["xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")


if __name__ == "__main__":
    App().mainloop()
