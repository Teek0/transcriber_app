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

# --- FFMPEG discovery/forcing (plug-and-play) ---
FFMPEG_EXE = None  # type: Optional[Path]
FFPROBE_EXE = None  # type: Optional[Path]

def _candidate_ffmpeg_dirs() -> list[Path]:
    """Posibles ubicaciones de ffmpeg dentro del build o proyecto."""
    here = Path(__file__).parent
    candidates = [here / "ffmpeg" / "bin", here / "ffmpeg"]
    # Si está empaquetado, usar el directorio del exe
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        candidates = [exe_dir / "ffmpeg" / "bin", exe_dir / "ffmpeg"] + candidates
    # Quitar duplicados preservando orden
    seen = set()
    out = []
    for p in candidates:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out

def _find_ffmpeg_in(folder: Path) -> tuple[Optional[Path], Optional[Path]]:
    ff = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    fp = folder / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
    return (ff if ff.exists() else None, fp if fp.exists() else None)

def ensure_ffmpeg_available(interactive: bool = True) -> None:
    """Fuerza la disponibilidad de ffmpeg/ffprobe:
    - Busca en rutas candidatas del build.
    - Si no los halla y 'interactive', pide carpeta al usuario.
    - Prepend la carpeta encontrada al PATH del proceso.
    - Setea FFMPEG_EXE/FFPROBE_EXE con rutas absolutas.
    """
    global FFMPEG_EXE, FFPROBE_EXE

    # 1) Buscar en candidatas
    for d in _candidate_ffmpeg_dirs():
        ff, fp = _find_ffmpeg_in(d)
        if ff and fp:
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            FFMPEG_EXE, FFPROBE_EXE = ff, fp
            return

    # 2) Si no, pedir carpeta (solo GUI)
    if interactive:
        try:
            from tkinter import filedialog, messagebox
            messagebox.showinfo(
                "FFmpeg requerido",
                "No encontré ffmpeg. Selecciona la carpeta que contiene ffmpeg.exe y ffprobe.exe."
            )
            chosen = filedialog.askdirectory(title="Selecciona la carpeta de ffmpeg")
            if chosen:
                d = Path(chosen)
                ff, fp = _find_ffmpeg_in(d)
                # También probar subcarpeta bin si eligieron la raíz
                if not (ff and fp) and (d / "bin").exists():
                    ff, fp = _find_ffmpeg_in(d / "bin")
                    if ff and fp:
                        d = d / "bin"
                if ff and fp:
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                    FFMPEG_EXE, FFPROBE_EXE = ff, fp
                    return
                else:
                    messagebox.showerror(
                        "FFmpeg no válido",
                        "La carpeta seleccionada no contiene ffmpeg.exe y ffprobe.exe."
                    )
        except Exception:
            pass

    # 3) Si seguimos sin encontrarlos, dejar variables en None (check_ffmpeg lo reportará)

# ==========================
#   Constantes y mapas
# ==========================
VALID_EXTENSIONS = {".mkv", ".mp4", ".mp3", ".wav", ".flac", ".webm", ".m4a"}
DEFAULT_MODEL = "small"  # tiny, base, small, medium, large-v3 (si disponible)
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
    """Comprueba presencia de ffmpeg/ffprobe. Usa rutas absolutas si las tenemos."""
    # 1) Priorizar rutas encontradas por ensure_ffmpeg_available()
    if FFMPEG_EXE and FFPROBE_EXE:
        try:
            out = subprocess.check_output([str(FFMPEG_EXE), "-version"], stderr=subprocess.STDOUT)
            first = out.decode(errors="ignore").splitlines()[0]
            return True, first
        except Exception as e:
            return False, f"No pude ejecutar ffmpeg embebido: {e}"

    # 2) Fallback: PATH del sistema/proceso
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return False, "FFmpeg/ffprobe no encontrados."
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return True, out.decode(errors="ignore").splitlines()[0]
    except Exception as e:
        return False, f"No pude ejecutar ffmpeg: {e}"


def get_duration_seconds(path: Path) -> Optional[float]:
    """Obtiene duración con ffprobe. Usa ruta absoluta si está disponible."""
    try:
        ffprobe_cmd = [str(FFPROBE_EXE)] if FFPROBE_EXE else ["ffprobe"]
        cmd = ffprobe_cmd + [
            "-v", "error",
            "-print_format", "json",
            "-show_entries", "format=duration",
            str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        return float(data["format"]["duration"])
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

        # === Ícono personalizado ===
        def _resource_path(*parts):
            if getattr(sys, "frozen", False):
                base = Path(sys.executable).parent
            else:
                base = Path(__file__).parent
            return base.joinpath(*parts)

        icon_path = _resource_path("assets", "lumi.ico")
        try:
            self.iconbitmap(str(icon_path))
        except Exception as e:
            print(f"No se pudo cargar el ícono: {e}")
        # ============================

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()  # reservado por si se añade cancelación real

        self._build_widgets()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        ensure_ffmpeg_available(interactive=True)

        ok, msg = check_ffmpeg()
        self._log(f"✅ FFmpeg detectado: {msg}" if ok else f"⚠️ {msg}")

        self.after(100, self._drain_log)
        self._update_start_enabled()

    def _on_close(self) -> None:
        """Cierra la app garantizando que los hilos se detengan."""
        self.stop_flag.set()
        if self.worker and self.worker.is_alive():
            # Intentar una espera breve para que el hilo termine limpio.
            self.worker.join(timeout=1)
        self.quit()
        self.destroy()

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
        self.cmb_model = ttk.Combobox(opts, state="readonly", values=["tiny", "base", "small", "medium", "large-v3"])
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

        self.btn_download_model = ttk.Button(
            opts,
            text="Descargar modelo seleccionado…",
            command=self._download_model,
        )
        self.btn_download_model.grid(row=2, column=0, sticky="w", pady=(10, 0))

        self.btn_manual_install = ttk.Button(
            opts,
            text="Instalar modelo manualmente…",
            command=self._manual_install_model,
        )
        self.btn_manual_install.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(10, 0))


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

    def _set_auxiliary_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        if hasattr(self, "btn_download_model"):
            self.btn_download_model.configure(state=state)
        if hasattr(self, "btn_manual_install"):
            self.btn_manual_install.configure(state=state)

    def _download_model(self) -> None:
        if self.worker:
            messagebox.showinfo(
                "Transcripción en curso",
                "Espera a que finalice la transcripción antes de descargar modelos.",
            )
            return

        model_name = self.cmb_model.get().strip()
        if not model_name:
            messagebox.showerror("Error", "Selecciona un modelo antes de descargarlo.")
            return

        repo_id = f"Systran/faster-whisper-{model_name}"
        if not messagebox.askyesno(
            "Descargar modelo",
            f"¿Deseas descargar el modelo {repo_id}?",
            icon=messagebox.QUESTION,
        ):
            return

        try:
            from huggingface_hub import snapshot_download
            try:
                # la excepción vive en utils (no en el paquete raíz)
                from huggingface_hub.utils import HfHubHTTPError
            except Exception:
                # fallback robusto si cambia el nombre/ruta en versiones futuras
                class HfHubHTTPError(Exception):
                    ...
        except Exception as e:
            messagebox.showerror(
                "Dependencia faltante",
                "No se pudo importar huggingface_hub. Instálalo con:\n"
                "    pip install --upgrade huggingface_hub\n"
                f"Detalle: {e}",
            )
            self._log(f"❌ No se pudo importar huggingface_hub: {e}")
            return


        cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf_lumi"))

        def worker() -> None:
            self._log(f"⬇️ Descargando modelo {repo_id}…")
            self.after(0, lambda: self._set_auxiliary_buttons_enabled(False))
            self.after(0, lambda: self._progress_busy(f"Descargando {model_name}…"))
            try:
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(cache_dir),
                    resume_download=True,
                )
            except HfHubHTTPError as e:  # type: ignore[misc]
                self._log(f"❌ Error de Hugging Face: {e}")
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Descarga fallida",
                        "No se pudo descargar el modelo desde Hugging Face.\n"
                        f"Detalle: {e}",
                    ),
                )
            except Exception as e:
                self._log(f"❌ Descarga interrumpida: {e}")
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Descarga fallida",
                        "Ocurrió un error inesperado al descargar el modelo.\n"
                        f"Detalle: {e}",
                    ),
                )
            else:
                self._log(f"✅ Modelo descargado correctamente: {repo_id}")
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Descarga completada",
                        "El modelo se descargó correctamente y quedará disponible para futuras transcripciones.",
                    ),
                )
            finally:
                self.after(
                    0,
                    lambda: self._progress_reset() if not self.worker else None,
                )
                self.after(0, lambda: self._set_auxiliary_buttons_enabled(True))

        threading.Thread(target=worker, daemon=True).start()

    def _manual_install_model(self) -> None:
        """Permite copiar una carpeta de modelo descargada manualmente al caché local."""
        try:
            messagebox.showinfo(
                "Instalación manual",
                "Selecciona la carpeta del modelo descargado (models--Systran--faster-whisper-<nombre>).\n"
                "Puedes copiarla desde otro equipo donde ya se haya bajado.",
            )
        except Exception:
            # En caso de ejecutarse sin entorno gráfico, continuar sin mostrar aviso previo.
            pass

        chosen = filedialog.askdirectory(title="Selecciona la carpeta del modelo")
        if not chosen:
            return

        source_dir = Path(chosen)
        if not source_dir.exists() or not source_dir.is_dir():
            messagebox.showerror("Error", "La carpeta seleccionada no es válida.")
            return

        folder_name = source_dir.name
        prefix = "models--Systran--faster-whisper-"
        if not folder_name.startswith(prefix):
            messagebox.showerror(
                "Carpeta inesperada",
                "La carpeta debe llamarse como models--Systran--faster-whisper-<modelo>.",
            )
            return

        has_weights = any(source_dir.rglob("model.bin")) or any(source_dir.rglob("ggml-model.bin"))
        if not has_weights:
            messagebox.showerror(
                "Modelo incompleto",
                "No se encontraron archivos del modelo (model.bin). Verifica que la descarga esté completa.",
            )
            return

        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf_lumi"))
        target_root = cache_root / "hub"
        target_root.mkdir(parents=True, exist_ok=True)

        target_dir = target_root / folder_name
        try:
            if target_dir.exists():
                if not messagebox.askyesno(
                    "Reemplazar modelo",
                    "Ya existe un modelo con ese nombre. ¿Deseas reemplazarlo?",
                    default=messagebox.NO,
                    icon=messagebox.WARNING,
                ):
                    return
                shutil.rmtree(target_dir)

            shutil.copytree(source_dir, target_dir)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo copiar el modelo:\n{e}")
            self._log(f"❌ Falló la copia manual del modelo: {e}")
            return

        self._log(f"📦 Modelo manual instalado: {folder_name}")
        messagebox.showinfo(
            "Instalación completada",
            "El modelo se copió correctamente. Ya puedes usarlo desde la lista de modelos.",
        )

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
