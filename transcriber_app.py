"""
Lumi Transcriber GUI — Tkinter + Faster-Whisper (solo TXT, con progreso real)

Cambios clave respecto a la versión anterior:
  • Cambiado a faster-whisper para recibir segmentos en streaming.
  • Progreso real por archivo: la barra refleja % = (seg.end / duración) * 100.
  • Carpeta: la barra muestra el progreso del archivo en curso (y el texto de estado indica i/N).
  • Arreglo del log: garantiza salto de línea.

Requisitos (para ejecutar desde código):
  - Python 3.9+
  - pip install faster-whisper
  - (opcional) pip install torch  # si quieres usar GPU, faster-whisper puede aprovecharla
  - FFmpeg disponible en PATH (ffmpeg/ffprobe)

Build (ejecutable):
  - pip install pyinstaller
  - pyinstaller --name "LumiTranscriber" --onefile --noconsole main_gui.py

Autor: Lumi.
"""

import os, sys
from pathlib import Path

# --- Configuración de HuggingFace para evitar symlinks ---
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf_lumi"))

# Si está congelado por PyInstaller, redirige HF_HOME a una carpeta junto al .exe
if getattr(sys, "frozen", False):
    exe_dir = Path(sys.executable).parent
    os.environ.setdefault("HF_HOME", str(exe_dir / "models_cache"))
# ---------------------------------------------------------

import threading
import queue
import json
import subprocess
import shutil

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# -------------------- Dependencias --------------------
try:
    from faster_whisper import WhisperModel
except Exception as e:
    WhisperModel = None  # type: ignore

VALID_EXTENSIONS = {".mkv", ".mp4", ".mp3", ".wav", ".flac", ".webm", ".m4a"}
DEFAULT_MODEL = "medium"  # tiny, base, small, medium, large-v2 (si está disponible en tu instalación)
LANG_MAP = {"Spanish":"es","English":"en","Portuguese":"pt","French":"fr","German":"de","Italian":"it"}
LANGUAGES = ["auto"] + list(LANG_MAP.keys())

# -------------------- Utilidades --------------------

def which(program: str) -> str | None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    exts = [""]
    if os.name == "nt":
        exts = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD").split(";")
    for p in paths:
        full = Path(p) / program
        for ext in exts:
            candidate = Path(str(full) + ext)
            if candidate.exists():
                return str(candidate)
    return None


def check_ffmpeg() -> tuple[bool, str]:
    ffmpeg_path = which("ffmpeg")
    ffprobe_path = which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        return False, "FFmpeg/ffprobe no encontrados en PATH."
    try:
        out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT)
        return True, out.decode(errors="ignore").splitlines()[0]
    except Exception as e:
        return False, f"No pude ejecutar ffmpeg: {e}"


def get_duration_seconds(path: str) -> float | None:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_entries", "format=duration", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        return float(data["format"]["duration"])  # type: ignore[index]
    except Exception:
        return None


# -------------------- Escritor --------------------

def write_txt(out_path: Path, text: str) -> None:
    out_path.write_text(text, encoding="utf-8")


# -------------------- Lógica de transcripción (faster-whisper) --------------------

def load_model(name: str, log: queue.Queue):
    if WhisperModel is None:
        raise RuntimeError("'faster-whisper' no está instalado. Ejecuta: pip install faster-whisper")

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf_lumi"))
    model_cache = cache_root / "hub" / f"models--Systran--faster-whisper-{name}"
    log.put(f"Cargando modelo: {name}… (primera vez puede tardar)\n")

    for attempt in range(2):  # un intento de recuperación máximo
        try:
            model = WhisperModel(name, compute_type="float32")
            log.put("✅ Modelo cargado correctamente.\n")
            return model
        except Exception as e:
            msg = str(e)
            if "WinError 1314" in msg and attempt == 0:
                log.put("⚠️ Se detectó error de permisos (WinError 1314). Limpiando caché y reintentando…\n")
                try:
                    if model_cache.exists():
                        shutil.rmtree(model_cache)
                        log.put(f"🧹 Caché borrado: {model_cache}\n")
                except Exception as ex:
                    log.put(f"❌ No pude borrar el caché: {ex}\n")
                continue  # intentar cargar nuevamente
            else:
                raise



def transcribe_file(model, path: Path, language: str, on_progress, log: queue.Queue) -> None:
    """
    Transcribe un archivo y actualiza la barra con progreso real por tiempo.
    on_progress: callback que recibe un float [0..100].
    """
    log.put(f"🎧 Transcribiendo: {path.name}")
    dur = get_duration_seconds(str(path))
    if dur:
        log.put(f"🎬 Duración estimada: ~{int(round(dur/60))} min ({int(dur)} s)")
    else:
        log.put("⚠️ No pude estimar duración con ffprobe.")

    lang = None if language == "auto" else language

    # Generador de segmentos
    segments, _info = model.transcribe(str(path), language=language)

    parts: list[str] = []
    last_end = 0.0
    for seg in segments:
        # seg.text incluye espacio inicial a veces; lo normalizamos
        text_piece = (seg.text or "").strip()
        if text_piece:
            parts.append(text_piece)
        # progreso por tiempo
        end = float(seg.end or last_end)
        last_end = end
        if dur and dur > 0:
            pct = max(0.0, min(100.0, end / dur * 100.0))
            on_progress(pct)

    # Guardar
    text = " ".join(parts).strip()
    out_base = path.with_suffix("")
    write_txt(out_base.with_suffix(".txt"), text)
    log.put("📄 Guardado TXT.")
    log.put("✅ Transcripción completada.")


# -------------------- GUI --------------------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lumi Transcriber")
        self.geometry("760x600")
        self.minsize(720, 560)

        self.log_q: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        self.create_widgets()
        self.after(100, self._drain_log)

        ok, msg = check_ffmpeg()
        if ok:
            self._log(f"✅ FFmpeg detectado: {msg}")
        else:
            self._log(f"⚠️ {msg}")

        self._update_start_enabled()

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

    # ---- UI layout ----
    def create_widgets(self) -> None:
        pad = {"padx": 10, "pady": 8}

        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", **pad)

        self.mode = tk.StringVar(value="file")
        r1 = ttk.Radiobutton(frm_top, text="Transcribir archivo", variable=self.mode, value="file", command=self._update_start_enabled)
        r2 = ttk.Radiobutton(frm_top, text="Transcribir carpeta", variable=self.mode, value="folder", command=self._update_start_enabled)
        r1.grid(row=0, column=0, sticky="w")
        r2.grid(row=0, column=1, sticky="w")

        self.path_var = tk.StringVar()
        self.path_var.trace_add("write", lambda *args: self._update_start_enabled())
        btn_browse = ttk.Button(frm_top, text="Elegir…", command=self.browse_path)
        ent_path = ttk.Entry(frm_top, textvariable=self.path_var)
        ent_path.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4,0))
        btn_browse.grid(row=1, column=3, sticky="e", padx=(6,0), pady=(4,0))
        frm_top.columnconfigure(2, weight=1)

        frm_opts = ttk.LabelFrame(self, text="Opciones")
        frm_opts.pack(fill="x", **pad)

        ttk.Label(frm_opts, text="Modelo").grid(row=0, column=0, sticky="w")
        self.cmb_model = ttk.Combobox(frm_opts, state="readonly", values=["tiny","base","small","medium","large"])
        self.cmb_model.set(DEFAULT_MODEL)
        self.cmb_model.grid(row=0, column=1, sticky="w")

        ttk.Label(frm_opts, text="Idioma").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.cmb_lang = ttk.Combobox(frm_opts, state="readonly", values=LANGUAGES)
        self.cmb_lang.set("Spanish")
        self.cmb_lang.grid(row=0, column=3, sticky="w")

        ttk.Label(
            frm_opts,
            text="* Si se selecciona un idioma distinto al original, se hace una traducción automática al idioma elegido.",
            foreground="#555",  # gris suave
            wraplength=680,     # evita que se salga del frame
            justify="left"
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6,0))

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)
        self.btn_start = ttk.Button(frm_actions, text="Iniciar", command=self.start, state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(frm_actions, text="Detener", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8,0))

        frm_log = ttk.LabelFrame(self, text="Registro")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(frm_log, wrap="word", height=16, state="disabled")
        self.txt_log.pack(fill="both", expand=True)

        # ---- Barra de progreso inferior (determinada para todo) ----
        frm_status = ttk.Frame(self)
        frm_status.pack(fill="x", padx=10, pady=(0,10))
        self.progress = ttk.Progressbar(frm_status, mode="determinate", maximum=100)
        self.progress.pack(fill="x", side="left", expand=True)
        self.lbl_status = ttk.Label(frm_status, text="Listo")
        self.lbl_status.pack(side="left", padx=(10,0))

    def browse_path(self) -> None:
        if self.mode.get() == "file":
            path = filedialog.askopenfilename(title="Selecciona un archivo de audio/video",
                                              filetypes=[("Medios", "*.mkv *.mp4 *.mp3 *.wav *.flac *.webm *.m4a")])
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
        else:
            return p.is_dir()

    def _update_start_enabled(self) -> None:
        self.btn_start.configure(state="normal" if self._valid_selection() and not self.worker else "disabled")

    def set_running(self, running: bool) -> None:
        self.btn_start.configure(state="disabled" if running else ("normal" if self._valid_selection() else "disabled"))
        self.btn_stop.configure(state="normal" if running else "disabled")

    # ---------- Progreso visual ----------
    def _progress_reset(self):
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.lbl_status.configure(text="Listo")

    def _progress_set(self, value: float, text: str):
        self.progress.configure(mode="determinate")
        self.progress.stop()
        self.progress["value"] = max(0, min(100, value))
        self.lbl_status.configure(text=text)

    def _progress_busy(self, text="Cargando…"):
        # Barra animada (indeterminada) con texto
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)  # velocidad del “marquee”
        self.lbl_status.configure(text=text)

    def _progress_to_determinate(self, text="Procesando…"):
        # Volver a barra real (determinada)
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.lbl_status.configure(text=text)

    def start(self) -> None:
        path = self.path_var.get().strip()
        if not self._valid_selection():
            messagebox.showwarning("Falta ruta", "Elige un archivo compatible o una carpeta válida.")
            return

        model_name = self.cmb_model.get()
        lang_label = self.cmb_lang.get()
        lang = None if lang_label == "auto" else LANG_MAP.get(lang_label, None)

        self.stop_flag.clear()
        self.set_running(True)

        def worker():
            try:
                self.after(0, lambda: self._progress_busy("Cargando modelo…"))
                model = load_model(model_name, self.log_q)
                p = Path(path)

                if p.is_file():
                    # Progreso real por segmentos
                    self._log("Iniciando transcripción…\n")

                    switched = {"done": False}  # banderita para cambiar una sola vez

                    def onp(v):
                        if not switched["done"]:
                            switched["done"] = True
                            # Cuando llegan los primeros segmentos: pasar a barra determinada
                            self.after(0, lambda: self._progress_to_determinate("Procesando…"))
                        self.after(0, lambda: self._progress_set(v, f"Procesando {v:.0f}%…"))

                    transcribe_file(model, p, lang, onp, self.log_q)
                    self._progress_set(100, "Completado")
                else:
                    files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VALID_EXTENSIONS])
                    self.after(0, lambda: self._progress_busy("Preparando…"))
                    total = len(files)
                    if not files:
                        self._log("No hay medios compatibles en la carpeta.\n")
                    for i, f in enumerate(files, start=1):
                        if self.stop_flag.is_set():
                            self._log("⏹️ Proceso detenido por el usuario.\n")
                            break

                        self._log(f"[{i}/{total}] -> {f.name}\n")

                        # mostrar “cargando” hasta que lleguen los primeros segmentos
                        self.after(0, lambda i=i, total=total: self._progress_busy(f"Preparando {i}/{total}…"))

                        switched = {"done": False}  # bandera por archivo

                        def onp(v, i=i, total=total):
                            # la primera vez que llega un segmento, pasamos a barra determinada
                            if not switched["done"]:
                                switched["done"] = True
                                self.after(0, lambda i=i, total=total: self._progress_to_determinate(f"Procesando {i}/{total}…"))
                            self.after(0, lambda v=v, i=i, total=total: self._progress_set(v, f"Procesando {i}/{total} — {v:.0f}%"))

                        transcribe_file(model, f, lang, onp, self.log_q)
                        self._progress_set(100, f"Procesando {i}/{total} — 100%")

                        # si quedan archivos, vuelve a “Preparando…” para que se note el cambio
                        if i < total:
                            self.after(0, lambda i=i+1, total=total: self._progress_busy(f"Preparando {i}/{total}…"))

                    if total:
                        self._progress_set(100, "Completado")
                self._log("✨ Trabajo terminado.\n*-------------------------------*\n")
            except Exception as e:
                self._log(f"💥 Error: {e}")
            finally:
                self.set_running(False)
                self.worker = None
                self.after(500, self._progress_reset)
                self.after(0, self._update_start_enabled)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
        self._update_start_enabled()

    def stop(self) -> None:
        self.stop_flag.set()
        self._log("Solicitando detener… espera a que finalice el archivo en curso.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
