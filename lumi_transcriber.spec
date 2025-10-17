# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Lumi Transcriber (Windows, onedir)
# Bundles ffmpeg + icon; tolerates ffmpeg in ffmpeg/ or ffmpeg/bin at runtime.

block_cipher = None

a = Analysis(
    ['transcriber_app.py'],
    pathex=[],
    binaries=[
        # Place into ffmpeg/bin in the dist to match upstream layout
        ('ffmpeg\\bin\\ffmpeg.exe', 'ffmpeg\\bin'),
        ('ffmpeg\\bin\\ffprobe.exe', 'ffmpeg\\bin'),
    ],
    datas=[
        ('assets\\lumi.ico', 'assets'),
    ],
    hiddenimports=['faster_whisper'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_ffmpeg_path.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LumiTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\lumi.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LumiTranscriber'
)
