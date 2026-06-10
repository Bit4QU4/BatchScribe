# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# Analysis binaries take (source, dest_dir) 2-tuples; post-Analysis TOC
# entries are 3-tuples, so the bundled ffmpeg must be declared here.
_extra_binaries = [("ffmpeg.exe", ".")] if os.path.exists("ffmpeg.exe") else []

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=_extra_binaries,
    datas=collect_data_files('faster_whisper'),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='TranscriptionHackery',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TranscriptionHackery',
)
