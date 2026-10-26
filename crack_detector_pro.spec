# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['crack_detector_pro.py'],
    pathex=[],
    binaries=[],
    datas=[('best.pt', '.')],
    hiddenimports=['ultralytics', 'ultralytics.nn.tasks', 'ultralytics.models.yolo.detect', 'cv2', 'PySide6'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='crack_detector_pro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
