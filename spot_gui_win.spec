# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui/spot_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/bartz/miniforge-pypy3/envs/spotGUI/Lib/site-packages/spotgui','spotgui/'),('C:/Users/bartz/miniforge-pypy3/envs/spotGUI/Lib/site-packages/customtkinter', 'customtkinter/'), ('C:/Users/bartz/miniforge-pypy3/envs/spotGUI/Lib/site-packages/lightning_fabric', 'lightning_fabric/'),  ( 'src/spotgui/ctk/images', 'spotgui/ctk/images/' ), ('C:/Users/bartz/miniforge-pypy3/envs/spotGUI/Lib/site-packages/spotPython/hyperdict', 'spotPython/hyperdict/'), ('C:/Users/bartz/miniforge-pypy3/envs/spotGUI/Lib/site-packages/spotRiver/hyperdict', 'spotRiver/hyperdict/')],
    hiddenimports=[],
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
    [],
    exclude_binaries=True,
    name='spot_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='spot_gui',
)
app = BUNDLE(
    coll,
    name='spot_gui.app',
    icon=None,
    bundle_identifier=None,
)
