# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui/spot_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/bartz/miniforge3/envs/spot/lib/python3.11/site-packages/customtkinter', 'customtkinter/'), ('/Users/bartz/miniforge3/envs/spot/lib/python3.11/site-packages/lightning_fabric', 'lightning_fabric/'),  ( 'src/spotgui/ctk/images', 'spotgui/ctk/images/' ), ('/Users/bartz/miniforge3/envs/spot/lib/python3.11/site-packages/spotpython/hyperdict', 'spotpython/hyperdict/'), ('/Users/bartz/miniforge3/envs/spot/lib/python3.11/site-packages/spotriver/hyperdict', 'spotriver/hyperdict/')],
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
    exclude_binaries=False,
    name='spot_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
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
