import os

block_cipher = None

ffmpeg_bin = ('ffmpeg.exe', '.')

# Define the path to the Whisper assets
whisper_assets_path = 'C:/Python310/Lib/site-packages/whisper/assets'
whisper_assets = (whisper_assets_path, 'whisper/assets')

# Define the path to the small.pt model in the models directory
small_pt_model = ('./models/small.pt', 'models/')

a = Analysis(['main.py'],
             pathex=['.'],
             binaries=[ffmpeg_bin],
             datas=[whisper_assets, small_pt_model],  # Include Whisper assets and small.pt model
             hiddenimports=['torch', 'whisper', 'tkinter', 'ttkbootstrap'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Whisper Transcriber',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          icon='./icon.ico')