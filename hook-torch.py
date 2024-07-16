from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect only the necessary submodules
hiddenimports = collect_submodules('torch')

# Collect only the necessary data files
datas = collect_data_files('torch', include_py_files=True)