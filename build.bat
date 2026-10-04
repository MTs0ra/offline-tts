@echo off
rem Builds dist\TTS\TTS.exe, which runs without Python installed.
setlocal
cd /d "%~dp0"

rem --collect-data piper bundles the espeak-ng phoneme data that Piper loads at runtime.
rem kokoro_onnx pulls in a chain of packages that each read their own bundled data file at
rem *import* time (not lazily), so all of them need --collect-data or the frozen exe raises
rem FileNotFoundError before app.py's import-guard even has a working engine to fall back
rem past: kokoro_onnx (config.json) -> phonemizer (share/festival, share/segments) ->
rem phonemizer.backend.segments -> segments -> csvw -> language_tags (data/json/*.json).
rem csvw and segments themselves ship no data files (checked - code only). espeakng_loader
rem bundles the actual espeak-ng.dll + espeak-ng-data, loaded via ctypes with a path computed
rem at runtime (espeakng_loader.get_library_path()/get_data_path()) - PyInstaller can't
rem discover that automatically since it's not a literal, statically-analyzable import.
rem --collect-binaries onnxruntime bundles whichever onnxruntime build this venv actually
rem has installed - if that's onnxruntime-directml (see requirements.txt), its capi\ folder
rem holds DirectML.dll alongside the usual onnxruntime.dll/onnxruntime_pybind11_state.pyd,
rem and it's loaded the same dynamic way those already needed explicit bundling for above.
rem --icon sets the .exe icon (and so the taskbar/desktop icon) to icon\tts.ico, a speaker
rem glyph taken from the Windows Sound resources (ddores.dll), not a custom asset.
pyinstaller --noconfirm --clean --windowed --name TTS --icon icon\tts.ico ^
    --collect-data piper --collect-data kokoro_onnx --collect-data phonemizer ^
    --collect-data language_tags --collect-data espeakng_loader --collect-binaries onnxruntime ^
    app.py || exit /b 1

rem Voices and Kokoro models live beside the exe (not inside it) so you can add, remove, or
rem rename them without rebuilding.
if exist voices xcopy /e /i /y voices dist\TTS\voices >nul
if exist models xcopy /e /i /y models dist\TTS\models >nul
rem app.py sets the window/taskbar icon at runtime too (ui_widgets.set_window_icon), not
rem just via this script's --icon flag above (which only sets the .exe file's own icon
rem resource - a separate thing from the *window's* icon) - it needs this file to actually
rem exist next to the exe at runtime to load from.
if exist icon xcopy /e /i /y icon dist\TTS\icon >nul
echo.
echo Built dist\TTS\TTS.exe
