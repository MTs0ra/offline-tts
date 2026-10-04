# TTS

Select text in any Windows program, press **F2**, and hear it read aloud by a neural voice.

- **Fully offline.** No cloud calls, no API keys, no telemetry. Windows 10/11 (x64) only.
- **Two engines, one voice list:**
  - **Kokoro** (default): Kokoro-82M, a high-quality neural voice, via `kokoro-onnx`. Runs on CPU by default; optional GPU acceleration through DirectML, with automatic CPU fallback.
  - **Piper**: lighter and faster, with a wide range of downloadable voices.
- Nothing runs at startup or in the background. Close the window and the hotkey is released.

## Requirements

- Windows 10/11, 64-bit
- Python 3.12 or newer, 64-bit (developed and tested on 3.14)
- `kokoro-onnx` is pinned to `==0.4.7` in `requirements.txt`; newer versions have not been tested with this app.

## Setup

```bat
git clone <this-repo-url> TTS
cd TTS
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then download the models (next section) and run:

```bat
python app.py
```

Click **Start** to enable the F2 hotkey. Select text anywhere and press **F2**; press F2 again (or click **Stop**) to cut it off. **Read** reads the current selection, or the last thing you copied if nothing is selected. **Ctrl+Alt+T** brings the window back if it is minimized.

## Get the models

No model or voice files are included in this repo. Download them once; after that the app never touches the network. The app runs with either engine alone, or both.

### Kokoro (default engine) → `models\kokoro\`

Download from the [`thewh1teagle/kokoro-onnx` `model-files-v1.0` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0):

| File | Size | Notes |
|---|---|---|
| [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx) | ~326 MB | Full precision. Preferred: in testing it was 3 to 3.5x faster per sentence than int8 on the dev CPU. |
| [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin) | ~28 MB | Required. The Kokoro voice bank. |
| [`kokoro-v1.0.int8.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx) | ~92 MB | Optional smaller alternative, used only if the fp32 file is absent. |

Keep the original filenames; no renaming is needed.

```bat
mkdir models\kokoro
curl -L -o models\kokoro\kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models\kokoro\voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Piper voices → `voices\`

Each voice is a `name.onnx` plus `name.onnx.json` pair. Browse and hear them at [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/), and download from [`rhasspy/piper-voices` on Hugging Face](https://huggingface.co/rhasspy/piper-voices). For example, the US English *Amy* voice is at [`en/en_US/amy/medium`](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium). Or let Piper fetch it for you:

```bat
python -m piper.download_voices en_US-amy-medium --data-dir voices
```

Voices are not redistributed here. Each carries its own licence (see [Licence](#licence)).

Kokoro voices (labelled `v2`) and Piper voices (`v1`) share one **Voice** dropdown; picking a voice picks the engine.

## Optional: GPU acceleration (DirectML)

```bat
pip uninstall onnxruntime -y
pip install onnxruntime-directml
```

The app detects this at the next launch and prefers the GPU for Kokoro, falling back to CPU automatically if DirectML fails. The active provider is logged to `tts.log` (for example `CPUExecutionProvider` or `DmlExecutionProvider`). This is a manual step because `pip` doesn't treat `onnxruntime-directml` as satisfying `onnxruntime`; listing both in `requirements.txt` would silently replace the GPU build with the CPU one.

On the dev machine (RTX 3050, `onnxruntime-directml` 1.24.4) Kokoro's vocoder currently fails on DirectML and the app falls back to CPU, so expect no speed-up there. Other GPUs and future `onnxruntime-directml` releases may differ.

To go back to CPU only: `pip uninstall onnxruntime-directml -y && pip install onnxruntime`.

## Build a standalone .exe (optional)

```bat
pip install pyinstaller
build.bat
```

Produces `dist\TTS\TTS.exe`, with `voices\`, `models\` and `icon\` copied beside it. Python is not needed on the target PC.

## Troubleshooting

- **F2 does nothing.** Click **Start** first. If another program already owns F2 (some games, MSI Afterburner, launcher overlays), the window says so; close it and click Start again.
- **F2 conflicts with other apps.** While TTS is running, F2 no longer reaches other programs, so File Explorer's rename and Excel's cell editing stop working. Change `HOTKEY` and `HOTKEY_LABEL` at the top of `app.py` to use another key.
- **No Kokoro voices in the dropdown.** Check that `kokoro-v1.0.onnx` (or the int8 file) and `voices-v1.0.bin` are in `models\kokoro\`. If the Kokoro import itself fails, the traceback is written to `tts.log` and the app carries on with Piper only.
- **No Piper voices.** Each voice needs both the `.onnx` and the `.onnx.json` file in `voices\`.
- **Nothing selected in a terminal.** The app captures text by sending Ctrl+C, which interrupts a running command if nothing is selected. Select text first.
- **Administrator windows.** Windows blocks simulated keystrokes into elevated windows, so the app falls back to whatever is on the clipboard.
- **Editors that copy the whole line** (such as VS Code) when nothing is selected will have that line read.
- **Clipboard.** Your previous clipboard text is restored after a capture, but images or files on the clipboard can't be.
- **Changed speakers or headphones.** The output device is chosen when the app opens; reopen it.
- **Something else.** Look at `tts.log` next to the app.

## Privacy

The app has no network code, telemetry or keys, and works the same with `TTS.exe` blocked in the firewall. One thing outside its control: selection capture sends an ordinary Ctrl+C. If Windows *Clipboard history → Sync across your devices* is on, Windows treats that copy like any other.

## Tests

```bat
python -m unittest discover -s tests
```

## Licence

The code in this repository is released under the [MIT License](LICENSE), copyright (c) 2026 MTs0ra.

Dependencies and models have their own licences:

- **[piper-tts](https://github.com/OHF-Voice/piper1-gpl)** is GPL-3.0. This repo only lists it as a dependency and does not include it, but if you distribute a built `TTS.exe` (which bundles Piper), distribute that binary under GPL-compatible terms with its source.
- **Piper voices** carry their own licences. The *en_US-amy-medium* model card points to [MycroftAI/mimic3-voices](https://github.com/MycroftAI/mimic3-voices) as the dataset and gives its licence only as "See URL". Its terms are not verified here, so check that page and the [voice's model card](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium) before using or redistributing it.
- **kokoro-onnx** is MIT; the **Kokoro-82M** weights are Apache-2.0.

## Acknowledgements

- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) (Open Home Foundation / Rhasspy) and the *en_US Amy* voice from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad (Apache-2.0) and [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle.
