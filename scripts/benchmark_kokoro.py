"""One-off measurement: synthesize + play a sample paragraph with Kokoro and report
peak/average CPU and GPU usage during that window.

Run after `pip install -r requirements.txt` and downloading the model files into
models\\kokoro\\ (see README.md, "Optional: add the Kokoro engine").

    python scripts\\benchmark_kokoro.py
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kokoro_backend import KokoroBackend  # noqa: E402  (path set up above)
from speech import Speaker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "kokoro" / "kokoro__v1.0-fp32__20260911.onnx"
VOICES_PATH = ROOT / "models" / "kokoro" / "kokoro__voices-v1.0__20260911.bin"
SAMPLE_TEXT = (
    "The old lighthouse stood at the edge of the cliff, its light sweeping slowly "
    "across the dark water. Every night, it warned ships away from the rocks below. "
    "Would anyone notice, the keeper wondered, if one evening the light simply didn't turn on?"
)
SAMPLE_INTERVAL_S = 0.2


def _gpu_utilization_percent() -> float:
    """0.0 if nvidia-smi isn't available or reports nothing - a real absence of GPU
    use looks the same as "couldn't measure", which is fine here since Kokoro has no
    CUDA execution provider installed and so cannot use the GPU either way.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def _sample_usage(proc: psutil.Process, samples: list, stop: threading.Event) -> None:
    ncpus = psutil.cpu_count() or 1
    proc.cpu_percent()  # first call only primes the internal baseline; discard it
    while not stop.is_set():
        time.sleep(SAMPLE_INTERVAL_S)
        cpu_percent_of_machine = proc.cpu_percent() / ncpus
        samples.append((cpu_percent_of_machine, _gpu_utilization_percent()))


def main() -> None:
    if not MODEL_PATH.is_file() or not VOICES_PATH.is_file():
        print("Missing Kokoro model files. Expected:")
        print(f"  {MODEL_PATH}")
        print(f"  {VOICES_PATH}")
        print("See README.md, \"Optional: add the Kokoro engine\", for the download commands.")
        return

    backend = KokoroBackend(MODEL_PATH, VOICES_PATH)
    print(f"Loading Kokoro ({MODEL_PATH.name}, voice={backend.voice}) - one-time warm-up, not measured...")
    backend.load(backend.voice)

    speaker = Speaker(print, backend)
    samples: list = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_usage, args=(psutil.Process(), samples, stop), daemon=True)
    sampler.start()

    print("Synthesizing and playing the sample paragraph...")
    started = time.monotonic()
    speaker.speak(SAMPLE_TEXT)
    time.sleep(0.3)  # give the session thread time to actually start before polling is_speaking
    while speaker.is_speaking:
        time.sleep(0.1)
    elapsed = time.monotonic() - started

    stop.set()
    sampler.join(timeout=1)

    if not samples:
        print("No usage samples were collected - the run was shorter than one sample interval.")
        return
    cpu_values = [cpu for cpu, _ in samples]
    gpu_values = [gpu for _, gpu in samples]
    print(f"\n{elapsed:.1f}s total, {len(samples)} samples at {SAMPLE_INTERVAL_S:.1f}s intervals")
    print(f"CPU (% of total machine capacity): peak {max(cpu_values):.1f}%  avg {sum(cpu_values) / len(cpu_values):.1f}%")
    print(f"GPU utilization:                   peak {max(gpu_values):.1f}%  avg {sum(gpu_values) / len(gpu_values):.1f}%")


if __name__ == "__main__":
    main()
