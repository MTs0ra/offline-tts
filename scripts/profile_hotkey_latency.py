"""Measures press-to-first-audio latency for Kokoro on a real multi-sentence paragraph
(exercising the sentence-streaming path, not a single-sentence one), and checks for
playback stalls between sentences: the gap between when one chunk's audio should finish
and when the next chunk starts writing should be ~0, not "wait for synthesis" dead air.

    python scripts\\profile_hotkey_latency.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kokoro_backend import DEFAULT_VOICE, KokoroBackend
from speech import Speaker

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "kokoro" / "kokoro__v1.0-fp32__20260911.onnx"
VOICES_PATH = ROOT / "models" / "kokoro" / "kokoro__voices-v1.0__20260911.bin"

PARAGRAPH = (
    "The meeting starts at three. Bring the updated slides and the budget numbers "
    "from last quarter. If anyone from finance asks about the delay, just say we're "
    "still waiting on legal. We should be done well before five."
)


def main() -> None:
    backend = KokoroBackend(MODEL_PATH, VOICES_PATH)
    print(f"Startup: loading {MODEL_PATH.name} + warm-up (not counted as a press)...")
    t0 = time.perf_counter()
    backend.load(DEFAULT_VOICE)
    print(f"  startup load took {time.perf_counter() - t0:.2f}s\n")

    speaker = Speaker(lambda text: None, backend)

    # Record every chunk write (real audio AND the deliberate inter-sentence Silence),
    # so a stall (synthesis falling behind playback) can be told apart from the silence
    # that's supposed to be there. is_real distinguishes the two for reporting.
    writes = []
    original_write = Speaker._write

    def timed_write(stream, chunk, cancel):
        frames = len(chunk.audio_int16_bytes) // (chunk.sample_width * chunk.sample_channels)
        duration = frames / chunk.sample_rate
        writes.append((time.perf_counter(), duration, any(chunk.audio_int16_bytes)))
        return original_write(stream, chunk, cancel)

    Speaker._write = staticmethod(timed_write)

    n_sentences = PARAGRAPH.count(".") + PARAGRAPH.count("?")
    print(f"Press: reading a {n_sentences}-sentence paragraph...")
    press_at = time.perf_counter()
    speaker.speak(PARAGRAPH)
    deadline = time.monotonic() + 60
    while speaker.is_speaking and time.monotonic() < deadline:
        time.sleep(0.005)
    total = time.perf_counter() - press_at

    first_audio_at = next(start for start, _dur, is_real in writes if is_real)
    print(f"\nTime from keypress (speak() call) to first audio: {first_audio_at - press_at:.3f}s")
    print(f"Total read (all sentences, playback included): {total:.2f}s\n")

    # A chunk should start writing right where the previous one's audio ends (that's what
    # "no gaps" means); any extra delay beyond ~50ms of scheduling jitter is synthesis
    # falling behind playback - a real stall, not the deliberate inter-sentence Silence
    # (which is already its own chunk in this list, accounted for on its own line).
    print("Timeline (stall = unexpected dead air *beyond* the deliberate pause, should be ~0):")
    prev_end = None
    max_stall = 0.0
    for i, (start, duration, is_real) in enumerate(writes, 1):
        stall = 0.0 if prev_end is None else max(0.0, start - prev_end)
        max_stall = max(max_stall, stall)
        kind = "sentence" if is_real else "pause"
        print(f"  {i}. {kind:8s} t={start - press_at:6.3f}s  duration={duration:.2f}s  stall_before={stall * 1000:.0f}ms")
        prev_end = start + duration
    print(f"\nLargest unexpected stall: {max_stall * 1000:.0f}ms "
          f"({'OK - no audible gap' if max_stall < 0.05 else 'STALL - synthesis fell behind playback'})")


if __name__ == "__main__":
    main()
