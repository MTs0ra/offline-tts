"""Offline speech: synthesis streamed straight to the default audio device.

Speaker owns playback, cancellation, and threading; it knows nothing about any
particular TTS engine. A backend (PiperBackend here; KokoroBackend in
kokoro_backend.py) only needs load(voice_id) and synthesize(text, cancel), the
same contract set_voice()/speak() already drove before Kokoro was added.
"""

import queue
import re
import threading
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import sounddevice as sd
from piper import PiperVoice

# Audio is written in slices of this length and the chunk queue is polled at the
# same interval, so a stop request is always honoured within about this long.
BLOCK_SECONDS = 0.1

# Pause lengths inserted between segments so speech breathes instead of running
# together. Both backends synthesize one chunk per sentence/clause and keep this
# split-and-splice-in-silence trick (mirroring the raw-silence-between-chunks
# technique Piper's own CLI uses for its --sentence-silence option) so Kokoro
# breathes at the same rhythm Piper does; Kokoro's tuning panel lets these be
# overridden per-backend (see kokoro_backend.py).
PAUSE_CLAUSE_S = 0.2      # comma, semicolon, colon
PAUSE_SENTENCE_S = 0.5    # period, exclamation mark
PAUSE_QUESTION_S = 1.0    # question mark: let it settle

# A run of sentence/clause punctuation followed by whitespace marks a break point;
# requiring the whitespace keeps mid-number punctuation ("3.14") from splitting.
# Abbreviations ("e.g. ", "Dr. ") still split - not perfect, but a slightly early
# pause there is a fair trade for not hand-maintaining an abbreviation list.
_BREAK_RE = re.compile(r"([.!?,;:]+)\s+")

# Each segment is synthesized independently, so the waveform at its very start/end has
# no relation to whatever comes right before/after it (typically a span of Silence) -
# without a fade, that discontinuity is an audible click. 6ms is short enough to be
# inaudible as a fade but long enough to remove the click; well under half a phoneme.
FADE_SECONDS = 0.006

VoiceId = Union[Path, str]


def pause_for(punctuation: str, clause_s: float = PAUSE_CLAUSE_S,
              sentence_s: float = PAUSE_SENTENCE_S, question_s: float = PAUSE_QUESTION_S) -> float:
    if "?" in punctuation:
        return question_s
    if "." in punctuation or "!" in punctuation:
        return sentence_s
    return clause_s


def split_into_segments(text: str, clause_s: float = PAUSE_CLAUSE_S,
                         sentence_s: float = PAUSE_SENTENCE_S,
                         question_s: float = PAUSE_QUESTION_S) -> List[Tuple[str, float]]:
    """Split text at sentence/clause punctuation, pairing each piece with the pause
    that should follow it. Punctuation stays on its segment, since both engines use
    it for prosody; the last segment never gets a pause, so speech doesn't end in silence.
    """
    segments = []
    pos = 0
    for match in _BREAK_RE.finditer(text):
        end = match.end()
        piece = text[pos:end].strip()
        if piece:
            segments.append([piece, pause_for(match.group(1), clause_s, sentence_s, question_s)])
        pos = end
    tail = text[pos:].strip()
    if tail:
        segments.append([tail, 0.0])
    if segments:
        segments[-1][1] = 0.0
    return [(piece, pause) for piece, pause in segments]


class Silence:
    """A span of digital silence, duck-typed as an AudioChunk (piper.voice.AudioChunk)
    so Speaker._play/_write can treat a pause exactly like real audio with no special-casing.
    """

    sample_width = 2  # both engines synthesize 16-bit PCM

    def __init__(self, seconds: float, sample_rate: int, sample_channels: int):
        self.sample_rate = sample_rate
        self.sample_channels = sample_channels
        # Whole samples first, then bytes, so an odd sample count can't misalign
        # the 16-bit frames of whatever chunk streams next.
        sample_count = int(seconds * sample_rate)
        self.audio_int16_bytes = bytes(sample_count * sample_channels * self.sample_width)


class Faded:
    """Wraps a chunk, linearly fading its first and/or last FADE_SECONDS to silence.
    Duck-typed the same as Silence, so it drops straight into the existing chunk protocol.
    Used only at segment boundaries - never on an engine's internal sub-chunks, which are
    already one continuous waveform and would only pick up artificial fades from this.
    """

    def __init__(self, chunk, fade_in: bool, fade_out: bool):
        self.sample_rate = chunk.sample_rate
        self.sample_channels = chunk.sample_channels
        self.sample_width = chunk.sample_width
        audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32)
        frames = len(audio) // chunk.sample_channels if chunk.sample_channels else 0
        fade_len = min(int(FADE_SECONDS * chunk.sample_rate), frames // 2)
        if fade_len > 0 and (fade_in or fade_out):
            audio = audio.reshape(frames, chunk.sample_channels)
            if fade_in:
                audio[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
            if fade_out:
                audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, None]
            audio = audio.reshape(-1)
        self.audio_int16_bytes = audio.astype(np.int16).tobytes()


class PiperBackend:
    """Loads a Piper .onnx voice and synthesizes text with it, sentence by sentence."""

    def __init__(self):
        self._voice: Optional[PiperVoice] = None
        self._voice_path: Optional[Path] = None

    def load(self, path: Path) -> None:
        if self._voice_path != path:
            self._voice = PiperVoice.load(str(path))
            self._voice_path = path

    def unload(self) -> None:
        """Releases the loaded voice so it can be garbage-collected. Called when switching
        to a different engine (see App._select_voice) so this app never holds more than one
        voice model resident at a time - see kokoro_backend.KokoroBackend.unload for the
        same pattern on that side, where it matters much more (a several-hundred-MB model
        versus this one's much smaller footprint)."""
        self._voice = None
        self._voice_path = None

    def synthesize(self, text: str, cancel: threading.Event) -> Iterator:
        fade_in_next = True  # the very first chunk of speech fades in from silence too
        for segment, pause in split_into_segments(text):
            if cancel.is_set():
                break
            # One lookahead slot: only the first and last of a segment's (possibly several,
            # contiguous) sub-chunks border silence and need a fade; the ones between are
            # one continuous waveform and must be left alone.
            chunk_iter = iter(self._voice.synthesize(segment))
            chunk = next(chunk_iter, None)
            last_chunk = None
            is_first = True
            while chunk is not None and not cancel.is_set():
                next_chunk = next(chunk_iter, None)
                fade_in, fade_out = is_first and fade_in_next, next_chunk is None
                if fade_in or fade_out:
                    chunk = Faded(chunk, fade_in, fade_out)
                yield chunk
                last_chunk, chunk, is_first = chunk, next_chunk, False
            fade_in_next = True
            if pause and last_chunk is not None and not cancel.is_set():
                yield Silence(pause, last_chunk.sample_rate, last_chunk.sample_channels)


class Speaker:
    """Speaks one text at a time with whatever backend is current. A new speak() or
    a stop() cancels the current session; set_backend() switches engines outright.
    """

    def __init__(self, notify: Callable[[str], None], backend):
        self._notify = notify                  # thread-safe status sink supplied by the UI
        self._engine_lock = threading.Lock()   # a backend's engine is not assumed re-entrant
        self._session_lock = threading.Lock()  # speak() may be called from the UI and hotkey threads
        self._backend = backend
        self._wanted_voice_id: Optional[VoiceId] = None
        self._cancel = threading.Event()
        self._session: Optional[threading.Thread] = None

    @property
    def is_speaking(self) -> bool:
        return self._session is not None and self._session.is_alive()

    def set_backend(self, backend) -> None:
        """Switch engines. Stops whatever is currently speaking first."""
        self.stop()
        with self._engine_lock:
            self._backend = backend
        self._wanted_voice_id = None

    def set_voice(self, voice_id: VoiceId) -> None:
        """Select a voice and load it in the background so the first read starts quickly."""
        self._wanted_voice_id = voice_id
        threading.Thread(target=self._preload, args=(voice_id,), daemon=True).start()

    def speak(self, text: str) -> None:
        with self._session_lock:
            self._cancel.set()
            if self._session is not None:
                self._session.join(timeout=2)  # never let two sessions play over each other
            self._cancel = threading.Event()
            self._session = threading.Thread(
                target=self._run_session, args=(text, self._wanted_voice_id, self._cancel), daemon=True
            )
            self._session.start()

    def stop(self) -> None:
        self._cancel.set()

    def _preload(self, voice_id: VoiceId) -> None:
        try:
            with self._engine_lock:
                if voice_id == self._wanted_voice_id:  # skip voices the user already scrolled past
                    self._backend.load(voice_id)
                    self._notify("Ready")
        except Exception as exc:
            self._notify(f"Could not load voice {_voice_label(voice_id)}: {exc}")

    def _run_session(self, text: str, voice_id: Optional[VoiceId], cancel: threading.Event) -> None:
        chunks: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._synthesize, args=(text, voice_id, chunks, cancel), daemon=True
        ).start()
        try:
            self._play(chunks, cancel)
        except Exception as exc:
            cancel.set()  # also tells the synthesis thread to quit
            self._notify(f"Speech failed: {exc}")
        else:
            self._notify("Stopped" if cancel.is_set() else "Ready")

    def _synthesize(self, text: str, voice_id: Optional[VoiceId], chunks: queue.Queue,
                     cancel: threading.Event) -> None:
        """Producer thread: synthesises the next sentence while the current one plays, so there are no gaps."""
        try:
            with self._engine_lock:
                self._backend.load(voice_id)
                for chunk in self._backend.synthesize(text, cancel):
                    if cancel.is_set():
                        break
                    chunks.put(chunk)
        except Exception as exc:
            chunks.put(exc)
        finally:
            chunks.put(None)  # end-of-speech marker

    def _play(self, chunks: queue.Queue, cancel: threading.Event) -> None:
        stream = None
        try:
            while not cancel.is_set():
                try:
                    chunk = chunks.get(timeout=BLOCK_SECONDS)
                except queue.Empty:
                    continue
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                if stream is None:
                    stream = sd.RawOutputStream(
                        samplerate=chunk.sample_rate, channels=chunk.sample_channels, dtype="int16"
                    )
                    stream.start()
                    self._notify("Speaking…")
                self._write(stream, chunk, cancel)
        finally:
            if stream is not None:
                if not cancel.is_set():
                    stream.stop()  # let the final buffer play out
                stream.close()     # discards anything still buffered when cancelled

    @staticmethod
    def _write(stream: sd.RawOutputStream, chunk, cancel: threading.Event) -> None:
        audio = memoryview(chunk.audio_int16_bytes)
        step = int(chunk.sample_rate * BLOCK_SECONDS) * chunk.sample_width * chunk.sample_channels
        for start in range(0, len(audio), step):
            if cancel.is_set():
                return
            stream.write(audio[start:start + step])


def _voice_label(voice_id: Optional[VoiceId]) -> str:
    return voice_id.stem if isinstance(voice_id, Path) else str(voice_id)
