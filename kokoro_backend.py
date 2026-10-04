"""Kokoro synthesis backend: same load(voice_id)/synthesize(text, cancel) contract
speech.PiperBackend implements, so Speaker can drive either engine identically.

GPU (DirectML) if available, CPU otherwise: _load_session tries the DmlExecutionProvider
first and falls back to CPUExecutionProvider if DirectML isn't compiled into this
environment's onnxruntime build, or fails to initialize on this machine (no DX12-capable
GPU, bad driver, etc). load() adds a second fallback on top: DirectML can construct a
session successfully and still fail on the *first real inference* - confirmed on this
project's dev machine (RTX 3050, onnxruntime-directml 1.24.4, the latest available for
this Python version): both the fp32 and int8 models throw a RUNTIME_EXCEPTION from a
ConvTranspose node in the vocoder's upsampling layer ("80070057 the parameter is
incorrect") - a DirectML kernel limitation for this op configuration, not something
fixable here. A failed warm-up (see _warm_up) is treated the same as a failed
initialization: rebuild CPU-only and retry. In practice, on this hardware, Kokoro
currently always ends up on CPU - the DirectML path is real and safe to leave on, but
won't speed anything up here until a future onnxruntime-directml fixes that kernel; it
may well work on other GPUs/driver versions where DirectML doesn't hit the same bug.
Getting the DirectML build in the first place is a manual swap (`pip install
onnxruntime-directml` in place of plain `onnxruntime`) - see requirements.txt and README
for why that can't be a normal requirements.txt line. Whichever provider a session
actually ends up using is logged (stdout + tts.log) on load - see _log_active_provider.
This stays fully offline either way: DirectML talks to whatever GPU is already in the
machine through Windows' own driver stack, no network calls, no CUDA toolkit or other
separate install.

Per-press latency on CPU: profiled at 1.7-2.5s/sentence before this file's fixes. The
model load itself was already cached correctly (not the cause). The real per-press bug
was phonemizer.phonemize() rebuilding its whole espeak backend - including a fresh copy
of espeak-ng.dll and a full espeak_Initialize() rescan of every language dictionary - on
every single call (see _patch_phonemizer_backend_caching below); that's now cached, and
_warm_up() absorbs both that and ONNX Runtime's first-run graph warmup at startup rather
than on the user's first keypress. What was left on CPU (~1.7-2s per create() call) is
inference time inherent to the model - even a 5-phoneme "Hi." took ~1.15s in isolation -
so it can't be lowered by chunking or caching alone; two things still helped: switching
the default model from int8 to fp32 (measured 3-3.5x faster per sentence on this CPU -
int8's ConvInteger kernel isn't well vectorized here, so the 3.5x larger fp32 file wins
anyway - see voices.txt), and splitting only the opening sentence at its first clause
boundary (see _split_opening_clause) so a long opening sentence doesn't gate first audio
on all of it, without adding a create() call to any sentence after it. intra_op_num_threads
tuning (below) recovers ~10% on top of whichever provider is active.
"""

import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

from speech import Faded, Silence, pause_for


def _patch_phonemizer_backend_caching() -> None:
    """kokoro_onnx's Tokenizer.phonemize() calls the top-level phonemizer.phonemize()
    once per synthesize() call. That function (phonemizer/phonemize.py) constructs a
    brand new EspeakBackend from scratch on *every* call - which, per
    phonemizer/backend/espeak/api.py's EspeakAPI.__init__, copies the whole espeak-ng.dll
    to a fresh temp directory and then calls espeak_Initialize() (which rescans every
    bundled language dictionary) again. That is the real multi-second per-press cost -
    profiled at 1.7-2.5s per call - not the ONNX model, which was already being cached
    correctly. Reusing one backend instance across calls removes it; this is applied once,
    at import time, to every KokoroBackend/Kokoro instance in the process.
    """
    import sys

    import phonemizer  # phonemizer/__init__.py does `from .phonemize import phonemize`,
    # which shadows the *submodule* name with that *function* as an attribute of the
    # `phonemizer` package - so `import phonemizer.phonemize as _pm` would silently bind
    # _pm to the function, not the module. Pulling the real submodule out of sys.modules
    # sidesteps that.
    _pm = sys.modules["phonemizer.phonemize"]

    if getattr(phonemizer.phonemize, "_kokoro_reader_patched", False):
        return  # a second KokoroBackend (or import) in the same process: already patched

    _original_phonemize = _pm.phonemize
    _backend_cache: dict = {}

    def _cached_phonemize(text, language="en-us", backend="espeak",
                           separator=_pm.default_separator, strip=False,
                           prepend_text=False, preserve_empty_lines=False,
                           preserve_punctuation=False,
                           punctuation_marks=_pm.Punctuation.default_marks(),
                           with_stress=False, tie=False, language_switch="keep-flags",
                           words_mismatch="ignore", njobs=1, logger=_pm.get_logger()):
        if backend != "espeak" or njobs != 1:
            # Anything off the exact path Kokoro uses falls back to the real, uncached
            # implementation rather than risk a subtly wrong fast path for it.
            return _original_phonemize(text, language, backend, separator, strip, prepend_text,
                                        preserve_empty_lines, preserve_punctuation, punctuation_marks,
                                        with_stress, tie, language_switch, words_mismatch, njobs, logger)
        _pm._check_arguments(backend, with_stress, tie, separator, language_switch, words_mismatch)
        key = (language, punctuation_marks, preserve_punctuation, with_stress, tie,
               language_switch, words_mismatch)
        espeak_backend = _backend_cache.get(key)
        if espeak_backend is None:
            espeak_backend = _pm.BACKENDS["espeak"](
                language, punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation, with_stress=with_stress, tie=tie,
                language_switch=language_switch, words_mismatch=words_mismatch, logger=logger,
            )
            _backend_cache[key] = espeak_backend
        return _pm._phonemize(espeak_backend, text, separator, strip, njobs,
                               prepend_text, preserve_empty_lines)

    _cached_phonemize._kokoro_reader_patched = True
    _pm.phonemize = _cached_phonemize
    phonemizer.phonemize = _cached_phonemize  # what kokoro_onnx.tokenizer actually calls


_patch_phonemizer_backend_caching()

# Best-graded English voices (see hexgrad/Kokoro-82M VOICES.md), shown in the UI
# before the model has finished loading; get_voices() replaces this with the full
# list (26 voices) once the real voices-v1.0.bin has been read.
FALLBACK_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah",
    "am_fenrir", "am_michael", "am_puck",
    "bf_emma", "bf_isabella", "bm_george", "bm_fable",
]
DEFAULT_VOICE = "af_heart"

# Sentence-ending punctuation only (not , ; : - unlike speech.split_into_segments' clause
# splitting). Each piece here becomes one Kokoro.create() call: see _split_into_sentences.
_SENTENCE_BREAK_RE = re.compile(r"([.!?]+)\s+")

# Clause-internal punctuation only, used just to shorten the opening sentence (see
# _split_opening_clause) - requires trailing whitespace for the same reason speech.py's
# break regex does: "3,500" has no space after the comma, so it's never mistaken for one.
_CLAUSE_BREAK_RE = re.compile(r"[,;:]+\s+")

# Below this many words, splitting the opening clause off isn't worth it - see
# _split_opening_clause's docstring for the full reasoning (a create() call's cost is
# dominated by fixed overhead, not length, so a short clause gains almost no time-to-
# first-audio but leaves a real dead-air gap once its own brief playback runs out).
_MIN_OPENING_CLAUSE_WORDS = 6


def _split_opening_clause(sentence: str):
    """Splits `sentence` at its first comma/semicolon/colon into (opening_clause,
    remainder), so only that short opening clause has to finish synthesizing before
    playback can start - instead of the whole (possibly long) first sentence. Returns
    (sentence, None) - i.e. no split - if there's no such boundary, or if the resulting
    opening clause is too short to be worth splitting off. Only ever applied to the first
    sentence of a read - see synthesize().

    The word-count floor fixes a real bug, not just a tuning tweak: found by measuring
    actual playback against actual create() timing. create()'s cost is dominated by a
    large fixed per-call overhead (documented above: even a 5-phoneme "Hi." took ~1.15s
    in isolation), not by how much text it's given - so splitting off a *short* opening
    clause (e.g. "Well," or "Actually,") doesn't meaningfully shorten that first create()
    call, but it DOES shrink how much audio comes back from it to just a word or two.
    That audio finishes playing in well under a second, while the *next* piece (the
    remainder of the sentence) still needs its own full ~1.5-2.5s create() call, which
    was only just starting when the first chunk began playing - so playback catches up to
    the queue and stalls on real dead air until the remainder is ready. This is exactly
    the "long pause after the first couple of words" symptom: normal sentence-to-sentence
    transitions don't have this problem because a whole sentence's worth of audio takes
    several seconds to play, comfortably outlasting the fixed cost of synthesizing the
    next one in the background - only this one specific transition (tiny opening clause
    versus a same-cost next piece) has too little playback runway to cover the gap.
    Below the word-count floor, treating the whole first sentence as one piece costs
    nothing in practice (same fixed floor either way, per the reasoning above) and removes
    the stall entirely.
    """
    match = _CLAUSE_BREAK_RE.search(sentence)
    if not match:
        return sentence, None
    opening = sentence[:match.end()].strip()
    if len(opening.split()) < _MIN_OPENING_CLAUSE_WORDS:
        return sentence, None
    return opening, sentence[match.end():].strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences for streamed synthesis: the first sentence is
    synthesized and handed to Speaker for immediate playback, then the rest follow one
    at a time in the same background producer thread while playback continues (see
    KokoroBackend.synthesize and speech.Speaker._synthesize/_play, which already stream
    whatever a backend yields - this only changes the chunk boundary those use).

    Deliberately coarser than speech.split_into_segments' comma/semicolon/colon splitting:
    each Kokoro.create() call carries a large fixed cost (~1s, independent of text length -
    see this module's docstring), so splitting at every comma would pay that fixed cost
    several times per sentence for no benefit; one call per whole sentence pays it once,
    and Kokoro's own trained prosody already handles comma pauses within a sentence.
    """
    sentences = []
    pos = 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        piece = text[pos:match.end()].strip()
        if piece:
            sentences.append(piece)
        pos = match.end()
    tail = text[pos:].strip()
    if tail:
        sentences.append(tail)
    return sentences


class KokoroChunk:
    """Wraps kokoro.create()'s (float32 samples, sample_rate) as an AudioChunk-shaped
    object, the same duck type Silence uses, so Speaker._play/_write need no changes.
    """

    sample_width = 2
    sample_channels = 1

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.sample_rate = sample_rate
        # Kokoro's float output is normally within [-1, 1]; scale for full 16-bit range and
        # only pull gain down on the rare peak that actually exceeds it, so ordinary speech
        # isn't quieted for no reason. Rounding (not truncating) avoids a quantization bias.
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        scale = 32767.0 if peak <= 1.0 else 32767.0 / peak
        self.audio_int16_bytes = np.round(samples * scale).clip(-32768, 32767).astype(np.int16).tobytes()


class KokoroBackend:
    """Loads the Kokoro ONNX model once, then synthesizes with whichever voice/speed/
    pause settings are current at the moment speak() is called (see app.py's Tuning panel).
    """

    def __init__(self, model_path: Path, voices_path: Path):
        self._model_path = model_path
        self._voices_path = voices_path
        self._kokoro: Optional[Kokoro] = None

        # Tuning: mutated directly by app.py's slider callbacks. Defaults mirror
        # speech.py's Piper pause constants, so switching engines doesn't change
        # the pacing you're used to until you deliberately move a slider. clause_pause_s
        # only ever applies to the one clause boundary synthesize() splits the opening
        # sentence at (see _split_opening_clause) - there are no other clause boundaries
        # in sentence-level streaming for it to apply to.
        self.voice = DEFAULT_VOICE
        self.speed = 1.0
        self.clause_pause_s = 0.2
        self.sentence_pause_s = 0.5
        self.question_pause_s = 1.0

    def load(self, voice_id) -> None:
        """voice_id (a str voice name) just selects the voice; the heavy one-time
        ONNX session + voice-bank load happens here too, but only the first time.
        Speaker always calls this under its own engine lock, so no lock is needed here.
        """
        if voice_id:
            self.voice = voice_id
        if self._kokoro is None:
            self._kokoro = self._load_session()
            if not self._warm_up():
                # Confirmed by testing: DmlExecutionProvider can construct a session
                # successfully and still fail the *first real inference* - this model's
                # vocoder has a ConvTranspose DirectML's kernel rejects at run time
                # ("the parameter is incorrect"), which session creation alone can't
                # detect. Treat a failed warm-up as if the GPU provider had failed to
                # initialize in the first place: rebuild CPU-only and warm up again.
                self._kokoro = self._load_session(force_cpu=True)
                self._warm_up()

    def unload(self) -> None:
        """Releases the loaded ONNX session (and the Kokoro wrapper holding it) so it can be
        garbage-collected. The fp32 model plus onnxruntime's own overhead is, by a wide
        margin, the single largest thing this app holds in memory (several hundred MB even
        with the arena disabled - see _load_session) - keeping it resident after the user
        has switched to Piper wastes all of that for no benefit, and is exactly the "more
        than one voice model loaded at a time" case worth avoiding. Called by
        App._select_voice when switching away from Kokoro.

        This is a deliberate memory-for-speed trade, not a free win: switching back to a
        Kokoro voice afterward pays the full session-construction + warm-up cost again
        (the same multi-second cost as a cold app launch), since load() only skips that
        work when self._kokoro is already set. Only call this once the caller has confirmed
        no synthesis is still in flight on the backend being unloaded (Speaker.set_backend
        already guarantees this - see its own locking).
        """
        self._kokoro = None

    def _load_session(self, force_cpu: bool = False) -> Kokoro:
        """Kokoro(model_path, voices_path) would use onnxruntime's default SessionOptions,
        which on this machine profiled ~10% slower than intra_op_num_threads=4 (measured
        1,2,4,12 threads on a 12-core CPU; 4 won, 12 was worse - oversubscription overhead
        outweighs the extra cores for a model this size). Not a fix for CPU inference's
        per-sentence floor itself (see this module's docstring), just a small, real,
        no-downside win on top of it - applies to CPU-executed nodes either way, so it's
        left in regardless of which provider below ends up active.

        GPU (DirectML): tried first (unless force_cpu, set by load() after a failed
        warm-up) if this environment's onnxruntime build has it compiled in (only true if
        `onnxruntime-directml` is installed instead of plain `onnxruntime` - see
        requirements.txt/README for why that's a manual swap, not a normal pip install).
        "Available" only means DirectML support was compiled in, not that it will actually
        work on *this* machine and model - construction can fail outright (no DX12-capable
        GPU, bad driver: caught here) or can succeed and still fail on first real use
        (caught by load()'s warm-up check above). Either way, the provider actually
        running is logged - see _log_active_provider.
        """
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Confirmed by measurement (repeated create() calls on a live session, tracking
        # process RSS): ORT's default CPU memory arena grows to whatever the largest
        # input/intermediate tensor size seen so far needed, then *keeps* that memory
        # allocated for reuse - it never shrinks back down, by design, even long after a
        # long read is done and memory needs have dropped back to nothing. Over a session
        # with reads of varying length this looks exactly like "memory keeps climbing,"
        # even though nothing is actually leaked - it's a retained high-water mark, not
        # unbounded growth (repeating the *same* text forever holds perfectly flat).
        # Disabling the arena makes ORT allocate/free per node with the plain heap instead,
        # so RSS tracks *current* need rather than historical peak - measured this to cut
        # the plateau after a long read from ~1.7GB back down to a few hundred MB, at an
        # acceptable cost given each read already takes over a second of inference time.
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False

        session = None
        if not force_cpu and "DmlExecutionProvider" in rt.get_available_providers():
            try:
                session = rt.InferenceSession(str(self._model_path), sess_options=opts,
                                               providers=["DmlExecutionProvider", "CPUExecutionProvider"])
            except Exception:
                session = None  # falls through to the CPU-only attempt below
        if session is None:
            session = rt.InferenceSession(str(self._model_path), sess_options=opts,
                                           providers=["CPUExecutionProvider"])

        self._log_active_provider(session)
        return Kokoro.from_session(session, str(self._voices_path))

    @staticmethod
    def _log_active_provider(session: rt.InferenceSession) -> None:
        active = session.get_providers()
        message = (f"{datetime.now().isoformat(timespec='seconds')} "
                   f"Kokoro ONNX Runtime provider: {active[0]} (full list: {active})\n")
        print(message.strip())
        log_path = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent / "tts.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(message)
        except OSError:
            pass  # best-effort; a missing log write must not block synthesis

    def _warm_up(self) -> bool:
        """Runs once, right after the model loads (in Speaker's background preload thread
        at app startup, not on a hotkey press): a throwaway synthesis absorbs the espeak
        backend's one-time construction (now cached - see _patch_phonemizer_backend_caching)
        and ONNX Runtime's own first-inference graph warmup, so the first *real* press is
        as fast as every press after it, instead of paying those costs on the user's keypress.

        Returns whether it actually succeeded: load() uses that to detect a GPU provider
        that constructed a session fine but can't actually run this model (see load()).
        """
        try:
            self._kokoro.create(".", voice=self.voice, speed=1.0)
            return True
        except Exception as exc:
            print(f"Kokoro warm-up failed: {exc!r}")
            return False

    def list_voices(self) -> List[str]:
        if self._kokoro is not None:
            return self._kokoro.get_voices()
        return list(FALLBACK_VOICES)

    def synthesize(self, text: str, cancel: threading.Event) -> Iterator:
        """Yields the opening clause's audio as soon as it's ready, then each following
        piece in turn - Speaker._synthesize (speech.py, unchanged) already runs this
        generator on its own producer thread and feeds a queue that _play drains on a
        separate thread, so whatever this yields starts playing immediately while this
        loop keeps producing the rest in the background. cancel is checked before and
        after every create() call, so Stop takes effect between pieces (create() itself
        can't be interrupted mid-call) and stops queuing any further ones.
        """
        sentences = _split_into_sentences(text)
        pieces: List[str] = []
        if sentences:
            # Only the first sentence is split - a long opening sentence would otherwise
            # gate first audio on synthesizing all of it. Sentences after it stay whole
            # (one create() call each), so this adds at most one extra call total, not
            # one per sentence - see this module's docstring for why that call is costly.
            opening, remainder = _split_opening_clause(sentences[0])
            pieces.append(opening)
            if remainder:
                pieces.append(remainder)
            pieces.extend(sentences[1:])

        for i, piece in enumerate(pieces):
            if cancel.is_set():
                break
            samples, sample_rate = self._kokoro.create(piece, voice=self.voice, speed=self.speed)
            if cancel.is_set():
                break
            # A lone, independently-synthesized chunk that always borders silence (or the
            # very start/end of speech) on both sides, so it always needs both fades to
            # avoid a click - same reasoning as PiperBackend's segment boundaries.
            chunk = Faded(KokoroChunk(samples, sample_rate), fade_in=True, fade_out=True)
            yield chunk
            is_last = i == len(pieces) - 1
            if not is_last and not cancel.is_set():
                # piece's own trailing punctuation decides the pause: "," -> clause_pause_s
                # (the opening-clause split above), "." / "!" -> sentence_pause_s, "?" ->
                # question_pause_s - pause_for reads exactly that off the last character.
                pause = pause_for(piece[-1:], self.clause_pause_s, self.sentence_pause_s,
                                   self.question_pause_s)
                yield Silence(pause, sample_rate, KokoroChunk.sample_channels)
