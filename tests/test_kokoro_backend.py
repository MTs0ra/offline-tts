import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kokoro_backend as kb
from speech import Faded, Silence


class FakeKokoro:
    """Stand-in for kokoro_onnx.Kokoro: no model files, no ONNX session, just enough
    to exercise KokoroBackend's own logic (voice/speed plumbing, segment boundaries).
    """

    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.voices_path = voices_path
        self.create_calls = []

    @classmethod
    def from_session(cls, session, voices_path, espeak_config=None):
        return cls(session, voices_path)  # keeps the session reachable for provider assertions

    def create(self, text, voice, speed):
        self.create_calls.append(text)
        return np.full(2000, 0.5, dtype=np.float32), 24000

    def get_voices(self):
        return ["af_heart", "af_bella"]


class FakeSession:
    """Stand-in for onnxruntime.InferenceSession: echoes back whatever `providers` it was
    constructed with, so _log_active_provider (and tests) can tell which attempt this was.
    """

    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return self._providers


class FlakyGpuKokoro(FakeKokoro):
    """Simulates the real DirectML failure mode found in testing: session construction
    succeeds, but the *first* create() call (the warm-up) raises, as if the provider
    initialized fine yet failed on actual first use. A second instance (after load()'s
    rebuild-CPU-only fallback) works normally.
    """

    instances_created = 0

    def __init__(self, model_path, voices_path):
        super().__init__(model_path, voices_path)
        FlakyGpuKokoro.instances_created += 1
        self._is_first_instance = FlakyGpuKokoro.instances_created == 1

    def create(self, text, voice, speed):
        if self._is_first_instance:
            raise RuntimeError("simulated DirectML ConvTranspose failure")
        return super().create(text, voice, speed)


class TestKokoroBackendLoad(unittest.TestCase):
    def setUp(self):
        self._real_kokoro = kb.Kokoro
        self._real_session = kb.rt.InferenceSession
        self._real_get_available_providers = kb.rt.get_available_providers
        kb.Kokoro = FakeKokoro
        kb.rt.InferenceSession = lambda model_path, sess_options=None, providers=None: FakeSession(providers)
        kb.rt.get_available_providers = lambda: ["CPUExecutionProvider"]
        FlakyGpuKokoro.instances_created = 0

    def tearDown(self):
        kb.Kokoro = self._real_kokoro
        kb.rt.InferenceSession = self._real_session
        kb.rt.get_available_providers = self._real_get_available_providers

    def test_gpu_provider_that_fails_on_first_use_falls_back_to_cpu(self):
        """Regression test for the real DirectML failure mode found in testing: session
        construction can succeed while the model still can't actually run on that
        provider. load() must detect that (via a failed warm-up) and end up on a working
        CPU session instead of leaving the backend silently broken.
        """
        kb.rt.get_available_providers = lambda: ["DmlExecutionProvider", "CPUExecutionProvider"]
        kb.Kokoro = FlakyGpuKokoro

        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)

        self.assertEqual(FlakyGpuKokoro.instances_created, 2)  # GPU attempt, then CPU rebuild
        self.assertEqual(backend._kokoro.model_path._providers, ["CPUExecutionProvider"])
        # The backend must be left in a genuinely working state, not just "not crashed".
        chunks = list(backend.synthesize("It works now.", threading.Event()))
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], Faded)

    def test_gpu_provider_unavailable_goes_straight_to_cpu(self):
        """When onnxruntime-directml isn't installed (the default - see requirements.txt),
        DmlExecutionProvider isn't even offered, so only one session is ever built.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))  # setUp's get_available_providers is CPU-only
        backend.load(kb.DEFAULT_VOICE)
        self.assertEqual(backend._kokoro.model_path._providers, ["CPUExecutionProvider"])

    def test_model_loaded_lazily_and_cached(self):
        backend = kb.KokoroBackend(Path("model.onnx"), Path("voices.bin"))
        self.assertIsNone(backend._kokoro)
        backend.load("af_bella")
        self.assertEqual(backend.voice, "af_bella")
        loaded = backend._kokoro
        backend.load(None)  # e.g. Speaker re-calling load() before a session with no voice_id
        self.assertIs(backend._kokoro, loaded)  # not reconstructed
        self.assertEqual(backend.voice, "af_bella")  # unchanged by a None voice_id

    def test_list_voices_falls_back_before_load(self):
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        self.assertEqual(backend.list_voices(), kb.FALLBACK_VOICES)
        backend.load(kb.DEFAULT_VOICE)
        self.assertEqual(backend.list_voices(), ["af_heart", "af_bella"])

    def test_unload_releases_the_session_and_forces_a_full_reload(self):
        """Memory-reduction check: unload() must actually drop the reference (so the ONNX
        session/model can be garbage-collected) rather than merely resetting a flag - the
        clearest proof is that the next load() rebuilds a session from scratch (a second
        _warm_up-triggering create() call) instead of reusing the old one.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        first_session = backend._kokoro
        self.assertIsNotNone(first_session)

        backend.unload()
        self.assertIsNone(backend._kokoro)

        backend.load(kb.DEFAULT_VOICE)
        self.assertIsNotNone(backend._kokoro)
        self.assertIsNot(backend._kokoro, first_session)  # a genuinely new session, not reused

    def test_opening_sentence_is_split_at_its_first_comma_when_long_enough(self):
        """The opening sentence is split at its first clause boundary so a short first
        chunk is available quickly, instead of gating first audio on the whole sentence -
        but only when that opening clause is long enough to be worth it (see
        _MIN_OPENING_CLAUSE_WORDS); this one has 7 words before the comma.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        backend._kokoro.create_calls.clear()  # drop the "." from _warm_up()'s own create() call
        text = "Given everything you explained to us earlier, this is fine."
        chunks = list(backend.synthesize(text, threading.Event()))
        self.assertEqual([type(c).__name__ for c in chunks], ["Faded", "Silence", "Faded"])
        self.assertEqual(backend._kokoro.create_calls,
                          ["Given everything you explained to us earlier,", "this is fine."])
        # The pause between the opening clause and the rest of that sentence should be
        # the short clause pause, not the longer full-sentence pause.
        expected_frames = int(backend.clause_pause_s * 24000)
        self.assertEqual(len(chunks[1].audio_int16_bytes), expected_frames * chunks[1].sample_width)

    def test_short_opening_clause_is_not_split_to_avoid_a_playback_stall(self):
        """Regression test for "long pause after the first couple of words": a short
        opening clause (below _MIN_OPENING_CLAUSE_WORDS) used to still get split off, but
        its own brief playback finished long before the next piece's same-fixed-cost
        create() call could catch up, leaving real dead air. Below the word-count floor,
        the whole first sentence must stay one piece instead - see _split_opening_clause.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        backend._kokoro.create_calls.clear()  # drop the "." from _warm_up()'s own create() call
        text = "Well, this is going to work out just fine in the end."
        chunks = list(backend.synthesize(text, threading.Event()))
        self.assertEqual([type(c).__name__ for c in chunks], ["Faded"])  # one piece, no mid-sentence pause
        self.assertEqual(backend._kokoro.create_calls, [text])

    def test_only_the_opening_sentence_is_split_not_the_rest(self):
        """A comma in a *later* sentence must not create an extra synthesis chunk boundary
        (unlike Piper's clause splitting) - each Kokoro.create() call carries a large fixed
        cost, so splitting every sentence at every comma would pay that cost repeatedly for
        no benefit. Only the opening sentence is worth shortening for first-audio latency.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        backend._kokoro.create_calls.clear()  # drop the "." from _warm_up()'s own create() call
        text = "Given everything you explained to us earlier, this is fine. Second one, with a comma too."
        chunks = list(backend.synthesize(text, threading.Event()))
        self.assertEqual(backend._kokoro.create_calls,
                          ["Given everything you explained to us earlier,", "this is fine.",
                           "Second one, with a comma too."])
        self.assertEqual([type(c).__name__ for c in chunks],
                          ["Faded", "Silence", "Faded", "Silence", "Faded"])

    def test_opening_sentence_without_a_comma_is_not_split(self):
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        backend._kokoro.create_calls.clear()
        chunks = list(backend.synthesize("Hello world.", threading.Event()))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(backend._kokoro.create_calls, ["Hello world."])

    def test_synthesize_streams_sentence_by_sentence(self):
        """The first sentence's chunk must be yielded (available to Speaker for immediate
        playback) before the second sentence is even synthesized - that's what lets
        Speaker._play start audio while this generator is still producing the rest.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        fake = backend._kokoro
        fake.create_calls.clear()  # drop the "." from _warm_up()'s own create() call
        gen = backend.synthesize("Hello world. This is fine. A third sentence.", threading.Event())

        first_chunk = next(gen)
        self.assertIsInstance(first_chunk, Faded)
        self.assertEqual(fake.create_calls, ["Hello world."])  # only the first was synthesized so far

        remaining = list(gen)
        # Silence, Faded, Silence, Faded - playback order preserved, one Silence between
        # each pair of sentences, none after the last.
        self.assertEqual([type(c).__name__ for c in remaining], ["Silence", "Faded", "Silence", "Faded"])
        self.assertEqual(fake.create_calls, ["Hello world.", "This is fine.", "A third sentence."])

    def test_question_mark_gets_the_longer_question_pause(self):
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        chunks = list(backend.synthesize("Really? Yes.", threading.Event()))
        silence = chunks[1]
        self.assertIsInstance(silence, Silence)
        expected_frames = int(backend.question_pause_s * 24000)
        self.assertEqual(len(silence.audio_int16_bytes), expected_frames * silence.sample_width)

    def test_cancel_stops_iteration(self):
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(backend.synthesize("Hello world.", cancel)), [])

    def test_cancel_after_first_sentence_stops_pending_background_synthesis(self):
        """Regression test for "stop hotkey cancels pending background synthesis, not just
        current playback": Speaker/Stop sets this same cancel Event mid-stream, and the
        generator must not go on to synthesize sentences 2+ once it's set.
        """
        backend = kb.KokoroBackend(Path("m"), Path("v"))
        backend.load(kb.DEFAULT_VOICE)
        fake = backend._kokoro
        fake.create_calls.clear()  # drop the "." from _warm_up()'s own create() call
        cancel = threading.Event()
        gen = backend.synthesize("Hello world. This is fine. A third sentence.", cancel)

        next(gen)  # first sentence's chunk
        cancel.set()  # simulate Stop being pressed while sentence 1 plays
        self.assertEqual(list(gen), [])  # no Silence, no further chunks
        self.assertEqual(fake.create_calls, ["Hello world."])  # sentences 2 and 3 never synthesized


class TestKokoroChunk(unittest.TestCase):
    def test_normal_signal_uses_full_scale_unattenuated(self):
        samples = np.array([0.5, -0.5, 0.9], dtype=np.float32)
        decoded = np.frombuffer(kb.KokoroChunk(samples, 24000).audio_int16_bytes, dtype=np.int16)
        self.assertEqual(decoded[0], round(0.5 * 32767))
        self.assertEqual(decoded[2], round(0.9 * 32767))

    def test_out_of_range_peak_is_scaled_down_not_hard_clipped(self):
        samples = np.array([1.5, -1.5, 0.75], dtype=np.float32)  # peak exceeds [-1, 1]
        decoded = np.frombuffer(kb.KokoroChunk(samples, 24000).audio_int16_bytes, dtype=np.int16)
        self.assertEqual(decoded[0], 32767)
        self.assertEqual(decoded[1], -32767)
        self.assertAlmostEqual(int(decoded[2]), round(0.75 / 1.5 * 32767), delta=1)


if __name__ == "__main__":
    unittest.main()
