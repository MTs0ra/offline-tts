import sys
import threading
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speech
from speech import Faded, PiperBackend, Silence, pause_for, split_into_segments


class FakeChunk:
    """Stand-in for piper.voice.AudioChunk: a constant-amplitude span, so a fade's
    effect on the edges (and lack of effect on the untouched middle) is easy to assert.
    """

    def __init__(self, value=1000, frames=1000, sample_rate=22050, sample_channels=1):
        self.sample_rate = sample_rate
        self.sample_channels = sample_channels
        self.sample_width = 2
        self.audio_int16_bytes = np.full(frames * sample_channels, value, dtype=np.int16).tobytes()


class FakeVoice:
    """Always returns three chunks per segment: first, an untouched middle, and last -
    enough to prove PiperBackend fades only the two chunks that actually border silence.
    """

    def synthesize(self, _text):
        return [FakeChunk(), FakeChunk(), FakeChunk()]


class TestPauseAndSegments(unittest.TestCase):
    def test_pause_for_by_punctuation(self):
        self.assertEqual(pause_for("?"), speech.PAUSE_QUESTION_S)
        self.assertEqual(pause_for("."), speech.PAUSE_SENTENCE_S)
        self.assertEqual(pause_for("!"), speech.PAUSE_SENTENCE_S)
        self.assertEqual(pause_for(","), speech.PAUSE_CLAUSE_S)

    def test_pause_for_custom_lengths(self):
        self.assertEqual(pause_for(",", clause_s=0.9), 0.9)

    def test_split_into_segments_pauses_and_last_segment(self):
        segments = split_into_segments("Hello world, this is fine.")
        self.assertEqual([s for s, _ in segments], ["Hello world,", "this is fine."])
        self.assertEqual(segments[0][1], speech.PAUSE_CLAUSE_S)
        self.assertEqual(segments[-1][1], 0.0)  # never ends in silence

    def test_split_into_segments_no_punctuation(self):
        self.assertEqual(split_into_segments("just one clause"), [("just one clause", 0.0)])


class TestSilence(unittest.TestCase):
    def test_byte_length_matches_duration(self):
        silence = Silence(0.5, sample_rate=22050, sample_channels=1)
        self.assertEqual(len(silence.audio_int16_bytes), int(0.5 * 22050) * 2)
        self.assertEqual(silence.audio_int16_bytes, bytes(len(silence.audio_int16_bytes)))


class TestFaded(unittest.TestCase):
    def test_fade_in_ramps_from_zero_leaves_tail_untouched(self):
        chunk = FakeChunk(value=1000, frames=1000)
        decoded = np.frombuffer(Faded(chunk, fade_in=True, fade_out=False).audio_int16_bytes, dtype=np.int16)
        self.assertEqual(decoded[0], 0)
        self.assertGreater(decoded[10], 0)
        self.assertEqual(decoded[-1], 1000)

    def test_fade_out_ramps_to_zero_leaves_head_untouched(self):
        chunk = FakeChunk(value=1000, frames=1000)
        decoded = np.frombuffer(Faded(chunk, fade_in=False, fade_out=True).audio_int16_bytes, dtype=np.int16)
        self.assertEqual(decoded[0], 1000)
        self.assertEqual(decoded[-1], 0)

    def test_short_chunk_does_not_crash(self):
        chunk = FakeChunk(value=500, frames=4)
        faded = Faded(chunk, fade_in=True, fade_out=True)
        self.assertEqual(len(faded.audio_int16_bytes), len(chunk.audio_int16_bytes))


class TestPiperBackendBoundaryFades(unittest.TestCase):
    def test_only_segment_edges_are_faded(self):
        backend = PiperBackend()
        backend._voice = FakeVoice()
        chunks = list(backend.synthesize("Hello world, this is fine.", threading.Event()))

        # 2 segments * 3 chunks each, plus one Silence between them.
        self.assertEqual(len(chunks), 7)
        first_seg, silence, second_seg = chunks[0:3], chunks[3], chunks[4:7]
        self.assertIsInstance(silence, Silence)

        for segment in (first_seg, second_seg):
            self.assertIsInstance(segment[0], Faded)     # borders silence (or speech start)
            self.assertIs(type(segment[1]), FakeChunk)   # continuous mid-sentence audio: untouched
            self.assertIsInstance(segment[2], Faded)      # borders silence (or speech end)

    def test_cancel_stops_iteration(self):
        backend = PiperBackend()
        backend._voice = FakeVoice()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(backend.synthesize("Hello world, this is fine.", cancel)), [])

    def test_unload_clears_voice_and_path_so_the_next_load_reloads(self):
        backend = PiperBackend()
        backend._voice = FakeVoice()
        backend._voice_path = Path("some-voice.onnx")

        backend.unload()

        self.assertIsNone(backend._voice)
        self.assertIsNone(backend._voice_path)


if __name__ == "__main__":
    unittest.main()
