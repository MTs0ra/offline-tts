import sys
import unittest
from pathlib import Path
from tkinter import Tk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
import kokoro_backend as kb


class FakeKokoroBackend:
    """Stand-in for kokoro_backend.KokoroBackend: no real ONNX session, so the test
    doesn't spend seconds loading the actual model just to check routing logic.
    """

    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = kb.DEFAULT_VOICE
        self.speed = 1.0
        self.clause_pause_s = 0.2
        self.sentence_pause_s = 0.5
        self.question_pause_s = 1.0
        self.loaded_with = []
        self.unloaded = False

    def load(self, voice_id):
        if voice_id:
            self.voice = voice_id
        self.loaded_with.append(voice_id)

    def unload(self):
        # Real KokoroBackend.unload() drops the session reference; loaded_with is call
        # history the tests assert against, not loaded state, so it's deliberately not
        # cleared here (matching the real backend, which keeps no such history at all).
        self.unloaded = True

    def list_voices(self):
        return list(kb.FALLBACK_VOICES)

    def synthesize(self, text, cancel):
        return iter(())


class TestVoiceMergingAndRouting(unittest.TestCase):
    def setUp(self):
        self._real_backend_cls = app.kokoro_backend.KokoroBackend
        app.kokoro_backend.KokoroBackend = FakeKokoroBackend
        self.root = Tk()
        self.root.withdraw()
        self.app = app.App(self.root)

    def tearDown(self):
        self.app._close()  # also destroys self.root
        app.kokoro_backend.KokoroBackend = self._real_backend_cls

    def test_kokoro_voices_listed_first_default_leading_piper_below(self):
        """Labels are compact now ("id · v#", no engine name in the text - see the info
        tooltip for what v1/v2 mean) so ordering/engine is checked via the choices dict's
        values, not by substring-matching the label text.
        """
        labels = list(self.app._voice_choices)
        first_engine, first_voice_id = self.app._voice_choices[labels[0]]
        self.assertEqual(first_engine, "kokoro")
        self.assertEqual(first_voice_id, kb.DEFAULT_VOICE)
        self.assertTrue(labels[0].startswith(kb.DEFAULT_VOICE))
        self.assertTrue(labels[0].endswith("v2"))

        engines = [self.app._voice_choices[label][0] for label in labels]
        kokoro_count = sum(1 for engine in engines if engine == "kokoro")
        self.assertTrue(all(engine == "kokoro" for engine in engines[:kokoro_count]))
        self.assertTrue(all(engine == "piper" for engine in engines[kokoro_count:]))

    def test_kokoro_is_the_default_engine_selected_on_launch(self):
        self.assertIs(self.app._speaker._backend, self.app._kokoro_backend)
        self.assertEqual(self.app._active_engine, "kokoro")

    def _label_for_engine(self, engine: str) -> str:
        return next(label for label, (eng, _voice_id) in self.app._voice_choices.items() if eng == engine)

    def test_selecting_a_piper_voice_routes_to_the_piper_backend(self):
        self.app._voice_name.set(self._label_for_engine("piper"))
        self.app._select_voice()
        self.assertIs(self.app._speaker._backend, self.app._piper_backend)
        self.assertEqual(self.app._active_engine, "piper")

    def test_switching_engines_unloads_the_one_switched_away_from(self):
        """Memory-reduction check: only one voice model should ever stay resident, so
        switching away from Kokoro must release it (see KokoroBackend.unload) rather than
        leaving both engines loaded at once.
        """
        self.app._voice_name.set(self._label_for_engine("piper"))
        self.app._select_voice()
        self.assertTrue(self.app._kokoro_backend.unloaded)

        self.app._voice_name.set(self._label_for_engine("kokoro"))
        self.app._select_voice()
        self.assertIsNone(self.app._piper_backend._voice)
        self.assertIsNone(self.app._piper_backend._voice_path)

    def test_selecting_a_kokoro_voice_routes_back_to_the_kokoro_backend(self):
        self.app._voice_name.set(self._label_for_engine("piper"))
        self.app._select_voice()

        self.app._voice_name.set(self._label_for_engine("kokoro"))
        self.app._select_voice()

        self.assertIs(self.app._speaker._backend, self.app._kokoro_backend)
        self.assertEqual(self.app._active_engine, "kokoro")
        self.assertIn(kb.DEFAULT_VOICE, self.app._kokoro_backend.loaded_with)

    def test_selecting_a_voice_plays_a_hi_i_am_preview(self):
        calls = []
        original_speak = app.Speaker.speak
        app.Speaker.speak = lambda self, text: calls.append(text)
        try:
            self.app._voice_name.set(self._label_for_engine("kokoro"))
            self.app._select_voice()
            self.app._preview_thread.join(timeout=2)  # preview speaks off the Tk thread
        finally:
            app.Speaker.speak = original_speak
        self.assertEqual(calls, ["Hi, I am Heart."])  # af_heart -> "Heart"

    def test_startup_voice_selection_does_not_play_a_preview(self):
        """_init_voice()'s own initial _select_voice() call must stay silent - a launch
        shouldn't immediately speak unprompted, only a deliberate dropdown pick should.
        """
        calls = []
        original_speak = app.Speaker.speak
        app.Speaker.speak = lambda self, text: calls.append(text)
        root2 = Tk()
        root2.withdraw()
        try:
            app2 = app.App(root2)
            try:
                self.assertEqual(calls, [])
            finally:
                app2._close()
        finally:
            app.Speaker.speak = original_speak


if __name__ == "__main__":
    unittest.main()
