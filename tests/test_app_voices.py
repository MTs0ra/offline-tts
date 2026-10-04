import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app


class TestSpokenName(unittest.TestCase):
    def test_kokoro_voice_id_uses_the_part_after_the_underscore(self):
        self.assertEqual(app._spoken_name("kokoro", "af_heart"), "Heart")
        self.assertEqual(app._spoken_name("kokoro", "bm_george"), "George")

    def test_piper_voice_path_uses_the_middle_hyphen_segment(self):
        path = Path("piper__en_US-lessac-medium__20260910.onnx")
        self.assertEqual(app._spoken_name("piper", path), "Lessac")

    def test_piper_voice_with_underscore_in_its_name_segment(self):
        path = Path("piper__en_GB-jenny_dioco-medium__20260910.onnx")
        self.assertEqual(app._spoken_name("piper", path), "Jenny")


class TestDisplayName(unittest.TestCase):
    def test_dated_filename_strips_bookkeeping_to_a_plain_id(self):
        """The dropdown shows a compact "id · v#" now (date lives in voices.txt only)."""
        self.assertEqual(app._display_name("piper__en_US-lessac-medium__20260910"),
                          "en_US-lessac-medium")

    def test_unrecognized_stem_passed_through(self):
        self.assertEqual(app._display_name("some_other_file"), "some_other_file")


class TestFindVoices(unittest.TestCase):
    def test_reads_renamed_files_not_a_stale_copy(self):
        """Regression test for the stale dist\\TTS\\voices bug: find_voices must reflect
        whatever is actually in the folder it's given, not some other cached listing.
        """
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "piper__en_US-amy-medium__20260910.onnx").write_bytes(b"")
            (folder / "piper__en_US-amy-medium__20260910.onnx.json").write_bytes(b"{}")
            (folder / "piper__orphan-no-config__20260910.onnx").write_bytes(b"")  # no .json: excluded

            voices = app.find_voices(folder)

            self.assertEqual(list(voices), ["en_US-amy-medium"])
            self.assertEqual(voices["en_US-amy-medium"],
                              folder / "piper__en_US-amy-medium__20260910.onnx")


if __name__ == "__main__":
    unittest.main()
