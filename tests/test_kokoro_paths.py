import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).write_bytes(b"x")


class TestResolveKokoroModel(unittest.TestCase):
    def resolve(self, folder):
        return app.resolve_kokoro_file(folder, app.KOKORO_MODEL_NAMES)

    def test_renamed_fp32_is_used_when_present(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro__v1.0-fp32__20260911.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__v1.0-fp32__20260911.onnx")

    def test_original_download_name_is_accepted(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro-v1.0.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro-v1.0.onnx")

    def test_renamed_name_wins_over_original_when_both_exist(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro-v1.0.onnx", "kokoro__v1.0-fp32__20260911.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__v1.0-fp32__20260911.onnx")

    def test_fp32_is_preferred_over_int8(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro-v1.0.onnx")

    def test_original_int8_is_used_when_it_is_the_only_model(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro-v1.0.int8.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro-v1.0.int8.onnx")

    def test_renamed_int8_is_used_when_it_is_the_only_model(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro__v1.0-int8__20260911.onnx")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__v1.0-int8__20260911.onnx")

    def test_missing_files_fall_back_to_the_preferred_renamed_path(self):
        with TemporaryDirectory() as d:
            self.assertEqual(self.resolve(Path(d)),
                             Path(d) / "kokoro__v1.0-fp32__20260911.onnx")


class TestResolveKokoroVoices(unittest.TestCase):
    def resolve(self, folder):
        return app.resolve_kokoro_file(folder, app.KOKORO_VOICES_NAMES)

    def test_renamed_voices_file_is_accepted(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "kokoro__voices-v1.0__20260911.bin")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__voices-v1.0__20260911.bin")

    def test_original_download_name_is_accepted(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "voices-v1.0.bin")
            self.assertEqual(self.resolve(Path(d)).name, "voices-v1.0.bin")

    def test_renamed_name_wins_over_original_when_both_exist(self):
        with TemporaryDirectory() as d:
            _touch(Path(d), "voices-v1.0.bin", "kokoro__voices-v1.0__20260911.bin")
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__voices-v1.0__20260911.bin")

    def test_missing_file_falls_back_to_the_preferred_renamed_path(self):
        with TemporaryDirectory() as d:
            self.assertEqual(self.resolve(Path(d)),
                             Path(d) / "kokoro__voices-v1.0__20260911.bin")

    def test_a_directory_with_a_matching_name_is_not_a_file(self):
        with TemporaryDirectory() as d:
            (Path(d) / "voices-v1.0.bin").mkdir()
            self.assertEqual(self.resolve(Path(d)).name, "kokoro__voices-v1.0__20260911.bin")


if __name__ == "__main__":
    unittest.main()
