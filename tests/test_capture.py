import unittest
from unittest import mock

import capture


class FakeClipboard:
    def __init__(self, text):
        self.text = text

    def paste(self):
        return self.text

    def copy(self, text):
        self.text = text


class TestCopySelection(unittest.TestCase):
    def _run(self, original, selection, copy_happens=True):
        clip = FakeClipboard(original)

        def send_ctrl_c():
            if copy_happens:
                clip.text = selection

        with mock.patch.object(capture, "pyperclip", clip), \
             mock.patch.object(capture, "_user32"), \
             mock.patch.object(capture, "_send_ctrl_c", send_ctrl_c), \
             mock.patch.object(capture, "_wait_until", side_effect=[True, copy_happens]), \
             mock.patch.object(capture.time, "sleep"):
            result = capture._copy_selection()
        return result, clip.text

    def test_restores_a_non_empty_clipboard(self):
        self.assertEqual(self._run("old", "picked"), (("picked", "old"), "old"))

    def test_restores_an_empty_clipboard(self):
        self.assertEqual(self._run("", "picked"), (("picked", ""), ""))

    def test_no_selection_leaves_clipboard_and_returns_none(self):
        self.assertEqual(self._run("old", "picked", copy_happens=False), ((None, "old"), "old"))


if __name__ == "__main__":
    unittest.main()
