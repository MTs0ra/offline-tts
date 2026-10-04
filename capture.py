"""Get the text to read: the current selection in any app, or the clipboard as a fallback."""

import ctypes
import time
from ctypes import wintypes
from typing import Callable, Optional, Tuple

import pyperclip

_MODIFIER_KEYS = (0x10, 0x11, 0x12, 0x5B, 0x5C)  # Shift, Ctrl, Alt, left Win, right Win
_VK_CONTROL, _VK_C = 0x11, 0x43
_INPUT_KEYBOARD, _KEYEVENTF_KEYUP = 1, 0x0002
_RELEASE_TIMEOUT = 1.5  # seconds to wait for the user to let go of the hotkey
_COPY_TIMEOUT = 0.6     # seconds to wait for the foreground app to answer Ctrl+C
_POLL_INTERVAL = 0.01


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t))


class _MOUSEINPUT(ctypes.Structure):
    # Never sent; it is the largest union member, so it gives INPUT the size SendInput checks.
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t))


class _INPUT(ctypes.Structure):
    class _Union(ctypes.Union):
        _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT))

    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _Union))


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT
_user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
_user32.GetAsyncKeyState.restype = wintypes.SHORT
_user32.GetClipboardSequenceNumber.restype = wintypes.DWORD


def clipboard_text() -> str:
    return pyperclip.paste()


def selection_or_clipboard() -> str:
    """Copy the foreground app's selection. If nothing is copied, return the clipboard unchanged."""
    selection, original = _copy_selection()
    return original if selection is None else selection


def selection_only() -> Optional[str]:
    """Copy the foreground app's selection. Returns None if nothing new was copied (no selection)."""
    selection, _original = _copy_selection()
    return selection


def _copy_selection() -> Tuple[Optional[str], str]:
    """Simulate Ctrl+C in the foreground app. Returns (new_text_or_None, original_clipboard).

    The clipboard is restored to its original content before returning either way; non-text
    content that was on it can't be restored, because the copy itself replaces it.
    """
    # Ctrl+C sent while Alt is still held would arrive as Ctrl+Alt+C and copy nothing.
    _wait_until(lambda: not any(_user32.GetAsyncKeyState(vk) & 0x8000 for vk in _MODIFIER_KEYS),
                _RELEASE_TIMEOUT)
    original = pyperclip.paste()
    sequence = _user32.GetClipboardSequenceNumber()
    _send_ctrl_c()
    if not _wait_until(lambda: _user32.GetClipboardSequenceNumber() != sequence, _COPY_TIMEOUT):
        return None, original
    time.sleep(0.05)  # the source app may still hold the clipboard open while it finishes writing
    selection = pyperclip.paste()
    if original:
        pyperclip.copy(original)
    return selection, original


def _send_ctrl_c() -> None:
    keys = ((_VK_CONTROL, 0), (_VK_C, 0), (_VK_C, _KEYEVENTF_KEYUP), (_VK_CONTROL, _KEYEVENTF_KEYUP))
    events = (_INPUT * len(keys))(
        *(_INPUT(type=_INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=vk, dwFlags=flags)) for vk, flags in keys)
    )
    _user32.SendInput(len(events), events, ctypes.sizeof(_INPUT))


def _wait_until(condition: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL)
    return True
