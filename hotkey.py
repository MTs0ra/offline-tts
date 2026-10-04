"""System-wide hotkey via Win32 RegisterHotKey.

Unlike a low-level keyboard hook, RegisterHotKey never sees any other keystrokes,
is less likely to trip antivirus heuristics, and is released by the OS as soon as
stop() runs or the process exits.
"""

import ctypes
import threading
from ctypes import wintypes
from typing import Callable, Optional

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN = 0x1, 0x2, 0x4, 0x8
_MOD_NOREPEAT = 0x4000  # holding the keys down fires once, not on every auto-repeat
_WM_QUIT, _WM_HOTKEY = 0x0012, 0x0312
_HOTKEY_ID = 1
_READY_TIMEOUT = 2.0  # seconds to wait for the message-loop thread to attempt registration

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_user32.RegisterHotKey.restype = wintypes.BOOL
_user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
_user32.UnregisterHotKey.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = (ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
_user32.GetMessageW.restype = wintypes.BOOL
_user32.PostThreadMessageW.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_user32.PostThreadMessageW.restype = wintypes.BOOL
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD


class GlobalHotkey:
    """Calls callback on a dedicated thread each time the key combination is pressed."""

    def __init__(self, modifiers: int, virtual_key: int, callback: Callable[[], None]):
        self._modifiers = modifiers | _MOD_NOREPEAT
        self._virtual_key = virtual_key
        self._callback = callback
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._registered = False

    def start(self) -> bool:
        """Register the hotkey. Returns False if another program already owns it, or if
        the message-loop thread didn't confirm registration within _READY_TIMEOUT (should
        never happen in practice - RegisterHotKey returns almost immediately - but an
        unbounded wait here would let a stuck thread hang the caller forever).

        Safe to call again after a failed attempt (e.g. once the other program has
        closed): a Thread can only ever be started once, so a failed attempt's
        thread (which exits immediately) is replaced rather than restarted.
        """
        if self._thread is None or not self._thread.is_alive():
            self._ready = threading.Event()
            self._thread = threading.Thread(target=self._message_loop, daemon=True)
            self._thread.start()
            if not self._ready.wait(timeout=_READY_TIMEOUT):
                return False
        return self._registered

    def stop(self) -> None:
        if self._registered:
            _user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
            self._thread.join(timeout=2)

    def _message_loop(self) -> None:
        # The hotkey belongs to the thread that registers it and arrives in that
        # thread's message queue, so registration, dispatch and release all happen here.
        self._thread_id = _kernel32.GetCurrentThreadId()
        self._registered = bool(_user32.RegisterHotKey(None, _HOTKEY_ID, self._modifiers, self._virtual_key))
        self._ready.set()
        if not self._registered:
            return
        msg = wintypes.MSG()
        try:
            while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:  # 0 = WM_QUIT, -1 = error
                if msg.message == _WM_HOTKEY:
                    self._callback()
        finally:
            _user32.UnregisterHotKey(None, _HOTKEY_ID)
