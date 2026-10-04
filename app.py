"""TTS: select text anywhere, press F2, hear it. Runs fully offline.

This file is the UI layer only. Synthesis (speech.py, kokoro_backend.py) and the global
hotkey (hotkey.py) were unchanged by the UI redesign - the UI just drives the same
Speaker/backend API it always did (speak(), stop(), set_voice(), set_backend()). A later
responsiveness pass did touch hotkey.py itself (a bounded wait in GlobalHotkey.start())
alongside this file - see _close(), _select_voice(), _on_hotkey(), and _poll_events().
"""

import ctypes
import queue
import re
import sys
import threading
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

import capture
from hotkey import MOD_ALT, MOD_CONTROL, GlobalHotkey
from speech import PiperBackend, Speaker
from ui_widgets import CanvasTooltip, CloseButton, DayNightToggle, DragHandle, GearIcon, HexButton, \
    HexGradientBackground, MinimizeButton, TRANSPARENT_KEY, draw_logo, lerp_color, make_app_window, \
    retheme_logo, set_window_icon

try:
    import kokoro_backend
except Exception:
    # Anything from a missing kokoro-onnx install to a PyInstaller build that didn't bundle
    # one of its data files (see build.bat) lands here - Piper must keep working regardless.
    kokoro_backend = None
    _log_path = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent / "tts.log"
    _message = f"{datetime.now().isoformat(timespec='seconds')} Kokoro engine disabled - import failed:\n"
    _message += traceback.format_exc()
    print(_message)
    try:
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(_message)
    except OSError:
        pass  # best-effort; a missing log write must not also take Piper down

# Change both lines together to use a different key (see hotkey.py for modifier flags).
HOTKEY = (0, 0x71)  # 0x71 is the F2 virtual-key code; 0 means no modifier is required
HOTKEY_LABEL = "F2"

# Restores and refocuses the window from anywhere - the only way back once minimized, since
# the borderless window (see _build_ui) has no taskbar button or Alt-Tab entry to click
# instead (see _minimize's docstring). 0x54 is 'T', chosen to mnemonically match "TTS" and
# because Ctrl+Alt+T isn't a standard Windows-reserved combo (unlike, say, Ctrl+Shift+T,
# which browsers already bind to reopen a closed tab).
RESTORE_HOTKEY = (MOD_CONTROL | MOD_ALT, 0x54)
RESTORE_HOTKEY_LABEL = "Ctrl+Alt+T"
STATUS_POLL_MS = 100

# A true regular hexagon (pointy top/bottom, flat vertical sides) needs waist=0.25 and
# height = width * 2/sqrt(3) - that's the only waist/aspect-ratio pair where all six sides
# come out equal (see ui_widgets.hexagon_points). 346x400 is the closest integer pixel pair
# to that ratio (346 * 2/sqrt(3) = 399.5).
WINDOW_WIDTH, WINDOW_HEIGHT = 346, 400
HEX_WAIST = 0.25

BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_GAP = 96, 42, 18  # Start/Read/Stop: one row, arch layout
ARCH_RISE = 26  # how far Start sits above Read/Stop's shared row

# The sunset scene (light mode only - see HexGradientBackground's class docstring for how
# the dome/bloom/clouds/horizon all fade and drift toward the night sky during a toggle,
# entirely as a function of App._on_theme_progress's amount, no separate animation loop).
# The dome sits centered ON the hexagon's own bottom point (SUN_Y_LIGHT == WINDOW_HEIGHT) so
# only its upper arc is ever visible - the rest falls off the canvas's bottom edge for free.
SUN_COLOR = "#c1301f"        # the dome itself: a deep sunset red, not orange
HORIZON_COLOR = "#fff1de"
CLOUD_COLOR = "#fff4e6"
SUN_RADIUS = 85
SUN_Y_LIGHT = WINDOW_HEIGHT  # sits right at the hexagon's bottom point
SUN_SINK = 95                # sinks fully below the canvas edge by the time a toggle finishes

# Stars start fading in at this fraction of the light-to-dark crossfade - well before amount
# reaches 1, so they're already gathering while the sun is still sinking, not just snapping
# in once the toggle is already done. Still just a function of amount, computed inline in
# App._on_theme_progress - see HexGradientBackground.update_stars().
STAR_FADE_START = 0.5

# The gear's settings panel: embedded directly in the main canvas (not a separate popup
# window) and confined to SETTINGS_WIDTH x whatever height its content needs, positioned so
# that box sits entirely inside the hexagon's full-width flat band (see _toggle_settings) -
# unlike a floating Toplevel, this can't ever show rectangular corners poking past the
# hexagon's slanted edges.
SETTINGS_WIDTH = 220  # close to the content's own natural width (~211px) - just enough
SETTINGS_TOP = 176  # just below the gear/toggle row
SETTINGS_ANIM_MS = 220
SETTINGS_ANIM_FRAMES = 12

# Two palettes; App crossfades between them frame-by-frame while the dark-mode toggle
# animates (see App._on_theme_progress), so the whole window "sunsets"/"moonrises", not
# just the switch itself. grad_top/grad_mid/grad_bottom are the window's own 3-stop vertical
# gradient (see HexGradientBackground.update()/_gradient_color) - a bluish sky near the top
# corners warming through a horizon band to deep sunset orange at the bottom in light mode;
# a uniformly dark, star-flecked night in dark mode (stars toggled in App._on_theme_done).
LIGHT = dict(
    panel="#fffaf5", text="#3a2c22", text_secondary="#8a7666",
    accent="#5eb8aa", accent_hover="#72c7ba", accent_press="#48897f",
    danger="#e8887a", danger_hover="#ee9d92", danger_press="#c96a5c",
    button_text="#ffffff", outline="#e7c2a0",
    grad_top="#6f93c4", grad_mid="#f0ba82", grad_bottom="#c9541f",
)
DARK = dict(
    panel="#262b39", text="#eef1f8", text_secondary="#9aa3bb",
    accent="#4fa79a", accent_hover="#63bbad", accent_press="#3d8a7e",
    danger="#c97b70", danger_hover="#d68f85", danger_press="#a8635a",
    button_text="#f5f7fa", outline="#3a4055",
    grad_top="#0d1020", grad_mid="#141b2c", grad_bottom="#1c2338",
)
FONT = ("Segoe UI", 10)          # no bold anywhere on the buttons - "lighter, less bold"
FONT_SMALL = ("Segoe UI", 9)


def app_dir() -> Path:
    """The folder containing this script, or the .exe once packaged by PyInstaller."""
    return Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent


VOICES_DIR = app_dir() / "voices"
KOKORO_DIR = app_dir() / "models" / "kokoro"
# fp32, not int8: measured 3-3.5x faster per sentence on this CPU (ConvInteger, int8's
# quantized conv kernel, isn't well vectorized here - see voices.txt). int8 file is kept
# on disk, just no longer the active default.
# Each list is in order of preference: the dated names this project uses, then the names the
# files carry when downloaded straight from the kokoro-onnx model-files-v1.0 release, so a
# fresh clone needs no manual rename. fp32 always beats int8 when both are present.
KOKORO_MODEL_NAMES = (
    "kokoro__v1.0-fp32__20260911.onnx",
    "kokoro-v1.0.onnx",
    "kokoro__v1.0-int8__20260911.onnx",
    "kokoro-v1.0.int8.onnx",
)
KOKORO_VOICES_NAMES = (
    "kokoro__voices-v1.0__20260911.bin",
    "voices-v1.0.bin",
)


def resolve_kokoro_file(kokoro_dir: Path, names: tuple) -> Path:
    """First existing file from `names` in `kokoro_dir`; the preferred name if none exist."""
    for name in names:
        if (kokoro_dir / name).is_file():
            return kokoro_dir / name
    return kokoro_dir / names[0]


KOKORO_MODEL_PATH = resolve_kokoro_file(KOKORO_DIR, KOKORO_MODEL_NAMES)
KOKORO_VOICES_PATH = resolve_kokoro_file(KOKORO_DIR, KOKORO_VOICES_NAMES)

# Files on disk follow the dated "engine__id__YYYYMMDD" scheme documented in voices.txt
# (see that file for what each one sounds like and when it was added, and for the full
# name<->version mapping); _display_name strips that bookkeeping back to a plain id for
# the compact Voice dropdown - the dropdown shows just "id - v#" (see App._build_voice_choices).
_FILE_NAME_RE = re.compile(r"^[a-z]+__(?P<id>.+)__\d{8}$")


def _display_name(stem: str) -> str:
    match = _FILE_NAME_RE.match(stem)
    return match.group("id") if match else stem


def find_voices(folder: Path) -> dict:
    """Map display name to model path. A voice needs both name.onnx and name.onnx.json."""
    return {_display_name(p.stem): p for p in sorted(folder.glob("*.onnx"))
            if p.with_suffix(".onnx.json").is_file()}


def _spoken_name(engine: str, voice_id) -> str:
    """A natural first name to say in the "Hi, I am ___" preview - Kokoro's "af_heart" ->
    "Heart", Piper's "en_US-lessac-medium" -> "Lessac" - rather than reading the raw id.
    """
    if engine == "kokoro":
        base = voice_id.split("_", 1)[-1] if "_" in voice_id else voice_id
    else:
        stem = _display_name(voice_id.stem)
        parts = stem.split("-")
        base = parts[1] if len(parts) >= 2 else stem
        base = base.split("_")[0]  # "jenny_dioco" -> "jenny"
    return base.capitalize()


# Tuning dropdown presets: (label, value). Collapsed from the old open sliders into
# closed dropdowns per the redesign; values still land on the same KokoroBackend attributes.
SPEED_OPTIONS = [("0.75x", 0.75), ("1.0x", 1.0), ("1.25x", 1.25), ("1.5x", 1.5), ("2.0x", 2.0)]
SENTENCE_PAUSE_OPTIONS = [("Short", 0.25), ("Medium", 0.5), ("Long", 0.75)]
CLAUSE_PAUSE_OPTIONS = [("Short", 0.1), ("Medium", 0.2), ("Long", 0.35)]


def _closest_label(options, value: float) -> str:
    return min(options, key=lambda pair: abs(pair[1] - value))[0]


class App:
    def __init__(self, root: tk.Tk):
        self._root = root
        # Worker threads report events through this queue; only the Tk thread touches widgets.
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._dark = False
        self._voices = find_voices(VOICES_DIR)
        self._piper_backend = PiperBackend()
        self._kokoro_backend = (kokoro_backend.KokoroBackend(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
                                 if kokoro_backend is not None else None)
        # None means "not yet established" - _init_voice's first _select_voice() call sets
        # the real starting backend (Kokoro, unless it's unavailable) via set_backend().
        self._active_engine: Optional[str] = None
        self._speaker = Speaker(lambda text: self._events.put(("status", text)), self._piper_backend)
        self._hotkey = GlobalHotkey(*HOTKEY, self._on_hotkey)
        self._restore_hotkey = GlobalHotkey(*RESTORE_HOTKEY, self._on_restore_hotkey)
        self._voice_choices = self._build_voice_choices()
        self._voice_name = tk.StringVar()
        self._status = tk.StringVar()
        self._themed_buttons = []  # [(HexButton, "accent"|"danger"), ...] - see _on_theme_progress
        self._tuning_frame: Optional[ttk.Frame] = None  # built lazily, see _build_settings_panel
        self._settings_frame: Optional[ttk.Frame] = None  # embedded in the canvas, not a popup window
        self._settings_item: Optional[int] = None  # the create_window id embedding it
        self._settings_open = False
        self._settings_animating = False
        self._settings_full_height = 0
        self._preview_thread: Optional[threading.Thread] = None  # see _select_voice
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._close)
        set_window_icon(root, app_dir() / "icon" / "tts.ico")
        self._init_voice()
        # Always on, unlike the F2 read-aloud hotkey (which only starts on Start) - this is
        # core window management (see _minimize's docstring), not an opt-in feature.
        if not self._restore_hotkey.start():
            self._events.put(("status", f"Note: {RESTORE_HOTKEY_LABEL} (restore) is taken by another program."))
        self._poll_events()

    def _build_voice_choices(self) -> dict:
        """One merged Voice list: Kokoro voices first (better-sounding, so its default -
        af_heart - leads the list and is what a fresh launch selects), the older Piper
        voices below where they're still reachable by scrolling. Labels are compact -
        just "id · v#" (v2 = Kokoro, v1 = Piper; see the info tooltip) - full detail on
        each voice lives in voices.txt. Maps label -> (engine, voice_id): voice_id is a
        Kokoro voice name (str) or a Piper .onnx Path, whichever _select_voice needs to
        hand to that engine's backend.
        """
        choices = {}
        if self._kokoro_backend is not None:
            ordered = [kokoro_backend.DEFAULT_VOICE] + [
                v for v in kokoro_backend.FALLBACK_VOICES if v != kokoro_backend.DEFAULT_VOICE
            ]
            for voice_id in ordered:
                choices[f"{voice_id} · v2"] = ("kokoro", voice_id)
        for name, path in self._voices.items():
            choices[f"{name} · v1"] = ("piper", path)
        return choices

    # ---------------------------------------------------------------- UI construction

    def _build_ui(self) -> None:
        """Borderless: overrideredirect(True) below removes the native title bar and frame,
        so with the rectangle's four corners keyed transparent (TRANSPARENT_KEY) the hexagon
        is the only thing on screen - no square chrome around it at all. This is the frameless
        approach ui_widgets' module docstring describes as previously abandoned; it's only
        safe to bring back now because DragHandle (move), CloseButton (close) and
        MinimizeButton (minimize) already exist and were already proven working *with* the
        title bar still present, one step at a time, before it was removed here. Known
        tradeoff: no taskbar entry (see this method's own comment below, and
        App._minimize's docstring for what that means for restoring a minimized window).
        Voice/tuning controls (ttk, since they need real dropdowns) live in a panel embedded
        in this same canvas, opened by the gear - see _build_settings_panel.
        """
        self._root.title("TTS")
        self._root.resizable(False, False)
        colors = LIGHT

        self._canvas = make_app_window(self._root, WINDOW_WIDTH, WINDOW_HEIGHT, colors["panel"])
        self._root.attributes("-transparentcolor", TRANSPARENT_KEY)
        # Removes the title bar/frame entirely - see this method's docstring. Every other
        # piece needed to keep using the window (moving, closing, minimizing) was already
        # built and verified working before this line was added.
        self._root.overrideredirect(True)

        self._background = HexGradientBackground(self._canvas, WINDOW_WIDTH, WINDOW_HEIGHT, HEX_WAIST,
                                                 colors["panel"], colors["outline"])
        self._background.update(colors["grad_top"], colors["grad_mid"], colors["grad_bottom"],
                                colors["outline"], colors["panel"])

        top = WINDOW_HEIGHT * HEX_WAIST
        cx = WINDOW_WIDTH / 2

        # Centered horizontally between the minimize and close buttons below, on their same
        # row - see those buttons' cx-70/cx+70 x positions right after this.
        self._logo_y = 55
        self._logo_items = draw_logo(self._canvas, cx, self._logo_y, colors["accent"], colors["text"])
        self._logo_tooltip = CanvasTooltip(self._canvas, self._logo_items, cx, self._logo_y + 20,
                                          self._help_text(), colors["panel"], colors["text"])

        # The only way left to close or minimize the window, now that overrideredirect (see
        # this method's docstring) has removed the native title bar that used to also do
        # this. Mirrored across the hexagon's top point: close on the right, minimize on
        # the left. Each flashes its own press color the instant it's pressed (_GlyphButton),
        # not on release, so a click always looks like it registered immediately. Created
        # after the logo so they stack on top of it if their edges ever touch.
        self._close_button = CloseButton(self._canvas, cx + 70, 55, colors["text_secondary"],
                                         colors["danger"], self._close)
        self._minimize_button = MinimizeButton(self._canvas, cx - 70, 55, colors["text_secondary"],
                                               colors["accent"], self._minimize)

        # The only way left to move the window, right at the hexagon's own top point, now
        # that the native title bar is gone.
        self._drag_handle = DragHandle(self._canvas, cx, 7, colors["text_secondary"])

        self._row_y = top + 56
        self._gear = GearIcon(self._canvas, cx - 34, self._row_y, colors["text_secondary"], self._toggle_settings)
        row_bg = self._background.color_at(self._row_y, colors["grad_top"], colors["grad_mid"], colors["grad_bottom"])
        self._toggle = DayNightToggle(self._canvas, row_bg, self._dark,
                                      self._on_theme_progress, self._on_theme_done)
        self._canvas.create_window(cx + 34, self._row_y, window=self._toggle.canvas)

        # Three uniform hexagonal buttons in one row - Start, Read and Stop evenly spaced
        # across the hexagon's centerline (cx), Start arched ARCH_RISE px above the
        # Read/Stop row so the group reads as a single balanced shape, not a straight line.
        buttons_top = top + 119  # Start's top edge - keeps the whole group clear of the
        start_y = buttons_top + BUTTON_HEIGHT / 2  # gear/toggle row above and the hexagon's
        row_y = start_y + ARCH_RISE                # lower point below
        step = BUTTON_WIDTH + BUTTON_GAP
        specs = (
            ("Start", self._start_listening, "accent", cx, start_y),
            ("Read", self._read_now, "accent", cx - step, row_y),
            ("Stop", self._speaker.stop, "danger", cx + step, row_y),
        )
        buttons = []
        for label, command, role, x, y in specs:
            fill, hover, press = self._role_colors(colors, role)
            button = HexButton(self._canvas, x, y, label, command, fill, hover, press,
                              colors["button_text"], width=BUTTON_WIDTH, height=BUTTON_HEIGHT, font=FONT)
            self._themed_buttons.append((button, role))
            buttons.append(button)
        self._start_button, self._read_button, self._stop_button = buttons

        self._style = ttk.Style(self._root)
        self._style.theme_use("clam")  # the default "vista" theme ignores custom colors on Windows
        self._apply_ttk_style(colors)  # only matters once the settings popup is built/opened

        # Primes everything - including the sun/glow, which have no other explicit "start
        # here" step - to its light-mode resting state. Reuses the exact same per-frame
        # logic a real toggle drives; harmlessly redundant for the items already built above
        # with LIGHT colors directly.
        self._on_theme_progress(0.0)

        self._root.update_idletasks()
        # Cached once (a window's HWND is stable for its whole life): _minimize and
        # _on_restore_hotkey both need it, and the latter runs on a different thread where
        # querying `wm frame` itself (a Tcl call) would not be safe - but reusing an int
        # already fetched on the Tk thread, in a plain ctypes call, is.
        self._wrapper_hwnd = int(self._root.tk.call("wm", "frame", self._root), 0)
        self._fix_taskbar_presence()

    def _fix_taskbar_presence(self) -> None:
        """overrideredirect(True), a few lines up, makes Tk set WS_EX_TOOLWINDOW on the
        wrapper frame HWND automatically on Windows - confirmed via direct GetWindowLong
        inspection (see _minimize's docstring and CLAUDE.md) - which is the exact flag
        Explorer's taskbar and Alt+Tab both use to exclude a window. Clearing that bit and
        adding WS_EX_APPWINDOW (which forces a taskbar entry even for a window Explorer might
        otherwise treat as auxiliary) restores a normal taskbar button without giving back
        the native title bar overrideredirect removed - those ex-style bits only control
        taskbar/Alt-Tab representation, not WS_CAPTION/WS_THICKFRAME.

        Windows only reconsiders a window's taskbar presence when it's (re)shown, not the
        instant its ex-style changes - so the SetWindowLong call alone would silently do
        nothing until some unrelated later hide/show happened to trigger it. The hide/show
        cycle right after is what makes it take effect immediately, confirmed live.
        """
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_APPWINDOW = 0x00040000
        SW_HIDE = 0
        SW_SHOW = 5
        user32 = ctypes.windll.user32
        ex_style = user32.GetWindowLongW(self._wrapper_hwnd, GWL_EXSTYLE)
        ex_style = (ex_style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
        user32.SetWindowLongW(self._wrapper_hwnd, GWL_EXSTYLE, ex_style)
        user32.ShowWindow(self._wrapper_hwnd, SW_HIDE)
        user32.ShowWindow(self._wrapper_hwnd, SW_SHOW)

    @staticmethod
    def _role_colors(colors: dict, role: str):
        return ((colors["accent"], colors["accent_hover"], colors["accent_press"]) if role == "accent"
                else (colors["danger"], colors["danger_hover"], colors["danger_press"]))

    def _toggle_settings(self) -> None:
        """The gear's click handler: builds the settings panel on first use, then reveals or
        hides it with a scroll-down/scroll-up animation - Voice and (for Kokoro) the tuning
        dropdowns live there now, out of the main window, per the redesign's "just the logo/
        buttons/gear/toggle" hexagon. The panel is embedded in the main canvas (see
        _build_settings_panel), confined to SETTINGS_WIDTH and an animated height entirely
        within the hexagon's flat band, so it can never show a corner poking past the
        hexagon's slanted edges the way a floating popup window could have.
        """
        if self._settings_frame is None:
            self._build_settings_panel()
        if self._settings_animating:
            return
        self._settings_animating = True
        opening = not self._settings_open
        if opening:
            if self._tuning_frame is not None:
                (self._tuning_frame.grid() if self._active_engine == "kokoro" else self._tuning_frame.grid_remove())
            self._settings_frame.update_idletasks()  # so reqheight reflects the tuning frame's current visibility
            self._settings_full_height = self._settings_frame.winfo_reqheight()
            # A window item shrunk to height=0 stays mapped and can leave stale pixels behind
            # on this layered (-transparentcolor) window instead of being erased - confirmed
            # live (the panel visually stayed on screen after closing despite height already
            # reading back as 0). Re-show it explicitly before growing back from 0.
            self._canvas.itemconfigure(self._settings_item, state="normal")
        start = 0 if opening else self._settings_full_height
        end = self._settings_full_height if opening else 0
        self._animate_settings(0, start, end)

    def _animate_settings(self, frame: int, start: int, end: int) -> None:
        """Reveals/hides the settings panel by animating the height of the create_window
        item embedding it (see _build_settings_panel) from 0 up to its full content height,
        or back - Tk clips an embedded window to that box, and anchor="n" pins the top edge
        in place, so growing height from 0 reads as the content scrolling down into view from
        the top, exactly mirroring DayNightToggle._animate's own frame/reschedule pattern.
        """
        try:
            t = frame / SETTINGS_ANIM_FRAMES
            height = start + (end - start) * t
            self._canvas.itemconfig(self._settings_item, height=max(0, int(height)))
            if frame < SETTINGS_ANIM_FRAMES:
                delay = SETTINGS_ANIM_MS // SETTINGS_ANIM_FRAMES
                self._canvas.after(delay, lambda: self._animate_settings(frame + 1, start, end))
            else:
                self._settings_open = end > 0
                self._settings_animating = False
                if not self._settings_open:
                    # Unmap the embedded window outright once fully closed - see the note in
                    # _toggle_settings about height=0 alone not being enough to erase it.
                    self._canvas.itemconfigure(self._settings_item, state="hidden")
        except tk.TclError:
            pass  # window closed mid-animation - a pending after() callback fired on a dead canvas

    def _build_settings_panel(self) -> None:
        colors = self._palette()
        frame = ttk.Frame(self._canvas, padding=3)

        ttk.Label(frame, text="Voice").grid(row=0, column=0, sticky="w")
        self._voice_box = ttk.Combobox(frame, textvariable=self._voice_name,
                                       values=list(self._voice_choices), state="readonly", width=15)
        self._voice_box.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._voice_box.bind("<<ComboboxSelected>>", lambda _e: self._select_voice())

        # None when Kokoro is unavailable (see the import guard at the top of this file) -
        # there's nothing to tune, and no Kokoro entry in the Voice list for _select_voice
        # to reach.
        if self._kokoro_backend is not None:
            self._tuning_frame = self._build_tuning_frame(frame)
            self._tuning_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
            if self._active_engine != "kokoro":
                self._tuning_frame.grid_remove()

        status_row = ttk.Frame(frame)
        status_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Label(status_row, textvariable=self._status, style="Status.TLabel",
                 wraplength=SETTINGS_WIDTH - 20).grid(sticky="w")

        self._settings_frame = frame
        self._settings_item = self._canvas.create_window(
            WINDOW_WIDTH / 2, SETTINGS_TOP, window=frame, anchor="n", width=SETTINGS_WIDTH, height=0)

    def _build_tuning_frame(self, parent: tk.Widget) -> ttk.Frame:
        """Kokoro's voice-tuning panel, collapsed into dropdowns (was open sliders). Kokoro
        has no separate "expressiveness" dial - that mostly comes from which voice you pick -
        so only the real knobs are here: speed (pacing) and the two pause lengths.
        """
        frame = ttk.Frame(parent)
        rows = (
            ("Speed", SPEED_OPTIONS, "speed", self._kokoro_backend.speed),
            ("Sentence pause", SENTENCE_PAUSE_OPTIONS, "sentence_pause_s", self._kokoro_backend.sentence_pause_s),
            ("Clause pause", CLAUSE_PAUSE_OPTIONS, "clause_pause_s", self._kokoro_backend.clause_pause_s),
        )
        for row, (label, options, attr, current) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 2, 0))
            var = tk.StringVar(value=_closest_label(options, current))
            box = ttk.Combobox(frame, textvariable=var, values=[lbl for lbl, _ in options],
                               state="readonly", width=7)
            box.grid(row=row, column=1, sticky="e", pady=(0 if row == 0 else 2, 0))
            box.bind("<<ComboboxSelected>>",
                    lambda _e, attr=attr, options=options, var=var: self._on_tuning_change(attr, options, var))
        frame.columnconfigure(0, weight=1)
        return frame

    def _on_tuning_change(self, attr: str, options, var: tk.StringVar) -> None:
        setattr(self._kokoro_backend, attr, dict(options)[var.get()])

    def _apply_ttk_style(self, colors: dict) -> None:
        """Styles the settings panel's ttk widgets (Frame/Label/Combobox) - the only ttk
        left after the redesign, since the main window is all canvas/raw-tk now. Applied
        once at the end of a dark-mode transition (see _on_theme_done): the panel snaps
        rather than crossfading frame-by-frame, unlike the always-visible main window.
        """
        # The scene's own gradient color at the panel's top edge, not the flat "panel" swatch
        # used elsewhere (tooltip, old corner fill) - so the panel reads as part of the
        # sunset/night background sitting behind it rather than a plain white card. Same
        # single-sample-point approach the day/night toggle's own background already uses
        # (see row_bg in _build_ui) - ttk can't itself paint a gradient across the panel's
        # full height.
        panel_bg = self._background.color_at(SETTINGS_TOP, colors["grad_top"], colors["grad_mid"],
                                             colors["grad_bottom"])
        self._style.configure("TFrame", background=panel_bg)
        self._style.configure("TLabel", background=panel_bg, foreground=colors["text"], font=FONT)
        self._style.configure("Status.TLabel", background=panel_bg, foreground=colors["text_secondary"],
                              font=FONT_SMALL)
        # Blended into the panel rather than standing out as its own box: field, arrow-button
        # background, and all three border shades all match panel_bg - only the arrow glyph
        # itself (arrowcolor) and the dropdown's own hover/press feedback still read as
        # interactive, no separate-colored square around the text.
        self._style.configure("TCombobox", fieldbackground=panel_bg, background=panel_bg,
                              foreground=colors["text"], arrowcolor=colors["text_secondary"],
                              bordercolor=panel_bg, lightcolor=panel_bg, darkcolor=panel_bg)
        self._style.map("TCombobox", fieldbackground=[("readonly", panel_bg)],
                        selectbackground=[("readonly", panel_bg)],
                        selectforeground=[("readonly", colors["text"])],
                        bordercolor=[("readonly", panel_bg)])

    def _palette(self) -> dict:
        return DARK if self._dark else LIGHT

    def _help_text(self) -> str:
        return (f"Press {HOTKEY_LABEL} anywhere to read the current selection (or the last "
                f"text you copied). Press {HOTKEY_LABEL} again, or Stop, to cut it off.\n\n"
                f"Press {RESTORE_HOTKEY_LABEL} anywhere to bring this window back if it's "
                "minimized.\n\n"
                "Voice list: v1 = Piper, v2 = Kokoro.")

    # -------------------------------------------------------------------- dark mode

    def _on_theme_progress(self, amount: float) -> None:
        """Called every animation frame (0=day/light .. 1=night/dark) by the DayNightToggle,
        so the whole window - the gradient background, logo, gear, buttons, and the whole
        sunset scene - crossfades in step with the sun/moon switch itself, sunrise-to-
        moonrise, not just the toggle icon. The settings panel's ttk widgets aren't touched
        here - they snap once at the end instead, see _on_theme_done.

        The sun dome sinks (SUN_Y_LIGHT toward SUN_Y_LIGHT + SUN_SINK, far enough that it's
        entirely below the canvas's own bottom edge by amount=1) while everything - dome,
        bloom, the horizon line, all the clouds - fades by blending its own light-mode color
        toward color_at() - whatever the background already looks like at that exact spot -
        rather than toward a fixed "off" color, so it all dissolves into the night sky instead
        of mismatching it. The bloom fades in lockstep with the dome since it's blended toward
        the same bg_at_sun target (see HexGradientBackground.update_bloom). The clouds also
        drift apart (see HexGradientBackground.update_clouds). No moon replaces any of it; by
        amount=1 everything here is fully blended into the dark background and stays there,
        static, until a toggle runs again.

        Stars fade in the opposite direction - from invisible toward full white - starting at
        STAR_FADE_START rather than only at amount=1, so they're already gathering while the
        sun is still going down (see HexGradientBackground.update_stars()). Same reasoning as
        everything else here: a plain function of amount, no separate timer or state to keep
        in sync with _on_theme_done.
        """
        colors = {key: lerp_color(LIGHT[key], DARK[key], amount) for key in LIGHT}
        self._background.update(colors["grad_top"], colors["grad_mid"], colors["grad_bottom"],
                                colors["outline"], colors["panel"])

        cx = WINDOW_WIDTH / 2

        sun_cy = SUN_Y_LIGHT + SUN_SINK * amount
        bg_at_sun = self._background.color_at(sun_cy, colors["grad_top"], colors["grad_mid"], colors["grad_bottom"])
        sun_color_now = lerp_color(SUN_COLOR, bg_at_sun, amount)
        self._background.update_sun(cx, sun_cy, SUN_RADIUS, sun_color_now)
        self._background.update_bloom(cx, sun_cy, SUN_RADIUS, sun_color_now, bg_at_sun)

        horizon_y = HexGradientBackground.GRAD_MID_T * WINDOW_HEIGHT
        bg_at_horizon = self._background.color_at(horizon_y, colors["grad_top"], colors["grad_mid"],
                                                   colors["grad_bottom"])
        self._background.update_horizon(lerp_color(HORIZON_COLOR, bg_at_horizon, amount))

        cloud_y = sum(y_frac for _, y_frac, _ in HexGradientBackground.CLOUD_SPECS) / \
            len(HexGradientBackground.CLOUD_SPECS) * WINDOW_HEIGHT
        bg_at_clouds = self._background.color_at(cloud_y, colors["grad_top"], colors["grad_mid"],
                                                 colors["grad_bottom"])
        self._background.update_clouds(amount, lerp_color(CLOUD_COLOR, bg_at_clouds, amount))

        star_fade = 0.0 if amount <= STAR_FADE_START else (amount - STAR_FADE_START) / (1 - STAR_FADE_START)
        self._background.update_stars(star_fade, colors["grad_top"], colors["grad_mid"], colors["grad_bottom"])

        retheme_logo(self._canvas, self._logo_items, colors["accent"], colors["text"])
        self._logo_tooltip.retheme(colors["panel"], colors["text"])
        self._gear.retheme(colors["text_secondary"])
        self._close_button.retheme(colors["text_secondary"], colors["danger"])
        self._minimize_button.retheme(colors["text_secondary"], colors["accent"])
        self._drag_handle.retheme(colors["text_secondary"])

        row_bg = self._background.color_at(self._row_y, colors["grad_top"], colors["grad_mid"], colors["grad_bottom"])
        self._toggle.retheme(row_bg)

        for button, role in self._themed_buttons:
            fill, hover, press = self._role_colors(colors, role)
            button.retheme(fill, hover, press, colors["button_text"])

    def _on_theme_done(self, is_dark: bool) -> None:
        self._dark = is_dark
        # Stars are already at their exact final value by the last _on_theme_progress frame
        # (see update_stars/STAR_FADE_START) - nothing left to snap here except ttk.
        self._apply_ttk_style(self._palette())  # snaps the settings panel, if it's been built

    # ------------------------------------------------------------------- app logic

    def _init_voice(self) -> None:
        if not self._voice_choices:
            self._start_button.set_enabled(False)
            self._read_button.set_enabled(False)
            self._events.put(("status", f"No voices found. Add Kokoro models to {KOKORO_DIR} or a "
                              f"Piper .onnx + .onnx.json pair to {VOICES_DIR}, then reopen TTS."))
            return
        # dict insertion order: Kokoro's default is always first when Kokoro is available,
        # so this is also what makes it the voice a fresh launch actually starts on.
        self._voice_name.set(next(iter(self._voice_choices)))
        self._select_voice(play_preview=False)  # a launch shouldn't immediately speak unprompted
        if self._kokoro_backend is None:
            self._events.put(("status", "Ready. Kokoro engine unavailable this session - see tts.log."))
        else:
            self._events.put(("status", "Ready."))

    def _start_listening(self) -> None:
        if self._hotkey.start():
            self._events.put(("status", f"{HOTKEY_LABEL} is on. Select text anywhere and press it."))
            self._start_button.set_enabled(False)
        else:
            self._events.put(("status", f"{HOTKEY_LABEL} is taken by another program. Read and Stop still work."))

    def _select_voice(self, play_preview: bool = True) -> None:
        """The single Voice list mixes both engines, so picking a voice is also what picks
        the engine: this is the one place that routes speak() to the right backend. Swapping
        backends only when the engine actually changed (not on every voice pick) means
        choosing a second Piper voice mid-sentence doesn't cut off whatever is still playing -
        set_backend() calls Speaker.stop(), which a same-engine voice change shouldn't do.
        """
        choice = self._voice_choices.get(self._voice_name.get())
        if choice is None:
            return
        engine, voice_id = choice
        if engine != self._active_engine:
            old_backend = {"kokoro": self._kokoro_backend, "piper": self._piper_backend}.get(self._active_engine)
            backend = self._kokoro_backend if engine == "kokoro" else self._piper_backend
            self._speaker.set_backend(backend)
            self._active_engine = engine
            if old_backend is not None:
                # Only one voice model resident at a time - see PiperBackend.unload /
                # KokoroBackend.unload for the memory-vs-switch-back-speed trade this makes.
                # Safe to call right here: set_backend() above blocks (via Speaker's own
                # engine_lock) until any synthesis still running on old_backend has fully
                # stopped, so nothing can still be mid-call against it below.
                old_backend.unload()
            if self._tuning_frame is not None:
                (self._tuning_frame.grid() if engine == "kokoro" else self._tuning_frame.grid_remove())
        self._events.put(("status", "Loading voice…"))
        self._speaker.set_voice(voice_id)
        if play_preview:
            # speak() can block its caller for up to 2s joining whatever session it's
            # replacing (see Speaker.speak) - this runs on the Tk thread (a Combobox
            # selection event), so it must not call speak() directly or picking a new
            # voice while one is still talking would freeze the window.
            self._preview_thread = threading.Thread(
                target=self._speaker.speak,
                args=(f"Hi, I am {_spoken_name(engine, voice_id)}.",),
                daemon=True,
            )
            self._preview_thread.start()

    def _read_now(self) -> None:
        """Runs on the Tk thread (a button click), so the capture itself needs its own thread -
        capture.selection_or_clipboard() can block for up to ~2s waiting on the clipboard."""
        threading.Thread(target=self._read, args=(capture.selection_or_clipboard,), daemon=True).start()

    def _on_hotkey(self) -> None:
        """Runs on hotkey.py's GlobalHotkey message-loop thread, so it must not touch Tk -
        and must not do capture/synthesis work here either: that thread only reaches
        GetMessageW again (and so only sees the next hotkey press, or stop()'s WM_QUIT)
        once this callback returns, so anything slow here delays both. Capture and speech
        are dispatched to their own thread, the same way _read_now() already does for the
        Read button. A press while speaking stops (stop() itself is instant - just sets an
        Event - so that path stays inline)."""
        if self._speaker.is_speaking:
            self._speaker.stop()
        else:
            threading.Thread(target=self._read, args=(capture.selection_or_clipboard,), daemon=True).start()

    def _read(self, get_text: Callable[[], str]) -> None:
        try:
            text = " ".join(get_text().split())  # join lines broken mid-sentence (PDFs, e-mail)
        except Exception as exc:  # e.g. another program is holding the clipboard open
            self._events.put(("status", f"Could not read the text: {exc}"))
            return
        self._speak(text)

    def _speak(self, text: str) -> None:
        # Safe from any thread: Speaker's methods and SimpleQueue.put are both thread-safe.
        if text:
            self._speaker.speak(text)
        else:
            self._events.put(("status", "Nothing to read. Select some text or copy it first."))

    def _poll_events(self) -> None:
        try:
            try:
                while True:
                    kind, *payload = self._events.get_nowait()
                    if kind == "status":
                        self._status.set(payload[0])
            except queue.Empty:
                pass
            self._root.after(STATUS_POLL_MS, self._poll_events)
        except tk.TclError:
            pass  # window closed mid-poll - a pending after() callback fired on a dead window

    def _minimize(self) -> None:
        """root.iconify() (plain Tk) is confirmed to be a no-op here: Tk's own manual says
        wm iconify "is ignored if window has been overrideredirected", and live-testing this
        exact window proved it - state stayed "normal" through an iconify() call. This uses
        the same direct-Win32, target-the-wrapper-frame-not-winfo_id() approach already used
        by set_window_icon to actually minimize it: ShowWindow(SW_MINIMIZE) on the cached
        wrapper-frame HWND (self._wrapper_hwnd, see _build_ui), confirmed live to genuinely
        take the window to Tk's "iconic" state.

        Tk sets WS_EX_TOOLWINDOW automatically on any overrideredirect() toplevel on Windows
        - the exact flag Explorer's taskbar and Alt+Tab both use to exclude a window - which
        used to mean nothing on screen at all to click once minimized, since the window
        itself is hidden while minimized. _fix_taskbar_presence (see _build_ui) clears that
        flag, so there's now a normal taskbar button to restore from too; _on_restore_hotkey
        below stays as a second, independent way back - a global hotkey whose only job is
        ShowWindow(SW_RESTORE) + SetForegroundWindow on this same HWND, useful if the taskbar
        itself is hard to reach (e.g. auto-hidden, or another window is fullscreen over it).
        """
        ctypes.windll.user32.ShowWindow(self._wrapper_hwnd, 6)  # SW_MINIMIZE

    def _on_restore_hotkey(self) -> None:
        """Runs on this hotkey's own GlobalHotkey message-loop thread - same must-not-touch-
        Tk constraint _on_hotkey documents, but nothing here needs Tk at all: self._wrapper_
        hwnd was fetched once on the Tk thread back in _build_ui (a window's HWND never
        changes), so restoring is just two plain, thread-safe Win32 calls on that cached int.
        """
        user32 = ctypes.windll.user32
        user32.ShowWindow(self._wrapper_hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(self._wrapper_hwnd)

    def _close(self) -> None:
        # hotkey.stop() posts WM_QUIT and then joins that thread (hotkey.py's own
        # _READY_TIMEOUT-scale bound) - it can take a while to return if _on_hotkey is
        # mid-capture/speech on that thread, so it runs off the Tk thread here. Otherwise
        # WM_DELETE_WINDOW (the native close button) would block the whole UI for however
        # long that join takes. Both hotkeys are released the moment the window closes
        # either way - by stop() if it finishes in time, or by the OS when the process exits.
        def _stop_hotkeys():
            self._hotkey.stop()
            self._restore_hotkey.stop()
        threading.Thread(target=_stop_hotkeys, daemon=True).start()
        self._speaker.stop()
        # Destroying Tk windows is deferred via after_idle rather than done here directly.
        # This is now invoked through the normal WM_DELETE_WINDOW protocol (the native
        # title bar's close button), which is itself a safe, standard place to call
        # destroy() from - but the previous, now-removed custom close glyph called this
        # same method synchronously from inside its own canvas <ButtonRelease-1> callback,
        # and destroying a window from inside its own event dispatch corrupted Tcl's memory
        # allocator on this Tk build (crashed with a native "Fatal Error: alloc: invalid
        # block" dialog, not a catchable Python exception) - confirmed with a minimal
        # reproduction: a timer-triggered destroy() never crashed, the same destroy() called
        # synchronously from a canvas click handler crashed every time. Keeping the
        # after_idle deferral here costs nothing and remains a safety margin against that
        # same class of bug if a canvas-based control is ever added back.
        self._root.after_idle(self._destroy_windows)

    def _destroy_windows(self) -> None:
        # The settings panel is no longer a separate window (see _build_settings_panel) -
        # it's embedded in self._canvas, so destroying the root below tears it down too.
        self._root.destroy()


def main() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on high-DPI displays
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
