"""Presentation-only building blocks for app.py's UI: a hexagonal card (drawn on a canvas
inside a perfectly normal, OS-managed window), hexagonal buttons, a hover tooltip, and an
animated day/night toggle. Nothing here touches synthesis or hotkey logic - nothing here
even imports Speaker/backends.

This used to make the *window itself* hexagonal via overrideredirect() + -transparentcolor
click-through masking. That was abandoned after repeated, escalating breakage: Tk gives an
override-redirect window a separate "wrapper frame" HWND that Explorer's taskbar looks at
(distinct from the HWND winfo_id() returns), so taskbar presence and the window icon both
silently failed no matter which handle-manipulation trick was tried; a transparent,
click-through, custom-dragged window also had no reliable native close affordance, which is
what previously required a hand-rolled close glyph and drag handler sharing canvas space
with the gear icon - the actual source of the clicks-get-swallowed bugs. A normal window
gets a real title bar, a working native close button, automatic taskbar presence, and a
correct icon entirely for free from the OS - none of that needs reimplementing by hand
anymore. The hexagon now lives purely as a drawn shape (HexGradientBackground) inside that
ordinary rectangular window.

-transparentcolor came back first, scoped much more narrowly than the old attempt: applied
to this same root, keyed to one fixed, reserved color (TRANSPARENT_KEY) that only
HexGradientBackground's four corner triangles ever use - so only the rectangle's corners
outside the hexagon become invisible (and click-through, onto whatever is behind the window -
that's this attribute's normal behavior, not a bug). CloseButton and DragHandle were added
as a second close/move affordance *alongside* the still-present native title bar.

overrideredirect() has now come back too, on top of all that - App._build_ui calls it after
those were already in place and proven working, specifically so the window's own hexagon
silhouette (no rectangular chrome at all) is the only thing on screen. CloseButton (now the
*only* close affordance) and MinimizeButton (new - a title bar's minimize has no equivalent
once the title bar is gone) both flash a press color the instant they're pressed, not on
release, so a click always gets an immediate visual acknowledgement.

The known tradeoff from before was taskbar presence, tied to override-redirect windows
getting a separate "wrapper frame" HWND that Explorer's taskbar looks at. See App._build_ui
and App._minimize's docstrings for what was actually re-confirmed this time and what that
means for restoring a minimized window.
"""

import ctypes
import math
import random
import tkinter as tk
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

Color = str

# Reserved for HexGradientBackground's corner triangles + root's -transparentcolor (see
# App._build_ui). Never used as a real fill anywhere else, and provably unreachable by
# lerp_color() between any two named colors in app.py's LIGHT/DARK palettes (none of them
# has a zero green channel), so nothing legitimate can ever accidentally turn invisible.
TRANSPARENT_KEY: Color = "#ff00ff"


def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    """Blend two "#rrggbb" colors; t=0 -> c1, t=1 -> c2."""
    def channels(c: Color) -> Tuple[int, int, int]:
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    r1, g1, b1 = channels(c1)
    r2, g2, b2 = channels(c2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _gradient_color(t: float, top: Color, mid: Color, bottom: Color, mid_t: float) -> Color:
    """A 3-stop vertical gradient: top->mid over [0, mid_t], then mid->bottom over
    [mid_t, 1] - top to bottom of the hexagon's own bounding rectangle. Used both by
    HexGradientBackground.update() (the actual stripe fills) and color_at() (matching an
    embedded widget's background to whatever's behind it), so both always agree exactly on
    what color lives at a given y.
    """
    if t <= mid_t:
        return lerp_color(top, mid, t / mid_t) if mid_t > 0 else mid
    return lerp_color(mid, bottom, (t - mid_t) / (1 - mid_t))


def hexagon_points(width: int, height: int, waist: float = 0.2) -> Sequence[float]:
    """Points for a pointy-top/bottom hexagon inscribed in width x height: pointed at the
    very top and bottom, flat vertical sides in the middle (1 - 2*waist) fraction of the
    height - that flat band is where app.py places its actual controls.
    """
    top = height * waist
    bottom = height * (1 - waist)
    return (
        width / 2, 0,
        width, top,
        width, bottom,
        width / 2, height,
        0, bottom,
        0, top,
    )


def regular_hexagon_points(cx: float, cy: float, r: float) -> Sequence[float]:
    """A small pointy-top regular hexagon centered at (cx, cy) with circumradius r - used
    for the logo's flanking accent marks (see draw_logo)."""
    points = []
    for i in range(6):
        angle = math.radians(60 * i - 90)  # -90 so the first vertex points straight up
        points.append(cx + r * math.cos(angle))
        points.append(cy + r * math.sin(angle))
    return tuple(points)


def draw_logo(canvas: tk.Canvas, cx: float, cy: float, accent_color: Color, text_color: Color) -> Tuple[int, int, int]:
    """The "TTS" wordmark flanked by two small hexagon accents - a small design treatment
    instead of plain text, echoing the window's own hexagon shape. Returns the three canvas
    item ids (word mark + both accents), used as a hover target for the hotkey-info tooltip
    and later recolored in place by retheme_logo during a dark-mode transition.
    """
    left = canvas.create_polygon(*regular_hexagon_points(cx - 50, cy, 5), fill=accent_color, outline="")
    text = canvas.create_text(cx, cy, text="TTS", font=("Segoe UI", 19, "bold"), fill=text_color)
    right = canvas.create_polygon(*regular_hexagon_points(cx + 50, cy, 5), fill=accent_color, outline="")
    return left, text, right


def retheme_logo(canvas: tk.Canvas, items: Tuple[int, int, int], accent_color: Color, text_color: Color) -> None:
    left, text, right = items
    canvas.itemconfig(left, fill=accent_color)
    canvas.itemconfig(right, fill=accent_color)
    canvas.itemconfig(text, fill=text_color)


def _point_in_convex_polygon(x: float, y: float, points: Sequence[float]) -> bool:
    """True if (x, y) is inside the convex polygon given as a flat (x0, y0, x1, y1, ...)
    sequence (the same shape hexagon_points returns) - used to keep the star field inside
    the hexagon outline instead of spilling into HexGradientBackground's corner triangles.
    """
    n = len(points) // 2
    sign = None
    for i in range(n):
        x1, y1 = points[2 * i], points[2 * i + 1]
        x2, y2 = points[2 * ((i + 1) % n)], points[2 * ((i + 1) % n) + 1]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross == 0:
            continue
        side = cross > 0
        if sign is None:
            sign = side
        elif side != sign:
            return False
    return True


class _Cloud:
    """A small stylized cloud - three overlapping ovals - that can drift horizontally and
    fade in place; see HexGradientBackground.update_clouds(). Anchor position is fixed at
    construction (clouds don't relocate, only drift relative to their own home spot); each
    update() call just recomputes every puff's coordinates and color from that anchor plus
    the caller's drift distance (scaled by this cloud's own fixed direction - +1 drifts
    right, -1 drifts left), the same "reposition/recolor an already-created item" pattern
    every other animated element in this file uses.
    """

    PUFFS = ((-0.5, 0.12, 0.55), (0.15, -0.18, 0.65), (0.62, 0.12, 0.5))  # (dx, dy, r) * scale

    def __init__(self, canvas: tk.Canvas, anchor_x: float, anchor_y: float, scale: float, direction: float):
        self._canvas = canvas
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self._scale = scale
        self._direction = direction
        self._items = [canvas.create_oval(0, 0, 0, 0, outline="") for _ in self.PUFFS]

    def update(self, drift_distance: float, color: Color) -> None:
        drift_x = self._direction * drift_distance
        for item, (dx, dy, r) in zip(self._items, self.PUFFS):
            x = self._anchor_x + drift_x + dx * self._scale
            y = self._anchor_y + dy * self._scale
            rad = r * self._scale
            self._canvas.coords(item, x - rad, y - rad, x + rad, y + rad)
            self._canvas.itemconfig(item, fill=color)


class HexGradientBackground:
    """The main window's background: a vertical color gradient clipped to a hexagon
    outline drawn on the canvas. Tkinter can't gradient-fill a polygon directly, so this
    draws many thin horizontal stripes across the full window, fills the four corners that
    fall outside the hexagon with the window's own plain background color (so the hexagon
    reads as a card sitting on a normal opaque window, not a cutout), and strokes a crisp
    outline on top. update() recolors the existing stripes in place (cheap - geometry never
    changes), called every frame during the day/night transition; color_at() lets app.py
    match embedded widgets' (buttons, the toggle) own backgrounds to whatever part of the
    gradient they sit over, so they blend in rather than showing as a flat box.

    The four corners are always filled with TRANSPARENT_KEY, not a theme color - paired with
    root.attributes("-transparentcolor", TRANSPARENT_KEY) in App._build_ui, this is what
    makes the corners disappear so only the hexagon itself shows. Fixed rather than
    theme-tracked so the day/night crossfade never needs to touch them (no flicker mid-
    transition from a corner color that briefly doesn't match the transparency key).

    Those same four corner polygons double as a clip mask for everything else drawn in this
    class: they're the exact complement of the hexagon within the canvas's own rectangle, so
    raising them above every other background element (done once, at the end of __init__)
    means anything that spills past the hexagon's slanted edges - the sun's own bloom, a
    drifting cloud - gets silently erased back to TRANSPARENT_KEY there, without needing each
    shape's own geometry hand-constrained to avoid the edges. Controls built afterward in
    App._build_ui (logo, buttons, gear, ...) still stack above all of this, since canvas
    z-order is otherwise just creation order and they're created later.

    A small field of static white dots (STAR_COUNT of them, fixed positions from a seeded
    Random so they don't reshuffle every launch) is created once here for the dark-mode night
    sky. Like the sunset elements below, they fade continuously rather than snapping - see
    update_stars() - each one blended between fully invisible (matching color_at() at its own
    position) and full white, driven every frame by App._on_theme_progress. That fade is
    timed to start well before the sunset elements finish fading out (see App's
    STAR_FADE_START), so dusk reads as one continuous scene - stars gathering while the sun
    is still going down - rather than a light half and a dark half stitched together.

    The light-mode counterpart is a small sunset scene - update()'s 3-stop gradient (sky-blue
    top, a horizon band at GRAD_MID_T, deep orange bottom - see _gradient_color), a horizon
    line (update_horizon), a sun "dome" sitting on the hexagon's own bottom point with a soft
    radial bloom instead of drawn rays (update_sun/update_bloom - the dome shape is just a big
    circle centered at/past y=height, so its lower half falls off the canvas's own bottom edge
    for free, no arc-drawing needed; the bloom is BLOOM_COUNT concentric rings, each a little
    further blended toward the background than the last, which reads as a smooth glow because
    no single ring is ever a hard step from its neighbors), and CLOUD_SPECS worth of small
    clouds (update_clouds). All of these are continuously interpolated - App._on_theme_progress
    moves/fades/drifts them every frame of the existing day/night crossfade, blending each
    one's color toward color_at() - whatever the background already looks like at that exact
    spot - rather than toward a fixed "off" color, so everything dissolves into the night sky
    instead of visibly mismatching it. No moon replaces any of it. Once a toggle finishes,
    nothing here keeps animating on its own.
    """

    STRIPES = 48
    STAR_COUNT = 45
    STAR_SEED = 20260911  # arbitrary but fixed, so the scatter is stable across runs
    GRAD_MID_T = 0.38     # where the horizon sits, as a fraction of the full height
    HORIZON_GAP_FRAC = (0.29, 0.72)  # as width fractions - clears the gear/toggle row (see below)

    # Concentric rings from outermost (index 0, most blended toward the background - nearly
    # invisible) to innermost (closest to the dome's own edge) - see update_bloom(). Small,
    # even steps between neighbors are what make it read as a soft glow instead of a ring.
    BLOOM_RADII_FRAC = (2.3, 2.0, 1.75, 1.55, 1.35, 1.15)
    BLOOM_BLEND = (0.92, 0.8, 0.68, 0.55, 0.4, 0.22)

    # (x_frac, y_frac, scale) for each cloud. Three sit up in the open band between the logo
    # and the gear/toggle row so the scene doesn't read as all crowded into the lower middle
    # zone; the remaining two stay down in that original warm/blue transition band. direction
    # (see update_clouds) is derived from x_frac at construction time, not listed here - left
    # of center drifts left on a toggle, right of center drifts right.
    CLOUD_SPECS = (
        (0.159, 0.27, 15),
        (0.341, 0.445, 13),
        (0.5, 0.22, 16),
        (0.659, 0.445, 13),
        (0.841, 0.27, 15),
    )
    CLOUD_DRIFT = 260      # px each cloud has moved by the time a toggle reaches full dark

    def __init__(self, canvas: tk.Canvas, width: int, height: int, waist: float,
                 bg_color: Color, outline_color: Color):
        # bg_color is no longer used for the corners (see class docstring) - kept as a
        # parameter so call sites don't need to change; harmless if unused.
        self._canvas = canvas
        self._height = height
        self._stripes = []
        for i in range(self.STRIPES):
            y0 = height * i / self.STRIPES
            y1 = height * (i + 1) / self.STRIPES
            self._stripes.append(canvas.create_rectangle(0, y0, width, y1 + 1, outline=""))
        top = height * waist
        bottom = height * (1 - waist)
        self._corners = [
            canvas.create_polygon(*points, fill=TRANSPARENT_KEY, outline=TRANSPARENT_KEY)
            for points in (
                (0, 0, width / 2, 0, 0, top),
                (width / 2, 0, width, 0, width, top),
                (0, bottom, 0, height, width / 2, height),
                (width, bottom, width, height, width / 2, height),
            )
        ]
        self._outline = canvas.create_polygon(*hexagon_points(width, height, waist), fill="",
                                              outline=outline_color, width=2)

        # The horizon: a thin line at GRAD_MID_T's height, where the gradient's sky-blue top
        # stop meets its warm bottom stop - see update(). Fixed position; only its color
        # crossfades (toward invisible - see App._on_theme_progress). Split into two segments
        # with a gap over the gear/toggle row (HORIZON_GAP_FRAC, centered like that row) -
        # GRAD_MID_T's height sits close enough to that row that a single unbroken line cut
        # straight across the gear glyph's open spokes and the toggle, reading as a stray
        # white line through both rather than the background element it actually is.
        horizon_y = self.GRAD_MID_T * height
        gap0, gap1 = width * self.HORIZON_GAP_FRAC[0], width * self.HORIZON_GAP_FRAC[1]
        self._horizon = (
            canvas.create_line(0, horizon_y, gap0, horizon_y, width=1),
            canvas.create_line(gap1, horizon_y, width, horizon_y, width=1),
        )

        # The setting sun: real position/size/color for all of these (and the horizon/clouds
        # above/below) are set by App._build_ui's priming call, same as before. Bloom rings
        # drawn outermost-first, dome disc last/on top, so the dome sits over their centers.
        self._bloom = [canvas.create_oval(0, 0, 0, 0, outline="") for _ in self.BLOOM_RADII_FRAC]
        self._sun = canvas.create_oval(0, 0, 0, 0, fill="", outline="")

        # A handful of small clouds sitting in the gradient's warm/blue transition band - they
        # drift apart and fade during a light-to-dark toggle, see update_clouds(). Each drifts
        # away from the hexagon's own centerline: anchors left of it drift further left,
        # anchors right of it drift right, splitting the group roughly in half either way.
        self._clouds = [
            _Cloud(canvas, width * x_frac, height * y_frac, scale,
                  direction=-1.0 if x_frac < 0.5 else 1.0)
            for x_frac, y_frac, scale in self.CLOUD_SPECS
        ]

        self._stars = self._make_stars(canvas, width, height, waist)

        # See class docstring: clips everything drawn above to the hexagon by sitting above
        # all of it - must be the last thing done in __init__.
        for corner in self._corners:
            canvas.tag_raise(corner)

    @classmethod
    def _make_stars(cls, canvas: tk.Canvas, width: int, height: int, waist: float) -> list:
        """Returns (item_id, y) pairs, not bare item ids - update_stars() needs each star's
        own y to know what "invisible" (matching the background right there) means for it.

        A star whose oval physically straddles two stripes can never fully match either one's
        color, no matter how well update_stars() matches its fill to the stripe its center
        falls in - confirmed live, over a quarter of the 45 stars crossed a stripe seam and
        stayed a faintly visible crescent at fade=0 even after that fix. So the oval's own
        extent is clamped here, at placement time, to stay entirely inside whichever stripe
        band its y lands in - guaranteeing every star can be made to match exactly, not just
        approximately.
        """
        hexagon = hexagon_points(width, height, waist)
        rng = random.Random(cls.STAR_SEED)
        stripe_h = height / cls.STRIPES
        stars = []
        while len(stars) < cls.STAR_COUNT:
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            r = rng.uniform(0.5, 1.3)
            stripe_i = min(cls.STRIPES - 1, int(y / height * cls.STRIPES))
            stripe_top, stripe_bottom = stripe_i * stripe_h, (stripe_i + 1) * stripe_h
            # A small extra margin, not just touching the stripe edges exactly - a star whose
            # rasterized pixels land exactly on the seam is one rounding hair from bleeding
            # into the next stripe's different color, which is fragile rather than fixed.
            margin = 0.5
            y = min(max(y, stripe_top + r + margin), stripe_bottom - r - margin)
            if not _point_in_convex_polygon(x, y, hexagon):
                continue
            item = canvas.create_oval(x - r, y - r, x + r, y + r, outline="")
            stars.append((item, y))
        return stars

    def update_stars(self, fade: float, top_color: Color, mid_color: Color, bottom_color: Color) -> None:
        """fade: 0 = fully invisible, 1 = full white. See App._on_theme_progress and
        STAR_FADE_START for why this starts rising well before amount reaches 1.

        At fade=0 every star is unmapped (state="hidden") outright rather than colored to
        match the background - color-matching only ever covers the plain gradient stripe a
        star's center happens to fall in, not the sun's bloom rings (or anything else) that
        might be layered on top of that same stripe at that position, and a star within the
        bloom's reach stayed a real, confirmed-live mismatch even after matching it to its
        exact stripe. A hidden item can't mismatch anything it isn't drawn over. Past fade=0
        it's shown and colored as before, blending from its own stripe's color toward white -
        any residual imprecision only shows up during the moving transition itself, not at
        rest, which is what was actually reported as visible.
        """
        for item, y in self._stars:
            if fade <= 0.0:
                self._canvas.itemconfig(item, state="hidden")
                continue
            stripe_i = min(self.STRIPES - 1, int(y / self._height * self.STRIPES))
            t = stripe_i / (self.STRIPES - 1)
            bg = _gradient_color(t, top_color, mid_color, bottom_color, self.GRAD_MID_T)
            self._canvas.itemconfig(item, state="normal", fill=lerp_color(bg, "#ffffff", fade))

    def color_at(self, y: float, top_color: Color, mid_color: Color, bottom_color: Color) -> Color:
        t = max(0.0, min(1.0, y / self._height))
        return _gradient_color(t, top_color, mid_color, bottom_color, self.GRAD_MID_T)

    def update(self, top_color: Color, mid_color: Color, bottom_color: Color,
               outline_color: Color, bg_color: Color) -> None:
        # bg_color is accepted (not just dropped) so call sites don't need to change if it's
        # ever needed again; the corners it used to recolor here are now always
        # TRANSPARENT_KEY (see __init__), so there's nothing for this method to do with it.
        for i, item in enumerate(self._stripes):
            t = i / (self.STRIPES - 1)
            self._canvas.itemconfig(item, fill=_gradient_color(t, top_color, mid_color, bottom_color, self.GRAD_MID_T))
        self._canvas.itemconfig(self._outline, outline=outline_color)

    def update_horizon(self, color: Color) -> None:
        for segment in self._horizon:
            self._canvas.itemconfig(segment, fill=color)

    def update_sun(self, cx: float, cy: float, radius: float, color: Color) -> None:
        """Repositions/recolors the setting sun - called every crossfade frame (see
        App._on_theme_progress) with a cy that sinks and a color that's already been blended
        toward whatever the background looks like at that position, so it reads as fading
        into the sky rather than just moving. The "dome" look isn't drawn specially - cy sits
        at/beyond the canvas's own bottom edge, so the circle's lower half is simply off-
        canvas and never rendered.
        """
        self._canvas.coords(self._sun, cx - radius, cy - radius, cx + radius, cy + radius)
        self._canvas.itemconfig(self._sun, fill=color)

    def update_bloom(self, cx: float, cy: float, base_radius: float, sun_color: Color, bg_color: Color) -> None:
        """A soft radial glow around the sun, faked the way this whole file fakes gradients
        Tkinter can't draw natively: several concentric rings, each blended a little further
        toward bg_color than the last (BLOOM_BLEND), sized as multiples of the dome's own
        radius (BLOOM_RADII_FRAC). sun_color should already be whatever App's own crossfade
        has faded the dome to at this frame - since every ring's blend target is the same
        bg_color the dome is fading toward, the whole bloom dissolves in lockstep with the
        dome itself, with no separate fade timer of its own.
        """
        for item, radius_frac, blend in zip(self._bloom, self.BLOOM_RADII_FRAC, self.BLOOM_BLEND):
            r = base_radius * radius_frac
            self._canvas.coords(item, cx - r, cy - r, cx + r, cy + r)
            self._canvas.itemconfig(item, fill=lerp_color(sun_color, bg_color, blend))

    def update_clouds(self, drift_t: float, color: Color) -> None:
        """drift_t: 0 = every cloud at its home anchor (light mode), 1 = fully split apart
        (full dark) - each drifts CLOUD_DRIFT px in its own fixed direction (see __init__),
        "splitting from the center" roughly evenly as App's docstring puts it.
        """
        for cloud in self._clouds:
            cloud.update(self.CLOUD_DRIFT * drift_t, color)


def make_app_window(root: tk.Tk, width: int, height: int, bg_color: str) -> tk.Canvas:
    """A normal, OS-managed top-level window (real title bar, working native close button,
    automatic taskbar presence) with a plain canvas filling its client area. Deliberately NOT
    overrideredirect() (the previous approach, see this module's docstring for why): a normal
    window gets dragging (via its own title bar), closing, a taskbar entry, and an icon
    entirely for free and reliably from the OS, none of which needed hand-rolled canvas
    click-routing that could - and repeatedly did - conflict with other things drawn on the
    same canvas. -transparentcolor is applied separately, by App._build_ui, to this same
    still-decorated root - see TRANSPARENT_KEY and HexGradientBackground's docstring.
    """
    root.geometry(f"{width}x{height}")
    canvas = tk.Canvas(root, width=width, height=height, bg=bg_color, highlightthickness=0, bd=0)
    canvas.place(x=0, y=0)
    return canvas


def set_window_icon(window: tk.Misc, icon_path: Union[str, Path]) -> None:
    """Sets the window's title-bar/taskbar icon from an .ico file. Uses WM_SETICON via
    ctypes rather than Tk's own iconbitmap(): on Windows, Tk creates a separate outer
    "wrapper frame" HWND for every toplevel in addition to the inner "content" HWND that
    winfo_id() returns, and iconbitmap() was confirmed (by direct WM_GETICON inspection) to
    set the icon on the content window while both the taskbar and the title bar read the
    wrapper frame's icon - so iconbitmap() silently produced no visible icon at all in
    testing. Targeting `wm frame` directly is a known-working fix, confirmed the same way.
    """
    path = Path(icon_path)
    if not path.is_file():
        return
    WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
    IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040

    window.update_idletasks()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendMessageW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_uint]

    hwnd = int(window.tk.call("wm", "frame", window), 0)
    big = user32.LoadImageW(None, str(path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
    small = user32.LoadImageW(None, str(path), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
    if big:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
    if small:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)


class HexButton:
    """A button shaped like a flattened hexagon (chamfered left/right ends) so it echoes
    the main window's hexagon silhouette - drawn directly on the shared canvas, the same way
    GearIcon/CloseButton/MinimizeButton/the logo already are, rather than on its own embedded
    Canvas.

    An embedded per-button Canvas needs its own opaque background behind the chamfered
    shape's cut corners (that's the whole point of a chamfer) - first a single flat color,
    then several stripe-matched rectangles, and both were confirmed live to leave a visibly
    mismatched rectangle/halo around every button regardless: a plain 48-stripe gradient
    sample can be replicated, but the sun's bloom rings and anything else layered on top of
    that gradient at the button's position can't be, short of re-deriving this whole file's
    entire background-compositing stack a second time just for a button's corners. Drawing
    the shape straight on the shared canvas - ttk/native buttons on Windows can't be shaped
    or reliably recolored either way (see the project's older RoundButton) - removes the bug
    at its root: there's no separate rectangle behind it left to mismatch anything.
    """

    def __init__(self, canvas: tk.Canvas, cx: float, cy: float, text: str, command: Callable[[], None],
                 fill: Color, hover: Color, press: Color, text_color: Color,
                 width: int = 140, height: int = 34, chamfer: int = 14,
                 font: Tuple = ("Segoe UI", 10)):
        self._canvas = canvas
        self._command = command
        self._fill, self._hover, self._press = fill, hover, press
        self._text_color = text_color
        self._enabled = True
        x0, y0 = cx - width / 2, cy - height / 2
        points = (x0 + chamfer, y0, x0 + width - chamfer, y0, x0 + width, cy,
                  x0 + width - chamfer, y0 + height, x0 + chamfer, y0 + height, x0, cy)
        self._shape = canvas.create_polygon(*points, outline="", fill=fill)
        self._label = canvas.create_text(cx, cy, text=text, fill=text_color, font=font)
        for item in (self._shape, self._label):
            for seq, handler in (("<Enter>", self._on_enter), ("<Leave>", self._on_leave),
                                  ("<ButtonPress-1>", self._on_press), ("<ButtonRelease-1>", self._on_release)):
                canvas.tag_bind(item, seq, handler)
        self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self._canvas.itemconfig(self._shape, fill=self._fill if enabled else lerp_color(self._fill, "#808080", 0.5))

    def retheme(self, fill: Color, hover: Color, press: Color, text_color: Color) -> None:
        """Called every frame during a dark-mode transition (see App._on_theme_progress),
        so this button's colors crossfade in step with the DayNightToggle's own animation."""
        self._fill, self._hover, self._press, self._text_color = fill, hover, press, text_color
        self._canvas.itemconfig(self._shape, fill=fill if self._enabled else lerp_color(fill, "#808080", 0.5))
        self._canvas.itemconfig(self._label, fill=text_color)

    def _on_enter(self, _event: tk.Event) -> None:
        if self._enabled:
            self._canvas.itemconfig(self._shape, fill=self._hover)
            self._canvas.config(cursor="hand2")

    def _on_leave(self, _event: tk.Event) -> None:
        if self._enabled:
            self._canvas.itemconfig(self._shape, fill=self._fill)
            self._canvas.config(cursor="arrow")

    def _on_press(self, _event: tk.Event) -> None:
        if self._enabled:
            self._canvas.itemconfig(self._shape, fill=self._press)

    def _on_release(self, _event: tk.Event) -> None:
        if self._enabled:
            self._canvas.itemconfig(self._shape, fill=self._hover)
            self._command()


class CanvasTooltip:
    """A small popup with `text`, shown near a group of canvas items after a short hover
    delay and hidden on mouse-leave - the standard way to expose info without permanent
    screen space. Targets canvas item ids (via tag_bind) rather than a widget, since the
    hover target here (the logo wordmark) is drawn directly on the shared main canvas,
    not a widget of its own.
    """

    DELAY_MS = 350

    def __init__(self, canvas: tk.Canvas, item_ids: Sequence[int], anchor_x: float, anchor_y: float,
                 text: str, bg: Color, fg: Color):
        self._canvas = canvas
        self._x, self._y = anchor_x, anchor_y
        self._text = text
        self._bg, self._fg = bg, fg
        self._win: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        for item in item_ids:
            canvas.tag_bind(item, "<Enter>", self._schedule)
            canvas.tag_bind(item, "<Leave>", self._hide)

    def set_text(self, text: str) -> None:
        self._text = text

    def retheme(self, bg: Color, fg: Color) -> None:
        self._bg, self._fg = bg, fg

    def _schedule(self, _event: tk.Event) -> None:
        self._after_id = self._canvas.after(self.DELAY_MS, self._show)

    def _show(self) -> None:
        if self._win is not None:
            return
        x = self._canvas.winfo_rootx() + int(self._x)
        y = self._canvas.winfo_rooty() + int(self._y)
        self._win = tk.Toplevel(self._canvas)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        label = tk.Label(self._win, text=self._text, bg=self._bg, fg=self._fg,
                         font=("Segoe UI", 9), justify="left", padx=10, pady=7,
                         wraplength=220)
        label.pack()
        self._win.geometry(f"+{x - 110}+{y}")

    def _hide(self, _event: Optional[tk.Event] = None) -> None:
        if self._after_id is not None:
            self._canvas.after_cancel(self._after_id)
            self._after_id = None
        if self._win is not None:
            self._win.destroy()
            self._win = None


class DragHandle:
    """A small dot at the hexagon's very top point - dragging it moves the whole window.
    The native title bar (still present, see make_app_window's docstring) already does
    this; this is a second, additive way to grab the window that lives on the hexagon's
    own silhouette instead of the OS chrome above it.

    Uses plain window-geometry dragging (record the mouse's screen position and the
    window's screen position on press, then re-geometry() the window by the mouse's delta
    on every drag motion) - the standard Tk technique for a custom drag handle on an
    ordinary, still-decorated toplevel. Nothing here touches overrideredirect or window
    styles; it just repositions the same window the title bar's own drag already moves.
    """

    RADIUS = 5

    def __init__(self, canvas: tk.Canvas, x: float, y: float, color: Color):
        self._canvas = canvas
        self._root = canvas.winfo_toplevel()
        self._start_root_x = 0
        self._start_root_y = 0
        self._start_win_x = 0
        self._start_win_y = 0
        r = self.RADIUS
        self.item = canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")
        canvas.tag_bind(self.item, "<Enter>", lambda _e: canvas.config(cursor="fleur"))
        canvas.tag_bind(self.item, "<Leave>", lambda _e: canvas.config(cursor="arrow"))
        canvas.tag_bind(self.item, "<ButtonPress-1>", self._on_press)
        canvas.tag_bind(self.item, "<B1-Motion>", self._on_drag)

    def retheme(self, color: Color) -> None:
        self._canvas.itemconfig(self.item, fill=color)

    def _on_press(self, event: tk.Event) -> None:
        self._start_root_x, self._start_root_y = event.x_root, event.y_root
        self._start_win_x, self._start_win_y = self._root.winfo_x(), self._root.winfo_y()

    def _on_drag(self, event: tk.Event) -> None:
        dx = event.x_root - self._start_root_x
        dy = event.y_root - self._start_root_y
        self._root.geometry(f"+{self._start_win_x + dx}+{self._start_win_y + dy}")


class GearIcon:
    """A small gear glyph drawn directly on a shared canvas (no background rectangle of
    its own to blend, same as the logo) that does a full spin animation on click,
    independent of whatever that click opens (see app.py's _toggle_settings). Rotation
    uses the canvas text item's own "angle" option (Tk 8.6+), not a rotated image.
    """

    SPIN_DEGREES = 360
    DURATION_MS = 350
    FRAMES = 14

    def __init__(self, canvas: tk.Canvas, x: float, y: float, color: Color, on_click: Callable[[], None]):
        self._canvas = canvas
        self._angle = 0.0
        self._spinning = False
        self._on_click = on_click
        self.item = canvas.create_text(x, y, text="⚙", font=("Segoe UI Symbol", 15), fill=color, angle=0)
        canvas.tag_bind(self.item, "<Enter>", lambda _e: canvas.config(cursor="hand2"))
        canvas.tag_bind(self.item, "<Leave>", lambda _e: canvas.config(cursor="arrow"))
        canvas.tag_bind(self.item, "<ButtonRelease-1>", lambda _e: self._clicked())

    def retheme(self, color: Color) -> None:
        self._canvas.itemconfig(self.item, fill=color)

    def _clicked(self) -> None:
        if not self._spinning:
            self._spinning = True
            self._animate(0)
        self._on_click()

    def _animate(self, frame: int) -> None:
        try:
            self._angle = (self._angle + self.SPIN_DEGREES / self.FRAMES) % 360
            self._canvas.itemconfig(self.item, angle=self._angle)
            if frame < self.FRAMES:
                self._canvas.after(self.DURATION_MS // self.FRAMES, lambda: self._animate(frame + 1))
            else:
                self._spinning = False
        except tk.TclError:
            pass  # window closed mid-spin - a pending after() callback fired on a dead canvas


class _GlyphButton:
    """Shared base for the small text-glyph controls drawn directly on the hexagon (close,
    minimize): hover cursor, a press color that changes the instant the mouse goes down -
    not when it's released, and not after whatever the click triggers has finished - so a
    click always gets an immediate visual acknowledgement no matter how long closing or
    minimizing then takes, plus the click itself on release.

    Safe to wire straight to a handler that destroys or hides the window even though this
    fires from a canvas <ButtonRelease-1> handler: App._close() already defers its actual
    destroy() to after_idle specifically so a canvas-based close control can't hit the
    synchronous-destroy-from-a-click-handler crash documented there, and iconify (used by
    MinimizeButton) was never part of that hazard - it doesn't destroy anything.
    """

    def __init__(self, canvas: tk.Canvas, x: float, y: float, text: str, color: Color,
                 press_color: Color, on_click: Callable[[], None]):
        self._canvas = canvas
        self._color = color
        self._press_color = press_color
        self._on_click = on_click
        self.item = canvas.create_text(x, y, text=text, font=("Segoe UI", 12), fill=color)
        canvas.tag_bind(self.item, "<Enter>", lambda _e: canvas.config(cursor="hand2"))
        canvas.tag_bind(self.item, "<Leave>", lambda _e: canvas.config(cursor="arrow"))
        canvas.tag_bind(self.item, "<ButtonPress-1>", self._on_press)
        canvas.tag_bind(self.item, "<ButtonRelease-1>", self._on_release)

    def retheme(self, color: Color, press_color: Optional[Color] = None) -> None:
        self._color = color
        if press_color is not None:
            self._press_color = press_color
        self._canvas.itemconfig(self.item, fill=color)

    def _on_press(self, _event: tk.Event) -> None:
        self._canvas.itemconfig(self.item, fill=self._press_color)

    def _on_release(self, _event: tk.Event) -> None:
        self._canvas.itemconfig(self.item, fill=self._color)
        self._on_click()


class CloseButton(_GlyphButton):
    """An "x" glyph near the hexagon's top-right - now the only way to close the window
    (see make_app_window's docstring: the native title bar this used to sit alongside is
    gone). Flashes to press_color (the danger role, by convention for a close control) the
    instant it's pressed - see _GlyphButton.
    """

    def __init__(self, canvas: tk.Canvas, x: float, y: float, color: Color, press_color: Color,
                 on_click: Callable[[], None]):
        super().__init__(canvas, x, y, "✕", color, press_color, on_click)


class MinimizeButton(_GlyphButton):
    """A "—" glyph mirroring CloseButton on the hexagon's top-left - now the only way to
    minimize the window, since removing the native title bar removed the standard minimize
    button along with it (see App._minimize and its docstring for the taskbar-restore
    tradeoff that follows from that).
    """

    def __init__(self, canvas: tk.Canvas, x: float, y: float, color: Color, press_color: Color,
                 on_click: Callable[[], None]):
        super().__init__(canvas, x, y, "—", color, press_color, on_click)


class DayNightToggle:
    """A small switch that slides between a sun and a moon, with the thumb crossfading
    between them - a literal sunset-to-moonrise animation, not just an instant flip.
    on_progress(t) fires every frame during the animation (t: 0=day, 1=night) so the caller
    can crossfade the rest of the app's colors in step; on_done(is_dark) fires once at the
    end so ttk widgets (which can't crossfade smoothly) can snap over.

    The track itself has no color of its own - it's always painted with whatever bg the
    caller passes in (see retheme), which App computes from the real hexagon scene at this
    row (the sunset gradient in light mode, the night sky in dark mode), the same way a
    button or the settings panel blends into the scene rather than sitting on a separate
    swatch. The crescent "moon bite" cutout uses that exact same bg, so it disappears
    seamlessly into the track instead of leaving a visible seam.
    """

    WIDTH, HEIGHT = 56, 28
    DURATION_MS = 450
    FRAMES = 20
    SUN = "#e8843e"
    MOON = "#eef0fa"

    def __init__(self, parent: tk.Widget, bg: Color, initial_dark: bool,
                 on_progress: Callable[[float], None], on_done: Callable[[bool], None]):
        self._dark = initial_dark
        self._animating = False
        self._on_progress = on_progress
        self._on_done = on_done
        self.canvas = tk.Canvas(parent, width=self.WIDTH, height=self.HEIGHT,
                                highlightthickness=0, bd=0, bg=bg)
        r = self.HEIGHT / 2
        self._track_shapes = (
            self.canvas.create_oval(0, 0, self.HEIGHT, self.HEIGHT, outline="", fill=bg),
            self.canvas.create_rectangle(r, 0, self.WIDTH - r, self.HEIGHT, outline="", fill=bg),
            self.canvas.create_oval(self.WIDTH - self.HEIGHT, 0, self.WIDTH, self.HEIGHT, outline="", fill=bg),
        )
        thumb_r = r - 3
        self._thumb = self.canvas.create_oval(0, 0, thumb_r * 2, thumb_r * 2, outline="")
        self._crescent = self.canvas.create_oval(0, 0, thumb_r * 2, thumb_r * 2, outline="", fill=bg, state="hidden")
        self._thumb_r = thumb_r
        self._bg = bg
        self.canvas.config(cursor="hand2")
        self.canvas.bind("<ButtonRelease-1>", lambda _e: self.toggle())
        self._render(1.0 if initial_dark else 0.0)

    def retheme(self, bg: Color) -> None:
        """bg is the real scene color at this row (see App._on_theme_progress's row_bg) -
        drives the track fill, the canvas's own backdrop, and the crescent cutout, so all
        three always match each other and whatever's actually behind the toggle."""
        self._bg = bg
        self.canvas.configure(bg=bg)
        for shape in self._track_shapes:
            self.canvas.itemconfig(shape, fill=bg)
        self.canvas.itemconfig(self._crescent, fill=bg)

    def toggle(self) -> None:
        if self._animating:
            return
        self._animating = True
        start = 1.0 if self._dark else 0.0
        end = 0.0 if self._dark else 1.0
        self._animate(0, start, end)

    def _animate(self, frame: int, start: float, end: float) -> None:
        try:
            t = frame / self.FRAMES
            amount = start + (end - start) * t
            self._render(amount)
            self._on_progress(amount)  # crossfades the whole window - see App._on_theme_progress
            if frame < self.FRAMES:
                delay = self.DURATION_MS // self.FRAMES
                self.canvas.after(delay, lambda: self._animate(frame + 1, start, end))
            else:
                self._dark = end > 0.5
                self._animating = False
                self._on_done(self._dark)
        except tk.TclError:
            pass  # window closed mid-transition - a pending after() callback fired on a dead canvas

    def _render(self, amount: float) -> None:
        """amount: 0.0 = full day (sun, left), 1.0 = full night (moon, right). Track/crescent
        colors aren't touched here - they follow self._bg exclusively, kept current by
        retheme() (see its docstring); this only moves and recolors the thumb itself."""
        thumb_color = lerp_color(self.SUN, self.MOON, amount)
        self.canvas.itemconfig(self._thumb, fill=thumb_color)
        cx = self._thumb_r + 3 + (self.WIDTH - 2 * (self._thumb_r + 3)) * amount
        cy = self.HEIGHT / 2
        r = self._thumb_r
        self.canvas.coords(self._thumb, cx - r, cy - r, cx + r, cy + r)

        # A moon crescent grows in as it rises - the same bg color as the track itself (see
        # retheme), so the "bite" reads as a seamless cutout rather than a visible patch.
        crescent_visible = amount > 0.55
        self.canvas.itemconfig(self._crescent, state="normal" if crescent_visible else "hidden")
        if crescent_visible:
            offset = r * 0.55
            self.canvas.coords(self._crescent, cx - r + offset, cy - r, cx + r + offset, cy + r)

        self.canvas.tag_raise(self._thumb)
        self.canvas.tag_raise(self._crescent)
