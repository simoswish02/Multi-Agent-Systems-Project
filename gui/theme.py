"""Shared visual theme for the GUI: palette, fonts and widget helpers.

Single source of truth for every colour used by ``gui/renderer.py``,
``gui/charts.py``, ``gui/terminal.py`` and ``gui/config_screen.py``. The
legacy ``COLOR_*`` names those modules used to define locally are re-exported
here so call sites keep reading naturally.

All UI strings are in English. Comments are in English.
"""

import pygame

# --- Core palette -----------------------------------------------------------
BG        = (17, 18, 24)      # window background
PANEL     = (23, 25, 32)      # sidebar background
PANEL_ALT = (33, 36, 46)      # section headers / inactive panels
SURFACE   = (42, 45, 58)      # buttons, slider tracks, fields
BORDER    = (62, 66, 82)
TEXT      = (233, 236, 244)
TEXT_DIM  = (150, 155, 170)
ACCENT    = (88, 200, 128)    # green: play / active / go
WARNING   = (242, 193, 74)    # amber: paused / near-timeout
DANGER    = (240, 92, 86)     # red: faults / errors
INFO      = (120, 190, 255)

# --- Map palette -------------------------------------------------------------
FOG         = (26, 28, 36)    # unexplored cells (team fog-of-war)
FOG_GRID    = (33, 36, 45)    # hairline on fog cells
FREE        = (199, 203, 213) # explored traversable cells
FREE_GRID   = (172, 177, 190) # hairline on explored cells
OBSTACLE    = (64, 68, 82)
OBSTACLE_HI = (90, 95, 112)   # bevel: light top/left edge
OBSTACLE_LO = (44, 47, 58)    # bevel: dark bottom/right edge
TARGET      = (255, 96, 88)
NAV_PATH    = (96, 224, 128)
COMM        = (255, 214, 64)  # comm activity: links, ripples, log lines
BROKEN      = (120, 124, 136) # wreck body
BROKEN_X    = (235, 84, 78)   # wreck accent / crash sites
SMOKE       = (170, 173, 182)

AGENT_COLORS = [
    (86, 168, 255),   # D0 azure
    (94, 222, 150),   # D1 green
    (255, 170, 84),   # D2 orange
    (196, 126, 255),  # D3 violet
    (255, 116, 200),  # D4 pink
]

# --- Legacy aliases (imported by gui/renderer.py and friends) ----------------
COLOR_UNKNOWN    = FOG
COLOR_FREE       = FREE
COLOR_OBSTACLE   = OBSTACLE
COLOR_TARGET     = TARGET
COLOR_GRID_LINE  = FREE_GRID
COLOR_BG         = BG
COLOR_TEXT       = TEXT
COLOR_COMM_FLASH = COMM
COLOR_REWARD_POS = ACCENT
COLOR_REWARD_NEG = DANGER
COLOR_BROKEN     = BROKEN
COLOR_BROKEN_X   = BROKEN_X
COLOR_NAV_PATH   = NAV_PATH

COLOR_TOOLBAR_BG     = (25, 27, 35)
COLOR_SIDEBAR_BG     = PANEL
COLOR_BORDER         = BORDER
COLOR_SECTION_HDR    = PANEL_ALT
COLOR_BTN            = SURFACE
COLOR_BTN_ACTIVE     = ACCENT
COLOR_BTN_DISABLED   = (30, 32, 40)
COLOR_PLAYING        = ACCENT
COLOR_PAUSED         = WARNING
COLOR_INACTIVE_PANEL = (30, 32, 40)
COLOR_LABEL          = TEXT_DIM

COLOR_LOG_HEADER = TEXT
COLOR_LOG_NORMAL = (178, 182, 194)
COLOR_LOG_COMM   = COMM
COLOR_LOG_TARGET = (255, 140, 64)
COLOR_LOG_NEG    = (214, 140, 140)
COLOR_LOG_BROKEN = (255, 102, 96)

# --- Chart palette (gui/charts.py) -------------------------------------------
CHART_PANEL  = (21, 23, 30)
CHART_BORDER = (52, 56, 70)
CHART_AXIS   = (96, 100, 116)
CHART_TITLE  = (214, 218, 228)
CHART_LABEL  = TEXT_DIM
CHART_DASH   = (110, 115, 132)
CHART_ZERO   = (70, 74, 90)

# --- Log widget (gui/terminal.py) ---------------------------------------------
LOG_BG = (14, 15, 20)

# --- Fonts --------------------------------------------------------------------
_FONT_UI   = "segoeui,verdana,arial"
_FONT_MONO = "consolas,couriernew,monospace"
_font_cache = {}


def font(size: int, bold: bool = False, mono: bool = False) -> "pygame.font.Font":
    """Cached SysFont with a fallback chain (no bundled TTF files)."""
    if not pygame.font.get_init():
        pygame.font.init()
    key = (size, bold, mono)
    f = _font_cache.get(key)
    if f is None:
        f = pygame.font.SysFont(_FONT_MONO if mono else _FONT_UI, size, bold=bold)
        _font_cache[key] = f
    return f


# --- Colour helpers -----------------------------------------------------------

def blend(c1, c2, t: float):
    """Linear blend between two RGB colours, t in [0, 1]."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def brighten(c, amt: int = 18):
    return tuple(min(255, x + amt) for x in c[:3])


def darken(c, amt: int = 18):
    return tuple(max(0, x - amt) for x in c[:3])


def with_alpha(c, a: int):
    return (c[0], c[1], c[2], a)


_mouse_override = None


def set_mouse_override(pos):
    """Force what `mouse_pos` reports, or clear it with None.

    Offscreen rendering has no real pointer, so a recorder drawing a synthetic
    cursor sets this to get the same hover states a user would see.
    """
    global _mouse_override
    _mouse_override = None if pos is None else (int(pos[0]), int(pos[1]))


def mouse_pos():
    """Mouse position, or offscreen when no display window exists (headless)."""
    if _mouse_override is not None:
        return _mouse_override
    if pygame.display.get_surface() is None:
        return (-1, -1)
    return pygame.mouse.get_pos()


# --- Widget helpers -------------------------------------------------------------

def draw_button(surf, rect, label, fnt, *, active=False, disabled=False, radius=6):
    """Rounded button with hover feedback. Caller keeps hit-testing the rect."""
    hover = (not disabled) and rect.collidepoint(mouse_pos())
    if disabled:
        base = COLOR_BTN_DISABLED
    elif active:
        base = ACCENT
    else:
        base = SURFACE
    if hover:
        base = brighten(base, 14)
    pygame.draw.rect(surf, base, rect, border_radius=radius)
    pygame.draw.rect(surf, brighten(BORDER, 24) if hover else BORDER, rect, 1,
                     border_radius=radius)
    if label:
        tcol = (18, 24, 20) if active else (TEXT_DIM if disabled else TEXT)
        ts = fnt.render(label, True, tcol)
        surf.blit(ts, (rect.centerx - ts.get_width() // 2,
                       rect.centery - ts.get_height() // 2))
    return hover


def draw_chip(surf, rect, label, fnt, on, *, radius=13):
    """Pill-shaped toggle chip: filled with accent when on, outlined when off."""
    hover = rect.collidepoint(mouse_pos())
    if on:
        base = brighten(ACCENT, 14) if hover else ACCENT
        pygame.draw.rect(surf, base, rect, border_radius=radius)
        tcol = (18, 24, 20)
    else:
        base = brighten(SURFACE, 14) if hover else SURFACE
        pygame.draw.rect(surf, base, rect, border_radius=radius)
        pygame.draw.rect(surf, brighten(BORDER, 24) if hover else BORDER, rect, 1,
                         border_radius=radius)
        tcol = TEXT
    ts = fnt.render(label, True, tcol)
    surf.blit(ts, (rect.centerx - ts.get_width() // 2,
                   rect.centery - ts.get_height() // 2))


def draw_panel(surf, rect, *, color=PANEL_ALT, alpha=None, radius=8, border=BORDER):
    """Rounded panel; with `alpha` it is composited translucently."""
    if alpha is None:
        pygame.draw.rect(surf, color, rect, border_radius=radius)
        pygame.draw.rect(surf, border, rect, 1, border_radius=radius)
    else:
        tmp = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(tmp, with_alpha(color, alpha), tmp.get_rect(),
                         border_radius=radius)
        pygame.draw.rect(tmp, with_alpha(border, min(255, alpha + 70)),
                         tmp.get_rect(), 1, border_radius=radius)
        surf.blit(tmp, rect.topleft)
