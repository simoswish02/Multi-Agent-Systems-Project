"""Pygame chart primitives for the multi-drone renderer.

Pure-pygame line and bar charts that draw into a caller-supplied rect on an
existing surface. No matplotlib: everything runs inside the same render loop.
All comments and labels are in English.

Each function is stateless — the renderer owns the history buffers and passes
them in on every frame.
"""

import pygame

from gui.theme import (
    CHART_AXIS   as COLOR_AXIS,
    CHART_TITLE  as COLOR_TITLE,
    CHART_LABEL  as COLOR_LABEL,
    CHART_DASH   as COLOR_DASH,
    CHART_PANEL  as COLOR_PANEL,
    CHART_BORDER as COLOR_BORDER,
    CHART_ZERO   as COLOR_ZERO,
)


def _fmt(v):
    """Compact numeric label for axis ticks."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _plot_rect(rect, has_title):
    """Inner plotting area: reserve top for the title, left for y-tick labels,
    bottom for the x-axis."""
    top    = rect.top + (15 if has_title else 3)
    left   = rect.left + 40
    right  = rect.right - 6
    bottom = rect.bottom - 12
    return pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))


def _draw_frame(surface, rect, font, title):
    pygame.draw.rect(surface, COLOR_PANEL, rect)
    pygame.draw.rect(surface, COLOR_BORDER, rect, 1)
    if title:
        surface.blit(font.render(title, True, COLOR_TITLE), (rect.left + 4, rect.top + 1))


def _span(values, y_min, y_max):
    """Resolve the y-range, autoscaling from `values` where not provided."""
    if y_min is None or y_max is None:
        lo = min(values) if values else 0.0
        hi = max(values) if values else 1.0
        if y_min is None:
            y_min = min(0.0, lo)
        if y_max is None:
            y_max = max(0.0, hi)
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0
    return y_min, y_max


def _draw_axes(surface, plot, font, y_min, y_max):
    pygame.draw.line(surface, COLOR_AXIS, (plot.left, plot.top), (plot.left, plot.bottom), 1)
    pygame.draw.line(surface, COLOR_AXIS, (plot.left, plot.bottom), (plot.right, plot.bottom), 1)
    surface.blit(font.render(_fmt(y_max), True, COLOR_LABEL), (plot.left - 38, plot.top - 3))
    surface.blit(font.render(_fmt(y_min), True, COLOR_LABEL), (plot.left - 38, plot.bottom - 9))
    # Zero reference line when the range straddles zero.
    if y_min < 0 < y_max:
        zy = int(plot.bottom - (0 - y_min) / (y_max - y_min) * plot.height)
        pygame.draw.line(surface, COLOR_ZERO, (plot.left, zy), (plot.right, zy), 1)


def draw_line_chart(surface, rect, series, font, title="", window=200,
                    y_min=None, y_max=None, hline=None, legend=None):
    """Draw one or more line series, right-aligned to the most recent `window`
    samples.

    Args:
        series:  list of (color, list_of_values). Each list is plotted against
                 its own index.
        hline:   optional y value drawn as a dashed horizontal reference line.
        legend:  optional list of (color, label) drawn top-right.
    """
    _draw_frame(surface, rect, font, title)
    plot = _plot_rect(rect, bool(title))

    flat = []
    for _color, vals in series:
        flat.extend(vals[-window:])
    y_min, y_max = _span(flat, y_min, y_max)
    _draw_axes(surface, plot, font, y_min, y_max)
    span = y_max - y_min

    def yof(v):
        return int(plot.bottom - (v - y_min) / span * plot.height)

    if hline is not None and y_min <= hline <= y_max:
        hy = yof(hline)
        x = plot.left
        while x < plot.right:
            pygame.draw.line(surface, COLOR_DASH, (x, hy), (min(x + 6, plot.right), hy), 1)
            x += 12

    step_x = plot.width / max(1, window - 1)
    for color, vals in series:
        data = vals[-window:]
        n = len(data)
        if n < 2:
            continue
        pts = []
        for k, v in enumerate(data):
            x = plot.right - (n - 1 - k) * step_x
            pts.append((int(x), yof(v)))
        pygame.draw.lines(surface, color, False, pts, 1)

    if legend:
        ly = plot.top + 1
        for color, label in legend:
            w = font.size(label)[0]
            surface.blit(font.render(label, True, color), (plot.right - w, ly))
            ly += 11


def draw_bar_chart(surface, rect, values, font, title="", window=100,
                   color=(90, 200, 110), y_max=None):
    """Draw a bar chart of the most recent `window` values, right-aligned."""
    _draw_frame(surface, rect, font, title)
    plot = _plot_rect(rect, bool(title))

    data = list(values[-window:])
    hi = max(data) if data else 1
    if y_max is not None:
        hi = max(hi, y_max)
    if hi <= 0:
        hi = 1
    _draw_axes(surface, plot, font, 0, hi)

    bw = plot.width / max(1, window)
    for k, v in enumerate(data):
        if v <= 0:
            continue
        x = plot.right - (len(data) - k) * bw
        h = (v / hi) * plot.height
        pygame.draw.rect(
            surface, color,
            pygame.Rect(int(x), int(plot.bottom - h), max(1, int(bw) - 1), int(h)),
        )
