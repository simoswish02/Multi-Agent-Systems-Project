"""Scrollable terminal-style log widget for the renderer (pure pygame).

The widget keeps a colour-coded scrollback buffer. By default it sticks to the
latest line (auto-scroll); the mouse wheel scrolls back into history and the
widget resumes auto-scrolling once the user returns to the bottom.
All comments and labels are in English.
"""

import pygame

from gui.theme import LOG_BG as COLOR_BG, BORDER as COLOR_BORDER


class TerminalLog:
    """A colour-coded scrollback text panel.

    `scroll` counts how many lines we are offset *up* from the bottom:
    0 means stuck to the latest line (auto-scroll on new entries).
    """

    def __init__(self, max_lines=4000, line_h=14):
        self.lines = []          # list of (text, color)
        self.scroll = 0
        self.max_lines = max_lines
        self.line_h = line_h
        self._auto = True

    def clear(self):
        self.lines.clear()
        self.scroll = 0
        self._auto = True

    def add(self, text, color):
        self.lines.append((text, color))
        if len(self.lines) > self.max_lines:
            del self.lines[: len(self.lines) - self.max_lines]
        if self._auto:
            self.scroll = 0   # keep pinned to the bottom

    def add_block(self, texts, color):
        for t in texts:
            self.add(t, color)

    def scroll_by(self, delta_lines):
        """Positive `delta_lines` scrolls up into history; negative scrolls down."""
        self.scroll = max(0, self.scroll + delta_lines)
        self._auto = (self.scroll == 0)

    def render(self, surface, rect, font):
        pygame.draw.rect(surface, COLOR_BG, rect)
        pygame.draw.rect(surface, COLOR_BORDER, rect, 1)

        visible = max(1, (rect.height - 6) // self.line_h)
        total = len(self.lines)
        # Clamp the scroll offset to the available history.
        self.scroll = min(self.scroll, max(0, total - visible))
        end = total - self.scroll
        start = max(0, end - visible)
        chunk = self.lines[start:end]

        prev_clip = surface.get_clip()
        surface.set_clip(rect)
        y = rect.bottom - 4 - len(chunk) * self.line_h
        for text, color in chunk:
            surface.blit(font.render(text, True, color), (rect.left + 6, y))
            y += self.line_h
        surface.set_clip(prev_clip)
