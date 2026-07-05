"""
config_screen.py — Pre-simulation setup GUI for ``--mode simulate``.

A standalone pygame screen with sliders + a weights-path field. ``run()`` blocks
until the user presses *Start simulation* (returns the chosen parameters as a
dict) or closes the window (returns ``None``). Styled to match
``gui/renderer.py`` (dark theme, monospace). In this mode the spawn is fixed to
the top-left corner — shown as a note.

All UI strings are in English; comments in English.
"""

import os
import pygame

# Palette consistent with gui/renderer.py
COLOR_BG       = (15, 15, 15)
COLOR_TEXT     = (235, 235, 235)
COLOR_LABEL    = (150, 150, 150)
COLOR_BORDER   = (70, 70, 78)
COLOR_ACCENT   = (90, 200, 120)
COLOR_TRACK    = (55, 55, 64)
COLOR_HANDLE   = (120, 200, 255)
COLOR_BTN      = (55, 55, 64)
COLOR_BTN_GO   = (90, 200, 120)
COLOR_ERR      = (230, 110, 110)
COLOR_FIELD    = (18, 18, 22)
COLOR_FIELD_ON = (30, 30, 40)


class Slider:
    """Horizontal integer slider with a draggable handle."""

    def __init__(self, x, y, w, label, lo, hi, value, unit=""):
        self.track = pygame.Rect(x, y, w, 6)
        self.label = label
        self.lo, self.hi = lo, hi
        self.value = int(min(hi, max(lo, value)))
        self.unit = unit
        self.dragging = False

    def _hx(self):
        t = (self.value - self.lo) / max(1, (self.hi - self.lo))
        return int(self.track.x + t * self.track.w)

    def _handle_rect(self):
        return pygame.Rect(self._hx() - 7, self.track.y - 7, 14, 20)

    def _set_from_x(self, px):
        t = (px - self.track.x) / max(1, self.track.w)
        t = min(1.0, max(0.0, t))
        self.value = int(round(self.lo + t * (self.hi - self.lo)))

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self._handle_rect().inflate(10, 10).collidepoint(ev.pos) or \
               self.track.inflate(0, 16).collidepoint(ev.pos):
                self.dragging = True
                self._set_from_x(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(ev.pos[0])

    def draw(self, surf, font):
        surf.blit(font.render(self.label, True, COLOR_LABEL), (self.track.x, self.track.y - 22))
        val = f"{self.value}{self.unit}"
        vw = font.size(val)[0]
        surf.blit(font.render(val, True, COLOR_TEXT), (self.track.right - vw, self.track.y - 22))
        pygame.draw.rect(surf, COLOR_TRACK, self.track)
        filled = pygame.Rect(self.track.x, self.track.y, self._hx() - self.track.x, self.track.h)
        pygame.draw.rect(surf, COLOR_ACCENT, filled)
        pygame.draw.rect(surf, COLOR_HANDLE, self._handle_rect())


class TextField:
    """Single-line editable text field (click to focus, type, backspace)."""

    def __init__(self, x, y, w, h, text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.focused = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.focused = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.KEYDOWN and self.focused:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
                self.focused = False
            elif ev.unicode and ev.unicode.isprintable():
                self.text += ev.unicode

    def draw(self, surf, font):
        pygame.draw.rect(surf, COLOR_FIELD_ON if self.focused else COLOR_FIELD, self.rect)
        pygame.draw.rect(surf, COLOR_ACCENT if self.focused else COLOR_BORDER, self.rect, 1)
        # Right-align the visible tail so the end of long paths stays readable.
        max_w = self.rect.w - 12
        txt = self.text
        prefix = ""
        while txt and font.size(prefix + txt)[0] > max_w:
            txt = txt[1:]
            prefix = "..."
        shown = prefix + txt
        surf.blit(font.render(shown, True, COLOR_TEXT),
                  (self.rect.x + 6, self.rect.y + (self.rect.h - 14) // 2))
        if self.focused:
            cx = self.rect.x + 6 + font.size(shown)[0] + 1
            pygame.draw.line(surf, COLOR_TEXT, (cx, self.rect.y + 5), (cx, self.rect.bottom - 5), 1)


def _browse(initialdir):
    """Native file picker via tkinter (stdlib). Returns a path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select network weights",
            initialdir=initialdir if os.path.isdir(initialdir) else ".",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


class ConfigScreen:
    """Setup screen. ``run()`` -> selections dict, or ``None`` if the user quits.

    Slider ranges for vision / comm / drones are bounded by the
    ``domain_randomization`` maxima so the run stays coherent with the context
    normalisation the network was trained with.
    """

    W, H = 560, 560

    def __init__(self, config: dict):
        self.config = config
        dr = config.get("domain_randomization", {})
        self.v_max = int(dr.get("vision_radius", [1, 6])[1])
        self.c_max = int(dr.get("comm_range", [1, 12])[1])
        self.a_max = int(dr.get("n_agents", [1, 4])[1])
        self.default_max_steps = int(config.get("env", {}).get("max_steps", 200))
        self.ckpt_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints/")
        self.default_weights = os.path.join(self.ckpt_dir, "best.pt")

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Simulation setup - Multi-Drone Search")
        clock = pygame.time.Clock()
        font   = pygame.font.SysFont("monospace", 14)
        font_b = pygame.font.SysFont("monospace", 18, bold=True)
        font_s = pygame.font.SysFont("monospace", 12)

        pad = 28
        x = pad
        w = self.W - 2 * pad
        gap = 56
        y = 84
        s_agents = Slider(x, y, w, "Number of drones", 1, self.a_max, min(2, self.a_max)); y += gap
        s_vision = Slider(x, y, w, "Vision radius", 1, self.v_max, min(3, self.v_max)); y += gap
        s_comm   = Slider(x, y, w, "Communication radius", 1, self.c_max, min(5, self.c_max)); y += gap
        s_steps  = Slider(x, y, w, "Max steps", 50, 600, self.default_max_steps); y += gap
        s_eps    = Slider(x, y, w, "Episodes to simulate", 1, 50, 10); y += gap
        sliders = [s_agents, s_vision, s_comm, s_steps, s_eps]

        wlabel_y = y + 2
        field = TextField(x, wlabel_y + 18, w - 110, 30, self.default_weights)
        browse_rect = pygame.Rect(field.rect.right + 8, field.rect.y, 102, 30)
        start_rect = pygame.Rect(x, self.H - 56, w, 38)

        error = ""
        result = None
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    result = None
                    break
                for s in sliders:
                    s.handle(ev)
                field.handle(ev)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if browse_rect.collidepoint(ev.pos):
                        picked = _browse(self.ckpt_dir)
                        if picked:
                            field.text = picked
                            error = ""
                    elif start_rect.collidepoint(ev.pos):
                        path = field.text.strip() or self.default_weights
                        if not os.path.isfile(path):
                            error = f"Weights file not found: {path}"
                        else:
                            result = {
                                "n_agents":      s_agents.value,
                                "vision_radius": s_vision.value,
                                "comm_range":    s_comm.value,
                                "max_steps":     s_steps.value,
                                "episodes":      s_eps.value,
                                "weights":       path,
                            }
                            running = False

            screen.fill(COLOR_BG)
            screen.blit(font_b.render("Simulation setup", True, COLOR_TEXT), (pad, 26))
            for s in sliders:
                s.draw(screen, font)
            screen.blit(font.render("Network weights (.pt)", True, COLOR_LABEL), (x, wlabel_y))
            field.draw(screen, font)
            pygame.draw.rect(screen, COLOR_BTN, browse_rect)
            pygame.draw.rect(screen, COLOR_BORDER, browse_rect, 1)
            blbl = "Browse..."
            bw = font.size(blbl)[0]
            screen.blit(font.render(blbl, True, COLOR_TEXT),
                        (browse_rect.centerx - bw // 2, browse_rect.y + 8))

            screen.blit(font_s.render("Spawn is forced to the top-left corner (0,0).",
                                      True, COLOR_LABEL), (x, start_rect.y - 26))
            if error:
                screen.blit(font_s.render(error, True, COLOR_ERR), (x, start_rect.y - 44))

            pygame.draw.rect(screen, COLOR_BTN_GO, start_rect)
            pygame.draw.rect(screen, COLOR_BORDER, start_rect, 1)
            slbl = "Start simulation"
            sw = font_b.size(slbl)[0]
            screen.blit(font_b.render(slbl, True, (15, 25, 18)),
                        (start_rect.centerx - sw // 2, start_rect.y + 8))

            pygame.display.flip()
            clock.tick(60)

        return result
