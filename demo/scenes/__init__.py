"""Scene API.

A scene is a module exposing ``render(ctx)``, a generator yielding finished
1920x1080 ``pygame.Surface`` frames. It is handed the narration beats that
belong to it, already timed, and must yield exactly ``ctx.frames(beat)`` frames
for each -- so the picture always fills the words.
"""

import importlib

from demo import config as C


class SceneCtx:
    """What a scene is given: the policy, its beats, and the clock."""

    def __init__(self, policy, beats, *, preview=False):
        self.policy  = policy
        self.beats   = beats            # list of narrate.synthesize() rows
        self.preview = preview

    # -- timing ------------------------------------------------------------
    @property
    def fps(self):
        return C.FPS

    def frames(self, beat):
        """How many video frames this beat's picture must cover."""
        return max(1, int(round(beat["duration"] * C.FPS)))

    def total_frames(self):
        return sum(self.frames(b) for b in self.beats)

    def beat(self, beat_id):
        for b in self.beats:
            if b["id"] == beat_id:
                return b
        raise KeyError("%s not in this scene" % beat_id)

    def text(self, beat_id):
        return self.beat(beat_id)["text"]


def load(scene_id):
    """Import a scene module by its id (``s2`` -> ``demo.scenes.s2_*``)."""
    for name in _MODULES:
        if name.startswith(scene_id + "_"):
            return importlib.import_module("demo.scenes." + name)
    raise KeyError("no scene module for %r" % scene_id)


_MODULES = [
    "s0_cold_open",
    "s1_setup",
    "s2_search",
    "s3_fault",
    "s4_network",
    "s5_generalize",
    "s6_outro",
]
