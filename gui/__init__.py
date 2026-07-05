"""GUI package for the multi-drone search visualiser.

`DroneRenderer` is the public entry point; `charts` and `terminal` are the
pure-pygame helper widgets it composes.
"""

from gui.renderer import DroneRenderer

__all__ = ["DroneRenderer"]
