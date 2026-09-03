"""Global knobs for the demo build.

Everything the film's look depends on lives here so scenes stay declarative.
"""

import os

# --- Paths -------------------------------------------------------------------
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG     = os.path.join(ROOT, "configs", "default.yaml")
WEIGHTS    = os.path.join(ROOT, "checkpoints", "mas_50k_dr_faults_ep50000.pt")
CSV_DIR    = os.path.join(ROOT, "testing_results", "csv")
OUT        = os.path.join(ROOT, "demo", "out")
AUDIO_DIR  = os.path.join(OUT, "audio")
STILLS_DIR = os.path.join(OUT, "stills")
ASSETS     = os.path.join(ROOT, "demo", "assets")
SEEDS_JSON = os.path.join(ROOT, "demo", "seeds.json")

# --- Video -------------------------------------------------------------------
W, H = 1920, 1080
FPS  = 60
CRF  = 18                     # libx264 quality; lower is better, 18 ~ visually lossless

# Preview mode renders at half size with no audio, for fast iteration.
PREVIEW_SCALE = 0.5

# --- Captured GUI ------------------------------------------------------------
# Headless DroneRenderer returns CELL_DEFAULT directly, so this is the exact
# on-screen cell size. 32*28 = 896 px of grid + a 500 px sidebar -> 1396x940,
# which sits 1:1 inside the 1080p canvas with room for captions.
CELL = 28

# Env rounds per second of film, per scene mood.
RPS_FAST   = 14.0
RPS_NORMAL = 6.0
RPS_SLOW   = 1.6

# --- Narration ---------------------------------------------------------------
VOICE      = "en-GB-RyanNeural"
VOICE_RATE = "-4%"            # a touch slower than default reads as more deliberate
LEAD_IN    = 0.35             # silence before each beat's speech
TAIL       = 0.45             # silence after

# --- Music -------------------------------------------------------------------
MUSIC        = os.path.join(ASSETS, "music.mp3")   # optional; skipped if absent
MUSIC_GAIN   = -21.0          # dB under the narration
MUSIC_DUCK   = -6.0           # extra dB while narration plays

# --- Type --------------------------------------------------------------------
TITLE_FONT_PX   = 76
SUB_FONT_PX     = 30
LOWER_FONT_PX   = 27
CALLOUT_FONT_PX = 21
MONO_FONT_PX    = 20
