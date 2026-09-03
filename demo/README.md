# `demo/` — the project's demo film

A ~4-minute narrated video of the system, generated end to end by code. There
is no editing step and no hand-made asset: every frame is rendered from the
real GUI driving the real policy, and every number on screen is read from the
real experiment CSVs at build time.

```bash
conda activate rl-drone
pip install -r demo/requirements.txt     # imageio-ffmpeg, edge-tts

python -m demo.build --all               # demo/out/demo_1080p.mp4 (+ demo.srt)
```

## What it shows

| # | Scene | Content |
|---|-------|---------|
| 0 | Cold open | The hero mission at speed, up to the first drone failing. Title. |
| 1 | Setup | The real `ConfigScreen`, sliders driven under a synthetic cursor. |
| 2 | Cooperative search | Fog, stigmergic dispersion, a map merge, the four private beliefs, the live charts. |
| 3 | Failure | A drone breaks; the news spreads teammate by teammate. |
| 4 | Inside the policy | The six belief planes, the 13×13 crop, the context vector, the trunk, the Q-values. |
| 5 | Generalization | Four worlds at once, two of them outside training, one checkpoint. |
| 6 | Results | Success rates read from `testing_results/csv/`, then the sign-off. |

## How it works

**Audio first.** `demo/script.py` holds the narration as a list of `Beat`s.
`demo/narrate.py` synthesises each one, measures its real length, and that
length is what the picture is cut to — so audio and video are in sync by
construction rather than by adjustment. Editing a line of narration and
re-running is all it takes to re-time the film.

```bash
python -m demo.build --list              # beat timings, render nothing
```

**The footage is the real GUI.** `demo/capture.py` drives `DroneSearchEnv` and
`gui.renderer.DroneRenderer` headlessly, in a loop that mirrors
`main.py::run_simulate` exactly — sequential per-drone turns, wrecks still
burning a turn, `set_domain_params` before every `reset`. The only difference
is that it renders several frames per round and glides the drones between
cells, instead of one static frame per round.

**The casting is searched, not staged.** Scenes 0, 2, 3 and 4 are the same
mission at different moments. `demo/seeds.py` searches seeds until a real
episode behaves the way the film needs (one or two failures, the first one late
enough that the team has spread out, still running afterwards) and records it in
`demo/seeds.json`. A fixed `reset(seed)` reproduces the layout *and* the fault
sequence, so the run is repeatable.

```bash
python -m demo.seeds --scout             # re-cast (slow: it runs real episodes)
python -m demo.seeds --show              # what is currently cast
```

**Scene 4 is checked, not asserted.** It claims to show what the network
actually receives and decides, so `demo/tensorviz/verify.py` re-derives all of
it — the planes against `obs`, the crop against
`CnnQNetwork._extract_local_patch`, the chips against `ctx_to_numpy`, the
highlighted bar against the action `select_action` returns — and the build
fails rather than ship a convincing picture of something false.

```bash
python -m demo.tensorviz.verify
```

## Iterating

```bash
python -m demo.build --scene s4 --preview          # 960x540, 30 fps, no audio
python -m demo.build --scene s3 --stills --every 60 # PNGs instead of a video
python -m demo.build --scene s2 s3                  # a couple of scenes, with audio
```

Look and pacing live in `demo/config.py` (resolution, frame rate, captured cell
size, voice, rounds-per-second presets). Type and chrome are in `demo/typo.py`;
camera and compositing in `demo/compositor.py`; the shot grammar the
GUI-footage scenes share is `demo/director.py`.

One trap worth knowing: `lower_third`, `callout` and `title_card` fade *out* at
the end of the value they are given, so hand them `typo.span(t, start, end)`
rather than a clamped expression that saturates at 1.0 — otherwise the element
parks at "fading out" and never appears.

## Narration

`edge-tts` is the default back end: free, no API key, neural quality. **It sends
the narration text to a Microsoft endpoint.** For a fully offline build:

```bash
pip install pyttsx3
python -m demo.build --all --tts sapi    # Windows SAPI5, lower quality
```

Synthesised beats are cached under `demo/out/audio/` by content, so re-running a
build only re-synthesises lines that actually changed.

## Publishing

```bash
python -m demo.thumbnail            # 1280x720 thumbnails -> demo/out/thumbs/
```

Three variants, all rendered from the real mission and the real observation
planes rather than mocked up: **a** the healthy swarm, **b** the wreck (the
strongest of the three), **c** the six belief planes.

`demo/out/youtube.md` holds a ready-to-paste title, description and chapter
list. The chapter timestamps are derived from `demo/out/timing.json`, so they
stay correct if the narration is re-cut — regenerate them rather than editing
by hand.

Subtitles reach a viewer three ways: a soft track inside the `.mp4`, the
`demo_1080p.srt` sidecar (same basename, so players auto-load it — this is also
what YouTube wants), and `demo_720p_subbed.mp4` with them burned in for places
that ignore both.

## Music

Optional and off by default. Drop a royalty-free track at
`demo/assets/music.mp3` and it will be mixed under the narration; without it the
film is voice-only.

## Notes

- Rendering the full film takes roughly ten minutes on CPU; the policy is a
  22 M-parameter network and there is no CUDA on this machine.
- `demo/out/` is gitignored.
- Nothing in `demo/` is imported by training or evaluation. The only changes it
  needed elsewhere were an optional virtual clock and tweened positions on
  `DroneRenderer.draw_offscreen`, a synthetic-cursor override in `gui/theme.py`,
  and the obstacle-density slider the setup screen was missing.
