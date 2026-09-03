"""Build the demo film.

    python -m demo.build --all                 full 1080p60 narrated render
    python -m demo.build --scene s2            one scene, with its narration
    python -m demo.build --scene s4 --preview  fast 960x540 30fps, no audio
    python -m demo.build --scene s3 --stills   dump frames instead of encoding
    python -m demo.build --list                beat timings, render nothing

Audio is synthesised first and its measured length decides how long each shot
runs, so the picture cannot drift out of sync with the words.
"""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from demo import config as C
from demo import narrate
from demo.script import SCENE_ORDER
from demo.scenes import SceneCtx, load as load_scene


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(frames, out_path, *, size, fps, audio=None, subs=None, scale=None,
           crf=None):
    """Pipe raw RGB frames into ffmpeg and mux the narration.

    Returns (frame_count, seconds_of_video).
    """
    w, h = size
    args = [
        narrate.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
        "-r", str(fps), "-i", "-",
    ]
    if audio:
        args += ["-i", audio]
    vf = []
    if scale and scale != 1.0:
        vf.append("scale=%d:%d" % (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2))
    if vf:
        args += ["-vf", ",".join(vf)]
    if subs:
        args += ["-i", subs]
    args += ["-c:v", "libx264", "-preset", "medium",
             "-crf", str(crf if crf is not None else C.CRF),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "192k"]
    if subs:
        # A soft track, so the file is self-contained and the subtitles stay
        # switchable. The sidecar .srt next to it is what upload forms want.
        args += ["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    if audio:
        args += ["-shortest"]
    args += [out_path]

    proc = subprocess.Popen(args, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    n, t0 = 0, time.time()
    try:
        for surf in frames:
            proc.stdin.write(pygame.image.tobytes(surf, "RGB"))
            n += 1
            if n % 120 == 0:
                el = time.time() - t0
                sys.stdout.write("\r    %5d frames  %6.1fs video  %4.1f fps"
                                 % (n, n / float(fps), n / max(el, 1e-6)))
                sys.stdout.flush()
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait()
    sys.stdout.write("\r" + " " * 62 + "\r")
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + err[-3000:])
    return n, n / float(fps)


def dump_stills(frames, out_dir, every=20, prefix="f"):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for k, surf in enumerate(frames):
        if k % every == 0:
            pygame.image.save(surf, os.path.join(out_dir, "%s_%05d.png"
                                                 % (prefix, k)))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Scene driving
# ---------------------------------------------------------------------------

def scene_frames(scene_ids, table, policy, preview=False):
    """Yield every frame of the requested scenes, in order, checking that each
    scene delivers exactly the number of frames its narration paid for."""
    for sid in scene_ids:
        beats = [r for r in table if r["scene"] == sid]
        if not beats:
            continue
        mod = load_scene(sid)
        ctx = SceneCtx(policy, beats, preview=preview)
        want = ctx.total_frames()
        print("  [%s] %-22s %5.1fs  %d frames"
              % (sid, mod.__name__.split(".")[-1], want / float(C.FPS), want))
        got = 0
        for surf in mod.render(ctx):
            got += 1
            yield surf
        if got != want:
            raise RuntimeError(
                "scene %s yielded %d frames, narration paid for %d "
                "(%.2fs of drift)" % (sid, got, want,
                                      abs(got - want) / float(C.FPS)))


def audio_for(scene_ids, table, path):
    """Narration track covering just these scenes."""
    sub = [r for r in table if r["scene"] in scene_ids]
    if not sub:
        return None
    track = narrate.build_track(sub, path)
    mixed = narrate.mix_music(track, path.replace(".wav", "_mixed.wav"))
    if mixed != track:
        print("  mixed in %s" % os.path.basename(C.MUSIC))
    return mixed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="render every scene")
    ap.add_argument("--scene", nargs="+", metavar="ID",
                    help="render only these scenes (s0 s1 ...)")
    ap.add_argument("--preview", action="store_true",
                    help="30 fps, half size, no audio -- fast iteration")
    ap.add_argument("--stills", action="store_true",
                    help="dump PNGs instead of encoding")
    ap.add_argument("--every", type=int, default=20,
                    help="with --stills, save one frame in N")
    ap.add_argument("--list", action="store_true",
                    help="print beat timings and exit")
    ap.add_argument("--tts", choices=sorted(narrate.BACKENDS), default="edge")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pygame.init()
    os.makedirs(C.OUT, exist_ok=True)

    print("narration (%s)..." % args.tts)
    table = narrate.synthesize(backend=args.tts, voice=args.voice)
    narrate.write_timing(table, os.path.join(C.OUT, "timing.json"))

    if args.list:
        print("\n%-5s %-4s %8s %8s  %s" % ("beat", "sc", "speech", "slot", "text"))
        for r in table:
            print("%-5s %-4s %7.2fs %7.2fs  %s"
                  % (r["id"], r["scene"], r["speech"], r["duration"],
                     r["text"][:60]))
        print("\ntotal %.1fs (%d:%02d)"
              % (table[-1]["start"] + table[-1]["duration"],
                 int(table[-1]["start"] + table[-1]["duration"]) // 60,
                 int(table[-1]["start"] + table[-1]["duration"]) % 60))
        return

    scene_ids = args.scene if args.scene else SCENE_ORDER
    if not (args.all or args.scene or args.stills):
        scene_ids = SCENE_ORDER

    if args.preview:
        C.FPS = 30

    print("loading policy...")
    from demo.capture import make_policy
    policy = make_policy()

    frames = scene_frames(scene_ids, table, policy, preview=args.preview)

    if args.stills:
        out_dir = args.out or os.path.join(C.STILLS_DIR, "_".join(scene_ids))
        n = dump_stills(frames, out_dir, every=args.every)
        print("wrote %d stills to %s" % (n, out_dir))
        return

    audio = None
    if not (args.preview or args.no_audio):
        print("mixing narration...")
        audio = audio_for(scene_ids, table,
                          os.path.join(C.AUDIO_DIR, "_narration.wav"))

    # The sidecar takes the video's own basename: players (VLC, mpv, MPC)
    # auto-load "<name>.srt" next to "<name>.mp4", and nothing else.
    tag = "demo" if scene_ids == SCENE_ORDER else "_".join(scene_ids)
    suffix = "_preview" if args.preview else "_1080p"
    out = args.out or os.path.join(C.OUT, tag + suffix + ".mp4")

    srt = None
    if not args.preview:
        srt = narrate.write_srt(
            [r for r in table if r["scene"] in scene_ids],
            os.path.splitext(out)[0] + ".srt",
            t0=next(r["start"] for r in table if r["scene"] in scene_ids))

    print("rendering...")
    t0 = time.time()
    n, secs = encode(frames, out, size=(C.W, C.H), fps=C.FPS, audio=audio,
                     subs=srt,
                     scale=C.PREVIEW_SCALE if args.preview else None,
                     crf=23 if args.preview else None)
    print("wrote %s -- %d frames, %.1fs of video, built in %.0fs"
          % (out, n, secs, time.time() - t0))
    if srt:
        print("       %s (sidecar; also embedded as a soft track)"
              % os.path.basename(srt))


if __name__ == "__main__":
    main()
