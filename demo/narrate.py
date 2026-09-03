"""Narration: turn the beat script into audio, durations and subtitles.

Audio comes first in the pipeline. Each beat is synthesised, its real length
measured, and that length is what the picture is then cut to -- so the film is
in sync by construction rather than by adjustment.

Two back ends:
  edge  (default) Microsoft Edge neural voices. Free, no API key, good quality.
                  It sends the narration text to a Microsoft endpoint.
  sapi            Offline Windows SAPI5 via pyttsx3. Lower quality, no network.

Synthesised beats are cached on disk by (text, voice, rate, backend), so
re-running a build only re-synthesises what actually changed.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import wave

from demo import config as C
from demo.script import BEATS


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args):
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                        "-y"] + args,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: %s"
                           % p.stderr.decode("utf-8", "replace")[-2000:])


def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------------------
# Back ends
# ---------------------------------------------------------------------------

def _synth_edge(text, out_wav, voice, rate):
    import edge_tts

    mp3 = out_wav + ".mp3"

    async def go():
        comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(mp3)

    asyncio.run(go())
    _run(["-i", mp3, "-ar", "48000", "-ac", "1", out_wav])
    os.remove(mp3)


def _synth_sapi(text, out_wav, voice, rate):
    import pyttsx3

    raw = out_wav + ".raw.wav"
    eng = pyttsx3.init()
    for v in eng.getProperty("voices"):
        if "en" in (v.id or "").lower() or "english" in (v.name or "").lower():
            eng.setProperty("voice", v.id)
            break
    eng.setProperty("rate", 170)
    eng.save_to_file(text, raw)
    eng.runAndWait()
    eng.stop()
    _run(["-i", raw, "-ar", "48000", "-ac", "1", out_wav])
    os.remove(raw)


BACKENDS = {"edge": _synth_edge, "sapi": _synth_sapi}


# ---------------------------------------------------------------------------
# Building the narration track
# ---------------------------------------------------------------------------

def _key(text, voice, rate, backend):
    h = hashlib.sha1(("%s|%s|%s|%s" % (backend, voice, rate, text))
                     .encode("utf-8")).hexdigest()[:16]
    return h


def synthesize(backend="edge", voice=None, rate=None, verbose=True):
    """Synthesise every beat (cached) and return the timing table.

    Returns a list of dicts: id, scene, text, speech (s), start (s),
    duration (s), wav path. `duration` is what the scene must fill:
    lead-in + speech + tail + the beat's own floor and padding.
    """
    voice = voice or C.VOICE
    rate  = rate or C.VOICE_RATE
    os.makedirs(C.AUDIO_DIR, exist_ok=True)
    synth = BACKENDS[backend]

    table, t = [], 0.0
    for b in BEATS:
        wav = os.path.join(C.AUDIO_DIR, "%s_%s.wav"
                           % (b.id, _key(b.text, voice, rate, backend)))
        if not os.path.exists(wav):
            if verbose:
                print("  tts %-5s %s" % (b.id, b.text[:58] + "..."))
            synth(b.text, wav, voice, rate)
        speech = wav_duration(wav)
        dur = max(b.min_s, C.LEAD_IN + speech + C.TAIL) + b.pad_s
        table.append({
            "id": b.id, "scene": b.scene, "text": b.text, "wav": wav,
            "speech": speech, "start": t, "duration": dur,
        })
        t += dur

    if verbose:
        print("  narration: %d beats, %.1fs total" % (len(table), t))
    return table


def build_track(table, out_wav):
    """Concatenate the beats into one narration track, each padded to its slot."""
    parts = []
    lst = os.path.join(C.AUDIO_DIR, "_concat.txt")
    for r in table:
        padded = os.path.join(C.AUDIO_DIR, "_slot_%s.wav" % r["id"])
        tail = max(0.0, r["duration"] - C.LEAD_IN - r["speech"])
        _run(["-f", "lavfi", "-t", "%.3f" % C.LEAD_IN,
              "-i", "anullsrc=r=48000:cl=mono",
              "-i", r["wav"],
              "-f", "lavfi", "-t", "%.3f" % tail,
              "-i", "anullsrc=r=48000:cl=mono",
              "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[a]",
              "-map", "[a]", "-ar", "48000", "-ac", "1", padded])
        parts.append(padded)

    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    _run(["-f", "concat", "-safe", "0", "-i", lst,
          "-ar", "48000", "-ac", "2", out_wav])
    for p in parts:
        os.remove(p)
    os.remove(lst)
    return out_wav


def mix_music(voice_wav, out_wav, music=None, *, gain=None, duck=None):
    """Lay a music bed under the narration, ducked while anyone is speaking.

    Returns `voice_wav` unchanged when there is no track to mix, so callers can
    use this unconditionally. The bed is looped to the narration's length and
    side-chain compressed against the voice, so the words always sit on top.
    """
    music = music or C.MUSIC
    if not music or not os.path.exists(music):
        return voice_wav

    gain = C.MUSIC_GAIN if gain is None else gain
    duck = C.MUSIC_DUCK if duck is None else duck
    _run([
        "-i", voice_wav,
        "-stream_loop", "-1", "-i", music,
        "-filter_complex",
        "[1:a]volume=%.1fdB,aformat=sample_rates=48000:channel_layouts=stereo[bed];"
        "[bed][0:a]sidechaincompress=threshold=0.02:ratio=8:attack=15:"
        "release=420:makeup=1[ducked];"
        "[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
        "volume=%.1fdB[out]" % (gain, -duck * 0.0),
        "-map", "[out]", "-ar", "48000", "-ac", "2", out_wav,
    ])
    return out_wav


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

def _ts(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, sec, ms)


def write_srt(table, path, t0=0.0):
    """Subtitles covering exactly the spoken span of each beat.

    `t0` is the film time the first listed beat starts at, so a render of a
    subset of scenes gets cues timed from the start of *that* file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for k, r in enumerate(table, 1):
            a = r["start"] - t0 + C.LEAD_IN
            b = a + r["speech"]
            f.write("%d\n%s --> %s\n%s\n\n"
                    % (k, _ts(a), _ts(b), _wrap_srt(r["text"])))
    return path


def _wrap_srt(text, width=46):
    """Wrap a cue, keeping every word.

    Long beats run to four lines rather than being clipped: a subtitle that
    quietly drops the end of a sentence is worse than a tall one.
    """
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width or not cur:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def write_timing(table, path):
    """Human-readable timing sheet -- handy when tuning `min_s` floors."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "wav"} for r in table],
                  f, indent=2)
    return path
