"""Shared test helpers."""

from __future__ import annotations

import array
import hashlib
import math
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
GOLDEN_DIR = Path(__file__).parent / "golden"


def midi_fingerprint(sequence) -> dict:
    """A stable, comparable summary of everything a MidiSequence sounds like.

    Events are rendered as compact strings so the golden fixture stays readable
    and produces a reviewable diff. Times are rounded to microseconds so that
    floating point noise does not make comparisons brittle, while still
    catching real timing changes.
    """

    def t(value: float) -> str:
        return f"{value:.6f}"

    return {
        "notes": [
            f"{t(n.start_time)} ch{n.channel} p{n.pitch} d{t(n.duration)} v{n.velocity}"
            for n in sorted(
                sequence.notes,
                key=lambda n: (n.start_time, n.channel, n.pitch, n.duration),
            )
        ],
        "program_changes": sorted(
            f"{t(p.time)} ch{p.channel} prog{int(p.program)}"
            for p in sequence.program_changes
        ),
        "control_changes": sorted(
            f"{t(c.time)} ch{c.channel} cc{c.control}={c.value}"
            for c in sequence.control_changes
        ),
        "tempo_changes": sorted(
            f"{t(x.time)} {x.bpm:.6f}bpm" for x in sequence.tempo_changes
        ),
    }


# ---------------------------------------------------------------------------
# Audio fingerprints
#
# The MIDI fingerprint above pins what the generator decided. It cannot say
# whether any of it was heard: a program change on the wrong channel, a part
# left on a channel another part had taken, or a pan that never reached the
# synthesizer all produce well-formed MIDI. Rendering the score and pinning
# the shape of the audio is the only check that covers that.
#
# What is pinned is deliberately coarse -- loudness per channel over quarter
# second windows, plus the peak and the length -- rather than the samples
# themselves. It survives the small numerical differences between platforms
# and compilers while still failing if an instrument drops out, a part moves
# in time, a pan flips sides, or the mix changes level.

#: The audio fixtures are rendered with one specific SoundFont, because the
#: samples are the SoundFont's, not aldakit's: rendered with a different one,
#: every number here would be different and none of it would mean anything.
#: This one is in the download catalog, is checksum-pinned, and is small
#: enough (6 MB) for CI to fetch.
AUDIO_SOUNDFONT = "TimGM6mb"

#: Length of each window the audio is measured over, in seconds.
AUDIO_WINDOW_SECONDS = 0.25

#: Gain the fixtures are rendered at. Low enough that the densest example
#: (rachmaninoff_piano_concerto_2_mvmt_2.alda, which peaks six times higher
#: than the next loudest) still has headroom, because a mix that clips is
#: clamped, and a clamped mix that grew louder looks unchanged.
AUDIO_GAIN = 0.15

#: Silence after the last note, matching the renderer's default.
AUDIO_TAIL_SECONDS = 0.5

#: How far a window may drift before it counts as a change. Rendering is
#: float arithmetic, so two platforms need not agree to the last bit.
AUDIO_RELATIVE_TOLERANCE = 0.02
AUDIO_ABSOLUTE_TOLERANCE = 4


def _rms_windows(samples: array.array, window: int) -> list[int]:
    """Loudness of each window, in the same units as the samples."""
    levels = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            break
        total = 0
        for sample in chunk:
            total += sample * sample
        levels.append(round(math.sqrt(total / len(chunk))))
    return levels


def audio_fingerprint(pcm: bytes, sample_rate: int, peak: float) -> dict:
    """A stable, comparable summary of what a rendered score sounds like.

    Args:
        pcm: Interleaved stereo 16-bit samples, as the renderer returns them.
        sample_rate: Samples per second.
        peak: The loudest sample before clamping, which is what reports a
            change in level that clipping would otherwise hide.

    Returns:
        Frame count, peak, and the loudness of each window in each channel.
        The two channels are kept apart so that panning is pinned too.
    """
    samples = array.array("h")
    samples.frombytes(pcm)
    window = int(sample_rate * AUDIO_WINDOW_SECONDS)

    return {
        "frames": len(samples) // 2,
        "sample_rate": sample_rate,
        "peak": round(peak, 3),
        "left": _rms_windows(samples[0::2], window),
        "right": _rms_windows(samples[1::2], window),
    }


def read_wav(path: Path) -> tuple[bytes, int]:
    """The raw frames and sample rate of a WAV file."""
    with wave.open(str(path)) as wav:
        return wav.readframes(wav.getnframes()), wav.getframerate()


def pinned_soundfont() -> Path | None:
    """The SoundFont the audio fixtures were rendered with, if it is here.

    Returns None both when it is not installed and when the file that is
    installed under that name is not the one the fixtures were made with.
    Either way there is nothing to compare against, so callers skip.
    """
    from aldakit.midi.soundfont import SOUNDFONT_CATALOG, get_soundfont_dir

    info = SOUNDFONT_CATALOG[AUDIO_SOUNDFONT]
    path = get_soundfont_dir() / str(info["filename"])
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path if digest == info["sha256"] else None
