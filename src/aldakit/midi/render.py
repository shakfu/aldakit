"""Offline rendering of a score to an audio file.

Playing a score puts it through the sound card in real time, which is what a
person wants and what a test cannot use: a five-minute score takes five minutes
and produces nothing that can be compared. This module runs the same
synthesizer with no audio device attached, as fast as the CPU allows, and
writes the result to a WAV file.

The synthesis itself is TinySoundFont in ``_tsf``, driven by exactly the loop
the audio callback uses, so a rendered file and the sound coming out of the
speakers cannot drift apart.
"""

from __future__ import annotations

import wave
from pathlib import Path

from ..constants import DEFAULT_SOUNDFONT_GAIN
from .soundfont import find_soundfont
from .types import MidiSequence

#: Audio rendered after the last note ends, for release tails and reverb.
#: A note that is still ringing when the score ends would otherwise be cut.
DEFAULT_TAIL_SECONDS = 0.5

#: Bytes per sample in the WAV files written here (16-bit PCM).
SAMPLE_WIDTH = 2

#: Interleaved stereo.
CHANNELS = 2


def render_pcm(
    sequence: MidiSequence,
    soundfont: str | Path | None = None,
    *,
    gain: float = DEFAULT_SOUNDFONT_GAIN,
    tail: float = DEFAULT_TAIL_SECONDS,
) -> tuple[bytes, int, float]:
    """Render a sequence to raw audio samples.

    Args:
        sequence: The MIDI sequence to render.
        soundfont: SoundFont to synthesize with. If None, the one playback
            would use is found the same way `aldakit play` finds it.
        gain: Global gain, 0.0 to 2.0.
        tail: Seconds rendered after the last note ends.

    Returns:
        ``(pcm, sample_rate, peak)``, where ``pcm`` is interleaved stereo
        16-bit little-endian samples and ``peak`` is the loudest sample
        before clamping. A peak above 1.0 means the render clipped and wants
        a lower gain.

    Raises:
        RuntimeError: If the TinySoundFont module was not built.
        FileNotFoundError: If no SoundFont is given and none can be found.
        ValueError: If the SoundFont cannot be loaded.
    """
    try:
        from .. import _tsf  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - depends on the build
        raise RuntimeError(
            "Rendering to audio needs the _tsf native module, which was not "
            "built or failed to load."
        ) from exc

    path = Path(soundfont) if soundfont is not None else find_soundfont()
    if path is None:
        raise FileNotFoundError(
            "No SoundFont found. Install one with 'aldakit soundfont install' "
            "or name one with --soundfont."
        )

    player = _tsf.TsfPlayer()
    if not player.load_soundfont(str(path)):
        raise ValueError(f"Could not load SoundFont: {path}")
    player.set_gain(gain)

    for change in sequence.program_changes:
        player.schedule_program(change.channel, change.program, change.time)
    for control in sequence.control_changes:
        player.schedule_control(
            control.channel, control.control, control.value, control.time
        )
    for note in sequence.notes:
        player.schedule_note(
            note.channel,
            note.pitch,
            note.velocity / 127.0,
            note.start_time,
            note.duration,
        )

    pcm = player.render_pcm16(tail)
    return pcm, player.sample_rate(), player.last_render_peak()


def render_wav(
    sequence: MidiSequence,
    path: str | Path,
    soundfont: str | Path | None = None,
    *,
    gain: float = DEFAULT_SOUNDFONT_GAIN,
    tail: float = DEFAULT_TAIL_SECONDS,
) -> tuple[Path, float]:
    """Render a sequence and write it to a WAV file.

    Args:
        sequence: The MIDI sequence to render.
        path: Output path. A ``.wav`` suffix is added if there is none.
        soundfont: SoundFont to synthesize with, or None to find one.
        gain: Global gain, 0.0 to 2.0.
        tail: Seconds rendered after the last note ends.

    Returns:
        ``(path, peak)``: the path written and the loudest sample before
        clamping, so a caller can report a render that clipped.
    """
    pcm, sample_rate, peak = render_pcm(sequence, soundfont, gain=gain, tail=tail)

    output = Path(path)
    if not output.suffix:
        output = output.with_suffix(".wav")

    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    return output, peak
