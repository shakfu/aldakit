"""High-level convenience functions for working with Alda music."""

from __future__ import annotations

from pathlib import Path

from .midi.backends import LibremidiBackend
from .score import PlaybackHandle, Score


def play(
    source: str, port: str | None = None, wait: bool = True
) -> PlaybackHandle | None:
    """Parse and play Alda source code.

    Args:
        source: Alda source code string.
        port: MIDI output port name. If None, uses the first available
            port or creates a virtual port named "AldakitMIDI".
        wait: If True (default), block until playback completes. If False,
            playback continues in the background and a handle is returned.

    Returns:
        None when ``wait`` is True, otherwise a
        :class:`~aldakit.score.PlaybackHandle`. Keep a reference to it:
        letting it be garbage collected stops playback.

    Examples:
        >>> import aldakit
        >>> aldakit.play("piano: c d e f g")
        >>> handle = aldakit.play("piano: c d e", port="FluidSynth", wait=False)
        >>> handle.stop()
    """
    score = Score(source)
    return score.play(port=port, wait=wait)


def play_file(
    path: str | Path, port: str | None = None, wait: bool = True
) -> PlaybackHandle | None:
    """Parse and play an Alda file.

    Args:
        path: Path to the Alda file.
        port: MIDI output port name. If None, uses the first available
            port or creates a virtual port named "AldakitMIDI".
        wait: If True (default), block until playback completes. If False,
            playback continues in the background and a handle is returned.

    Returns:
        None when ``wait`` is True, otherwise a
        :class:`~aldakit.score.PlaybackHandle`.

    Examples:
        >>> import aldakit
        >>> aldakit.play_file("song.alda")
    """
    score = Score.from_file(path)
    return score.play(port=port, wait=wait)


def save(source: str, path: str | Path) -> None:
    """Parse Alda source code and save as a MIDI file.

    Args:
        source: Alda source code string.
        path: Output MIDI file path.

    Examples:
        >>> import aldakit
        >>> aldakit.save("piano: c d e f g", "output.mid")
    """
    score = Score(source)
    score.save(path)


def save_file(source_path: str | Path, output_path: str | Path) -> None:
    """Parse an Alda file and save as a MIDI file.

    Args:
        source_path: Path to the Alda file.
        output_path: Output MIDI file path.

    Examples:
        >>> import aldakit
        >>> aldakit.save_file("song.alda", "song.mid")
    """
    score = Score.from_file(source_path)
    score.save(output_path)


def render(
    source: str,
    path: str | Path,
    soundfont: str | Path | None = None,
) -> Path:
    """Parse Alda source code and render it to a WAV file.

    Rendering needs no audio device and runs as fast as the CPU allows,
    unlike `play()`, which takes as long as the score lasts.

    Args:
        source: Alda source code string.
        path: Output WAV file path.
        soundfont: SoundFont to synthesize with, or None to find one.

    Returns:
        The path written.

    Examples:
        >>> import aldakit
        >>> aldakit.render("piano: c d e f g", "scale.wav")
    """
    return Score(source).render(path, soundfont)


def render_file(
    source_path: str | Path,
    output_path: str | Path,
    soundfont: str | Path | None = None,
) -> Path:
    """Parse an Alda file and render it to a WAV file.

    Args:
        source_path: Path to the Alda file.
        output_path: Output WAV file path.
        soundfont: SoundFont to synthesize with, or None to find one.

    Returns:
        The path written.

    Examples:
        >>> import aldakit
        >>> aldakit.render_file("song.alda", "song.wav")
    """
    return Score.from_file(source_path).render(output_path, soundfont)


def list_ports() -> list[str]:
    """List available MIDI output ports.

    Returns:
        List of MIDI output port names.

    Examples:
        >>> import aldakit
        >>> ports = aldakit.list_ports()
        >>> print(ports)
        ['IAC Driver Bus 1', 'FluidSynth virtual port']
    """
    backend = LibremidiBackend()
    return backend.list_output_ports()
