"""High-level convenience functions for working with Alda music."""

from __future__ import annotations

from pathlib import Path

from .score import Score
from .midi.backends import LibremidiBackend


def play(source: str, port: str | None = None, wait: bool = True) -> None:
    """Parse and play Alda source code.

    Args:
        source: Alda source code string.
        port: MIDI output port name. If None, uses the first available
            port or creates a virtual port named "AldaPyMIDI".
        wait: If True (default), block until playback completes.

    Examples:
        >>> import aldakit
        >>> aldakit.play("piano: c d e f g")
        >>> aldakit.play("piano: c d e", port="FluidSynth", wait=False)
    """
    score = Score(source)
    score.play(port=port, wait=wait)


def play_file(path: str | Path, port: str | None = None, wait: bool = True) -> None:
    """Parse and play an Alda file.

    Args:
        path: Path to the Alda file.
        port: MIDI output port name. If None, uses the first available
            port or creates a virtual port named "AldaPyMIDI".
        wait: If True (default), block until playback completes.

    Examples:
        >>> import aldakit
        >>> aldakit.play_file("song.alda")
    """
    score = Score.from_file(path)
    score.play(port=port, wait=wait)


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
