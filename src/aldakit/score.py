"""High-level Score class for working with Alda music."""

from __future__ import annotations

import time
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from .parser import parse
from .midi.generator import generate_midi
from .midi.backends import LibremidiBackend

if TYPE_CHECKING:
    from .ast_nodes import RootNode
    from .midi.types import MidiSequence


class Score:
    """A parsed Alda score with lazy MIDI generation.

    The Score class provides a high-level interface for working with Alda music.
    It lazily parses and generates MIDI only when needed, caching results for
    repeated access.

    Examples:
        >>> score = Score("piano: c d e f g")
        >>> score.play()

        >>> score = Score.from_file("song.alda")
        >>> score.save("output.mid")

        >>> # Access internals when needed
        >>> print(score.duration)
        >>> print(score.ast)
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        """Create a Score from Alda source code.

        Args:
            source: Alda source code string.
            filename: Optional filename for error messages.
        """
        self._source = source
        self._filename = filename

    @classmethod
    def from_file(cls, path: str | Path) -> Score:
        """Create a Score from an Alda file.

        Args:
            path: Path to the Alda file.

        Returns:
            A new Score instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        source = path.read_text(encoding="utf-8")
        return cls(source, filename=str(path))

    @property
    def source(self) -> str:
        """The original Alda source code."""
        return self._source

    @cached_property
    def ast(self) -> RootNode:
        """The parsed AST (lazily computed and cached)."""
        return parse(self._source, self._filename)

    @cached_property
    def midi(self) -> MidiSequence:
        """The generated MIDI sequence (lazily computed and cached)."""
        return generate_midi(self.ast)

    @property
    def duration(self) -> float:
        """Total duration of the score in seconds."""
        return self.midi.duration()

    def play(self, port: str | None = None, wait: bool = True) -> None:
        """Play the score through a MIDI port.

        Args:
            port: MIDI output port name. If None, uses the first available
                port or creates a virtual port named "AldaPyMIDI".
            wait: If True (default), block until playback completes.
        """
        with LibremidiBackend(port_name=port) as backend:
            backend.play(self.midi)
            if wait:
                while backend.is_playing():
                    time.sleep(0.1)

    def save(self, path: str | Path) -> None:
        """Save the score as a MIDI file.

        Args:
            path: Output file path.
        """
        backend = LibremidiBackend()
        backend.save(self.midi, path)

    def __repr__(self) -> str:
        # Show first 50 chars of source, truncated if longer
        preview = self._source[:50]
        if len(self._source) > 50:
            preview += "..."
        preview = preview.replace("\n", "\\n")
        return f"Score({preview!r})"
