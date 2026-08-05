"""High-level Score class for working with Alda music."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ast_nodes import EventSequenceNode, PartNode, RootNode
from .constants import POLL_INTERVAL_PLAYBACK
from .midi.backends import LibremidiBackend
from .midi.generator import Diagnostic, MidiGenerator
from .midi.smf import write_midi_file
from .parser import parse

if TYPE_CHECKING:
    from .compose.base import ComposeElement
    from .midi.types import MidiSequence


# Mode constants for internal state
_MODE_SOURCE = "source"
_MODE_ELEMENTS = "elements"
_MODE_MIDI = "midi"


def _ast_to_alda(ast: RootNode) -> str:
    """Convert an AST back to Alda source code.

    Deprecated alias for :func:`aldakit.serialize.write_alda`, kept for
    backwards compatibility.
    """
    from .serialize import write_alda

    return write_alda(ast)


class PlaybackHandle:
    """Controls playback that was started without waiting for it to finish.

    Returned by ``Score.play(wait=False)``. The handle owns the backend: the
    MIDI port stays open and the playback threads keep running until you call
    :meth:`stop` or :meth:`close`, or until the handle is used as a context
    manager and the block exits.

    Examples:
        >>> handle = Score("piano: c1 d1 e1").play(wait=False)
        >>> handle.is_playing()
        True
        >>> handle.stop()
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._closed = False

    @property
    def backend(self) -> Any:
        """The underlying MIDI or audio backend."""
        return self._backend

    def is_playing(self) -> bool:
        """Whether playback is still in progress."""
        if self._closed:
            return False
        return self._backend.is_playing()

    def wait(self, poll_interval: float = POLL_INTERVAL_PLAYBACK) -> None:
        """Block until playback finishes."""
        if self._closed:
            return
        self._backend.wait(poll_interval)

    def stop(self) -> None:
        """Stop playback and release the backend."""
        self.close()

    def close(self) -> None:
        """Stop playback and release the backend. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._backend, "close", None)
        if close is not None:
            close()
        else:
            self._backend.stop()

    def __enter__(self) -> PlaybackHandle:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class Score:
    """A unified score class for parsing, building, and playing music.

    The Score class provides a high-level interface for working with Alda music.
    It supports multiple construction methods:

    - From Alda source code: `Score("piano: c d e")` or `Score.from_source(...)`
    - From Alda file: `Score.from_file("song.alda")`
    - From compose elements: `Score.from_elements(part("piano"), note("c"), ...)`

    Examples:
        >>> # From source code
        >>> score = Score("piano: c d e f g")
        >>> score.play()

        >>> # From file
        >>> score = Score.from_file("song.alda")
        >>> score.save("output.mid")

        >>> # From compose elements
        >>> from aldakit.compose import part, note, tempo
        >>> score = Score.from_elements(
        ...     part("piano"),
        ...     tempo(120),
        ...     note("c", duration=4),
        ...     note("d"),
        ...     note("e"),
        ... )
        >>> score.play()

        >>> # Builder pattern
        >>> score = Score.from_elements(part("piano"))
        >>> score.add(note("c"), note("d"), note("e"))
        >>> score.play()
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        """Create a Score from Alda source code.

        Args:
            source: Alda source code string.
            filename: Optional filename for error messages.
        """
        self._mode = _MODE_SOURCE
        self._source = source
        self._filename = filename
        self._elements: list[ComposeElement] = []
        self._imported_ast: RootNode | None = None
        self._diagnostics: list[Diagnostic] = []
        self._playback: PlaybackHandle | None = None

    @classmethod
    def from_source(cls, source: str, filename: str = "<input>") -> Score:
        """Create a Score from Alda source code.

        This is equivalent to calling Score(source, filename) directly.

        Args:
            source: Alda source code string.
            filename: Optional filename for error messages.

        Returns:
            A new Score instance.
        """
        return cls(source, filename)

    @classmethod
    def from_file(cls, path: str | Path) -> Score:
        """Create a Score from an Alda or MIDI file.

        Args:
            path: Path to the Alda (.alda) or MIDI (.mid, .midi) file.

        Returns:
            A new Score instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        path = Path(path)

        if path.suffix.lower() in (".mid", ".midi"):
            return cls.from_midi_file(path)
        elif path.suffix.lower() == ".alda":
            source = path.read_text(encoding="utf-8")
            return cls(source, filename=str(path))
        else:
            # Try to read as Alda source
            source = path.read_text(encoding="utf-8")
            return cls(source, filename=str(path))

    @classmethod
    def from_midi_file(
        cls,
        path: str | Path,
        *,
        quantize_grid: float = 0.25,
    ) -> Score:
        """Create a Score by importing a MIDI file.

        This converts the MIDI file to an AST representation, allowing
        it to be played, exported to Alda, or further manipulated.

        Args:
            path: Path to the MIDI file.
            quantize_grid: Grid size in beats for quantization (0.25 = 16th notes).
                Set to 0 to disable quantization.

        Returns:
            A new Score instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            MidiParseError: If the MIDI file cannot be parsed.

        Examples:
            >>> score = Score.from_midi_file("recording.mid")
            >>> print(score.to_alda())
            >>> score.play()
        """
        from .midi.midi_to_ast import midi_to_ast
        from .midi.smf_reader import read_midi_file

        path = Path(path)
        midi_sequence = read_midi_file(path)
        ast = midi_to_ast(midi_sequence, quantize_grid=quantize_grid)

        score = cls.__new__(cls)
        score._mode = _MODE_MIDI
        score._source = ""
        score._filename = str(path)
        score._elements = []
        score._imported_ast = ast
        score._diagnostics = []
        score._playback = None
        return score

    @classmethod
    def from_elements(cls, *elements: ComposeElement) -> Score:
        """Create a Score from compose domain objects.

        Args:
            *elements: Compose elements (notes, rests, parts, tempo, etc.).

        Returns:
            A new Score instance.

        Examples:
            >>> from aldakit.compose import part, note, tempo
            >>> score = Score.from_elements(
            ...     part("piano"),
            ...     tempo(120),
            ...     note("c", duration=4),
            ...     note("d"),
            ...     note("e"),
            ... )
        """
        score = cls.__new__(cls)
        score._mode = _MODE_ELEMENTS
        score._source = ""
        score._filename = "<compose>"
        score._elements = list(elements)
        score._imported_ast = None
        score._diagnostics = []
        score._playback = None
        return score

    @classmethod
    def from_parts(cls, *parts: Any) -> Score:
        """Create a Score from Part objects.

        This is a convenience method for creating a score from parts only.

        Args:
            *parts: Part objects.

        Returns:
            A new Score instance.
        """
        return cls.from_elements(*parts)

    @property
    def source(self) -> str:
        """The original Alda source code (if created from source)."""
        return self._source

    @cached_property
    def ast(self) -> RootNode:
        """The parsed AST (lazily computed and cached)."""
        if self._mode == _MODE_SOURCE:
            return parse(self._source, self._filename)
        elif self._mode == _MODE_MIDI:
            # AST was imported from MIDI file
            assert self._imported_ast is not None
            return self._imported_ast
        else:
            return self._build_ast_from_elements()

    @cached_property
    def midi(self) -> MidiSequence:
        """The generated MIDI sequence (lazily computed and cached)."""
        generator = MidiGenerator()
        sequence = generator.generate(self.ast)
        self._diagnostics = list(generator.diagnostics)
        return sequence

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """Non-fatal problems found while generating MIDI.

        Includes unknown instrument names, undefined variable and marker
        references, and MIDI channel exhaustion. Generating the MIDI sequence
        is what produces them, so accessing this triggers generation.

        Examples:
            >>> score = Score("bogus-instrument: c d e")
            >>> [str(d) for d in score.diagnostics]
            ["<input>:1:1: Unknown instrument 'bogus-instrument'; falling back to acoustic grand piano."]
        """
        self.midi  # Ensure generation has happened
        return self._diagnostics

    @property
    def duration(self) -> float:
        """Total duration of the score in seconds."""
        return self.midi.duration()

    def _build_ast_from_elements(self) -> RootNode:
        """Build AST directly from compose elements."""
        from .ast_nodes import PartDeclarationNode

        from .compose.base import OctaveContext
        from .compose.part import Part

        octave_ctx = OctaveContext()
        children = []
        current_events: list = []
        current_part_decl: PartDeclarationNode | None = None

        def flush_part():
            """Flush accumulated events, wrapping in PartNode if there's a declaration."""
            nonlocal current_events, current_part_decl
            if current_part_decl is not None:
                # Wrap declaration and events in a PartNode
                children.append(
                    PartNode(
                        declaration=current_part_decl,
                        events=EventSequenceNode(events=current_events, position=None),
                        position=None,
                    )
                )
                current_part_decl = None
                current_events = []
            elif current_events:
                # No part declaration - bare event sequence
                children.append(EventSequenceNode(events=current_events, position=None))
                current_events = []

        for element in self._elements:
            # to_events() threads octave state, so notes that declare an
            # octave emit the octave change the AST needs to reproduce them.
            ast_nodes = element.to_events(octave_ctx)

            if isinstance(element, Part):
                # Flush previous part first
                flush_part()
                # Start new part - Part.to_events() returns a PartDeclarationNode
                assert len(ast_nodes) == 1
                assert isinstance(ast_nodes[0], PartDeclarationNode)
                current_part_decl = ast_nodes[0]
            else:
                # Accumulate events
                current_events.extend(ast_nodes)

        # Flush remaining content
        flush_part()

        return RootNode(children=children, position=None)

    def _invalidate_cache(self) -> None:
        """Invalidate cached properties after modification."""
        # Delete cached properties if they exist
        for attr in ("ast", "midi"):
            if attr in self.__dict__:
                del self.__dict__[attr]

    # Builder methods

    def add(self, *elements: ComposeElement) -> Score:
        """Add elements to the score.

        This method modifies the score in place and returns self for chaining.
        Only works with scores created via from_elements().

        Args:
            *elements: Compose elements to add.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the score was created from source code.
        """
        if self._mode != _MODE_ELEMENTS:
            raise ValueError(
                "Cannot add elements to this score. "
                "Use Score.from_elements() to create a modifiable score."
            )
        self._elements.extend(elements)
        self._invalidate_cache()
        return self

    def with_part(self, *instruments: str, alias: str | None = None) -> Score:
        """Add a part declaration.

        Args:
            *instruments: Instrument names.
            alias: Optional alias for the part.

        Returns:
            Self for method chaining.
        """
        from .compose.part import Part

        return self.add(Part(instruments=instruments, alias=alias))

    def with_tempo(self, bpm: int | float, global_: bool = False) -> Score:
        """Add a tempo attribute.

        Args:
            bpm: Beats per minute.
            global_: If True, applies globally to all parts.

        Returns:
            Self for method chaining.
        """
        from .compose.attributes import Tempo

        return self.add(Tempo(bpm=bpm, global_=global_))

    def with_volume(self, level: int | float) -> Score:
        """Add a volume attribute.

        Args:
            level: Volume level (0-100).

        Returns:
            Self for method chaining.
        """
        from .compose.attributes import Volume

        return self.add(Volume(level=level))

    # Output methods

    def to_alda(self) -> str:
        """Export as Alda source code.

        Returns:
            Alda source code string.
        """
        if self._mode == _MODE_SOURCE:
            return self._source
        elif self._mode == _MODE_MIDI:
            # Generate Alda from AST
            return _ast_to_alda(self.ast)
        else:
            from .compose.base import OctaveContext

            ctx = OctaveContext()
            parts: list[str] = []
            for element in self._elements:
                parts.extend(element.to_alda_parts(ctx))
            return " ".join(parts)

    def _make_backend(self, backend: str, port: str | None, soundfont: str | None):
        """Create the requested playback backend."""
        if backend == "audio":
            from .midi.backends import HAS_TSF, TsfBackend

            if not HAS_TSF:
                raise RuntimeError(
                    "Audio backend not available. The _tsf module was not built."
                )
            return TsfBackend(soundfont=soundfont)
        return LibremidiBackend(port_name=port)

    def play(
        self,
        port: str | None = None,
        wait: bool = True,
        backend: str = "midi",
        soundfont: str | None = None,
    ) -> PlaybackHandle | None:
        """Play the score.

        Args:
            port: MIDI output port name (for backend="midi"). If None, uses
                the first available port or creates a virtual port.
            wait: If True (default), block until playback completes and release
                the backend. If False, playback continues in the background and
                a :class:`PlaybackHandle` is returned; the caller owns it and
                should call ``stop()`` or ``close()`` when finished.
            backend: Backend to use: "midi" (default) or "audio".
                - "midi": Send to MIDI port via libremidi (requires external synth)
                - "audio": Direct audio output via TinySoundFont (requires SoundFont)
            soundfont: Path to SoundFont file (for backend="audio"). If None,
                searches common locations or uses ALDAKIT_SOUNDFONT env var.

        Returns:
            None when ``wait`` is True, otherwise a :class:`PlaybackHandle`
            controlling the background playback.

        Raises:
            RuntimeError: If the selected backend is not available.
            FileNotFoundError: If backend="audio" and no SoundFont is found.

        Examples:
            >>> score = Score("piano: c d e")
            >>> score.play()  # blocks until finished
            >>> handle = score.play(wait=False)  # returns immediately
            >>> handle.stop()  # ... and stops when you say so
        """
        player = self._make_backend(backend, port, soundfont)

        if not wait:
            # Ownership passes to the handle: closing the backend here would
            # cut playback off before it had a chance to start.
            player.play(self.midi)
            handle = PlaybackHandle(player)
            self._playback = handle
            return handle

        with player as active:
            active.play(self.midi)
            active.wait()
        return None

    def stop(self) -> None:
        """Stop background playback started by ``play(wait=False)``."""
        if self._playback is not None:
            self._playback.close()
            self._playback = None

    def save(self, path: str | Path) -> None:
        """Save the score to a file.

        Supports both MIDI (.mid, .midi) and Alda (.alda) formats.

        Args:
            path: Output file path.
        """
        path = Path(path)
        if path.suffix in (".mid", ".midi"):
            write_midi_file(self.midi, path)
        elif path.suffix == ".alda":
            path.write_text(self.to_alda(), encoding="utf-8")
        else:
            # Default to MIDI
            write_midi_file(self.midi, path)

    def __repr__(self) -> str:
        if self._mode == _MODE_SOURCE:
            # Show first 50 chars of source, truncated if longer
            preview = self._source[:50]
            if len(self._source) > 50:
                preview += "..."
            preview = preview.replace("\n", "\\n")
            return f"Score({preview!r})"
        elif self._mode == _MODE_MIDI:
            return f"Score.from_midi_file({self._filename!r})"
        else:
            n = len(self._elements)
            return f"Score.from_elements(<{n} elements>)"
