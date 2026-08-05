"""Core compose elements: Note, Rest, Chord, Seq."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .base import ComposeElement, OctaveContext

if TYPE_CHECKING:
    from ..ast_nodes import ASTNode

from ..ast_nodes import (
    AtMarkerNode,
    ChordNode,
    CramNode,
    DurationNode,
    EventSequenceNode,
    MarkerNode,
    NoteLengthMsNode,
    NoteLengthNode,
    NoteLengthSecondsNode,
    NoteNode,
    OctaveSetNode,
    RepeatNode,
    RestNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    VoiceGroupNode,
    VoiceNode,
)

# Accidental characters Alda understands: sharp, flat, natural.
_VALID_ACCIDENTALS = frozenset("+-_")

# Pitch to semitone offset from C
_PITCH_OFFSETS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_SEMITONE_TO_PITCH = ["c", "c", "d", "d", "e", "f", "f", "g", "g", "a", "a", "b"]
_SEMITONE_ACCIDENTALS = ["", "+", "", "+", "", "", "+", "", "+", "", "+", ""]


@dataclass(frozen=True)
class Note(ComposeElement):
    """A musical note.

    All parameters except pitch are keyword-only to avoid ambiguity.

    Examples:
        >>> note("c")                              # C quarter note
        >>> note("c", duration=4)                  # C quarter note (explicit)
        >>> note("c", duration=8, accidental="+")  # C# eighth note
        >>> note("c", octave=5)                    # C in octave 5
        >>> note("c", ms=500)                      # C for 500 milliseconds
    """

    pitch: str
    duration: int | None = None
    octave: int | None = None
    accidental: str | None = None
    dots: int = 0
    ms: float | None = None
    seconds: float | None = None
    slurred: bool = False

    def __post_init__(self) -> None:
        # Validate pitch
        if self.pitch.lower() not in _PITCH_OFFSETS:
            raise ValueError(f"Invalid pitch: {self.pitch}. Must be a-g.")

        # Validate accidentals
        if self.accidental:
            invalid = set(self.accidental) - _VALID_ACCIDENTALS
            if invalid:
                raise ValueError(
                    f"Invalid accidental: {self.accidental!r}. "
                    "Use '+' (sharp), '-' (flat) or '_' (natural), "
                    "repeated for double accidentals."
                )

    def to_ast(self) -> NoteNode:
        """Convert to AST NoteNode."""
        # Build accidentals list
        accidentals: list[str] = []
        if self.accidental:
            accidentals = list(self.accidental)

        # Build duration node
        duration_node = self._build_duration_node()

        return NoteNode(
            letter=self.pitch.lower(),
            accidentals=accidentals,
            duration=duration_node,
            slurred=self.slurred,
            position=None,
        )

    def _build_duration_node(self) -> DurationNode | None:
        """Build duration node from duration parameters."""
        if self.ms is not None:
            return DurationNode(
                components=[NoteLengthMsNode(ms=self.ms, position=None)],
                position=None,
            )
        elif self.seconds is not None:
            return DurationNode(
                components=[NoteLengthSecondsNode(seconds=self.seconds, position=None)],
                position=None,
            )
        elif self.duration is not None:
            return DurationNode(
                components=[
                    NoteLengthNode(
                        denominator=self.duration, dots=self.dots, position=None
                    )
                ],
                position=None,
            )
        elif self.dots > 0:
            # Dots without explicit duration - use default quarter note
            return DurationNode(
                components=[
                    NoteLengthNode(denominator=4, dots=self.dots, position=None)
                ],
                position=None,
            )
        return None

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        result = self.pitch.lower()

        # Accidentals
        if self.accidental:
            result += self.accidental

        # Duration
        if self.ms is not None:
            result += f"{int(self.ms)}ms"
        elif self.seconds is not None:
            result += f"{self.seconds}s"
        elif self.duration is not None:
            result += str(self.duration)
            result += "." * self.dots

        # Slur
        if self.slurred:
            result += "~"

        return result

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to AST events, prefixing an octave change when needed."""
        events: list[ASTNode] = []
        octave = ctx.shift_to(self.octave)
        if octave is not None:
            events.append(OctaveSetNode(octave=octave, position=None))
        events.append(self.to_ast())
        return events

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, prefixing an octave change when needed."""
        parts: list[str] = []
        octave = ctx.shift_to(self.octave)
        if octave is not None:
            parts.append(f"o{octave}")
        parts.append(self.to_alda())
        return parts

    @property
    def midi_pitch(self) -> int:
        """Calculate MIDI pitch number.

        Uses octave 4 as default if not specified.
        """
        oct = self.octave if self.octave is not None else 4
        base = _PITCH_OFFSETS[self.pitch.lower()]

        # Apply accidentals
        offset = 0
        if self.accidental:
            for acc in self.accidental:
                if acc == "+":
                    offset += 1
                elif acc == "-":
                    offset -= 1
                # "_" (natural) doesn't change offset

        return (oct + 1) * 12 + base + offset

    # Transformation methods (return new Note instances)

    def sharpen(self) -> Note:
        """Return a new note raised by one semitone."""
        new_acc = (self.accidental or "") + "+"
        return replace(self, accidental=new_acc)

    def flatten(self) -> Note:
        """Return a new note lowered by one semitone."""
        new_acc = (self.accidental or "") + "-"
        return replace(self, accidental=new_acc)

    def transpose(self, semitones: int) -> Note:
        """Return a new note transposed by the given number of semitones."""
        current_midi = self.midi_pitch
        new_midi = current_midi + semitones

        # Calculate new octave and pitch
        new_octave = (new_midi // 12) - 1
        semitone_in_octave = new_midi % 12

        new_pitch = _SEMITONE_TO_PITCH[semitone_in_octave]
        new_accidental = _SEMITONE_ACCIDENTALS[semitone_in_octave] or None

        return replace(
            self, pitch=new_pitch, octave=new_octave, accidental=new_accidental
        )

    def with_duration(self, duration: int) -> Note:
        """Return a new note with the given duration."""
        return replace(self, duration=duration, ms=None, seconds=None)

    def with_octave(self, octave: int) -> Note:
        """Return a new note in the given octave."""
        return replace(self, octave=octave)

    def with_dots(self, dots: int) -> Note:
        """Return a new note with the given number of dots."""
        return replace(self, dots=dots)

    def slur(self) -> Note:
        """Return a new slurred note."""
        return replace(self, slurred=True)

    def __mul__(self, n: int) -> Repeat:
        """Repeat this note n times."""
        return Repeat(element=self, times=n)

    def __rmul__(self, n: int) -> Repeat:
        """Repeat this note n times (reverse order)."""
        return self.__mul__(n)


@dataclass(frozen=True)
class Rest(ComposeElement):
    """A musical rest (silence).

    Examples:
        >>> rest()                    # Quarter rest
        >>> rest(duration=2)          # Half rest
        >>> rest(ms=1000)             # One second rest
    """

    duration: int | None = None
    dots: int = 0
    ms: float | None = None
    seconds: float | None = None

    def to_ast(self) -> RestNode:
        """Convert to AST RestNode."""
        duration_node = self._build_duration_node()
        return RestNode(duration=duration_node, position=None)

    def _build_duration_node(self) -> DurationNode | None:
        """Build duration node from duration parameters."""
        if self.ms is not None:
            return DurationNode(
                components=[NoteLengthMsNode(ms=self.ms, position=None)],
                position=None,
            )
        elif self.seconds is not None:
            return DurationNode(
                components=[NoteLengthSecondsNode(seconds=self.seconds, position=None)],
                position=None,
            )
        elif self.duration is not None:
            return DurationNode(
                components=[
                    NoteLengthNode(
                        denominator=self.duration, dots=self.dots, position=None
                    )
                ],
                position=None,
            )
        return None

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        result = "r"

        if self.ms is not None:
            result += f"{int(self.ms)}ms"
        elif self.seconds is not None:
            result += f"{self.seconds}s"
        elif self.duration is not None:
            result += str(self.duration)
            result += "." * self.dots

        return result


@dataclass(frozen=True)
class Chord(ComposeElement):
    """A chord (multiple notes played simultaneously).

    Examples:
        >>> chord("c", "e", "g")                    # C major
        >>> chord(note("c"), note("e"), note("g"))  # Same, using Note objects
        >>> chord("c", "e", "g", duration=1)        # Whole note C major
    """

    notes: tuple[Note, ...] = field(default_factory=tuple)
    duration: int | None = None
    dots: int = 0

    def to_ast(self) -> ChordNode:
        """Convert to AST ChordNode."""
        return ChordNode(notes=[n.to_ast() for n in self._chord_notes()], position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return "/".join(self._alda_parts(OctaveContext()))

    def _chord_notes(self) -> list[Note]:
        """Return the chord's notes with the chord duration applied to the first."""
        notes: list[Note] = []
        for i, n in enumerate(self.notes):
            if i == 0 and self.duration is not None:
                n = n.with_duration(self.duration)
                if self.dots:
                    n = n.with_dots(self.dots)
            notes.append(n)
        return notes

    def _alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Render each chord member, inserting octave changes between them."""
        parts: list[str] = []
        for i, n in enumerate(self._chord_notes()):
            prefix = ""
            octave = ctx.shift_to(n.octave)
            if octave is not None:
                prefix = f"o{octave} "
            if (
                i == 0
                or n.duration is not None
                or n.ms is not None
                or n.seconds is not None
            ):
                parts.append(prefix + n.to_alda())
            else:
                # Subsequent notes don't repeat the chord duration
                parts.append(prefix + n.pitch.lower() + (n.accidental or ""))
        return parts

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a ChordNode, inserting octave changes between members."""
        elements: list = []
        for n in self._chord_notes():
            octave = ctx.shift_to(n.octave)
            if octave is not None:
                elements.append(OctaveSetNode(octave=octave, position=None))
            elements.append(n.to_ast())
        return [ChordNode(notes=elements, position=None)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, inserting octave changes between members."""
        return ["/".join(self._alda_parts(ctx))]


@dataclass
class Seq(ComposeElement):
    """A sequence of musical elements.

    Examples:
        >>> seq(note("c"), note("d"), note("e"))
        >>> seq.from_alda("c d e f g")
        >>> seq(note("c"), note("d")) * 4  # Repeat 4 times
    """

    elements: list[ComposeElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ast(self) -> EventSequenceNode:
        """Convert to AST EventSequenceNode."""
        return EventSequenceNode(events=self.to_events(OctaveContext()), position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return " ".join(self.to_alda_parts(OctaveContext()))

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert child elements to AST events, threading octave context."""
        events: list[ASTNode] = []
        for element in self.elements:
            events.extend(element.to_events(ctx))
        return events

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert child elements to Alda fragments, threading octave context."""
        parts: list[str] = []
        for element in self.elements:
            parts.extend(element.to_alda_parts(ctx))
        return parts

    @classmethod
    def from_alda(cls, source: str) -> Seq:
        """Create a Seq by parsing Alda source code.

        Args:
            source: Alda source code snippet.

        Returns:
            Seq containing parsed elements.
        """
        from ..parser import parse

        ast = parse(source)
        # Wrap the AST in a ParsedSeq that delegates to_ast to the parsed result
        return _ParsedSeq(ast=ast, source=source)

    def __mul__(self, n: int) -> Repeat:
        """Repeat this sequence n times."""
        return Repeat(element=self, times=n)

    def __rmul__(self, n: int) -> Repeat:
        """Repeat this sequence n times (reverse order)."""
        return self.__mul__(n)

    def __add__(self, other: Seq) -> Seq:
        """Concatenate two sequences."""
        if isinstance(other, Seq):
            merged = dict(self.metadata)
            for key, value in other.metadata.items():
                merged.setdefault(key, value)
            return Seq(
                elements=list(self.elements) + list(other.elements), metadata=merged
            )
        return NotImplemented


@dataclass
class _ParsedSeq(Seq):
    """A sequence created from parsed Alda source.

    This wraps a parsed AST to avoid re-parsing.
    """

    ast: object = None  # RootNode
    source: str = ""
    elements: list[ComposeElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ast(self) -> EventSequenceNode:
        """Return the parsed AST."""
        # The parsed AST is a RootNode, return its contents as EventSequenceNode
        from ..ast_nodes import EventSequenceNode as ESN
        from ..ast_nodes import RootNode

        if isinstance(self.ast, RootNode):
            # Flatten RootNode children into EventSequenceNode
            return ESN(events=list(self.ast.children), position=None)
        return ESN(events=[], position=None)

    def to_alda(self) -> str:
        """Return the original source."""
        return self.source

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Return the parsed events; parsed source carries its own octave state."""
        ctx.reset(None)
        return list(self.to_ast().events)

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Return the original source; it carries its own octave state."""
        ctx.reset(None)
        return [self.source]


@dataclass(frozen=True)
class Repeat(ComposeElement):
    """A repeated element.

    Examples:
        >>> note("c") * 4        # Repeat note 4 times
        >>> seq(...) * 8         # Repeat sequence 8 times
    """

    element: ComposeElement
    times: int

    def to_ast(self) -> RepeatNode:
        """Convert to AST RepeatNode."""
        return self._build(OctaveContext())

    def _build(self, ctx: OctaveContext) -> RepeatNode:
        from ..ast_nodes import BracketedSequenceNode

        inner_events = self.element.to_events(ctx)
        if len(inner_events) == 1 and not isinstance(
            inner_events[0], EventSequenceNode
        ):
            inner_ast = inner_events[0]
        else:
            inner_ast = EventSequenceNode(
                events=(
                    list(inner_events[0].events)
                    if len(inner_events) == 1
                    and isinstance(inner_events[0], EventSequenceNode)
                    else inner_events
                ),
                position=None,
            )

        # Wrap in a bracketed sequence so the repeat applies to the whole group
        if isinstance(inner_ast, EventSequenceNode):
            bracketed = BracketedSequenceNode(events=inner_ast, position=None)
            return RepeatNode(event=bracketed, times=self.times, position=None)
        return RepeatNode(event=inner_ast, times=self.times, position=None)

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a RepeatNode, threading octave context into the body."""
        return [self._build(ctx)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, threading octave context into the body."""
        inner = " ".join(self.element.to_alda_parts(ctx))
        if isinstance(self.element, Seq) and len(self.element.elements) > 1:
            return [f"[{inner}]*{self.times}"]
        if " " in inner:
            # Octave prefix or multi-token body needs bracketing to repeat as a unit
            return [f"[{inner}]*{self.times}"]
        return [f"{inner}*{self.times}"]

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        inner = self.element.to_alda()
        # Wrap in brackets if it's a sequence
        if isinstance(self.element, Seq) and len(self.element.elements) > 1:
            return f"[{inner}]*{self.times}"
        return f"{inner}*{self.times}"


# Factory functions


def note(
    pitch: str,
    *,
    duration: int | None = None,
    octave: int | None = None,
    accidental: str | None = None,
    dots: int = 0,
    ms: float | None = None,
    seconds: float | None = None,
    slurred: bool = False,
) -> Note:
    """Create a note.

    Args:
        pitch: Note pitch (a-g).
        duration: Note duration denominator (1=whole, 2=half, 4=quarter, etc.).
        octave: Octave number (typically 0-8).
        accidental: Accidental string ("+", "-", "_", "++", etc.).
        dots: Number of dots (0-3).
        ms: Duration in milliseconds.
        seconds: Duration in seconds.
        slurred: Whether the note is slurred to the next.

    Returns:
        Note element.
    """
    return Note(
        pitch=pitch,
        duration=duration,
        octave=octave,
        accidental=accidental,
        dots=dots,
        ms=ms,
        seconds=seconds,
        slurred=slurred,
    )


def rest(
    *,
    duration: int | None = None,
    dots: int = 0,
    ms: float | None = None,
    seconds: float | None = None,
) -> Rest:
    """Create a rest.

    Args:
        duration: Rest duration denominator (1=whole, 2=half, 4=quarter, etc.).
        dots: Number of dots (0-3).
        ms: Duration in milliseconds.
        seconds: Duration in seconds.

    Returns:
        Rest element.
    """
    return Rest(duration=duration, dots=dots, ms=ms, seconds=seconds)


def chord(
    *notes_or_pitches: Note | str, duration: int | None = None, dots: int = 0
) -> Chord:
    """Create a chord.

    Args:
        *notes_or_pitches: Note objects or pitch strings.
        duration: Chord duration (applied to first note).
        dots: Number of dots to apply to the chord duration.

    Returns:
        Chord element.
    """
    notes = []
    for item in notes_or_pitches:
        if isinstance(item, Note):
            notes.append(item)
        elif isinstance(item, str):
            notes.append(Note(pitch=item))
        else:
            raise TypeError(f"Expected Note or str, got {type(item)}")

    return Chord(notes=tuple(notes), duration=duration, dots=dots)


def seq(*elements: ComposeElement, metadata: dict[str, Any] | None = None) -> Seq:
    """Create a sequence.

    Args:
        *elements: Compose elements to include in the sequence.
        metadata: Optional metadata dictionary for timing/annotations.

    Returns:
        Seq element.
    """
    return Seq(elements=list(elements), metadata=dict(metadata or {}))


# =============================================================================
# Cram (Tuplets)
# =============================================================================


@dataclass(frozen=True)
class Cram(ComposeElement):
    """A cram expression (tuplet) - fit multiple notes into a duration.

    Cram expressions fit a group of notes into a specific duration,
    creating tuplets like triplets, quintuplets, etc.

    Examples:
        >>> cram(note("c"), note("d"), note("e"), duration=4)  # Triplet in quarter note
        >>> cram(note("c"), note("d"), note("e"), note("f"), note("g"), duration=2)  # Quintuplet
    """

    elements: list[ComposeElement]
    duration: int | None = None
    dots: int = 0

    def to_ast(self) -> CramNode:
        """Convert to AST CramNode."""
        return self._build(OctaveContext())

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a CramNode, threading octave context into the body."""
        return [self._build(ctx)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, threading octave context into the body."""
        inner_parts: list[str] = []
        for elem in self.elements:
            inner_parts.extend(elem.to_alda_parts(ctx))
        inner = " ".join(inner_parts)
        if self.duration is not None:
            dots = "." * self.dots
            return [f"{{{inner}}}{self.duration}{dots}"]
        return [f"{{{inner}}}"]

    def _build(self, ctx: OctaveContext) -> CramNode:
        # Build event sequence from elements
        events: list[ASTNode] = []
        for elem in self.elements:
            events.extend(elem.to_events(ctx))

        event_seq = EventSequenceNode(events=events, position=None)

        # Build duration node
        duration_node = None
        if self.duration is not None:
            duration_node = DurationNode(
                components=[
                    NoteLengthNode(
                        denominator=self.duration, dots=self.dots, position=None
                    )
                ],
                position=None,
            )

        return CramNode(events=event_seq, duration=duration_node, position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        inner = " ".join(elem.to_alda() for elem in self.elements)
        if self.duration is not None:
            dots = "." * self.dots
            return f"{{{inner}}}{self.duration}{dots}"
        return f"{{{inner}}}"


def cram(*elements: ComposeElement, duration: int | None = None, dots: int = 0) -> Cram:
    """Create a cram expression (tuplet).

    Args:
        *elements: Notes/rests to fit into the duration.
        duration: Duration denominator to fit notes into.
        dots: Number of dots on the duration.

    Returns:
        Cram element.

    Examples:
        >>> cram(note("c"), note("d"), note("e"), duration=4)  # Triplet
        >>> cram(note("c"), note("d"), note("e"), note("f"), note("g"), duration=2)
    """
    return Cram(elements=list(elements), duration=duration, dots=dots)


# =============================================================================
# Voice (Parallel Lines)
# =============================================================================


@dataclass(frozen=True)
class Voice(ComposeElement):
    """A voice within a part for polyphonic writing.

    Voices allow multiple parallel lines within a single part.
    Voice numbers start at 1; V0 ends the voice section.

    Examples:
        >>> voice(1, note("c"), note("d"), note("e"))  # Voice 1
        >>> voice(2, note("e"), note("f"), note("g"))  # Voice 2
    """

    number: int
    elements: list[ComposeElement]

    def __post_init__(self) -> None:
        if self.number < 0:
            raise ValueError("Voice number must be non-negative (0 ends voices)")

    def to_ast(self) -> VoiceNode:
        """Convert to AST VoiceNode."""
        return self._build(OctaveContext())

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return self.to_alda_parts(OctaveContext())[0]

    def _build(self, ctx: OctaveContext) -> VoiceNode:
        events: list[ASTNode] = []
        for elem in self.elements:
            events.extend(elem.to_events(ctx))
        event_seq = EventSequenceNode(events=events, position=None)
        return VoiceNode(number=self.number, events=event_seq, position=None)

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a VoiceNode, threading octave context into the body."""
        return [self._build(ctx)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, threading octave context into the body."""
        parts: list[str] = []
        for elem in self.elements:
            parts.extend(elem.to_alda_parts(ctx))
        return [f"V{self.number}: {' '.join(parts)}"]


@dataclass(frozen=True)
class VoiceGroup(ComposeElement):
    """A group of parallel voices.

    Examples:
        >>> voice_group(
        ...     voice(1, note("c"), note("d")),
        ...     voice(2, note("e"), note("f")),
        ... )
    """

    voices: list[Voice]

    def to_ast(self) -> VoiceGroupNode:
        """Convert to AST VoiceGroupNode."""
        return self.to_events(OctaveContext())[0]  # type: ignore[return-value]

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return self.to_alda_parts(OctaveContext())[0]

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a VoiceGroupNode.

        Each voice starts from the octave in force when the group begins, so
        they are converted independently. After the group the octave is treated
        as unknown, forcing the next element to state it explicitly.
        """
        entry = ctx.octave
        voice_nodes = []
        for v in self.voices:
            voice_nodes.append(v._build(OctaveContext(octave=entry)))
        ctx.reset(None)
        return [VoiceGroupNode(voices=voice_nodes, position=None)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source, converting each voice independently."""
        entry = ctx.octave
        lines = [v.to_alda_parts(OctaveContext(octave=entry))[0] for v in self.voices]
        lines.append("V0:")  # End voices
        ctx.reset(None)
        return ["\n".join(lines)]


def voice(number: int, *elements: ComposeElement) -> Voice:
    """Create a voice.

    Args:
        number: Voice number (1, 2, etc.; 0 ends voice section).
        *elements: Notes/rests in this voice.

    Returns:
        Voice element.
    """
    return Voice(number=number, elements=list(elements))


def voice_group(*voices: Voice) -> VoiceGroup:
    """Create a voice group from multiple voices.

    Args:
        *voices: Voice elements to group.

    Returns:
        VoiceGroup element.
    """
    return VoiceGroup(voices=list(voices))


# =============================================================================
# Variables
# =============================================================================


@dataclass(frozen=True)
class Variable(ComposeElement):
    """A variable definition.

    Variables allow naming a sequence for reuse.

    Examples:
        >>> var("riff", note("c"), note("d"), note("e"))
        >>> var_ref("riff")  # Use the variable
    """

    name: str
    elements: list[ComposeElement]

    def to_ast(self) -> VariableDefinitionNode:
        """Convert to AST VariableDefinitionNode."""
        return self.to_events(OctaveContext())[0]  # type: ignore[return-value]

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return self.to_alda_parts(OctaveContext())[0]

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to a VariableDefinitionNode.

        A definition produces no sound where it appears, so it does not affect
        the surrounding octave context. Its body is converted with its own
        context because it may be referenced from anywhere.
        """
        body = OctaveContext()
        events: list[ASTNode] = []
        for elem in self.elements:
            events.extend(elem.to_events(body))
        event_seq = EventSequenceNode(events=events, position=None)
        return [VariableDefinitionNode(name=self.name, events=event_seq, position=None)]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source with a self-contained body context."""
        body = OctaveContext()
        parts: list[str] = []
        for elem in self.elements:
            parts.extend(elem.to_alda_parts(body))
        return [f"{self.name} = {' '.join(parts)}"]


@dataclass(frozen=True)
class VariableRef(ComposeElement):
    """A reference to a previously defined variable.

    Examples:
        >>> var_ref("riff")  # Use a variable named 'riff'
    """

    name: str

    def to_ast(self) -> VariableReferenceNode:
        """Convert to AST VariableReferenceNode."""
        return VariableReferenceNode(name=self.name, position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return self.name

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Expanding a variable may change the octave, so mark it unknown."""
        ctx.reset(None)
        return [self.to_ast()]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Expanding a variable may change the octave, so mark it unknown."""
        ctx.reset(None)
        return [self.to_alda()]


def var(name: str, *elements: ComposeElement) -> Variable:
    """Define a variable.

    Args:
        name: Variable name.
        *elements: Notes/rests to assign to the variable.

    Returns:
        Variable definition element.

    Examples:
        >>> var("riff", note("c"), note("d"), note("e"), note("f"))
    """
    return Variable(name=name, elements=list(elements))


def var_ref(name: str) -> VariableRef:
    """Reference a variable.

    Args:
        name: Variable name to reference.

    Returns:
        Variable reference element.

    Examples:
        >>> var_ref("riff")
    """
    return VariableRef(name=name)


# =============================================================================
# Markers
# =============================================================================


@dataclass(frozen=True)
class Marker(ComposeElement):
    """A marker definition for synchronization points.

    Markers allow different parts to synchronize at named points.

    Examples:
        >>> marker("chorus")  # Define marker
        >>> at_marker("chorus")  # Jump to marker in another part
    """

    name: str

    def to_ast(self) -> MarkerNode:
        """Convert to AST MarkerNode."""
        return MarkerNode(name=self.name, position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return f"%{self.name}"


@dataclass(frozen=True)
class AtMarker(ComposeElement):
    """A marker reference (jump to marker).

    Examples:
        >>> at_marker("chorus")  # Jump to the 'chorus' marker
    """

    name: str

    def to_ast(self) -> AtMarkerNode:
        """Convert to AST AtMarkerNode."""
        return AtMarkerNode(name=self.name, position=None)

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        return f"@{self.name}"


def marker(name: str) -> Marker:
    """Create a marker.

    Args:
        name: Marker name.

    Returns:
        Marker element.

    Examples:
        >>> marker("verse")
        >>> marker("chorus")
    """
    return Marker(name=name)


def at_marker(name: str) -> AtMarker:
    """Create a marker reference (jump to marker).

    Args:
        name: Marker name to jump to.

    Returns:
        AtMarker element.

    Examples:
        >>> at_marker("chorus")  # Jump to chorus marker
    """
    return AtMarker(name=name)
