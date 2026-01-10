"""AST node definitions for the Alda parser."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .tokens import SourcePosition


class ASTNode(ABC):
    """Base class for all AST nodes."""

    position: SourcePosition | None = None

    @abstractmethod
    def accept(self, visitor: "ASTVisitor") -> object:
        """Accept a visitor for tree traversal."""
        pass

    def __repr__(self) -> str:
        return self._repr_helper(0)

    @abstractmethod
    def _repr_helper(self, indent: int) -> str:
        """Helper for indented representation."""
        pass


class ASTVisitor(ABC):
    """Visitor interface for AST traversal."""

    def visit(self, node: ASTNode) -> object:
        """Visit a node by dispatching to the appropriate method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> None:
        """Default visitor implementation."""
        pass


# Top-level nodes


@dataclass
class RootNode(ASTNode):
    """Root of the AST, contains all parts and events."""

    children: list[ASTNode] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = ["RootNode"]
        for child in self.children:
            lines.append("  " * (indent + 1) + child._repr_helper(indent + 1))
        return "\n".join(lines)


@dataclass
class PartNode(ASTNode):
    """A part (instrument) declaration with its events."""

    declaration: "PartDeclarationNode"
    events: "EventSequenceNode"
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = ["PartNode"]
        lines.append("  " * (indent + 1) + self.declaration._repr_helper(indent + 1))
        lines.append("  " * (indent + 1) + self.events._repr_helper(indent + 1))
        return "\n".join(lines)


@dataclass
class PartDeclarationNode(ASTNode):
    """Part declaration (e.g., 'piano:', 'violin "v1":')."""

    names: list[str]
    alias: str | None = None
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if self.alias:
            return f'PartDeclarationNode(names={self.names}, alias="{self.alias}")'
        return f"PartDeclarationNode(names={self.names})"


@dataclass
class EventSequenceNode(ASTNode):
    """A sequence of musical events."""

    events: list[ASTNode] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if not self.events:
            return "EventSequenceNode()"
        lines = ["EventSequenceNode"]
        for event in self.events:
            lines.append("  " * (indent + 1) + event._repr_helper(indent + 1))
        return "\n".join(lines)


# Musical event nodes


@dataclass
class NoteNode(ASTNode):
    """A musical note."""

    letter: str  # a-g
    accidentals: list[str] = field(default_factory=list)  # list of '+', '-', '_'
    duration: "DurationNode | None" = None
    slurred: bool = False  # True if followed by ~ connecting to next note
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        parts = [f"NoteNode(letter={self.letter!r}"]
        if self.accidentals:
            parts.append(f", accidentals={self.accidentals}")
        if self.duration:
            parts.append(f", duration={self.duration._repr_helper(0)}")
        if self.slurred:
            parts.append(", slurred=True")
        parts.append(")")
        return "".join(parts)


@dataclass
class RestNode(ASTNode):
    """A rest."""

    duration: "DurationNode | None" = None
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if self.duration:
            return f"RestNode(duration={self.duration._repr_helper(0)})"
        return "RestNode()"


@dataclass
class ChordNode(ASTNode):
    """A chord (multiple notes played simultaneously)."""

    notes: list[
        NoteNode
        | RestNode
        | OctaveUpNode
        | OctaveDownNode
        | OctaveSetNode
        | LispListNode
    ]
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = ["ChordNode"]
        for note in self.notes:
            lines.append("  " * (indent + 1) + note._repr_helper(indent + 1))
        return "\n".join(lines)


# Duration nodes


@dataclass
class DurationNode(ASTNode):
    """A duration specification."""

    components: list["DurationComponentNode"] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if len(self.components) == 1:
            return f"DurationNode({self.components[0]._repr_helper(0)})"
        comps = ", ".join(c._repr_helper(0) for c in self.components)
        return f"DurationNode([{comps}])"


class DurationComponentNode(ASTNode):
    """Base class for duration components."""

    pass


@dataclass
class NoteLengthNode(DurationComponentNode):
    """A note length (e.g., 4 for quarter, 8 for eighth)."""

    denominator: int | float
    dots: int = 0
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if self.dots:
            return f"NoteLengthNode({self.denominator}, dots={self.dots})"
        return f"NoteLengthNode({self.denominator})"


@dataclass
class NoteLengthMsNode(DurationComponentNode):
    """A duration in milliseconds."""

    ms: float
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"NoteLengthMsNode({self.ms}ms)"


@dataclass
class NoteLengthSecondsNode(DurationComponentNode):
    """A duration in seconds."""

    seconds: float
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"NoteLengthSecondsNode({self.seconds}s)"


@dataclass
class BarlineNode(ASTNode):
    """A barline (|) - mainly for visual organization."""

    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return "BarlineNode()"


# Octave nodes


@dataclass
class OctaveSetNode(ASTNode):
    """Set octave to absolute value (e.g., o4)."""

    octave: int
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"OctaveSetNode({self.octave})"


@dataclass
class OctaveUpNode(ASTNode):
    """Increase octave by one (>)."""

    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return "OctaveUpNode()"


@dataclass
class OctaveDownNode(ASTNode):
    """Decrease octave by one (<)."""

    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return "OctaveDownNode()"


# S-expression (Lisp) nodes


@dataclass
class LispListNode(ASTNode):
    """A Lisp S-expression (e.g., (tempo 120))."""

    elements: list["LispNode"] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        if not self.elements:
            return "LispListNode()"
        elems = " ".join(e._repr_helper(0) for e in self.elements)
        return f"LispListNode({elems})"


class LispNode(ASTNode):
    """Base class for Lisp expression elements."""

    pass


@dataclass
class LispSymbolNode(LispNode):
    """A Lisp symbol."""

    name: str
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return self.name


@dataclass
class LispNumberNode(LispNode):
    """A Lisp number."""

    value: int | float
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return str(self.value)


@dataclass
class LispStringNode(LispNode):
    """A Lisp string."""

    value: str
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f'"{self.value}"'


@dataclass
class LispQuotedNode(LispNode):
    """A quoted Lisp expression (e.g., '(g minor) or 'up)."""

    value: "LispListNode | LispSymbolNode"
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"'{self.value._repr_helper(indent)}"


# Variable nodes


@dataclass
class VariableDefinitionNode(ASTNode):
    """A variable definition (e.g., 'myMotif = c d e')."""

    name: str
    events: EventSequenceNode
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = [f"VariableDefinitionNode(name={self.name!r})"]
        lines.append("  " * (indent + 1) + self.events._repr_helper(indent + 1))
        return "\n".join(lines)


@dataclass
class VariableReferenceNode(ASTNode):
    """A reference to a variable."""

    name: str
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"VariableReferenceNode({self.name!r})"


# Marker nodes


@dataclass
class MarkerNode(ASTNode):
    """A marker definition (e.g., '%verse')."""

    name: str
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"MarkerNode({self.name!r})"


@dataclass
class AtMarkerNode(ASTNode):
    """A marker reference (e.g., '@verse')."""

    name: str
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        return f"AtMarkerNode({self.name!r})"


# Voice nodes


@dataclass
class VoiceNode(ASTNode):
    """A single voice within a voice group."""

    number: int
    events: EventSequenceNode
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = [f"VoiceNode(number={self.number})"]
        lines.append("  " * (indent + 1) + self.events._repr_helper(indent + 1))
        return "\n".join(lines)


@dataclass
class VoiceGroupNode(ASTNode):
    """A group of voices (V1:, V2:, etc. until V0:)."""

    voices: list[VoiceNode] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = ["VoiceGroupNode"]
        for voice in self.voices:
            lines.append("  " * (indent + 1) + voice._repr_helper(indent + 1))
        return "\n".join(lines)


# Cram expression node


@dataclass
class CramNode(ASTNode):
    """A cram expression (e.g., '{c d e}2')."""

    events: EventSequenceNode
    duration: DurationNode | None = None
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        parts = ["CramNode"]
        if self.duration:
            parts[0] += f"(duration={self.duration._repr_helper(0)})"
        lines = [parts[0]]
        lines.append("  " * (indent + 1) + self.events._repr_helper(indent + 1))
        return "\n".join(lines)


# Repeat nodes


@dataclass
class RepeatNode(ASTNode):
    """A repeated event or sequence (e.g., '[c d e]*4')."""

    event: ASTNode
    times: int
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = [f"RepeatNode(times={self.times})"]
        lines.append("  " * (indent + 1) + self.event._repr_helper(indent + 1))
        return "\n".join(lines)


@dataclass
class RepetitionRange:
    """A range of repetition numbers (e.g., 1-3 or just 5)."""

    first: int
    last: int | None = None  # None means single number, not a range

    def __repr__(self) -> str:
        if self.last is None:
            return str(self.first)
        return f"{self.first}-{self.last}"


@dataclass
class OnRepetitionsNode(ASTNode):
    """An event with repetition conditions (e.g., \"c'1-3,5\")."""

    event: ASTNode
    ranges: list[RepetitionRange] = field(default_factory=list)
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        ranges_str = ",".join(str(r) for r in self.ranges)
        lines = [f"OnRepetitionsNode(ranges=[{ranges_str}])"]
        lines.append("  " * (indent + 1) + self.event._repr_helper(indent + 1))
        return "\n".join(lines)


# Bracketed event sequence (can be repeated)


@dataclass
class BracketedSequenceNode(ASTNode):
    """A bracketed event sequence (e.g., '[c d e]')."""

    events: EventSequenceNode
    position: SourcePosition | None = None

    def accept(self, visitor: ASTVisitor) -> object:
        return visitor.visit(self)

    def _repr_helper(self, indent: int) -> str:
        lines = ["BracketedSequenceNode"]
        lines.append("  " * (indent + 1) + self.events._repr_helper(indent + 1))
        return "\n".join(lines)
