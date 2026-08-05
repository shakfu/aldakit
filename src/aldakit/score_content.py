"""Where a :class:`~aldakit.score.Score` gets its music from.

A score can be built three ways -- from Alda source, from compose elements, or
by importing a MIDI file -- and each answers the same three questions
differently: what AST does this produce, what Alda source does it export as,
and how should it print. Those answers live here, one class each, so ``Score``
holds a single ``_content`` attribute instead of a mode tag plus a set of
fields that only apply in one mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .ast_nodes import EventSequenceNode, PartDeclarationNode, PartNode, RootNode
from .parser import parse

if TYPE_CHECKING:
    from .compose.base import ComposeElement

# Characters of source shown by Score.__repr__ before truncating.
_REPR_PREVIEW_CHARS = 50


class ScoreContent(ABC):
    """The music a score is built from."""

    #: Name used in parse errors and diagnostics.
    filename: str

    @abstractmethod
    def build_ast(self) -> RootNode:
        """Produce the AST for this content."""

    @abstractmethod
    def to_alda(self) -> str:
        """Export this content as Alda source."""

    @abstractmethod
    def describe(self) -> str:
        """The repr of a score holding this content."""

    @property
    def source(self) -> str:
        """The Alda source this content was created from, if any."""
        return ""

    @property
    def elements(self) -> list[ComposeElement]:
        """The compose elements this content holds, if any.

        Returns a list that is only mutable for element-based content;
        :class:`ElementsContent` overrides this with its live list.
        """
        return []

    @property
    def is_mutable(self) -> bool:
        """Whether elements can be appended to this content."""
        return False


@dataclass
class SourceContent(ScoreContent):
    """A score written as Alda source."""

    text: str
    filename: str = "<input>"

    def build_ast(self) -> RootNode:
        return parse(self.text, self.filename)

    def to_alda(self) -> str:
        return self.text

    def describe(self) -> str:
        preview = self.text[:_REPR_PREVIEW_CHARS]
        if len(self.text) > _REPR_PREVIEW_CHARS:
            preview += "..."
        preview = preview.replace("\n", "\\n")
        return f"Score({preview!r})"

    @property
    def source(self) -> str:
        return self.text


@dataclass
class ImportedContent(ScoreContent):
    """A score whose AST came from somewhere other than the Alda parser.

    Currently that means a MIDI file read by ``Score.from_midi_file``.
    """

    ast: RootNode
    filename: str

    def build_ast(self) -> RootNode:
        return self.ast

    def to_alda(self) -> str:
        from .serialize import write_alda

        return write_alda(self.ast)

    def describe(self) -> str:
        return f"Score.from_midi_file({self.filename!r})"


@dataclass
class ElementsContent(ScoreContent):
    """A score assembled from compose elements."""

    element_list: list[ComposeElement] = field(default_factory=list)
    filename: str = "<compose>"

    def build_ast(self) -> RootNode:
        from .compose.base import OctaveContext
        from .compose.part import Part

        octave_ctx = OctaveContext()
        children: list = []
        current_events: list = []
        current_part_decl: PartDeclarationNode | None = None

        def flush_part() -> None:
            """Emit the accumulated events, as a part when one was declared."""
            nonlocal current_events, current_part_decl
            if current_part_decl is not None:
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
                children.append(EventSequenceNode(events=current_events, position=None))
                current_events = []

        for element in self.element_list:
            # to_events() threads octave state, so notes that declare an
            # octave emit the octave change the AST needs to reproduce them.
            ast_nodes = element.to_events(octave_ctx)

            if isinstance(element, Part):
                flush_part()
                # Part.to_events() returns exactly one PartDeclarationNode.
                assert len(ast_nodes) == 1
                assert isinstance(ast_nodes[0], PartDeclarationNode)
                current_part_decl = ast_nodes[0]
            else:
                current_events.extend(ast_nodes)

        flush_part()

        return RootNode(children=children, position=None)

    def to_alda(self) -> str:
        from .compose.base import OctaveContext

        ctx = OctaveContext()
        parts: list[str] = []
        for element in self.element_list:
            parts.extend(element.to_alda_parts(ctx))
        return " ".join(parts)

    def describe(self) -> str:
        return f"Score.from_elements(<{len(self.element_list)} elements>)"

    @property
    def elements(self) -> list[ComposeElement]:
        return self.element_list

    @property
    def is_mutable(self) -> bool:
        return True
