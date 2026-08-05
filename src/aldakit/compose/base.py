"""Base classes and protocols for compose elements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ast_nodes import ASTNode

# Alda parts start in octave 4 unless told otherwise.
DEFAULT_OCTAVE = 4


@dataclass
class OctaveContext:
    """Tracks the current octave while flattening compose elements.

    Octave is stateful in Alda: ``o5 c d`` puts both notes in octave 5. Compose
    elements instead carry an absolute octave per note, so conversion to AST or
    to Alda source has to emit an octave change only where the octave actually
    changes. This class holds that state.
    """

    octave: int | None = None

    def shift_to(self, octave: int | None) -> int | None:
        """Record a move to ``octave``.

        Args:
            octave: The absolute octave a note declares, or None if it does
                not declare one.

        Returns:
            The octave to emit an explicit change for, or None when no change
            is needed (either the note declared no octave, or the context is
            already there).
        """
        if octave is None or octave == self.octave:
            return None
        self.octave = octave
        return octave

    def reset(self, octave: int | None = None) -> None:
        """Reset the tracked octave, e.g. at a part boundary."""
        self.octave = octave


class ComposeElement(ABC):
    """Base class for all compose elements.

    All compose elements can generate AST nodes directly and
    serialize to Alda source code.
    """

    @abstractmethod
    def to_ast(self) -> ASTNode:
        """Convert this element to an AST node.

        Note:
            This returns a single node and therefore cannot carry the octave
            changes that surrounding context implies. Use :meth:`to_events`
            (which containers such as ``Seq`` and ``Score`` call for you) when
            the octave a note declares must be preserved.
        """
        ...

    @abstractmethod
    def to_alda(self) -> str:
        """Convert this element to Alda source code.

        Note:
            As with :meth:`to_ast`, octave context is not applied. Use
            :meth:`to_alda_parts` for a context-aware conversion.
        """
        ...

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """Convert to AST events, emitting octave changes required by ``ctx``.

        Args:
            ctx: The octave context, mutated as elements are visited.

        Returns:
            One or more AST nodes.
        """
        return [self.to_ast()]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """Convert to Alda source fragments, emitting octave changes.

        Args:
            ctx: The octave context, mutated as elements are visited.

        Returns:
            Source fragments to be joined with spaces.
        """
        return [self.to_alda()]
