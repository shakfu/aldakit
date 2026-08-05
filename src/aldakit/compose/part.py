"""Part class for instrument declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ast_nodes import PartDeclarationNode
from .base import ComposeElement, OctaveContext

if TYPE_CHECKING:
    from ..ast_nodes import ASTNode


@dataclass(frozen=True)
class Part(ComposeElement):
    """An instrument part declaration.

    Examples:
        >>> part("piano")
        >>> part("violin", alias="v1")
        >>> part("violin", "viola", "cello", alias="strings")
    """

    instruments: tuple[str, ...]
    alias: str | None = None

    def to_ast(self) -> PartDeclarationNode:
        """Convert to AST PartDeclarationNode."""
        return PartDeclarationNode(
            names=list(self.instruments), alias=self.alias, position=None
        )

    def to_alda(self) -> str:
        """Convert to Alda source code."""
        names = "/".join(self.instruments)
        if self.alias:
            return f'{names} "{self.alias}":'
        return f"{names}:"

    def to_events(self, ctx: OctaveContext) -> list[ASTNode]:
        """A part boundary makes the octave unknown until a note states one."""
        ctx.reset(None)
        return [self.to_ast()]

    def to_alda_parts(self, ctx: OctaveContext) -> list[str]:
        """A part boundary makes the octave unknown until a note states one."""
        ctx.reset(None)
        return [self.to_alda()]


def part(*instruments: str, alias: str | None = None) -> Part:
    """Create a part declaration.

    Args:
        *instruments: Instrument names (e.g., "piano", "violin").
        alias: Optional alias for the part.

    Returns:
        Part element.

    Examples:
        >>> part("piano")
        >>> part("violin", alias="v1")
        >>> part("violin", "viola", "cello", alias="strings")
    """
    if not instruments:
        raise ValueError("At least one instrument name is required")
    return Part(instruments=instruments, alias=alias)
