"""Attribute functions for compose elements (tempo, volume, dynamics, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import ComposeElement

if TYPE_CHECKING:
    pass

from ..ast_nodes import (
    LispListNode,
    LispNumberNode,
    LispSymbolNode,
    OctaveDownNode,
    OctaveSetNode,
    OctaveUpNode,
)


@dataclass(frozen=True)
class Tempo(ComposeElement):
    """Tempo attribute."""

    bpm: int | float
    global_: bool = False

    def to_ast(self) -> LispListNode:
        symbol = "tempo!" if self.global_ else "tempo"
        return LispListNode(
            elements=[
                LispSymbolNode(name=symbol, position=None),
                LispNumberNode(value=self.bpm, position=None),
            ],
            position=None,
        )

    def to_alda(self) -> str:
        symbol = "tempo!" if self.global_ else "tempo"
        return f"({symbol} {self.bpm})"


@dataclass(frozen=True)
class Volume(ComposeElement):
    """Volume attribute."""

    level: int | float

    def to_ast(self) -> LispListNode:
        return LispListNode(
            elements=[
                LispSymbolNode(name="volume", position=None),
                LispNumberNode(value=self.level, position=None),
            ],
            position=None,
        )

    def to_alda(self) -> str:
        return f"(volume {self.level})"


@dataclass(frozen=True)
class Quant(ComposeElement):
    """Quantization attribute."""

    level: int | float

    def to_ast(self) -> LispListNode:
        return LispListNode(
            elements=[
                LispSymbolNode(name="quant", position=None),
                LispNumberNode(value=self.level, position=None),
            ],
            position=None,
        )

    def to_alda(self) -> str:
        return f"(quant {self.level})"


@dataclass(frozen=True)
class Panning(ComposeElement):
    """Panning attribute."""

    level: int | float

    def to_ast(self) -> LispListNode:
        return LispListNode(
            elements=[
                LispSymbolNode(name="panning", position=None),
                LispNumberNode(value=self.level, position=None),
            ],
            position=None,
        )

    def to_alda(self) -> str:
        return f"(panning {self.level})"


@dataclass(frozen=True)
class OctaveSet(ComposeElement):
    """Set octave to specific value."""

    value: int

    def to_ast(self) -> OctaveSetNode:
        return OctaveSetNode(octave=self.value, position=None)

    def to_alda(self) -> str:
        return f"o{self.value}"


@dataclass(frozen=True)
class OctaveUp(ComposeElement):
    """Increase octave by one."""

    def to_ast(self) -> OctaveUpNode:
        return OctaveUpNode(position=None)

    def to_alda(self) -> str:
        return ">"


@dataclass(frozen=True)
class OctaveDown(ComposeElement):
    """Decrease octave by one."""

    def to_ast(self) -> OctaveDownNode:
        return OctaveDownNode(position=None)

    def to_alda(self) -> str:
        return "<"


@dataclass(frozen=True)
class Dynamic(ComposeElement):
    """Dynamic marking (pp, p, mp, mf, f, ff)."""

    marking: str

    def to_ast(self) -> LispListNode:
        return LispListNode(
            elements=[LispSymbolNode(name=self.marking, position=None)],
            position=None,
        )

    def to_alda(self) -> str:
        return f"({self.marking})"


# Factory functions


def tempo(bpm: int | float, global_: bool = False) -> Tempo:
    """Create a tempo attribute.

    Args:
        bpm: Beats per minute.
        global_: If True, applies globally to all parts.

    Returns:
        Tempo element.
    """
    return Tempo(bpm=bpm, global_=global_)


def volume(level: int | float) -> Volume:
    """Create a volume attribute.

    Args:
        level: Volume level (0-100).

    Returns:
        Volume element.
    """
    return Volume(level=level)


# Alias for volume
vol = volume


def quant(level: int | float) -> Quant:
    """Create a quantization attribute.

    Args:
        level: Quantization level (0-100).

    Returns:
        Quant element.
    """
    return Quant(level=level)


def panning(level: int | float) -> Panning:
    """Create a panning attribute.

    Args:
        level: Pan position (0=left, 50=center, 100=right).

    Returns:
        Panning element.
    """
    return Panning(level=level)


def octave(n: int) -> OctaveSet:
    """Set the octave.

    Args:
        n: Octave number (typically 0-8).

    Returns:
        OctaveSet element.
    """
    return OctaveSet(value=n)


def octave_up() -> OctaveUp:
    """Increase octave by one.

    Returns:
        OctaveUp element.
    """
    return OctaveUp()


def octave_down() -> OctaveDown:
    """Decrease octave by one.

    Returns:
        OctaveDown element.
    """
    return OctaveDown()


# Dynamic markings


def pp() -> Dynamic:
    """Pianissimo (very soft)."""
    return Dynamic(marking="pp")


def p() -> Dynamic:
    """Piano (soft)."""
    return Dynamic(marking="p")


def mp() -> Dynamic:
    """Mezzo-piano (moderately soft)."""
    return Dynamic(marking="mp")


def mf() -> Dynamic:
    """Mezzo-forte (moderately loud)."""
    return Dynamic(marking="mf")


def f() -> Dynamic:
    """Forte (loud)."""
    return Dynamic(marking="f")


def ff() -> Dynamic:
    """Fortissimo (very loud)."""
    return Dynamic(marking="ff")
