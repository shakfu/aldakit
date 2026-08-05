"""Duration handling shared by compose elements.

``Note`` and ``Rest`` both accept the same four ways of expressing a length --
a note-length denominator with optional dots, a millisecond value, or a value
in seconds -- and both have to turn that into a :class:`DurationNode` and into
Alda source. Those two conversions live here so the two elements cannot drift
apart.
"""

from __future__ import annotations

from ..ast_nodes import (
    DurationNode,
    NoteLengthMsNode,
    NoteLengthNode,
    NoteLengthSecondsNode,
)

# Denominator assumed when dots are given without a note length: ``dots=1``
# means a dotted quarter, matching Alda's default note length.
DEFAULT_DENOMINATOR = 4


def build_duration_node(
    *,
    duration: int | None,
    dots: int = 0,
    ms: float | None = None,
    seconds: float | None = None,
) -> DurationNode | None:
    """Build the AST duration node for a compose element.

    Args:
        duration: Note-length denominator (4 for a quarter note), or None.
        dots: Number of augmentation dots.
        ms: Length in milliseconds, if given.
        seconds: Length in seconds, if given.

    Returns:
        A DurationNode, or None when the element inherits the part's current
        default duration.
    """
    if ms is not None:
        return DurationNode(
            components=[NoteLengthMsNode(ms=ms, position=None)],
            position=None,
        )
    if seconds is not None:
        return DurationNode(
            components=[NoteLengthSecondsNode(seconds=seconds, position=None)],
            position=None,
        )
    if duration is not None:
        return DurationNode(
            components=[NoteLengthNode(denominator=duration, dots=dots, position=None)],
            position=None,
        )
    if dots > 0:
        return DurationNode(
            components=[
                NoteLengthNode(
                    denominator=DEFAULT_DENOMINATOR, dots=dots, position=None
                )
            ],
            position=None,
        )
    return None


def format_duration(
    *,
    duration: int | None,
    dots: int = 0,
    ms: float | None = None,
    seconds: float | None = None,
) -> str:
    """Format the same duration parameters as an Alda source suffix.

    The result is what follows the pitch letter or ``r``: ``"4."``, ``"500ms"``,
    ``"1.5s"``, or the empty string when the element inherits the default
    duration. It mirrors :func:`build_duration_node` exactly, including the
    quarter-note assumption for dots without a denominator.
    """
    if ms is not None:
        return f"{int(ms)}ms"
    if seconds is not None:
        return f"{seconds}s"
    if duration is not None:
        return f"{duration}{'.' * dots}"
    if dots > 0:
        return f"{DEFAULT_DENOMINATOR}{'.' * dots}"
    return ""
