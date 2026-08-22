"""Serialize an Alda AST back to Alda source code.

This is the inverse of :func:`aldakit.parser.parse`. Round-tripping through it
preserves musical meaning: ``parse(write(parse(src)))`` generates the same MIDI
as ``parse(src)``. Formatting (whitespace, comments, line breaks) is not
preserved, since the AST does not record it.

Example:
    >>> from aldakit import parse
    >>> from aldakit.serialize import write_alda
    >>> write_alda(parse("piano: c d e"))
    'piano:\\n  c d e'
"""

from __future__ import annotations

from typing import NoReturn

from .ast_nodes import (
    ASTNode,
    ASTVisitor,
    AtMarkerNode,
    BarlineNode,
    BracketedSequenceNode,
    ChordNode,
    CramNode,
    DurationNode,
    EventSequenceNode,
    LispListNode,
    LispNumberNode,
    LispQuotedNode,
    LispStringNode,
    LispSymbolNode,
    MarkerNode,
    NoteLengthMsNode,
    NoteLengthNode,
    NoteLengthSecondsNode,
    NoteNode,
    OctaveDownNode,
    OctaveSetNode,
    OctaveUpNode,
    OnRepetitionsNode,
    PartDeclarationNode,
    PartNode,
    RepeatNode,
    RestNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    VoiceGroupNode,
    VoiceNode,
)

# Nodes that must occupy a line of their own because the parser terminates them
# at a newline (variable definitions) or at the next voice marker (voices).
_LINE_NODES = (VariableDefinitionNode, VoiceGroupNode)

# Indent applied to the body of a part.
_INDENT = "  "


def _format_number(value: int | float) -> str:
    """Render a number without a trailing ``.0`` for whole values."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class AldaWriter(ASTVisitor):
    """Renders an AST as Alda source code.

    Every node type produced by the parser is handled. Unknown node types raise
    :class:`TypeError` rather than being silently dropped, so that adding an AST
    node without teaching the writer about it fails loudly.
    """

    def write(self, node: ASTNode) -> str:
        """Render ``node`` as Alda source code."""
        return self._render(node)

    # -- fallback ---------------------------------------------------------

    def generic_visit(self, node: ASTNode) -> NoReturn:
        raise TypeError(
            f"{type(self).__name__} cannot serialize {type(node).__name__}. "
            "Add a visit_ method for it."
        )

    # -- helpers ----------------------------------------------------------

    def _render(self, node: ASTNode) -> str:
        """Dispatch to the visit_ method for ``node`` and return its text."""
        result = self.visit(node)
        if not isinstance(result, str):
            raise TypeError(
                f"visit_{type(node).__name__} returned {type(result).__name__}, "
                "expected str"
            )
        return result

    def _join_events(self, events: list[ASTNode]) -> str:
        """Join events with spaces, giving line-scoped nodes their own line."""
        lines: list[str] = []
        current: list[str] = []

        for event in events:
            text = self._render(event)
            if isinstance(event, _LINE_NODES):
                if current:
                    lines.append(" ".join(current))
                    current = []
                lines.append(text)
            else:
                current.append(text)

        if current:
            lines.append(" ".join(current))
        return "\n".join(lines)

    # -- top level --------------------------------------------------------

    def visit_RootNode(self, node: RootNode) -> str:
        blocks = [self._render(child) for child in node.children]
        return "\n".join(b for b in blocks if b)

    def visit_PartNode(self, node: PartNode) -> str:
        declaration = self._render(node.declaration)
        body = self._render(node.events)
        if not body:
            return declaration
        indented = "\n".join(
            _INDENT + line if line else line for line in body.split("\n")
        )
        return f"{declaration}\n{indented}"

    def visit_PartDeclarationNode(self, node: PartDeclarationNode) -> str:
        names = "/".join(node.names)
        if node.alias:
            return f'{names} "{node.alias}":'
        return f"{names}:"

    def visit_EventSequenceNode(self, node: EventSequenceNode) -> str:
        return self._join_events(node.events)

    # -- notes and rests --------------------------------------------------

    def visit_NoteNode(self, node: NoteNode) -> str:
        result = node.letter + "".join(node.accidentals)
        if node.duration is not None:
            result += self._render(node.duration)
        if node.slurred:
            result += "~"
        return result

    def visit_RestNode(self, node: RestNode) -> str:
        result = "r"
        if node.duration is not None:
            result += self._render(node.duration)
        return result

    def visit_ChordNode(self, node: ChordNode) -> str:
        # Octave changes and attribute calls may sit between chord members.
        # They belong to the member that follows, so they are attached to it
        # rather than separated by another "/".
        members: list[str] = []
        prefix = ""
        for element in node.notes:
            text = self._render(element)
            if isinstance(
                element,
                (OctaveSetNode, OctaveUpNode, OctaveDownNode, LispListNode),
            ):
                prefix += text if text in (">", "<") else text + " "
                continue
            members.append(prefix + text)
            prefix = ""
        if prefix:
            # Trailing modifier with no note after it; keep it as its own member
            members.append(prefix.strip())
        return "/".join(members)

    # -- durations --------------------------------------------------------

    def visit_DurationNode(self, node: DurationNode) -> str:
        # Multiple components are tied together: "1~1" is a double whole note.
        # A tie that crossed a barline is written back as "~|", the one form
        # the parser reads identically however the source spelled it.
        parts = []
        for index, component in enumerate(node.components):
            if index:
                parts.append("~" + "|" * node.barlines_before.get(index, 0))
            parts.append(self._render(component))
        return "".join(parts)

    def visit_NoteLengthNode(self, node: NoteLengthNode) -> str:
        return _format_number(node.denominator) + "." * node.dots

    def visit_NoteLengthMsNode(self, node: NoteLengthMsNode) -> str:
        return f"{_format_number(node.ms)}ms"

    def visit_NoteLengthSecondsNode(self, node: NoteLengthSecondsNode) -> str:
        return f"{_format_number(node.seconds)}s"

    # -- octaves and barlines ---------------------------------------------

    def visit_OctaveSetNode(self, node: OctaveSetNode) -> str:
        return f"o{node.octave}"

    def visit_OctaveUpNode(self, node: OctaveUpNode) -> str:
        return ">"

    def visit_OctaveDownNode(self, node: OctaveDownNode) -> str:
        return "<"

    def visit_BarlineNode(self, node: BarlineNode) -> str:
        return "|"

    # -- S-expressions ----------------------------------------------------

    def visit_LispListNode(self, node: LispListNode) -> str:
        return "(" + " ".join(self._render(e) for e in node.elements) + ")"

    def visit_LispSymbolNode(self, node: LispSymbolNode) -> str:
        return node.name

    def visit_LispNumberNode(self, node: LispNumberNode) -> str:
        return _format_number(node.value)

    def visit_LispStringNode(self, node: LispStringNode) -> str:
        escaped = node.value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def visit_LispQuotedNode(self, node: LispQuotedNode) -> str:
        return "'" + self._render(node.value)

    # -- variables and markers --------------------------------------------

    def visit_VariableDefinitionNode(self, node: VariableDefinitionNode) -> str:
        # The parser ends a definition at the newline, so the body must stay on
        # one line and the definition must not share a line with other events.
        body = " ".join(self._render(e) for e in node.events.events)
        return f"{node.name} = {body}"

    def visit_VariableReferenceNode(self, node: VariableReferenceNode) -> str:
        return node.name

    def visit_MarkerNode(self, node: MarkerNode) -> str:
        return f"%{node.name}"

    def visit_AtMarkerNode(self, node: AtMarkerNode) -> str:
        return f"@{node.name}"

    # -- voices -----------------------------------------------------------

    def visit_VoiceGroupNode(self, node: VoiceGroupNode) -> str:
        lines = [self._render(v) for v in node.voices]
        lines.append("V0:")  # Explicitly close the group
        return "\n".join(lines)

    def visit_VoiceNode(self, node: VoiceNode) -> str:
        body = " ".join(self._render(e) for e in node.events.events)
        return f"V{node.number}: {body}" if body else f"V{node.number}:"

    # -- grouping and repetition ------------------------------------------

    def visit_CramNode(self, node: CramNode) -> str:
        body = " ".join(self._render(e) for e in node.events.events)
        result = "{" + body + "}"
        if node.duration is not None:
            result += self._render(node.duration)
        return result

    def visit_BracketedSequenceNode(self, node: BracketedSequenceNode) -> str:
        body = " ".join(self._render(e) for e in node.events.events)
        return "[" + body + "]"

    def visit_RepeatNode(self, node: RepeatNode) -> str:
        return f"{self._as_single_event(node.event)}*{node.times}"

    def visit_OnRepetitionsNode(self, node: OnRepetitionsNode) -> str:
        ranges = ",".join(str(r) for r in node.ranges)
        return f"{self._as_single_event(node.event)}'{ranges}"

    def _as_single_event(self, node: ASTNode) -> str:
        """Render a node so a postfix operator binds to all of it.

        Postfix operators (``*4``, ``'1-3``) attach to the single preceding
        event, so anything that renders as more than one token is bracketed.
        """
        text = self._render(node)
        if isinstance(node, (BracketedSequenceNode, CramNode)):
            return text
        if " " in text or "\n" in text:
            return "[" + text + "]"
        return text


def write_alda(node: ASTNode) -> str:
    """Render an AST node as Alda source code.

    Args:
        node: Any AST node; typically the :class:`RootNode` from
            :func:`aldakit.parser.parse`.

    Returns:
        Alda source code.

    Raises:
        TypeError: If the tree contains a node type the writer cannot render.
    """
    return AldaWriter().write(node)
