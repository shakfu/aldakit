"""Score inspection and linting.

The parser, the AST and the MIDI generator already know everything needed to
answer "what is in this score?" and "what about it is likely wrong?" -- but
until now nothing exposed it. This module turns both into values that the CLI
prints and that tools (an editor plugin, a CI check) can consume directly.

Two entry points:

- :func:`inspect_score` summarises a score: its parts, their instruments and
  channels, the tempo map, the key signatures in force, and totals.
- :func:`lint_score` reports problems. Most come from the generator's own
  diagnostics channel -- an unknown instrument, an undefined variable, a note
  clamped into range -- and the rest are static checks on the AST.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ast_nodes import (
    ASTNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
)
from .midi.generator import MELODIC_CHANNELS, MidiGenerator
from .midi.types import MidiSequence, canonical_name
from .parser import parse

# Severity levels, ordered by how much they should worry the reader.
ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One problem found in a score."""

    code: str
    message: str
    severity: str = WARNING
    position: object = None  # SourcePosition | None

    def __str__(self) -> str:
        where = f"{self.position}: " if self.position is not None else ""
        return f"{where}{self.severity}: {self.message} [{self.code}]"


@dataclass
class PartInfo:
    """What one part of a score turned into."""

    name: str
    program: int
    channel: int
    percussion: bool = False
    note_count: int = 0
    key_signature: dict[str, str] = field(default_factory=dict)
    transpose: int = 0

    @property
    def instrument(self) -> str:
        """The General MIDI instrument name, or "percussion"."""
        if self.percussion:
            return "percussion"
        try:
            return canonical_name(self.program)
        except ValueError:
            return f"program {self.program}"


@dataclass
class ScoreInfo:
    """A summary of what a score contains."""

    filename: str
    parts: list[PartInfo] = field(default_factory=list)
    note_count: int = 0
    duration: float = 0.0
    tempos: list[tuple[float, float]] = field(default_factory=list)  # (time, bpm)
    control_change_count: int = 0
    variables: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    sequence: MidiSequence | None = None


def _walk(node: object):
    """Yield every AST node reachable from ``node``.

    Node children live in differently named fields (``children``, ``events``,
    ``notes``, ``elements``), so this walks attributes generically rather than
    encoding the shape of every node type.
    """
    if isinstance(node, ASTNode):
        yield node
        for value in vars(node).values():
            yield from _walk(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item)


def inspect_score(source: str, filename: str = "<input>") -> ScoreInfo:
    """Summarise a score: parts, instruments, channels, tempo map, totals.

    Args:
        source: Alda source code.
        filename: Name used in positions and in the summary.

    Returns:
        A ScoreInfo. Generation problems are included as ``findings``.
    """
    ast = parse(source, filename)
    generator = MidiGenerator()
    sequence = generator.generate(ast)

    notes_per_channel: dict[int, int] = {}
    for note in sequence.notes:
        notes_per_channel[note.channel] = notes_per_channel.get(note.channel, 0) + 1

    parts = [
        PartInfo(
            name=name,
            program=state.program,
            channel=state.channel,
            percussion=state.percussion,
            note_count=notes_per_channel.get(state.channel, 0),
            key_signature=dict(state.key_signature),
            transpose=state.transpose,
        )
        for name, state in generator.state.parts.items()
    ]

    return ScoreInfo(
        filename=filename,
        parts=parts,
        note_count=len(sequence.notes),
        duration=sequence.duration(),
        tempos=[(t.time, t.bpm) for t in sequence.tempo_changes],
        control_change_count=len(sequence.control_changes),
        variables=sorted(generator.state.variables),
        markers=sorted(generator.state.markers),
        findings=lint_ast(ast, generator),
        sequence=sequence,
    )


def lint_score(source: str, filename: str = "<input>") -> list[Finding]:
    """Report problems in a score, most severe first.

    Args:
        source: Alda source code.
        filename: Name used in positions.

    Returns:
        Findings, ordered errors first and then by position.
    """
    ast = parse(source, filename)
    generator = MidiGenerator()
    generator.generate(ast)
    return lint_ast(ast, generator)


def lint_ast(ast: RootNode, generator: MidiGenerator) -> list[Finding]:
    """Collect findings from a generator that has already run, plus static ones.

    Args:
        ast: The parsed score.
        generator: A generator whose ``generate()`` has been called on ``ast``.

    Returns:
        Findings, most severe first.
    """
    findings = [
        Finding(
            code=diagnostic.code or "generator",
            message=diagnostic.message,
            severity=_severity_for(diagnostic.code),
            position=diagnostic.position,
        )
        for diagnostic in generator.diagnostics
    ]
    findings.extend(_static_findings(ast))
    findings.extend(_sequence_findings(generator))

    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), _sort_key(f)))


# Diagnostics that change which notes are heard, rather than merely how.
_ERROR_CODES = frozenset(
    {
        "undefined-variable",
        "undefined-marker",
        "unknown-group-member",
        "channel-exhaustion",
    }
)


def _severity_for(code: str) -> str:
    return ERROR if code in _ERROR_CODES else WARNING


def _sort_key(finding: Finding) -> tuple:
    position = finding.position
    line = getattr(position, "line", 0) or 0
    column = getattr(position, "column", 0) or 0
    return (line, column, finding.code)


def _static_findings(ast: RootNode) -> list[Finding]:
    """Checks that need only the AST, not a MIDI sequence."""
    findings: list[Finding] = []

    defined: dict[str, VariableDefinitionNode] = {}
    referenced: set[str] = set()
    for node in _walk(ast):
        if isinstance(node, VariableDefinitionNode):
            if node.name in defined:
                findings.append(
                    Finding(
                        code="variable-redefined",
                        message=(
                            f"Variable {node.name!r} is defined more than once; "
                            "the last definition wins."
                        ),
                        position=node.position,
                    )
                )
            defined[node.name] = node
        elif isinstance(node, VariableReferenceNode):
            referenced.add(node.name)

    for name, node in defined.items():
        if name not in referenced:
            findings.append(
                Finding(
                    code="unused-variable",
                    message=f"Variable {name!r} is defined but never used.",
                    severity=INFO,
                    position=node.position,
                )
            )

    return findings


def _sequence_findings(generator: MidiGenerator) -> list[Finding]:
    """Checks on the generated sequence and the generator's part states."""
    findings: list[Finding] = []
    sequence = generator.sequence

    if not sequence.notes:
        findings.append(
            Finding(
                code="no-notes",
                message="The score generates no notes.",
                severity=WARNING,
            )
        )

    melodic_parts = [s for s in generator.state.parts.values() if not s.percussion]
    if len(melodic_parts) > len(MELODIC_CHANNELS):
        # The generator also reports this; keep the count in the message here.
        findings.append(
            Finding(
                code="too-many-parts",
                message=(
                    f"{len(melodic_parts)} melodic parts declared but only "
                    f"{len(MELODIC_CHANNELS)} MIDI channels are available."
                ),
                severity=ERROR,
            )
        )

    channels: dict[int, list[str]] = {}
    for name, state in generator.state.parts.items():
        channels.setdefault(state.channel, []).append(name)
    for channel, names in sorted(channels.items()):
        if len(names) > 1:
            findings.append(
                Finding(
                    code="shared-channel",
                    message=(
                        f"Parts {', '.join(sorted(names))} share MIDI channel "
                        f"{channel}; their instruments will collide."
                    ),
                    severity=ERROR,
                )
            )

    return findings
