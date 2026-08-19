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
from enum import Enum

from .ast_nodes import (
    ASTNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
)
from .midi.generator import MELODIC_CHANNELS, MidiGenerator
from .midi.types import MidiSequence, canonical_name
from .parser import parse


class Severity(str, Enum):
    """How much a finding should worry the reader, worst first.

    Members are declared in order of severity and that declaration is the
    only place the order is recorded; :attr:`rank` reads it back rather than
    repeating it. A plain ``str`` subclass, so that ``finding.severity ==
    "error"`` keeps working and findings still serialise as their names.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    def __str__(self) -> str:
        # Enum's own __format__ changed in 3.11: without this, an f-string
        # gives "error" on 3.10 and "Severity.ERROR" from 3.11 on, which
        # would put the interpreter version into what the linter prints.
        return self.value

    @property
    def rank(self) -> int:
        """Position in the ordering. Lower means more serious."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {severity: index for index, severity in enumerate(Severity)}

# The names these levels were exported under before they became an enum.
# They compare equal to the strings they used to be, so callers do not care.
ERROR = Severity.ERROR
WARNING = Severity.WARNING
INFO = Severity.INFO


def severity_rank(severity: str) -> int:
    """Ordering position for a severity, including one we do not know.

    Anything unrecognised sorts after everything recognised rather than
    raising: a finding with an odd severity is still worth reporting.
    """
    try:
        return _SEVERITY_RANK[Severity(severity)]
    except ValueError:
        return len(_SEVERITY_RANK)


@dataclass(frozen=True)
class Finding:
    """One problem found in a score."""

    code: str
    message: str
    severity: Severity = WARNING
    position: object = None  # SourcePosition | None

    def __str__(self) -> str:
        where = f"{self.position}: " if self.position is not None else ""
        return f"{where}{self.severity}: {self.message} [{self.code}]"


@dataclass
class PartInfo:
    """What one part of a score turned into."""

    name: str
    program: int
    #: The channel the part sounds on first, or -1 if it never sounds. A part
    #: can move between channels in a score that has to reuse them; see
    #: ``channels``.
    channel: int
    percussion: bool = False
    note_count: int = 0
    #: Every channel the part sounds on, in the order it uses them.
    channels: tuple[int, ...] = ()
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

    # Counted before channels were assigned: once a channel can carry two
    # parts at different points in the score, the notes on it no longer say
    # which part played them.
    assignment = generator.channel_assignment
    notes_per_part = assignment.note_counts

    parts = [
        PartInfo(
            name=name,
            program=state.program,
            channel=state.channel,
            percussion=state.percussion,
            note_count=notes_per_part.get(state.allocated_channel, 0),
            key_signature=dict(state.key_signature),
            transpose=state.transpose,
            channels=tuple(
                assignment.channels.get(state.allocated_channel, [state.channel])
            ),
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

    return sorted(findings, key=lambda f: (severity_rank(f.severity), _sort_key(f)))


# Diagnostics that change which notes are heard, rather than merely how.
_ERROR_CODES = frozenset(
    {
        "undefined-variable",
        "undefined-marker",
        "unknown-group-member",
        "channel-exhaustion",
    }
)


def _severity_for(code: str) -> Severity:
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

    # A score may declare far more parts than there are channels and still be
    # fine, because a part only holds a channel while it is sounding. What has
    # to fit in the 15 melodic channels is how many parts play at once.
    assignment = generator.channel_assignment
    if assignment.overflowed:
        findings.append(
            Finding(
                code="too-many-parts",
                message=(
                    f"{assignment.max_concurrent} melodic parts sound at the "
                    f"same time but only {len(MELODIC_CHANNELS)} MIDI channels "
                    "are available."
                ),
                severity=ERROR,
            )
        )

    names_by_channel = {
        state.allocated_channel: name for name, state in generator.state.parts.items()
    }
    for channel, first, second in assignment.conflicts:
        names = sorted(
            names_by_channel.get(owner, f"channel {owner}") for owner in (first, second)
        )
        findings.append(
            Finding(
                code="shared-channel",
                message=(
                    f"Parts {names[0]} and {names[1]} play at the same time on "
                    f"MIDI channel {channel}; their instruments will collide."
                ),
                severity=ERROR,
            )
        )

    return findings
