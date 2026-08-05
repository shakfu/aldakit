"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES = PROJECT_ROOT / "examples"
GOLDEN_DIR = Path(__file__).parent / "golden"


def midi_fingerprint(sequence) -> dict:
    """A stable, comparable summary of everything a MidiSequence sounds like.

    Events are rendered as compact strings so the golden fixture stays readable
    and produces a reviewable diff. Times are rounded to microseconds so that
    floating point noise does not make comparisons brittle, while still
    catching real timing changes.
    """

    def t(value: float) -> str:
        return f"{value:.6f}"

    return {
        "notes": [
            f"{t(n.start_time)} ch{n.channel} p{n.pitch} d{t(n.duration)} v{n.velocity}"
            for n in sorted(
                sequence.notes,
                key=lambda n: (n.start_time, n.channel, n.pitch, n.duration),
            )
        ],
        "program_changes": sorted(
            f"{t(p.time)} ch{p.channel} prog{int(p.program)}"
            for p in sequence.program_changes
        ),
        "control_changes": sorted(
            f"{t(c.time)} ch{c.channel} cc{c.control}={c.value}"
            for c in sequence.control_changes
        ),
        "tempo_changes": sorted(
            f"{t(x.time)} {x.bpm:.6f}bpm" for x in sequence.tempo_changes
        ),
    }
