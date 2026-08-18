"""Tests for score inspection and linting.

The MIDI generator has always known when a score was wrong -- an unknown
instrument, an undefined variable, a note pushed out of MIDI range -- but the
only way to see it was to play the score and listen. These tests cover the
module that turns those diagnostics, plus static checks on the AST, into
findings a user or a tool can read.
"""

from __future__ import annotations

import pytest

from aldakit.analysis import (
    ERROR,
    INFO,
    WARNING,
    Finding,
    inspect_score,
    lint_score,
)


def codes(findings: list[Finding]) -> list[str]:
    return [f.code for f in findings]


class TestInspect:
    def test_counts_parts_and_notes(self):
        info = inspect_score("piano: c d e\nviolin: e f g", "song.alda")
        assert info.filename == "song.alda"
        assert [p.name for p in info.parts] == ["piano", "violin"]
        assert info.note_count == 6
        assert all(p.note_count == 3 for p in info.parts)

    def test_reports_instrument_and_channel(self):
        info = inspect_score("piano: c\ncello: c")
        piano, cello = info.parts
        assert (piano.program, piano.channel) == (0, 0)
        assert piano.instrument == "midi-acoustic-grand-piano"
        assert (cello.program, cello.channel) == (42, 1)
        assert cello.instrument == "midi-cello"

    def test_percussion_is_named_and_on_the_drum_channel(self):
        info = inspect_score("midi-percussion: c d e")
        part = info.parts[0]
        assert part.percussion is True
        assert part.channel == 9
        assert part.instrument == "percussion"

    def test_duration_and_tempo_map(self):
        info = inspect_score("piano: (tempo 60) c4 d4")
        assert info.duration == pytest.approx(1.9, abs=0.01)  # 1s + 0.9s quantized
        assert [bpm for _, bpm in info.tempos] == [120.0, 60.0]

    def test_key_signature_and_transposition_are_reported(self):
        info = inspect_score("piano: (key-sig '(g minor)) (transpose 2) c")
        part = info.parts[0]
        assert part.key_signature == {"b": "-", "e": "-"}
        assert part.transpose == 2

    def test_variables_and_markers_are_listed(self):
        info = inspect_score("theme = c d\npiano: %start theme @start theme")
        assert info.variables == ["theme"]
        assert info.markers == ["start"]

    def test_control_changes_are_counted(self):
        info = inspect_score("piano: (panning 25) c (pan 75) d")
        assert info.control_change_count == 2

    def test_findings_travel_with_the_summary(self):
        info = inspect_score("bogus-instrument: c")
        assert "unknown-instrument" in codes(info.findings)


class TestLintDiagnostics:
    """Findings that come from the generator's diagnostics channel."""

    def test_unknown_instrument(self):
        findings = lint_score("bogus: c d e")
        assert codes(findings) == ["unknown-instrument"]
        assert findings[0].severity == WARNING

    def test_undefined_variable_is_an_error(self):
        findings = lint_score("piano: nosuchvar")
        assert "undefined-variable" in codes(findings)
        assert [f.severity for f in findings if f.code == "undefined-variable"] == [
            ERROR
        ]

    def test_undefined_marker_is_an_error(self):
        findings = lint_score("piano: c @nowhere d")
        assert "undefined-marker" in codes(findings)

    def test_unknown_attribute(self):
        findings = lint_score("piano: (frobnicate 3) c")
        assert "unknown-attribute" in codes(findings)

    def test_note_out_of_range(self):
        findings = lint_score("piano: o9 > > c")
        out_of_range = [f for f in findings if f.code == "note-out-of-range"]
        assert out_of_range
        assert "outside the MIDI range" in out_of_range[0].message

    def test_notes_in_range_are_not_reported(self):
        assert codes(lint_score("piano: o0 c\npiano: o9 g")) == []

    def test_channel_exhaustion(self):
        parts = "\n".join(f"midi-instrument-{i}: c" for i in range(1))
        # 16 distinct melodic parts: one more than there are melodic channels
        source = "\n".join(f"{name}: c" for name in _sixteen_instruments())
        findings = lint_score(source + parts)
        assert "channel-exhaustion" in codes(findings)
        assert "too-many-parts" in codes(findings)

    def test_positions_are_preserved(self):
        findings = lint_score("piano: c\nbogus: d")
        finding = next(f for f in findings if f.code == "unknown-instrument")
        assert getattr(finding.position, "line", None) == 2


def _sixteen_instruments() -> list[str]:
    return [
        "piano",
        "violin",
        "viola",
        "cello",
        "flute",
        "oboe",
        "clarinet",
        "bassoon",
        "trumpet",
        "trombone",
        "tuba",
        "harp",
        "organ",
        "banjo",
        "harmonica",
        "accordion",
    ]


class TestLintStaticChecks:
    """Findings that need only the AST."""

    def test_unused_variable(self):
        findings = lint_score("unused = c d\npiano: c")
        unused = [f for f in findings if f.code == "unused-variable"]
        assert len(unused) == 1
        assert unused[0].severity == INFO
        assert "unused" in unused[0].message

    def test_used_variable_is_not_reported(self):
        assert "unused-variable" not in codes(lint_score("theme = c d\npiano: theme"))

    def test_redefined_variable(self):
        findings = lint_score("theme = c\ntheme = d\npiano: theme")
        assert "variable-redefined" in codes(findings)

    def test_score_with_no_notes(self):
        assert "no-notes" in codes(lint_score("piano:"))

    def test_clean_score_has_no_findings(self):
        assert lint_score("piano: (tempo 120) c d e f g") == []


class TestOrdering:
    def test_errors_come_first(self):
        findings = lint_score("unused = c\nbogus: nosuchvar")
        severities = [f.severity for f in findings]
        assert severities == sorted(severities, key=[ERROR, WARNING, INFO].index)

    def test_same_severity_orders_by_position(self):
        findings = lint_score("piano: (frob 1) c\nbogus: d")
        warnings = [f for f in findings if f.severity == WARNING]
        lines = [getattr(f.position, "line", 0) for f in warnings]
        assert lines == sorted(lines)


class TestFindingFormatting:
    def test_includes_position_severity_and_code(self):
        finding = next(f for f in lint_score("bogus: c") if f.code)
        text = str(finding)
        assert "warning" in text
        assert "[unknown-instrument]" in text
        assert "<input>:1:1" in text

    def test_formats_without_a_position(self):
        assert str(Finding("no-notes", "Nothing here.")) == (
            "warning: Nothing here. [no-notes]"
        )


class TestExamples:
    """What linting the bundled examples reports.

    Every bundled example must lint without errors. Two of them declare far
    more parts than there are MIDI channels -- all-instruments.alda plays 128
    instruments -- and they pass because channels are handed on once a part
    has stopped sounding, which is what those two examples exist to show.
    """

    def _examples(self):
        from pathlib import Path

        return sorted((Path(__file__).parent.parent / "examples").glob("*.alda"))

    def test_no_example_has_errors(self):
        offenders = {}
        for path in self._examples():
            findings = lint_score(path.read_text(encoding="utf-8"), path.name)
            errors = [f for f in findings if f.severity == ERROR]
            if errors:
                offenders[path.name] = sorted({f.code for f in errors})

        assert offenders == {}

    def test_every_example_can_be_linted(self):
        for path in self._examples():
            lint_score(path.read_text(encoding="utf-8"), path.name)
