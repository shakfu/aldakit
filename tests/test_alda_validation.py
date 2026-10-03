"""Input Alda refuses: reported as diagnostics, or raised in strict mode.

Each case was checked against `alda parse -o data` 2.4.7, which rejects it.
See docs/dev/alda-deviations.md.
"""

from __future__ import annotations

import pytest

from aldakit import Score, generate_midi, parse
from aldakit.analysis import ERROR, lint_score
from aldakit.cli import main
from aldakit.errors import AldaGenerationError
from aldakit.midi.generator import MidiGenerator


def codes(source: str) -> list[str]:
    generator = MidiGenerator()
    generator.generate(parse(source))
    return [d.code for d in generator.diagnostics]


class TestAttributeRanges:
    @pytest.mark.parametrize(
        "attribute",
        [
            "(vol 120)",
            "(volume -1)",
            "(track-volume 120)",
            "(pan 101)",
            "(quant -5)",
            "(tempo 0)",
            "(set-duration 0)",
            "(set-duration-ms 0)",
            "(set-note-length 0)",
        ],
    )
    def test_out_of_range_is_reported(self, attribute):
        assert codes(f"piano: {attribute} c") == ["invalid-attribute-value"]

    @pytest.mark.parametrize(
        "attribute", ["(vol 0)", "(vol 100)", "(pan 0)", "(quant 0)", "(quant 200)"]
    )
    def test_boundaries_are_accepted(self, attribute):
        assert codes(f"piano: {attribute} c") == []

    def test_an_invalid_value_is_ignored(self):
        seq = generate_midi(parse("piano: (vol 50) c (vol 120) d"))
        assert [n.velocity for n in seq.notes] == [64, 64]

    def test_lint_reports_it_as_an_error(self):
        findings = lint_score("piano: (vol 120) c")
        assert [(f.code, f.severity) for f in findings] == [
            ("invalid-attribute-value", ERROR)
        ]


class TestInstanceNaming:
    @pytest.mark.parametrize(
        "source",
        [
            'violin: c\nviolin "v2": d',
            'violin "v1": c\nviolin: d',
            'violin: c\nviolin/viola "s": d\nviolin: e',
        ],
    )
    def test_unnamed_and_named_instances_are_ambiguous(self, source):
        assert codes(source) == ["ambiguous-instance"]

    @pytest.mark.parametrize(
        "source",
        [
            'violin "v1": c\nviolin/viola "s": d',
            'piano "pno": c\npno: d',
            "piano: c\npiano: d",
            'violin "v1": c\nviolin "v2": d',
        ],
    )
    def test_consistent_naming_is_accepted(self, source):
        assert codes(source) == []


class TestStrict:
    def test_generator_raises_on_the_first_diagnostic(self):
        with pytest.raises(AldaGenerationError) as raised:
            generate_midi(parse("piano: (vol 120) c"), strict=True)
        assert raised.value.diagnostic.code == "invalid-attribute-value"

    def test_any_diagnostic_raises(self):
        with pytest.raises(AldaGenerationError):
            generate_midi(parse("piano: (frobnicate 3) c"), strict=True)

    def test_a_clean_score_generates(self):
        assert generate_midi(parse("piano: c d"), strict=True).notes

    def test_score_is_lenient_by_default(self):
        score = Score("piano: (vol 120) c")
        assert score.midi.notes
        assert [d.code for d in score.diagnostics] == ["invalid-attribute-value"]

    def test_score_strict_raises(self):
        with pytest.raises(AldaGenerationError):
            Score("piano: (vol 120) c", strict=True).midi

    def test_cli_strict_fails(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        assert main(["eval", "--strict", "piano: (vol 120) c", "-o", str(out)]) == 1
        assert "Error:" in capsys.readouterr().err
        assert not out.exists()

    def test_cli_without_strict_warns_and_saves(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        assert main(["eval", "piano: (vol 120) c", "-o", str(out)]) == 0
        assert "Warning:" in capsys.readouterr().err
        assert out.exists()
