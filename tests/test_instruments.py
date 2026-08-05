"""Tests for the General MIDI instrument table.

Regression cover for D1: the table used to omit Alda's canonical ``midi-``
prefixed names, so 128 of the 129 instruments in examples/all-instruments.alda
silently resolved to acoustic grand piano, and 47 GM programs were unreachable.
"""

import re
from pathlib import Path

import pytest

from aldakit import generate_midi, parse
from aldakit.midi._instruments import PROGRAM_NAMES, canonical_name
from aldakit.midi.types import (
    INSTRUMENT_PROGRAMS,
    LEGACY_INSTRUMENT_ALIASES,
    PERCUSSION_NAMES,
    is_percussion,
    lookup_instrument,
    normalize_instrument_name,
)

DOC_PATH = Path(__file__).parent.parent / "docs/alda-language/list-of-instruments.md"
EXAMPLES = Path(__file__).parent.parent / "examples"


class TestTableCompleteness:
    """The table must cover the whole General MIDI sound set."""

    def test_all_128_programs_are_reachable(self):
        """Every GM program has at least one name that resolves to it."""
        reachable = set(INSTRUMENT_PROGRAMS.values())
        missing = sorted(set(range(128)) - reachable)
        assert missing == [], f"GM programs with no Alda name: {missing}"

    def test_program_names_has_128_entries(self):
        assert len(PROGRAM_NAMES) == 128

    def test_canonical_names_are_unique(self):
        assert len(set(PROGRAM_NAMES)) == 128

    def test_canonical_names_round_trip(self):
        """canonical_name(p) resolves back to p."""
        for program in range(128):
            assert lookup_instrument(canonical_name(program)) == program

    def test_canonical_name_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            canonical_name(128)
        with pytest.raises(ValueError):
            canonical_name(-1)


class TestCanonicalNames:
    """Alda's canonical names are the midi- prefixed ones."""

    @pytest.mark.parametrize(
        "name,program",
        [
            ("midi-acoustic-grand-piano", 0),
            ("midi-harpsichord", 6),
            ("midi-clavi", 7),
            ("midi-celesta", 8),
            ("midi-church-organ", 19),
            ("midi-acoustic-guitar-nylon", 24),
            ("midi-violin", 40),
            ("midi-cello", 42),
            ("midi-trumpet", 56),
            ("midi-alto-saxophone", 65),
            ("midi-flute", 73),
            ("midi-square-lead", 80),
            ("midi-saw-wave", 81),
            ("midi-synth-pad-new-age", 88),
            ("midi-fx-rain", 96),
            ("midi-sitar", 104),
            ("midi-tinkle-bell", 112),
            ("midi-gunshot", 127),
        ],
    )
    def test_canonical_name_program(self, name, program):
        assert lookup_instrument(name) == program

    @pytest.mark.parametrize(
        "alias,program",
        [
            ("piano", 0),
            ("midi-piano", 0),
            ("harpsichord", 6),
            ("clavinet", 7),
            ("celeste", 8),
            ("vibes", 11),
            ("organ", 19),
            ("guitar", 24),
            ("upright-bass", 32),
            ("double-bass", 43),
            ("harp", 46),
            ("timpani", 47),
            ("trumpet", 56),
            ("alto-sax", 65),
            ("bari-sax", 67),
            ("flute", 73),
            ("square", 80),
            ("sawtooth", 81),
            ("chiff", 83),
            ("sitar", 104),
            ("bagpipes", 109),
            ("shanai", 111),
            ("steel-drum", 114),
        ],
    )
    def test_alias_program(self, alias, program):
        assert lookup_instrument(alias) == program


class TestLookupBehaviour:
    def test_names_are_case_insensitive(self):
        assert lookup_instrument("PIANO") == lookup_instrument("piano")

    def test_underscores_normalize_to_hyphens(self):
        assert lookup_instrument("midi_square_lead") == 80

    def test_unknown_name_returns_none(self):
        assert lookup_instrument("nonexistent-instrument") is None

    def test_normalize_instrument_name(self):
        assert normalize_instrument_name("Midi_Square_Lead") == "midi-square-lead"

    def test_percussion_names_are_not_programs(self):
        for name in PERCUSSION_NAMES:
            assert lookup_instrument(name) is None
            assert is_percussion(name)

    def test_percussion_detection(self):
        assert is_percussion("midi-percussion")
        assert is_percussion("percussion")
        assert is_percussion("PERCUSSION")
        assert not is_percussion("piano")


class TestLegacyAliases:
    """Names accepted by earlier releases keep working."""

    def test_legacy_aliases_still_resolve(self):
        for name, program in LEGACY_INSTRUMENT_ALIASES.items():
            assert lookup_instrument(name) == int(program), name

    @pytest.mark.parametrize(
        "name,program",
        [
            ("acoustic-grand-piano", 0),
            ("electric-piano-1", 4),
            ("bass", 32),
            ("string-ensemble-1", 48),
            ("choir-aahs", 52),
            ("brass-section", 61),
            ("blown-bottle", 76),
        ],
    )
    def test_specific_legacy_alias(self, name, program):
        assert lookup_instrument(name) == program

    def test_canonical_wins_over_legacy_on_conflict(self):
        """'organ' is Alda's alias for church organ, not drawbar organ."""
        assert lookup_instrument("organ") == 19


class TestGeneratedTableMatchesDocs:
    """The table is generated from the language docs; keep them in sync."""

    def test_every_documented_name_resolves(self):
        assert DOC_PATH.exists(), "instrument list doc is missing"
        text = DOC_PATH.read_text(encoding="utf-8")
        bullet = re.compile(
            r"^\*\s+(?P<name>[a-z0-9+\-]+)\s*(?:\((?P<aliases>[^)]*)\)?)?\s*$"
        )
        documented: list[str] = []
        for line in text.splitlines():
            match = bullet.match(line.strip())
            if not match:
                continue
            documented.append(match.group("name"))
            documented.extend(
                a.strip()
                for a in (match.group("aliases") or "").split(",")
                if a.strip()
            )

        assert len(documented) >= 128
        unresolved = [n for n in documented if lookup_instrument(n) is None]
        assert unresolved == [], f"documented but unmapped: {unresolved}"


class TestExampleFiles:
    """The bundled examples must not silently degrade to piano."""

    def test_all_instruments_example_uses_distinct_programs(self):
        """examples/all-instruments.alda must reach all 128 GM programs."""
        source = (EXAMPLES / "all-instruments.alda").read_text()
        sequence = generate_midi(parse(source))
        programs = {int(pc.program) for pc in sequence.program_changes}
        # 128 melodic programs; midi-percussion emits no program change.
        assert len(programs) == 128

    def test_no_example_has_unknown_instruments(self):
        """Every instrument named in examples/ resolves to a real program."""
        from aldakit.midi.generator import MidiGenerator

        offenders = {}
        for path in sorted(EXAMPLES.glob("*.alda")):
            generator = MidiGenerator()
            generator.generate(parse(path.read_text(), str(path)))
            unknown = [
                str(d)
                for d in generator.diagnostics
                if "Unknown instrument" in d.message
            ]
            if unknown:
                offenders[path.name] = unknown
        assert offenders == {}
