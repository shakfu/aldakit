"""Tests that the shared constant and theory tables stay single-sourced.

The review these tests come from found four facts encoded twice with different
values (a dynamics table, mode intervals, SoundFont filenames, scale
intervals). Duplication is only dangerous when the copies disagree, and a copy
that nothing imports is what lets them disagree unnoticed -- so these tests
check both: that every constant is actually used, and that the places which
used to hold a second copy now read the shared one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aldakit import constants, theory
from aldakit.compose import scales
from aldakit.midi import generator, types

SRC = Path(__file__).resolve().parent.parent / "src" / "aldakit"


def _module_sources() -> str:
    """Every first-party source file, excluding vendored code and constants."""
    return "\n".join(
        path.read_text()
        for path in SRC.rglob("*.py")
        if "ext" not in path.parts and path.name != "constants.py"
    )


def _public_constants() -> list[str]:
    return [
        match.group(1)
        for line in (SRC / "constants.py").read_text().splitlines()
        if (match := re.match(r"^([A-Z][A-Z0-9_]*)\s*[:=]", line))
    ]


class TestConstantsAreLive:
    """Every constant must have a caller, or it is not a source of truth."""

    def test_constants_are_referenced(self):
        sources = _module_sources()
        unused = [
            name
            for name in _public_constants()
            if not re.search(rf"\b{name}\b", sources)
        ]
        assert unused == [], (
            f"constants.py defines {unused} but nothing imports them. "
            "Wire them into the code that hardcodes the value, or delete them."
        )

    def test_constants_module_defines_something(self):
        # Guards the test above against silently passing on an empty list.
        assert len(_public_constants()) > 20


class TestDynamicsSingleSource:
    """The generator must read the dynamics table, not carry its own."""

    @pytest.mark.parametrize(
        ("marking", "velocity"),
        [("pppppp", 1), ("pp", 39), ("mf", 69), ("f", 79), ("ffffff", 127)],
    )
    def test_dynamic_marking_uses_constant_velocity(self, marking, velocity):
        from aldakit import Score

        assert constants.DYNAMICS_VELOCITY[marking] == velocity
        score = Score(f"piano: ({marking}) c")
        assert score.midi.notes[0].velocity == velocity

    def test_generator_has_no_second_dynamics_table(self):
        source = (SRC / "midi" / "generator.py").read_text()
        assert '"pppppp"' not in source
        assert "DYNAMICS_VELOCITY" in source

    def test_default_volume_matches_mf(self):
        assert constants.DEFAULT_VOLUME == constants.DYNAMICS_VELOCITY["mf"]
        assert generator.PartState().volume == constants.DEFAULT_VOLUME


class TestTheoryIsShared:
    """Pitch and scale tables must be the same objects everywhere."""

    def test_pitch_tables_are_shared(self):
        assert types.NOTE_OFFSETS is theory.PITCH_SEMITONES
        assert scales.PITCH_TO_OFFSET is theory.PITCH_SEMITONES
        assert scales.OFFSET_TO_PITCH is theory.SEMITONE_PITCHES
        assert scales.SCALE_INTERVALS is theory.SCALE_INTERVALS

    def test_mode_intervals_agree_with_scale_intervals(self):
        # A mode's offset within its parent major must be the interval the
        # scale table gives for that degree of the major scale.
        major = theory.SCALE_INTERVALS["major"]
        for mode, offset in theory.MODE_INTERVALS.items():
            degree = list(theory.MODE_INTERVALS).index(mode)
            assert offset == major[degree], mode
            rotated = tuple(
                (interval - offset) % 12 for interval in major[degree:] + major[:degree]
            )
            assert rotated == theory.SCALE_INTERVALS[mode], mode

    def test_part_state_defaults_come_from_constants(self):
        state = generator.PartState()
        assert state.octave == constants.DEFAULT_OCTAVE
        assert state.tempo == constants.DEFAULT_TEMPO
        assert state.quantization == constants.DEFAULT_QUANTIZATION
        assert state.default_duration == constants.DEFAULT_DURATION


class TestTheoryCalculations:
    """The key-signature helpers moved out of MidiGenerator."""

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("f+ c+ g+", {"f": "+", "c": "+", "g": "+"}),
            ("F# C#", {"f": "+", "c": "+"}),
            ("bb eb", {"b": "-", "e": "-"}),
            ("", {}),
            ("x+ q-", {}),
        ],
    )
    def test_key_signature_from_string(self, spec, expected):
        assert theory.key_signature_from_string(spec) == expected

    @pytest.mark.parametrize(
        ("symbols", "expected"),
        [
            (["g", "minor"], {"b": "-", "e": "-"}),
            (["c", "ionian"], {}),
            (["d", "dorian"], {}),
            (["e", "flat", "b", "flat"], {"e": "-", "b": "-"}),
            (["f", "sharp"], {"f": "+"}),
            (["g"], None),
            (["h", "minor"], None),
        ],
    )
    def test_key_signature_from_symbols(self, symbols, expected):
        assert theory.key_signature_from_symbols(symbols) == expected

    @pytest.mark.parametrize(
        ("root", "mode", "expected"),
        [
            ("d", "dorian", {}),  # parent C major
            ("e", "phrygian", {}),
            ("g", "mixolydian", {}),
            ("a", "dorian", {"f": "+"}),  # parent G major
            ("d", "aeolian", {"b": "-"}),  # parent F major
            ("f+", "dorian", {"f": "+", "c": "+", "g": "+", "d": "+"}),  # E major
            ("c", "not-a-mode", None),
            ("h", "dorian", None),
        ],
    )
    def test_mode_key_signature(self, root, mode, expected):
        assert theory.mode_key_signature(root, mode) == expected

    def test_mode_key_signature_returns_a_copy(self):
        signature = theory.mode_key_signature("d", "dorian")
        assert signature is not None
        signature["f"] = "+"
        assert theory.KEY_SIGNATURES["c major"] == {}

    @pytest.mark.parametrize(
        ("root", "semitone"),
        [("c", 0), ("C", 0), ("f+", 6), ("f#", 6), ("bb", 10), ("b-", 10), ("b", 11)],
    )
    def test_parse_root(self, root, semitone):
        assert theory.parse_root(root) == semitone

    @pytest.mark.parametrize("root", ["", "h", "x+"])
    def test_parse_root_rejects_non_notes(self, root):
        assert theory.parse_root(root) is None

    @pytest.mark.parametrize(
        ("accidentals", "offset"),
        [("", 0), ("+", 1), ("++", 2), ("-", -1), ("--", -2), ("_", 0), (["+"], 1)],
    )
    def test_accidental_offset(self, accidentals, offset):
        assert theory.accidental_offset(accidentals) == offset
