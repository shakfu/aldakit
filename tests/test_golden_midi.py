"""Golden-output tests: every bundled example must keep sounding the same.

These pin the exact notes, channels, programs, timings and velocities that each
file in ``examples/`` generates. Structural tests elsewhere check that the AST
has the right shape; these check that the music does not change underneath us.

Most of the defects this suite was written for -- silent instrument fallback,
percussion on the wrong channel, melodic parts on the drum channel, octaves
being dropped -- produced perfectly well-formed output that simply sounded
wrong. Only an end-to-end comparison catches that class of bug.

When a change to the generator is intentional, regenerate with::

    python scripts/gen_golden_midi.py

and review the diff: it shows exactly which notes changed.
"""

from __future__ import annotations

import json

import pytest

from aldakit import generate_midi, parse
from tests.helpers import EXAMPLES, GOLDEN_DIR, midi_fingerprint

GOLDEN_PATH = GOLDEN_DIR / "examples.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. Run: python scripts/gen_golden_midi.py"
    )
    return json.loads(GOLDEN_PATH.read_text())


def _examples():
    return sorted(EXAMPLES.glob("*.alda"))


class TestGoldenCoverage:
    def test_every_example_has_a_fixture(self, golden):
        missing = [p.name for p in _examples() if p.name not in golden]
        assert missing == [], (
            f"examples without a golden fixture: {missing}. "
            "Run: python scripts/gen_golden_midi.py"
        )

    def test_no_stale_fixtures(self, golden):
        names = {p.name for p in _examples()}
        stale = sorted(set(golden) - names)
        assert stale == [], f"fixtures for deleted examples: {stale}"

    def test_fixtures_are_not_empty(self, golden):
        """A fixture with no notes would silently pass any comparison."""
        empty = [name for name, data in golden.items() if not data["notes"]]
        assert empty == []


class TestGoldenOutput:
    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_example_matches_golden(self, path, golden):
        expected = golden[path.name]
        actual = midi_fingerprint(generate_midi(parse(path.read_text(), str(path))))

        # Compare event kinds separately so failures point at the right thing
        assert actual["program_changes"] == expected["program_changes"]
        assert actual["control_changes"] == expected["control_changes"]
        assert actual["tempo_changes"] == expected["tempo_changes"]
        assert len(actual["notes"]) == len(expected["notes"])
        assert actual["notes"] == expected["notes"]


class TestGoldenInvariants:
    """Properties every fixture must satisfy, independent of its exact values."""

    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_pitches_and_velocities_in_range(self, path):
        sequence = generate_midi(parse(path.read_text(), str(path)))
        for note in sequence.notes:
            assert 0 <= note.pitch <= 127, f"pitch out of range: {note}"
            assert 0 <= note.velocity <= 127, f"velocity out of range: {note}"
            assert 0 <= note.channel <= 15, f"channel out of range: {note}"

    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_no_negative_or_zero_durations(self, path):
        sequence = generate_midi(parse(path.read_text(), str(path)))
        for note in sequence.notes:
            assert note.duration > 0, f"non-positive duration: {note}"
            assert note.start_time >= 0, f"negative start time: {note}"

    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_programs_in_range(self, path):
        sequence = generate_midi(parse(path.read_text(), str(path)))
        for change in sequence.program_changes:
            assert 0 <= int(change.program) <= 127

    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_generation_is_deterministic(self, path):
        """Two runs of the generator must agree."""
        source = path.read_text()
        first = midi_fingerprint(generate_midi(parse(source, str(path))))
        second = midi_fingerprint(generate_midi(parse(source, str(path))))
        assert first == second
