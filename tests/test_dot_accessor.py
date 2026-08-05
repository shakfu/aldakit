"""Tests for the group-member dot accessor (``strings.cello:``).

Alda lets an aliased group of instruments be addressed member by member via the
dot operator (docs/alda-language/scores-and-parts.md). aldakit did not implement
it: the scanner split ``strings.cello`` into three tokens, the part-declaration
lookahead stopped at the dot, and the result was a brand new ``cello`` instance
on its own channel starting at t=0 rather than the group's existing cello
continuing where it left off. examples/dot_accessor.alda was listed as
compatible while producing that wrong output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aldakit import parse
from aldakit.midi.generator import MidiGenerator
from aldakit.serialize import write_alda
from aldakit.tokens import TokenType

EXAMPLES = Path(__file__).parent.parent / "examples"


def generate(source: str):
    generator = MidiGenerator()
    return generator, generator.generate(parse(source))


class TestScanning:
    def test_dotted_name_is_one_token(self):
        from aldakit.scanner import Scanner

        tokens = [t for t in Scanner("strings.cello: c").scan()]
        names = [t.literal for t in tokens if t.type == TokenType.NAME]
        assert names == ["strings.cello"]

    def test_duration_dot_is_still_a_dot(self):
        from aldakit.scanner import Scanner

        tokens = Scanner("piano: c4.").scan()
        assert any(t.type == TokenType.DOT for t in tokens)

    def test_double_duration_dot(self):
        _, sequence = generate("piano: c2..")
        # Double-dotted half note = 3.5 beats = 1.75s at 120bpm, times 0.9 quant
        assert sequence.notes[0].duration == pytest.approx(1.75 * 0.9)

    def test_trailing_dot_is_not_absorbed(self):
        from aldakit.scanner import Scanner

        tokens = Scanner("piano: c4. d").scan()
        assert [t.literal for t in tokens if t.type == TokenType.NAME] == ["piano"]

    def test_dotted_durations_unaffected(self):
        _, plain = generate("piano: c4. d2. e8.")
        assert len(plain.notes) == 3
        assert plain.notes[0].duration == pytest.approx(1.5 * 0.5 * 0.9)


class TestGroupMemberResolution:
    SOURCE = 'violin/viola/cello "strings": g1\nstrings.cello: c1'

    def test_no_diagnostics(self):
        generator, _ = generate(self.SOURCE)
        assert generator.diagnostics == []

    def test_no_extra_part_is_created(self):
        """The reference must reuse the group's cello, not make a new one."""
        generator, _ = generate(self.SOURCE)
        assert len(generator.state.parts) == 3

    def test_group_membership_is_recorded(self):
        generator, _ = generate(self.SOURCE)
        assert generator.state.groups["strings"] == {
            "violin": "strings_0",
            "viola": "strings_1",
            "cello": "strings_2",
        }

    def test_member_reuses_the_group_channel(self):
        generator, sequence = generate(self.SOURCE)
        cello_channel = generator.state.parts["strings_2"].channel
        follow_up = [n for n in sequence.notes if n.pitch == 60]
        assert follow_up and all(n.channel == cello_channel for n in follow_up)

    def test_member_continues_after_the_group_phrase(self):
        """The cello resumes where the group left off, not at t=0."""
        _, sequence = generate(self.SOURCE)
        group_note = next(n for n in sequence.notes if n.pitch == 67)
        member_note = next(n for n in sequence.notes if n.pitch == 60)
        assert member_note.start_time == pytest.approx(2.0)
        assert group_note.start_time == pytest.approx(0.0)

    def test_no_new_program_change(self):
        _, sequence = generate(self.SOURCE)
        assert len(sequence.program_changes) == 3

    def test_each_member_addressable(self):
        generator, sequence = generate(
            'violin/viola/cello "strings": g1\n'
            "strings.violin: c1\n"
            "strings.viola: d1\n"
            "strings.cello: e1"
        )
        assert generator.diagnostics == []
        assert len(generator.state.parts) == 3
        by_channel = {}
        for note in sequence.notes:
            by_channel.setdefault(note.channel, []).append(note.pitch)
        assert sorted(by_channel[0]) == [60, 67]
        assert sorted(by_channel[1]) == [62, 67]
        assert sorted(by_channel[2]) == [64, 67]

    def test_member_name_is_case_insensitive(self):
        generator, _ = generate('violin/cello "strings": g1\nstrings.CELLO: c1')
        assert generator.diagnostics == []
        assert len(generator.state.parts) == 2


class TestUnresolvedReferences:
    def test_unknown_group_warns(self):
        generator, _ = generate("piano: c\nnosuchgroup.cello: d")
        messages = [str(d) for d in generator.diagnostics]
        assert any("Unknown group member 'nosuchgroup.cello'" in m for m in messages)

    def test_unknown_group_does_not_double_warn(self):
        """The group-member message already explains the problem."""
        generator, _ = generate("piano: c\nnosuchgroup.cello: d")
        messages = [str(d) for d in generator.diagnostics]
        assert not any("Unknown instrument" in m for m in messages)

    def test_unknown_member_of_known_group_warns(self):
        generator, _ = generate('violin/viola "strings": g\nstrings.tuba: c')
        assert any("Unknown group member" in str(d) for d in generator.diagnostics)

    def test_notes_are_not_lost_when_unresolved(self):
        """An unresolvable reference must still produce sound, with a warning."""
        _, sequence = generate("piano: c\nnosuchgroup.cello: d")
        assert len(sequence.notes) == 2


class TestSerialization:
    def test_dotted_part_round_trips(self):
        source = 'violin/cello "strings": g1\nstrings.cello: c1'
        written = write_alda(parse(source))
        assert "strings.cello:" in written

        _, original = generate(source)
        _, reparsed = generate(written)
        assert [(n.pitch, n.channel) for n in original.notes] == [
            (n.pitch, n.channel) for n in reparsed.notes
        ]


class TestExampleFile:
    def test_dot_accessor_example_has_no_diagnostics(self):
        generator, _ = generate(
            (EXAMPLES / "dot_accessor.alda").read_text(encoding="utf-8")
        )
        assert generator.diagnostics == []

    def test_dot_accessor_example_uses_three_parts(self):
        generator, _ = generate(
            (EXAMPLES / "dot_accessor.alda").read_text(encoding="utf-8")
        )
        assert len(generator.state.parts) == 3

    def test_dot_accessor_example_channels(self):
        """Four notes across three channels, not four."""
        _, sequence = generate(
            (EXAMPLES / "dot_accessor.alda").read_text(encoding="utf-8")
        )
        assert len(sequence.notes) == 4
        assert sorted({n.channel for n in sequence.notes}) == [0, 1, 2]
