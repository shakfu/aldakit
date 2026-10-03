"""Tests for the comparison logic in scripts/alda_diff.py.

The script's value depends on classifying differences correctly, so these
cover the matcher with synthetic sequences. None of them needs Alda installed.
"""

from __future__ import annotations

from aldakit.midi.types import (
    MidiControlChange,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
)

from tests.helpers import load_alda_diff

alda_diff = load_alda_diff()

TICKS = 1


def seq(*notes: MidiNote, **events) -> MidiSequence:
    return MidiSequence(notes=list(notes), **events)


def note(pitch=60, start=0.0, duration=0.45, velocity=69, channel=0) -> MidiNote:
    return MidiNote(pitch, velocity, start, duration, channel)


def counts(ref: MidiSequence, ours: MidiSequence) -> dict[str, int]:
    return dict(alda_diff.compare(ref, ours, TICKS, limit=5).counts)


class TestNotes:
    def test_identical_sequences_match(self):
        s = seq(note(60, 0.0), note(62, 0.5))
        assert counts(s, s) == {}

    def test_tick_rounding_within_tolerance_matches(self):
        assert counts(seq(note(duration=0.449219)), seq(note(duration=0.45))) == {}

    def test_each_field_is_reported_separately(self):
        ref = seq(note(duration=0.45, velocity=69))
        ours = seq(note(duration=0.9, velocity=70))
        assert counts(ref, ours) == {"note duration": 1, "note velocity": 1}

    def test_channel_numbers_are_not_compared(self):
        assert counts(seq(note(channel=1)), seq(note(channel=0))) == {}

    def test_a_shifted_note_is_one_difference(self):
        assert counts(seq(note(start=0.66)), seq(note(start=0.674))) == {
            "note start": 1
        }

    def test_a_different_pitch_is_missing_plus_extra(self):
        assert counts(seq(note(71)), seq(note(70))) == {
            "note missing": 1,
            "note extra": 1,
        }

    def test_unison_pairs_by_duration_before_channel(self):
        # The two sides number channels differently; pairing by channel alone
        # would report two duration differences.
        ref = MidiSequence(
            notes=[note(duration=3.119, channel=1), note(duration=4.156, channel=2)],
            ticks_per_beat=128,
        )
        ours = seq(note(duration=4.154, channel=1), note(duration=3.115, channel=2))
        assert counts(ref, ours) == {}

    def test_tolerance_widens_at_slow_tempo(self):
        # 128 ticks per beat at 52 BPM: one tick is about 9 ms
        slow = [MidiTempoChange(52, 0.0)]
        ref = MidiSequence(notes=[note(start=1.0)], tempo_changes=slow, ticks_per_beat=128)
        ours = MidiSequence(notes=[note(start=1.008)], tempo_changes=slow)
        assert counts(ref, ours) == {}

    def test_unison_in_two_parts_pairs_by_program(self):
        programs = [MidiProgramChange(0, 0.0, 0), MidiProgramChange(42, 0.0, 1)]
        ref = seq(note(channel=0), note(channel=1), program_changes=programs)
        swapped = [MidiProgramChange(42, 0.0, 0), MidiProgramChange(0, 0.0, 1)]
        ours = seq(note(channel=1), note(channel=0), program_changes=swapped)
        assert counts(ref, ours) == {}


class TestEvents:
    def test_duplicate_tempo_is_extra(self):
        ref = seq(tempo_changes=[MidiTempoChange(120, 0.0)])
        ours = seq(tempo_changes=[MidiTempoChange(120, 0.0), MidiTempoChange(120, 0.0)])
        assert counts(ref, ours) == {"tempo extra": 1}


class TestNoteState:
    def test_program_sent_earlier_is_the_same_state(self):
        ref = seq(note(start=2.0, channel=1), program_changes=[MidiProgramChange(40, 2.0, 1)])
        ours = seq(note(start=2.0, channel=3), program_changes=[MidiProgramChange(40, 0.0, 3)])
        assert counts(ref, ours) == {}

    def test_different_program_is_reported(self):
        ref = seq(note(), program_changes=[MidiProgramChange(40, 0.0, 0)])
        ours = seq(note(), program_changes=[MidiProgramChange(41, 0.0, 0)])
        assert counts(ref, ours) == {"note program": 1}

    def test_control_value_difference_is_reported(self):
        ref = seq(note(), control_changes=[MidiControlChange(10, 64, 0.0)])
        ours = seq(note(), control_changes=[MidiControlChange(10, 63, 0.0)])
        assert counts(ref, ours) == {"note cc10": 1}

    def test_control_never_sent_is_reported(self):
        ref = seq(note(), control_changes=[MidiControlChange(11, 100, 0.0)])
        assert counts(ref, seq(note())) == {"note cc11": 1}

    def test_control_on_another_channel_does_not_apply(self):
        ref = seq(note(channel=0), control_changes=[MidiControlChange(10, 64, 0.0, 1)])
        ours = seq(note(channel=0))
        assert counts(ref, ours) == {}

    def test_change_after_the_note_starts_does_not_apply(self):
        ref = seq(note(start=0.0), control_changes=[MidiControlChange(10, 20, 0.5)])
        assert counts(ref, seq(note(start=0.0))) == {}
