"""Tests for per-part timing when several parts are addressed at once.

Regression cover for R9: ``_process_note`` returned the duration of whichever
part the loop visited last, and ``visit_ChordNode`` then advanced *every*
active part by that one value. The same defect was in ``visit_CramNode``,
which took its total length from the first active part. Parts in a group can
be at different tempi or carry different default durations, so one written
event is a different number of seconds long in each of them.
"""

from __future__ import annotations

import pytest

from aldakit import parse
from aldakit.midi.generator import MidiGenerator


def generate(source: str):
    return MidiGenerator().generate(parse(source))


def starts(sequence, channel: int) -> list[float]:
    return sorted(n.start_time for n in sequence.notes if n.channel == channel)


def durations(sequence, channel: int) -> list[float]:
    return [
        n.duration
        for n in sorted(
            (n for n in sequence.notes if n.channel == channel),
            key=lambda n: (n.start_time, n.pitch),
        )
    ]


class TestChordTiming:
    """A chord advances each part by its own longest note."""

    def test_divergent_tempo_parts_keep_their_own_clock(self):
        # violin: quarter = 1.0s, viola: quarter = 0.25s.
        sequence = generate(
            """
            violin: (tempo 60) o4 c4
            viola: (tempo 240) o4 c4
            violin/viola: c4/e4/g4
            violin: c4
            viola: c4
            """
        )
        # violin: note at 0, chord at 1.0, next note one violin quarter later.
        assert starts(sequence, 0) == pytest.approx([0.0, 1.0, 1.0, 1.0, 2.0])
        # viola: note at 0, chord at 0.25, next note one viola quarter later.
        assert starts(sequence, 1) == pytest.approx([0.0, 0.25, 0.25, 0.25, 0.5])

    def test_chord_length_is_the_longest_note_per_part(self):
        # The chord's longest note is a half note, which is 2.0s for the
        # violin and 0.5s for the viola.
        sequence = generate(
            """
            violin: (tempo 60) o4
            viola: (tempo 240) o4
            violin/viola: c4/e2/g4
            violin: c4
            viola: c4
            """
        )
        assert starts(sequence, 0)[-1] == pytest.approx(2.0)
        assert starts(sequence, 1)[-1] == pytest.approx(0.5)

    def test_divergent_default_duration_parts_keep_their_own_clock(self):
        # Same tempo, but set-duration differs, so an undotted chord is a
        # whole note in one part and a half note in the other.
        sequence = generate(
            """
            piano: (set-duration 4)
            harp: (set-duration 2)
            piano/harp: c/e/g
            piano: c4
            harp: c4
            """
        )
        # At 120 BPM a beat is 0.5s: 4 beats = 2.0s, 2 beats = 1.0s.
        assert starts(sequence, 0)[-1] == pytest.approx(2.0)
        assert starts(sequence, 1)[-1] == pytest.approx(1.0)

    def test_single_part_chord_is_unchanged(self):
        sequence = generate("piano: (tempo 120) c4/e4/g4 c4")
        assert starts(sequence, 0) == pytest.approx([0.0, 0.0, 0.0, 0.5])


class TestCramTiming:
    """A cram fills each part's own duration, not the first part's."""

    def test_divergent_default_duration_gives_a_polyrhythm(self):
        # multi-poly.alda in miniature: the harp crams three notes into a
        # half note while the piano crams three into a whole note.
        sequence = generate(
            """
            piano: (set-duration 4)
            harp: (set-duration 2)
            piano/harp: {c d e}
            """
        )
        assert starts(sequence, 0) == pytest.approx([0.0, 2 / 3, 4 / 3])
        assert starts(sequence, 1) == pytest.approx([0.0, 1 / 3, 2 / 3])

    def test_divergent_tempo_parts_keep_their_own_clock(self):
        sequence = generate(
            """
            violin: (tempo 60) o4
            viola: (tempo 240) o4
            violin/viola: {c d e}4
            violin: c4
            viola: c4
            """
        )
        # The cram spans one quarter note in each part: 1.0s and 0.25s.
        assert starts(sequence, 0)[-1] == pytest.approx(1.0)
        assert starts(sequence, 1)[-1] == pytest.approx(0.25)

    def test_explicit_cram_duration_is_read_per_part(self):
        sequence = generate(
            """
            violin: (tempo 60) o4
            viola: (tempo 240) o4
            violin/viola: {c d}2
            """
        )
        # A half note is 2.0s for the violin and 0.5s for the viola, so each
        # crammed note is half of that.
        assert starts(sequence, 0) == pytest.approx([0.0, 1.0])
        assert starts(sequence, 1) == pytest.approx([0.0, 0.25])

    def test_default_duration_is_restored_after_the_cram(self):
        sequence = generate(
            """
            piano: (set-duration 4)
            harp: (set-duration 2)
            piano/harp: {c d e} c
            """
        )
        # The note after the cram is a whole note for the piano (2.0s) and a
        # half note for the harp (1.0s), at the default quantization of 0.9.
        assert durations(sequence, 0)[-1] == pytest.approx(2.0 * 0.9)
        assert durations(sequence, 1)[-1] == pytest.approx(1.0 * 0.9)


class TestMultiPolyExample:
    """The multi-poly.alda example lines up only when both fixes are in."""

    def test_both_parts_end_together(self):
        sequence = generate(
            """
            piano: (set-duration 4)
            harp:  (set-duration 2) (octave 3)
            piano/harp:
              {e f g} {a b > c}
            harp:
              {d e f} {g a b}
            """
        )
        piano = [n for n in sequence.notes if n.channel == 0]
        harp = [n for n in sequence.notes if n.channel == 1]
        assert len(piano) == 6
        assert len(harp) == 12
        piano_end = max(n.start_time for n in piano)
        harp_end = max(n.start_time for n in harp)
        # Two whole notes at 120 BPM is 4.0s; the last onset is one crammed
        # note before that in each part.
        assert piano_end == pytest.approx(4.0 - 2 / 3)
        assert harp_end == pytest.approx(4.0 - 1 / 3)
