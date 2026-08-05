"""Tests that the compose API preserves the octave a note declares.

Regression cover for D4: ``Note.octave`` was stored and used by
``Note.midi_pitch`` but discarded by both ``to_ast()`` and ``to_alda()``, so
``note("c", octave=5)`` generated middle C. That silently broke ``voicing()``,
``build_chord(octave=...)``, transposition across octave boundaries and the
register of every transcription.

The strongest check here is equivalence: a compose tree and the Alda source it
claims to represent must generate the same MIDI.
"""

import pytest

from aldakit import Score, parse
from aldakit.compose import (
    arpeggiate,
    build_chord,
    chord,
    cram,
    major,
    maj7,
    note,
    octave,
    octave_down,
    octave_up,
    part,
    rest,
    scale_notes,
    seq,
    tempo,
    var,
    var_ref,
    voice,
    voice_group,
    voicing,
)
from aldakit.compose.base import OctaveContext
from aldakit.midi.generator import generate_midi


def pitches(*elements):
    """MIDI pitches produced by a compose element sequence."""
    return [n.pitch for n in Score.from_elements(part("piano"), *elements).midi.notes]


def assert_matches_source(elements, source: str):
    """A compose tree must generate the same MIDI as its Alda rendering."""
    score = Score.from_elements(part("piano"), *elements)
    from_elements = [
        (round(n.start_time, 6), n.pitch, round(n.duration, 6))
        for n in score.midi.notes
    ]
    reference = generate_midi(parse(f"piano: {source}"))
    from_source = [
        (round(n.start_time, 6), n.pitch, round(n.duration, 6)) for n in reference.notes
    ]
    assert from_elements == from_source, f"compose tree != {source!r}"


class TestOctaveContext:
    def test_first_octave_is_emitted(self):
        ctx = OctaveContext()
        assert ctx.shift_to(5) == 5

    def test_repeated_octave_is_not_re_emitted(self):
        ctx = OctaveContext()
        ctx.shift_to(5)
        assert ctx.shift_to(5) is None

    def test_changed_octave_is_emitted(self):
        ctx = OctaveContext()
        ctx.shift_to(5)
        assert ctx.shift_to(6) == 6

    def test_absent_octave_emits_nothing(self):
        ctx = OctaveContext(octave=4)
        assert ctx.shift_to(None) is None

    def test_reset_forces_re_emission(self):
        ctx = OctaveContext(octave=5)
        ctx.reset(None)
        assert ctx.shift_to(5) == 5


class TestNoteOctave:
    def test_octave_reaches_midi(self):
        assert pitches(note("c", octave=5)) == [72]

    def test_default_octave_is_four(self):
        assert pitches(note("c")) == [60]

    def test_octave_persists_to_following_notes(self):
        assert pitches(note("c", octave=5), note("d"), note("e")) == [72, 74, 76]

    def test_octave_changes_are_tracked(self):
        assert pitches(
            note("c", octave=3), note("c", octave=4), note("c", octave=5)
        ) == [48, 60, 72]

    def test_octave_is_not_re_emitted_redundantly(self):
        alda = Score.from_elements(
            part("piano"), note("c", octave=5), note("d", octave=5)
        ).to_alda()
        assert alda.count("o5") == 1

    def test_to_alda_emits_octave(self):
        alda = Score.from_elements(part("piano"), note("c", octave=5)).to_alda()
        assert "o5" in alda

    def test_midi_pitch_agrees_with_generated_midi(self):
        for octave_number in range(1, 8):
            n = note("c", octave=octave_number)
            assert pitches(n) == [n.midi_pitch]

    def test_equivalent_to_source(self):
        assert_matches_source([note("c", octave=5), note("d"), note("e")], "o5 c d e")


class TestTranspose:
    def test_transpose_across_octave_boundary(self):
        """b4 up a semitone is c5, not c4."""
        assert pitches(note("b").transpose(1)) == [72]

    def test_transpose_down_across_boundary(self):
        """c4 down a semitone is b3."""
        assert pitches(note("c").transpose(-1)) == [59]

    def test_transpose_octave(self):
        assert pitches(note("c").transpose(12)) == [72]

    @pytest.mark.parametrize("semitones", range(-24, 25, 3))
    def test_transpose_matches_midi_pitch(self, semitones):
        n = note("c", octave=4).transpose(semitones)
        assert pitches(n) == [n.midi_pitch]


class TestChordOctaves:
    def test_chord_members_keep_their_octaves(self):
        c = chord(note("c", octave=3), note("e", octave=4), note("g", octave=5))
        assert sorted(pitches(c)) == [48, 64, 79]

    def test_voicing_spreads_across_octaves(self):
        """README documents voicing(major("c"), [3, 4, 5]) as C3 E4 G5."""
        assert sorted(pitches(voicing(major("c"), [3, 4, 5]))) == [48, 64, 79]

    def test_build_chord_octave_argument(self):
        assert sorted(pitches(build_chord("c", "major", octave=5))) == [72, 76, 79]

    def test_build_chord_default_octave(self):
        assert sorted(pitches(build_chord("c", "major"))) == [60, 64, 67]

    def test_chord_inversion_raises_lower_notes(self):
        root = sorted(pitches(build_chord("c", "major", octave=4)))
        first = sorted(pitches(build_chord("c", "major", octave=4, inversion=1)))
        assert first[-1] > root[-1]

    def test_arpeggio_keeps_octaves(self):
        # arpeggiate() returns a list of notes, not a Seq
        arp = arpeggiate(maj7("c", octave=4), pattern=[0, 1, 2, 3])
        assert pitches(*arp) == [60, 64, 67, 71]

    def test_arpeggio_in_higher_octave(self):
        arp = arpeggiate(maj7("c", octave=6), pattern=[0, 1, 2, 3])
        assert pitches(*arp) == [84, 88, 91, 95]

    def test_chord_to_alda_round_trips(self):
        c = chord(note("c", octave=3), note("e", octave=4), note("g", octave=5))
        score = Score.from_elements(part("piano"), c)
        reference = generate_midi(parse(score.to_alda()))
        assert sorted(n.pitch for n in reference.notes) == sorted(
            n.pitch for n in score.midi.notes
        )


class TestContainersThreadOctave:
    def test_seq_threads_octave(self):
        assert pitches(seq(note("c", octave=5), note("d"))) == [72, 74]

    def test_seq_to_alda_threads_octave(self):
        assert seq(note("c", octave=5), note("d")).to_alda() == "o5 c d"

    def test_cram_threads_octave(self):
        assert pitches(cram(note("c", octave=5), note("d"), note("e"), duration=4)) == [
            72,
            74,
            76,
        ]

    def test_repeat_threads_octave(self):
        assert pitches(seq(note("c", octave=5), note("d")) * 2) == [72, 74, 72, 74]

    def test_voice_threads_octave(self):
        score = Score.from_elements(
            part("piano"),
            voice_group(
                voice(1, note("c", octave=5), note("d")),
                voice(2, note("e", octave=3), note("f")),
            ),
        )
        assert sorted(n.pitch for n in score.midi.notes) == [52, 53, 72, 74]

    def test_voices_do_not_leak_octave_to_each_other(self):
        """Voice 2 must state its own octave, not inherit voice 1's."""
        group = voice_group(
            voice(1, note("c", octave=7)),
            voice(2, note("c", octave=2)),
        )
        alda = Score.from_elements(part("piano"), group).to_alda()
        assert "o7" in alda and "o2" in alda

    def test_variable_body_is_self_contained(self):
        """A variable used after an octave change must not depend on it."""
        score = Score.from_elements(
            part("piano"),
            var("riff", note("c", octave=3), note("d")),
            note("c", octave=6),
            var_ref("riff"),
        )
        assert sorted(n.pitch for n in score.midi.notes) == [48, 50, 84]

    def test_scale_notes_are_playable(self):
        result = pitches(scale_notes("c", "major", duration=8))
        assert result == [60, 62, 64, 65, 67, 69, 71]


class TestExplicitOctaveAttributes:
    def test_octave_attribute(self):
        assert pitches(octave(5), note("c")) == [72]

    def test_octave_up_attribute(self):
        assert pitches(octave(4), note("c"), octave_up(), note("c")) == [60, 72]

    def test_octave_down_attribute(self):
        assert pitches(octave(4), note("c"), octave_down(), note("c")) == [60, 48]

    def test_explicit_octave_then_note_octave(self):
        assert pitches(octave(5), note("c"), note("d", octave=6)) == [72, 86]

    def test_note_matching_explicit_octave_is_not_re_emitted(self):
        alda = Score.from_elements(
            part("piano"), octave(5), note("c", octave=5)
        ).to_alda()
        assert alda.count("o5") == 1


class TestPartBoundaries:
    def test_octave_resets_at_part_boundary(self):
        score = Score.from_elements(
            part("piano"),
            note("c", octave=5),
            part("violin"),
            note("c", octave=5),
        )
        # Both notes must sound at o5 even though the octave was already 5
        assert sorted(n.pitch for n in score.midi.notes) == [72, 72]

    def test_parts_stay_on_separate_channels(self):
        score = Score.from_elements(
            part("piano"), note("c", octave=5), part("violin"), note("c", octave=5)
        )
        assert len({n.channel for n in score.midi.notes}) == 2


class TestAccidentalValidation:
    """Regression cover for D9: any string was accepted as an accidental."""

    @pytest.mark.parametrize("accidental", ["+", "-", "_", "++", "--", "+_"])
    def test_valid_accidentals_accepted(self, accidental):
        assert note("c", accidental=accidental).accidental == accidental

    @pytest.mark.parametrize("accidental", ["sharp", "flat", "#", "b", "x", "+#"])
    def test_invalid_accidentals_rejected(self, accidental):
        with pytest.raises(ValueError, match="Invalid accidental"):
            note("c", accidental=accidental)

    def test_invalid_pitch_still_rejected(self):
        with pytest.raises(ValueError, match="Invalid pitch"):
            note("h")


class TestComposeSourceEquivalence:
    """A compose tree must agree with the Alda source it renders to."""

    @pytest.mark.parametrize(
        "elements,source",
        [
            ([note("c"), note("d"), note("e")], "c d e"),
            ([note("c", octave=5), note("d"), note("e")], "o5 c d e"),
            ([note("c", duration=8), note("d")], "c8 d"),
            ([note("c", accidental="+"), note("d", accidental="-")], "c+ d-"),
            ([rest(duration=4), note("c")], "r4 c"),
            ([chord("c", "e", "g", duration=2)], "c2/e/g"),
            ([tempo(90), note("c")], "(tempo 90) c"),
            ([cram(note("c"), note("d"), note("e"), duration=4)], "{c d e}4"),
            ([seq(note("c"), note("d")) * 3], "[c d]*3"),
            ([note("c", octave=6), note("d", octave=2)], "o6 c o2 d"),
            ([octave(3), note("c"), octave_up(), note("c")], "o3 c > c"),
        ],
    )
    def test_equivalent(self, elements, source):
        assert_matches_source(elements, source)

    def test_to_alda_output_reparses_identically(self):
        score = Score.from_elements(
            part("piano"),
            tempo(100),
            note("c", octave=5, duration=8),
            note("e"),
            chord(note("g", octave=4), note("b", octave=4)),
        )
        reparsed = generate_midi(parse(score.to_alda()))
        assert sorted(n.pitch for n in reparsed.notes) == sorted(
            n.pitch for n in score.midi.notes
        )
