"""Tests for MIDI generation."""

import pytest

from aldakit import parse, generate_midi
from aldakit.midi import (
    MidiSequence,
    MidiTempoChange,
    note_to_midi,
    INSTRUMENT_PROGRAMS,
)
from aldakit.midi.smf import TempoMap


class TestNoteToMidi:
    """Test note to MIDI conversion."""

    def test_middle_c(self):
        # C4 = MIDI 60
        assert note_to_midi("c", 4, []) == 60

    def test_a440(self):
        # A4 = MIDI 69 (440 Hz)
        assert note_to_midi("a", 4, []) == 69

    def test_c_sharp(self):
        assert note_to_midi("c", 4, ["+"]) == 61

    def test_b_flat(self):
        assert note_to_midi("b", 4, ["-"]) == 70

    def test_double_sharp(self):
        assert note_to_midi("c", 4, ["+", "+"]) == 62

    def test_octave_0(self):
        # C0 = MIDI 12
        assert note_to_midi("c", 0, []) == 12

    def test_octave_8(self):
        # C8 = MIDI 108
        assert note_to_midi("c", 8, []) == 108


class TestMidiGenerator:
    """Test MIDI generation from AST."""

    def test_single_note(self):
        ast = parse("c")
        seq = generate_midi(ast)
        assert len(seq.notes) == 1
        assert seq.notes[0].pitch == 60  # C4

    def test_note_with_octave(self):
        ast = parse("o5 c")
        seq = generate_midi(ast)
        assert seq.notes[0].pitch == 72  # C5

    def test_note_with_accidental(self):
        ast = parse("c+")
        seq = generate_midi(ast)
        assert seq.notes[0].pitch == 61  # C#4

    def test_octave_up(self):
        ast = parse("> c")
        seq = generate_midi(ast)
        assert seq.notes[0].pitch == 72  # C5

    def test_octave_down(self):
        ast = parse("< c")
        seq = generate_midi(ast)
        assert seq.notes[0].pitch == 48  # C3

    def test_multiple_notes(self):
        ast = parse("c d e")
        seq = generate_midi(ast)
        assert len(seq.notes) == 3
        assert seq.notes[0].pitch == 60  # C4
        assert seq.notes[1].pitch == 62  # D4
        assert seq.notes[2].pitch == 64  # E4

    def test_rest_advances_time(self):
        ast = parse("c r d")
        seq = generate_midi(ast)
        assert len(seq.notes) == 2
        # D should start later than C's end
        assert seq.notes[1].start_time > seq.notes[0].start_time + seq.notes[0].duration


class TestDurations:
    """Test duration calculations."""

    def test_quarter_note(self):
        ast = parse("c4")
        seq = generate_midi(ast)
        # At 120 BPM, quarter note = 0.5 seconds
        assert (
            abs(seq.notes[0].duration - 0.5 * 0.9) < 0.01
        )  # 0.9 is default quantization

    def test_half_note(self):
        ast = parse("c2")
        seq = generate_midi(ast)
        # At 120 BPM, half note = 1.0 seconds
        assert abs(seq.notes[0].duration - 1.0 * 0.9) < 0.01

    def test_whole_note(self):
        ast = parse("c1")
        seq = generate_midi(ast)
        # At 120 BPM, whole note = 2.0 seconds
        assert abs(seq.notes[0].duration - 2.0 * 0.9) < 0.01

    def test_dotted_note(self):
        ast = parse("c4.")
        seq = generate_midi(ast)
        # Dotted quarter = quarter + eighth = 0.75 seconds at 120 BPM
        expected = 0.75 * 0.9
        assert abs(seq.notes[0].duration - expected) < 0.01

    def test_ms_duration(self):
        ast = parse("c500ms")
        seq = generate_midi(ast)
        assert abs(seq.notes[0].duration - 0.5 * 0.9) < 0.01

    def test_seconds_duration(self):
        ast = parse("c2s")
        seq = generate_midi(ast)
        assert abs(seq.notes[0].duration - 2.0 * 0.9) < 0.01


class TestChords:
    """Test chord generation."""

    def test_simple_chord(self):
        ast = parse("c/e/g")
        seq = generate_midi(ast)
        assert len(seq.notes) == 3
        # All notes start at the same time
        assert seq.notes[0].start_time == seq.notes[1].start_time
        assert seq.notes[1].start_time == seq.notes[2].start_time
        # Check pitches (C, E, G)
        pitches = sorted(n.pitch for n in seq.notes)
        assert pitches == [60, 64, 67]

    def test_chord_with_octave(self):
        ast = parse("c/>e/g")
        seq = generate_midi(ast)
        pitches = sorted(n.pitch for n in seq.notes)
        # C4, E5, G5
        assert pitches == [60, 76, 79]


class TestTempo:
    """Test tempo handling."""

    def test_tempo_attribute(self):
        ast = parse("(tempo 60) c4")
        seq = generate_midi(ast)
        # At 60 BPM, quarter note = 1.0 seconds
        assert abs(seq.notes[0].duration - 1.0 * 0.9) < 0.01

    def test_global_tempo(self):
        ast = parse("(tempo! 240) c4")
        seq = generate_midi(ast)
        # At 240 BPM, quarter note = 0.25 seconds
        assert abs(seq.notes[0].duration - 0.25 * 0.9) < 0.01


class TestGlobalAttributes:
    """Attributes written with a trailing "!" apply to the whole score.

    Alda applies a global attribute to every part, including parts declared
    after it. Only tempo used to do that: a `(quant! 95)` or
    `(key-sig! ...)` above the first part declaration was silently dropped,
    which is how examples/across_the_sea.alda and examples/debussy_quartet.alda
    played with the wrong note lengths and the wrong key.
    """

    def test_global_quantization_reaches_later_parts(self):
        seq = generate_midi(parse("(quant! 50)\npiano: c4"))
        # 50% of a quarter note at 120 BPM
        assert abs(seq.notes[0].duration - 0.25) < 0.001

    def test_global_key_signature_reaches_later_parts(self):
        seq = generate_midi(parse("(key-sig! '(g minor))\npiano: b e"))
        assert [n.pitch for n in seq.notes] == [70, 63]  # b- and e-

    def test_global_volume_reaches_later_parts(self):
        seq = generate_midi(parse("(vol! 50)\npiano: c"))
        assert seq.notes[0].velocity == 63

    def test_global_transposition_reaches_later_parts(self):
        seq = generate_midi(parse("(transpose! 12)\npiano: c"))
        assert seq.notes[0].pitch == 72

    def test_leading_global_attribute_leaves_channel_0_free(self):
        # An attribute above the first part declaration used to create an
        # implicit part, which took channel 0 and pushed the score's first
        # instrument onto channel 1.
        seq = generate_midi(parse("(tempo! 160)\npiano: c"))
        assert seq.notes[0].channel == 0
        assert [pc.channel for pc in seq.program_changes] == [0]

    def test_global_attribute_reaches_every_part(self):
        seq = generate_midi(parse("(quant! 50)\npiano: c4\nviolin: c4"))
        assert [round(n.duration, 3) for n in seq.notes] == [0.25, 0.25]

    def test_parts_do_not_share_a_key_signature_dict(self):
        seq = generate_midi(
            parse('(key-sig! \'(g minor))\npiano: (key-sig "f+") f\nviolin: b')
        )
        assert [n.pitch for n in seq.notes] == [66, 70]  # piano f+, violin b-

    def test_local_attribute_does_not_leak_to_later_parts(self):
        seq = generate_midi(parse("piano: (quant! 50) c4\nviolin: c4"))
        # The global form applies from where it appears, so the violin
        # declared afterwards inherits it...
        assert [round(n.duration, 3) for n in seq.notes] == [0.25, 0.25]

        seq = generate_midi(parse("piano: (quant 50) c4\nviolin: c4"))
        # ...but the non-global form only touches the part it is written in.
        assert [round(n.duration, 3) for n in seq.notes] == [0.25, 0.45]


class TestDurationAttributes:
    """(set-duration), (set-note-length) and (set-duration-ms).

    These set the length a note gets when it does not spell one out. They
    parsed but did nothing until 0.2.x, which is why poly.alda and
    multi-poly.alda played with quarter-note defaults throughout.
    """

    def test_set_duration_is_in_beats(self):
        seq = generate_midi(parse("piano: (set-duration 2) c"))
        # Two beats at 120 BPM = 1.0s, quantized to 90%
        assert seq.notes[0].duration == pytest.approx(0.9)

    def test_set_duration_accepts_fractions(self):
        seq = generate_midi(parse("piano: (set-duration 2.5) c"))
        assert seq.notes[0].duration == pytest.approx(1.125)

    def test_set_note_length_is_a_note_value(self):
        seq = generate_midi(parse("piano: (set-note-length 1) c"))
        # A whole note is 4 beats = 2.0s at 120 BPM
        assert seq.notes[0].duration == pytest.approx(1.8)

    def test_set_duration_ms_converts_at_the_part_tempo(self):
        seq = generate_midi(parse("piano: (tempo 60) (set-duration-ms 2000) c"))
        # 2 seconds, quantized to 90%
        assert seq.notes[0].duration == pytest.approx(1.8)

    def test_an_explicit_note_length_still_wins(self):
        seq = generate_midi(parse("piano: (set-duration 4) c1"))
        assert seq.notes[0].duration == pytest.approx(1.8)  # the whole note

    def test_nonsense_values_are_ignored(self):
        for source in ("(set-duration 0)", "(set-duration -1)", "(set-note-length 0)"):
            seq = generate_midi(parse(f"piano: {source} c"))
            assert seq.notes[0].duration == pytest.approx(0.45)  # unchanged default

    def test_global_form_reaches_later_parts(self):
        seq = generate_midi(parse("(set-duration! 2)\npiano: c\nviolin: c"))
        assert [round(n.duration, 3) for n in seq.notes] == [0.9, 0.9]


class TestTrackVolume:
    """(track-volume) is the channel level, as opposed to note velocity."""

    def test_emits_a_channel_volume_control_change(self):
        seq = generate_midi(parse("piano: (track-volume 100) c"))
        assert [(cc.control, cc.value) for cc in seq.control_changes] == [(7, 127)]

    def test_abbreviation(self):
        seq = generate_midi(parse("piano: (track-vol 50) c"))
        assert [(cc.control, cc.value) for cc in seq.control_changes] == [(7, 63)]

    def test_does_not_change_note_velocity(self):
        seq = generate_midi(parse("piano: (track-volume 10) c"))
        assert seq.notes[0].velocity == 69  # still mf

    def test_scores_that_never_set_it_emit_no_cc7(self):
        seq = generate_midi(parse("piano: (vol 50) c"))
        assert [cc for cc in seq.control_changes if cc.control == 7] == []

    def test_is_emitted_at_the_current_time(self):
        seq = generate_midi(parse("piano: c4 (track-volume 20) d4"))
        assert seq.control_changes[0].time == pytest.approx(0.5)

    def test_global_form_reaches_later_parts(self):
        seq = generate_midi(parse("(track-volume! 80)\npiano: c\nviolin: c"))
        emitted = sorted((cc.channel, cc.value) for cc in seq.control_changes)
        assert emitted == [(0, 101), (1, 101)]


class TestMidiChannel:
    """(midi-channel N) pins a part to a channel of its choosing."""

    def test_assigns_the_requested_channel(self):
        seq = generate_midi(parse("piano: (midi-channel 5) c"))
        assert seq.notes[0].channel == 5

    def test_program_change_follows_the_part(self):
        seq = generate_midi(parse("cello: (midi-channel 4) c"))
        assert [(pc.program, pc.channel) for pc in seq.program_changes] == [(42, 4)]

    def test_switching_mid_part_moves_later_notes_only(self):
        seq = generate_midi(parse("piano: (midi-channel 2) c (midi-channel 3) d"))
        assert [n.channel for n in seq.notes] == [2, 3]
        assert sorted(pc.channel for pc in seq.program_changes) == [2, 3]

    def test_two_parts_may_share_a_channel(self):
        seq = generate_midi(
            parse("piano: (midi-channel 2) c\nguitar: (midi-channel 2) r4 d")
        )
        assert {n.channel for n in seq.notes} == {2}

    def test_drum_channel_is_refused_for_melodic_parts(self):
        from aldakit.midi.generator import MidiGenerator

        generator = MidiGenerator()
        sequence = generator.generate(parse("piano: (midi-channel 9) c"))
        assert sequence.notes[0].channel != 9
        assert any(d.code == "invalid-midi-channel" for d in generator.diagnostics)

    def test_percussion_may_ask_for_the_drum_channel(self):
        seq = generate_midi(parse("midi-percussion: (midi-channel 9) c"))
        assert seq.notes[0].channel == 9

    def test_out_of_range_channel_is_reported(self):
        from aldakit.midi.generator import MidiGenerator

        generator = MidiGenerator()
        sequence = generator.generate(parse("piano: (midi-channel 42) c"))
        assert sequence.notes[0].channel == 0
        assert any(d.code == "invalid-midi-channel" for d in generator.diagnostics)


class TestAttributeRegistry:
    """Attribute names the generator accepts, and what it does with the rest."""

    def test_documented_abbreviations_are_handled(self):
        # docs/alda-language/attributes.md lists these abbreviations.
        seq = generate_midi(parse("piano: (pan 25) c"))
        assert [(cc.control, cc.value) for cc in seq.control_changes] == [(10, 31)]

        seq = generate_midi(parse("piano: (transposition 12) c"))
        assert seq.notes[0].pitch == 72

        seq = generate_midi(parse("piano: (quantize 50) c4"))
        assert abs(seq.notes[0].duration - 0.25) < 0.001

    def test_unknown_attribute_is_reported(self):
        from aldakit.midi.generator import MidiGenerator

        generator = MidiGenerator()
        generator.generate(parse("piano: (frobnicate 3) c"))
        assert any("frobnicate" in str(d) for d in generator.diagnostics)

    def test_unknown_attribute_does_not_stop_generation(self):
        seq = generate_midi(parse("piano: (frobnicate 3) c d"))
        assert len(seq.notes) == 2


class TestVolume:
    """Test volume handling."""

    def test_volume_attribute(self):
        ast = parse("(vol 50) c")
        seq = generate_midi(ast)
        # 50% of 127 ~ 63
        assert seq.notes[0].velocity == 63

    def test_dynamic_marking(self):
        ast = parse("(ff) c")
        seq = generate_midi(ast)
        # ff = 88 velocity (official Alda spec)
        assert seq.notes[0].velocity == 88


class TestParts:
    """Test part/instrument handling."""

    def test_piano_part(self):
        ast = parse("piano: c d e")
        seq = generate_midi(ast)
        assert len(seq.notes) == 3
        assert len(seq.program_changes) >= 1
        # Piano = program 0
        assert seq.program_changes[0].program == 0

    def test_violin_part(self):
        ast = parse("violin: c d e")
        seq = generate_midi(ast)
        # Violin = program 40
        assert any(pc.program == 40 for pc in seq.program_changes)

    def test_multiple_parts(self):
        ast = parse("piano: c d e\nviolin: f g a")
        seq = generate_midi(ast)
        assert len(seq.notes) == 6
        # Should have program changes for both
        programs = [pc.program for pc in seq.program_changes]
        assert 0 in programs  # Piano
        assert 40 in programs  # Violin


class TestVariables:
    """Test variable handling."""

    def test_variable_definition_and_reference(self):
        ast = parse("theme = c d e\ntheme theme")
        seq = generate_midi(ast)
        # Definition stores but doesn't emit; 3 + 3 from two references = 6
        assert len(seq.notes) == 6


class TestRepeats:
    """Test repeat handling."""

    def test_repeat_note(self):
        ast = parse("c*4")
        seq = generate_midi(ast)
        assert len(seq.notes) == 4

    def test_repeat_sequence(self):
        ast = parse("[c d]*3")
        seq = generate_midi(ast)
        assert len(seq.notes) == 6


class TestVoices:
    """Test voice handling."""

    def test_two_voices(self):
        ast = parse("V1: c4 d4 V2: e4 f4 V0:")
        seq = generate_midi(ast)
        # Both voices should have 2 notes
        assert len(seq.notes) == 4
        # Notes should overlap in time
        times = [n.start_time for n in seq.notes]
        # First notes of both voices should start at the same time
        assert times.count(0.0) == 2


class TestCram:
    """Test cram expression handling."""

    def test_cram(self):
        ast = parse("{c d e}2")
        seq = generate_midi(ast)
        assert len(seq.notes) == 3
        # Total duration should be a half note at default tempo
        total_duration = (
            seq.notes[-1].start_time + seq.notes[-1].duration - seq.notes[0].start_time
        )
        # At 120 BPM, half note = 1.0 seconds (before quantization)
        assert total_duration < 1.1  # Allow some tolerance


class TestSequenceProperties:
    """Test MidiSequence properties."""

    def test_duration(self):
        ast = parse("c4 d4 e4")
        seq = generate_midi(ast)
        # 3 quarter notes at 120 BPM = 1.5 seconds
        assert 1.4 < seq.duration() < 1.6

    def test_empty_sequence_duration(self):
        seq = MidiSequence()
        assert seq.duration() == 0.0


class TestInstrumentMapping:
    """Test instrument name to MIDI program mapping."""

    def test_common_instruments(self):
        assert INSTRUMENT_PROGRAMS["piano"] == 0
        assert INSTRUMENT_PROGRAMS["violin"] == 40
        assert INSTRUMENT_PROGRAMS["flute"] == 73
        assert INSTRUMENT_PROGRAMS["trumpet"] == 56
        assert INSTRUMENT_PROGRAMS["cello"] == 42


class TestVariableSemantics:
    """Test variable definition semantics."""

    def test_variable_definition_does_not_emit_sound(self):
        """Regression: variable definition should only store, not emit notes."""
        ast = parse("theme = c d e")
        seq = generate_midi(ast)
        # Definition alone should not emit any notes
        assert len(seq.notes) == 0

    def test_variable_only_plays_when_referenced(self):
        """Variable content should only play on reference."""
        ast = parse("theme = c d e\ntheme")
        seq = generate_midi(ast)
        # Only one reference = 3 notes
        assert len(seq.notes) == 3


class TestTempoMap:
    """Test TempoMap for accurate MIDI timing across tempo changes."""

    def test_no_tempo_changes_uses_default(self):
        """With no tempo changes, use default 120 BPM."""
        seq = MidiSequence(ticks_per_beat=480)
        tempo_map = TempoMap(seq)
        # At 120 BPM, 1 second = 2 beats = 960 ticks
        assert tempo_map.seconds_to_ticks(1.0) == 960

    def test_single_tempo_at_start(self):
        """Single tempo change at t=0."""
        seq = MidiSequence(
            tempo_changes=[MidiTempoChange(bpm=60.0, time=0.0)],
            ticks_per_beat=480,
        )
        tempo_map = TempoMap(seq)
        # At 60 BPM, 1 second = 1 beat = 480 ticks
        assert tempo_map.seconds_to_ticks(1.0) == 480
        assert tempo_map.seconds_to_ticks(2.0) == 960

    def test_tempo_change_mid_score(self):
        """Regression: tempo changes after t=0 must integrate correctly."""
        seq = MidiSequence(
            tempo_changes=[
                MidiTempoChange(bpm=120.0, time=0.0),  # 120 BPM for first 2 sec
                MidiTempoChange(bpm=60.0, time=2.0),  # 60 BPM after
            ],
            ticks_per_beat=480,
        )
        tempo_map = TempoMap(seq)

        # t=0: tick 0
        assert tempo_map.seconds_to_ticks(0.0) == 0

        # t=1 sec at 120 BPM: 1 sec * 2 beats/sec * 480 ticks/beat = 960
        assert tempo_map.seconds_to_ticks(1.0) == 960

        # t=2 sec at 120 BPM: 2 sec * 2 beats/sec * 480 ticks/beat = 1920
        assert tempo_map.seconds_to_ticks(2.0) == 1920

        # t=3 sec: 2 sec at 120 BPM (1920 ticks) + 1 sec at 60 BPM (480 ticks) = 2400
        assert tempo_map.seconds_to_ticks(3.0) == 2400

        # t=4 sec: 1920 + 2 sec at 60 BPM (960 ticks) = 2880
        assert tempo_map.seconds_to_ticks(4.0) == 2880

    def test_multiple_tempo_changes(self):
        """Multiple tempo changes integrate correctly."""
        seq = MidiSequence(
            tempo_changes=[
                MidiTempoChange(bpm=120.0, time=0.0),
                MidiTempoChange(bpm=60.0, time=1.0),
                MidiTempoChange(bpm=240.0, time=2.0),
            ],
            ticks_per_beat=480,
        )
        tempo_map = TempoMap(seq)

        # 0-1 sec at 120 BPM: 960 ticks
        # 1-2 sec at 60 BPM: 480 ticks (total: 1440)
        # 2-3 sec at 240 BPM: 1920 ticks (total: 3360)
        assert tempo_map.seconds_to_ticks(1.0) == 960
        assert tempo_map.seconds_to_ticks(2.0) == 1440
        assert tempo_map.seconds_to_ticks(3.0) == 3360


class TestKeySignature:
    """Test key signature application."""

    def test_no_key_signature(self):
        """Without key signature, notes are natural."""
        ast = parse("piano: c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        assert pitches == [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B natural

    def test_g_major_key_signature_string_format(self):
        """G major: F#. String format "f+"."""
        ast = parse('piano: (key-sig "f+") c d e f g a b')
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # F should be sharp
        assert pitches == [60, 62, 64, 66, 67, 69, 71]  # C D E F# G A B

    def test_g_major_key_signature_quoted_list(self):
        """G major via quoted list format '(g major)'."""
        ast = parse("piano: (key-sig '(g major)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # F should be sharp
        assert pitches == [60, 62, 64, 66, 67, 69, 71]  # C D E F# G A B

    def test_d_major_key_signature(self):
        """D major: F#, C#."""
        ast = parse("piano: (key-sig '(d major)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # F and C should be sharp
        assert pitches == [61, 62, 64, 66, 67, 69, 71]  # C# D E F# G A B

    def test_f_major_key_signature(self):
        """F major: Bb."""
        ast = parse("piano: (key-sig '(f major)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # B should be flat
        assert pitches == [60, 62, 64, 65, 67, 69, 70]  # C D E F G A Bb

    def test_g_minor_key_signature(self):
        """G minor: Bb, Eb."""
        ast = parse("piano: (key-sig '(g minor)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # B and E should be flat
        assert pitches == [60, 62, 63, 65, 67, 69, 70]  # C D Eb F G A Bb

    def test_a_minor_key_signature(self):
        """A minor: no accidentals (like C major)."""
        ast = parse("piano: (key-sig '(a minor)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # All natural
        assert pitches == [60, 62, 64, 65, 67, 69, 71]  # C D E F G A B

    def test_natural_overrides_key_signature(self):
        """Explicit natural cancels key signature."""
        ast = parse("piano: (key-sig '(g major)) f f_")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # First F is sharp from key sig, second F is natural from explicit _
        assert pitches == [66, 65]  # F#, F

    def test_explicit_accidental_overrides_key_signature(self):
        """Explicit accidental on note overrides key signature."""
        ast = parse("piano: (key-sig '(g major)) f f-")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # First F is sharp from key sig, second F is flat from explicit -
        assert pitches == [66, 64]  # F#, Fb

    def test_key_signature_with_mode_ionian(self):
        """Ionian mode (same as major)."""
        ast = parse("piano: (key-sig '(c ionian)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C ionian = C major = no accidentals
        assert pitches == [60, 62, 64, 65, 67, 69, 71]

    def test_key_signature_with_mode_dorian(self):
        """D dorian has same notes as C major."""
        ast = parse("piano: (key-sig '(d dorian)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # D dorian has same accidentals as C major
        assert pitches == [60, 62, 64, 65, 67, 69, 71]

    def test_key_signature_explicit_accidentals_format(self):
        """Explicit accidentals format: '(b (flat) e (flat))'."""
        ast = parse("piano: (key-sig '(b (flat) e (flat))) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # B and E should be flat
        assert pitches == [60, 62, 63, 65, 67, 69, 70]  # C D Eb F G A Bb

    def test_key_signature_per_part(self):
        """Each part can have its own key signature."""
        ast = parse("""
            piano: (key-sig '(g major)) f
            violin: (key-sig '(f major)) b
        """)
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # Piano F is sharp (G major), violin B is flat (F major)
        assert 66 in pitches  # F#
        assert 70 in pitches  # Bb

    def test_key_signature_string_multiple_accidentals(self):
        """Multiple accidentals in string format."""
        ast = parse('piano: (key-sig "f+ c+ g+") c d e f g a b')
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # A major: C#, F#, G#
        assert pitches == [61, 62, 64, 66, 68, 69, 71]  # C# D E F# G# A B

    def test_key_signature_c_major_no_effect(self):
        """C major has no accidentals."""
        ast = parse("piano: (key-sig '(c major)) c d e f g a b")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # All natural
        assert pitches == [60, 62, 64, 65, 67, 69, 71]


class TestTranspose:
    """Test transposition functionality."""

    def test_no_transpose(self):
        """Without transpose, notes play at normal pitch."""
        ast = parse("piano: c d e")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        assert pitches == [60, 62, 64]  # C4 D4 E4

    def test_transpose_up_5(self):
        """Transpose up 5 semitones (perfect fourth)."""
        ast = parse("piano: (transpose 5) c d e")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C D E transposed up 5 -> F G A
        assert pitches == [65, 67, 69]

    def test_transpose_up_12(self):
        """Transpose up 12 semitones (one octave)."""
        ast = parse("piano: (transpose 12) c d e")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C4 D4 E4 -> C5 D5 E5
        assert pitches == [72, 74, 76]

    def test_transpose_down(self):
        """Transpose down (negative semitones)."""
        ast = parse("piano: (transpose -5) c d e")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C D E transposed down 5 -> G3 A3 B3
        assert pitches == [55, 57, 59]

    def test_transpose_reset(self):
        """Transpose 0 resets transposition."""
        ast = parse("piano: (transpose 5) c d (transpose 0) e f")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C D transposed +5, then E F with no transpose
        assert pitches == [65, 67, 64, 65]  # F G E F

    def test_transpose_change_mid_score(self):
        """Transpose can change during the score."""
        ast = parse("piano: c (transpose 5) d (transpose 7) e")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C normal, D +5, E +7
        assert pitches == [60, 67, 71]  # C4, G4, B4

    def test_transpose_with_accidentals(self):
        """Transpose applies after accidentals."""
        ast = parse("piano: (transpose 5) c c+ c-")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # C, C#, Cb all transposed up 5
        assert pitches == [65, 66, 64]  # F, F#, E

    def test_transpose_with_key_signature(self):
        """Transpose applies after key signature."""
        ast = parse("piano: (key-sig '(g major)) (transpose 5) c d e f")
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # G major has F#, then transpose up 5
        # C D E F# -> (60 62 64 66) + 5 -> 65 67 69 71
        assert pitches == [65, 67, 69, 71]

    def test_transpose_per_part(self):
        """Each part can have its own transposition."""
        ast = parse("""
            piano: (transpose 5) c
            violin: (transpose -5) c
        """)
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # Piano C +5 = 65, Violin C -5 = 55
        assert 65 in pitches  # F
        assert 55 in pitches  # G3

    def test_transpose_clamps_to_valid_range(self):
        """Transpose clamps MIDI notes to 0-127."""
        ast = parse(
            "piano: o8 (transpose 20) c"
        )  # C8 = 108, +20 = 128 -> clamped to 127
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        assert pitches == [127]

    def test_transpose_clamps_low(self):
        """Transpose clamps low notes to 0."""
        ast = parse("piano: o1 (transpose -30) c")  # C1 = 24, -30 = -6 -> clamped to 0
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        assert pitches == [0]

    def test_transpose_in_variable(self):
        """Transpose affects variable playback."""
        ast = parse("""
            motif = c d e
            piano:
                motif
                (transpose 5) motif
        """)
        seq = generate_midi(ast)
        pitches = [n.pitch for n in seq.notes]
        # First motif: C D E, second motif: F G A
        assert pitches == [60, 62, 64, 65, 67, 69]
