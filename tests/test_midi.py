"""Tests for MIDI generation."""

from aldapy import parse, generate_midi
from aldapy.midi import (
    MidiSequence,
    note_to_midi,
    INSTRUMENT_PROGRAMS,
)


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
        assert seq.notes[0].velocity == 100


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
        # 3 notes from definition + 3 + 3 from two references = 9
        assert len(seq.notes) == 9


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
