"""Tests for the compose module."""

import pytest

from aldakit import Score
from aldakit.compose import (
    # Core
    Note,
    Seq,
    Repeat,
    note,
    rest,
    chord,
    seq,
    # Part
    part,
    # Attributes
    tempo,
    volume,
    vol,
    quant,
    panning,
    octave,
    octave_up,
    octave_down,
    pp,
    p,
    mp,
    mf,
    f,
    ff,
)
from aldakit.ast_nodes import (
    NoteNode,
    RestNode,
    ChordNode,
    EventSequenceNode,
    PartDeclarationNode,
    PartNode,
    LispListNode,
    OctaveSetNode,
    OctaveUpNode,
    OctaveDownNode,
    RepeatNode,
)


class TestNote:
    """Test Note class and note() factory."""

    def test_basic_note(self):
        n = note("c")
        assert n.pitch == "c"
        assert n.duration is None
        assert n.octave is None
        assert n.accidental is None

    def test_note_with_duration(self):
        n = note("c", duration=4)
        assert n.duration == 4

    def test_note_with_octave(self):
        n = note("c", octave=5)
        assert n.octave == 5

    def test_note_with_accidental(self):
        n = note("c", accidental="+")
        assert n.accidental == "+"

    def test_note_with_dots(self):
        n = note("c", duration=4, dots=1)
        assert n.dots == 1

    def test_note_with_ms(self):
        n = note("c", ms=500)
        assert n.ms == 500

    def test_note_with_seconds(self):
        n = note("c", seconds=2)
        assert n.seconds == 2

    def test_note_slurred(self):
        n = note("c", slurred=True)
        assert n.slurred is True

    def test_note_invalid_pitch(self):
        with pytest.raises(ValueError):
            note("x")

    def test_note_to_ast(self):
        n = note("c", duration=4)
        ast = n.to_ast()
        assert isinstance(ast, NoteNode)
        assert ast.letter == "c"
        assert ast.duration is not None

    def test_note_to_ast_with_accidentals(self):
        n = note("c", accidental="++")
        ast = n.to_ast()
        assert ast.accidentals == ["+", "+"]

    def test_note_to_alda(self):
        assert note("c").to_alda() == "c"
        assert note("c", duration=4).to_alda() == "c4"
        assert note("c", accidental="+").to_alda() == "c+"
        assert note("c", duration=4, dots=1).to_alda() == "c4."
        assert note("c", ms=500).to_alda() == "c500ms"
        assert note("c", seconds=2).to_alda() == "c2s"
        assert note("c", slurred=True).to_alda() == "c~"

    def test_note_midi_pitch(self):
        assert note("c").midi_pitch == 60  # C4
        assert note("c", octave=5).midi_pitch == 72  # C5
        assert note("c", accidental="+").midi_pitch == 61  # C#4
        assert note("c", accidental="-").midi_pitch == 59  # Cb4 (B3)

    def test_note_sharpen(self):
        n = note("c")
        sharp = n.sharpen()
        assert sharp.accidental == "+"
        assert n.accidental is None  # Original unchanged

    def test_note_flatten(self):
        n = note("c")
        flat = n.flatten()
        assert flat.accidental == "-"

    def test_note_transpose(self):
        n = note("c", octave=4)
        up = n.transpose(semitones=2)
        assert up.pitch == "d"
        assert up.octave == 4

    def test_note_transpose_with_accidental(self):
        n = note("c", octave=4)
        up = n.transpose(semitones=1)
        assert up.pitch == "c"
        assert up.accidental == "+"

    def test_note_with_duration_method(self):
        n = note("c")
        n2 = n.with_duration(8)
        assert n2.duration == 8
        assert n.duration is None  # Original unchanged

    def test_note_with_octave_method(self):
        n = note("c")
        n2 = n.with_octave(5)
        assert n2.octave == 5

    def test_note_slur_method(self):
        n = note("c")
        slurred = n.slur()
        assert slurred.slurred is True


class TestRest:
    """Test Rest class and rest() factory."""

    def test_basic_rest(self):
        r = rest()
        assert r.duration is None

    def test_rest_with_duration(self):
        r = rest(duration=2)
        assert r.duration == 2

    def test_rest_with_ms(self):
        r = rest(ms=1000)
        assert r.ms == 1000

    def test_rest_to_ast(self):
        r = rest(duration=4)
        ast = r.to_ast()
        assert isinstance(ast, RestNode)

    def test_rest_to_alda(self):
        assert rest().to_alda() == "r"
        assert rest(duration=2).to_alda() == "r2"
        assert rest(ms=1000).to_alda() == "r1000ms"


class TestChord:
    """Test Chord class and chord() factory."""

    def test_chord_from_strings(self):
        c = chord("c", "e", "g")
        assert len(c.notes) == 3
        assert all(isinstance(n, Note) for n in c.notes)

    def test_chord_from_notes(self):
        c = chord(note("c"), note("e"), note("g"))
        assert len(c.notes) == 3

    def test_chord_with_duration(self):
        c = chord("c", "e", "g", duration=1)
        assert c.duration == 1

    def test_chord_to_ast(self):
        c = chord("c", "e", "g")
        ast = c.to_ast()
        assert isinstance(ast, ChordNode)
        assert len(ast.notes) == 3

    def test_chord_to_alda(self):
        assert chord("c", "e", "g").to_alda() == "c/e/g"
        assert chord("c", "e", "g", duration=1).to_alda() == "c1/e/g"


class TestSeq:
    """Test Seq class and seq() factory."""

    def test_basic_seq(self):
        s = seq(note("c"), note("d"), note("e"))
        assert len(s.elements) == 3

    def test_seq_to_ast(self):
        s = seq(note("c"), note("d"))
        ast = s.to_ast()
        assert isinstance(ast, EventSequenceNode)
        assert len(ast.events) == 2

    def test_seq_to_alda(self):
        s = seq(note("c"), note("d"), note("e"))
        assert s.to_alda() == "c d e"

    def test_seq_multiply(self):
        s = seq(note("c"), note("d"))
        repeated = s * 4
        assert isinstance(repeated, Repeat)
        assert repeated.times == 4

    def test_seq_rmultiply(self):
        s = seq(note("c"), note("d"))
        repeated = 4 * s
        assert isinstance(repeated, Repeat)
        assert repeated.times == 4

    def test_seq_add(self):
        s1 = seq(note("c"), note("d"))
        s2 = seq(note("e"), note("f"))
        combined = s1 + s2
        assert len(combined.elements) == 4

    def test_seq_from_alda(self):
        s = Seq.from_alda("c d e f g")
        ast = s.to_ast()
        assert isinstance(ast, EventSequenceNode)


class TestRepeat:
    """Test Repeat class."""

    def test_repeat_note(self):
        r = note("c") * 4
        assert r.times == 4

    def test_repeat_to_ast(self):
        r = note("c") * 4
        ast = r.to_ast()
        assert isinstance(ast, RepeatNode)

    def test_repeat_to_alda(self):
        r = note("c") * 4
        assert r.to_alda() == "c*4"

    def test_repeat_seq_to_alda(self):
        s = seq(note("c"), note("d"))
        r = s * 4
        assert r.to_alda() == "[c d]*4"


class TestPart:
    """Test Part class and part() factory."""

    def test_basic_part(self):
        p = part("piano")
        assert p.instruments == ("piano",)
        assert p.alias is None

    def test_part_with_alias(self):
        p = part("violin", alias="v1")
        assert p.alias == "v1"

    def test_multi_instrument_part(self):
        p = part("violin", "viola", "cello", alias="strings")
        assert p.instruments == ("violin", "viola", "cello")

    def test_part_empty_raises(self):
        with pytest.raises(ValueError):
            part()

    def test_part_to_ast(self):
        p = part("piano")
        ast = p.to_ast()
        assert isinstance(ast, PartDeclarationNode)
        assert ast.names == ["piano"]

    def test_part_to_alda(self):
        assert part("piano").to_alda() == "piano:"
        assert part("violin", alias="v1").to_alda() == 'violin "v1":'
        assert part("violin", "viola").to_alda() == "violin/viola:"


class TestTempo:
    """Test tempo attribute."""

    def test_tempo(self):
        t = tempo(120)
        assert t.bpm == 120
        assert t.global_ is False

    def test_tempo_global(self):
        t = tempo(120, global_=True)
        assert t.global_ is True

    def test_tempo_to_ast(self):
        t = tempo(120)
        ast = t.to_ast()
        assert isinstance(ast, LispListNode)

    def test_tempo_to_alda(self):
        assert tempo(120).to_alda() == "(tempo 120)"
        assert tempo(120, global_=True).to_alda() == "(tempo! 120)"


class TestVolume:
    """Test volume attribute."""

    def test_volume(self):
        v = volume(80)
        assert v.level == 80

    def test_vol_alias(self):
        v = vol(80)
        assert v.level == 80

    def test_volume_to_alda(self):
        assert volume(80).to_alda() == "(volume 80)"


class TestOctave:
    """Test octave attributes."""

    def test_octave_set(self):
        o = octave(5)
        assert o.value == 5

    def test_octave_set_to_ast(self):
        o = octave(5)
        ast = o.to_ast()
        assert isinstance(ast, OctaveSetNode)
        assert ast.octave == 5

    def test_octave_set_to_alda(self):
        assert octave(5).to_alda() == "o5"

    def test_octave_up(self):
        o = octave_up()
        ast = o.to_ast()
        assert isinstance(ast, OctaveUpNode)
        assert o.to_alda() == ">"

    def test_octave_down(self):
        o = octave_down()
        ast = o.to_ast()
        assert isinstance(ast, OctaveDownNode)
        assert o.to_alda() == "<"


class TestDynamics:
    """Test dynamic markings."""

    def test_dynamics(self):
        assert pp().to_alda() == "(pp)"
        assert p().to_alda() == "(p)"
        assert mp().to_alda() == "(mp)"
        assert mf().to_alda() == "(mf)"
        assert f().to_alda() == "(f)"
        assert ff().to_alda() == "(ff)"

    def test_dynamics_to_ast(self):
        d = mf()
        ast = d.to_ast()
        assert isinstance(ast, LispListNode)


class TestOtherAttributes:
    """Test quant and panning."""

    def test_quant(self):
        q = quant(90)
        assert q.level == 90
        assert q.to_alda() == "(quant 90)"

    def test_panning(self):
        p = panning(50)
        assert p.level == 50
        assert p.to_alda() == "(panning 50)"


class TestScoreFromElements:
    """Test Score.from_elements() and related functionality."""

    def test_from_elements_basic(self):
        score = Score.from_elements(
            part("piano"), tempo(120), note("c", duration=4), note("d"), note("e")
        )
        assert score._mode == "elements"
        assert len(score._elements) == 5

    def test_from_elements_ast(self):
        score = Score.from_elements(part("piano"), note("c"), note("d"), note("e"))
        ast = score.ast
        assert ast is not None
        # Should have part declaration + event sequence
        assert len(ast.children) >= 1

    def test_from_elements_midi(self):
        score = Score.from_elements(part("piano"), note("c"), note("d"), note("e"))
        midi = score.midi
        assert midi is not None
        assert len(midi.notes) == 3

    def test_from_elements_duration(self):
        score = Score.from_elements(part("piano"), note("c", duration=4))
        assert score.duration > 0

    def test_from_elements_to_alda(self):
        score = Score.from_elements(part("piano"), note("c"), note("d"))
        alda = score.to_alda()
        assert "piano:" in alda
        assert "c" in alda
        assert "d" in alda

    def test_from_parts(self):
        score = Score.from_parts(part("piano"), part("violin"))
        assert len(score._elements) == 2

    def test_add_elements(self):
        score = Score.from_elements(part("piano"))
        score.add(note("c"), note("d"), note("e"))
        assert len(score._elements) == 4

    def test_add_returns_self(self):
        score = Score.from_elements(part("piano"))
        result = score.add(note("c"))
        assert result is score

    def test_add_to_source_raises(self):
        score = Score("piano: c d e")
        with pytest.raises(ValueError):
            score.add(note("f"))

    def test_with_part(self):
        score = Score.from_elements()
        score.with_part("piano")
        assert len(score._elements) == 1

    def test_with_tempo(self):
        score = Score.from_elements(part("piano"))
        score.with_tempo(120)
        assert len(score._elements) == 2

    def test_with_volume(self):
        score = Score.from_elements(part("piano"))
        score.with_volume(80)
        assert len(score._elements) == 2

    def test_method_chaining(self):
        score = (
            Score.from_elements()
            .with_part("piano")
            .with_tempo(120)
            .add(note("c"), note("d"), note("e"))
        )
        assert len(score._elements) == 5

    def test_repr_elements(self):
        score = Score.from_elements(part("piano"), note("c"))
        r = repr(score)
        assert "from_elements" in r
        assert "2 elements" in r

    def test_save_alda(self, tmp_path):
        score = Score.from_elements(part("piano"), note("c"))
        output = tmp_path / "test.alda"
        score.save(output)
        assert output.exists()
        content = output.read_text()
        assert "piano:" in content

    def test_save_midi(self, tmp_path):
        score = Score.from_elements(part("piano"), note("c"))
        output = tmp_path / "test.mid"
        score.save(output)
        assert output.exists()
        # MIDI files start with "MThd"
        with open(output, "rb") as f:
            assert f.read(4) == b"MThd"

    def test_part_generates_partnode_in_ast(self):
        """Regression: compose API must wrap parts in PartNode for MIDI generator."""
        score = Score.from_elements(part("violin"), note("c"), note("d"))
        ast = score.ast
        # Should have exactly one PartNode wrapping declaration + events
        part_nodes = [c for c in ast.children if isinstance(c, PartNode)]
        assert len(part_nodes) == 1
        assert part_nodes[0].declaration.names == ["violin"]
        assert len(part_nodes[0].events.events) == 2

    def test_part_instrument_honored_in_midi(self):
        """Regression: instrument from part() must affect MIDI program change."""
        score = Score.from_elements(part("violin"), note("c"))
        midi = score.midi
        # Violin should be MIDI program 40, not 0 (piano)
        assert any(pc.program == 40 for pc in midi.program_changes)

    def test_multiple_parts_generate_multiple_partnodes(self):
        """Regression: multiple parts should each get their own PartNode."""
        score = Score.from_elements(
            part("piano"),
            note("c"),
            part("violin"),
            note("d"),
        )
        ast = score.ast
        part_nodes = [c for c in ast.children if isinstance(c, PartNode)]
        assert len(part_nodes) == 2
        assert part_nodes[0].declaration.names == ["piano"]
        assert part_nodes[1].declaration.names == ["violin"]

    def test_to_alda_with_chord_no_crash(self):
        """Regression: to_alda() must not crash on chords (ChordNode has no duration)."""
        score = Score.from_elements(part("piano"), chord("c", "e", "g", duration=4))
        # This used to raise AttributeError: 'ChordNode' has no attribute 'duration'
        alda = score.to_alda()
        assert "c" in alda
        assert "e" in alda
        assert "g" in alda

    def test_to_alda_preserves_part_structure(self):
        """Regression: to_alda() must render PartNode with declaration + events."""
        score = Score.from_elements(
            part("violin"),
            note("c"),
            note("d"),
            part("piano"),
            note("e"),
            note("f"),
        )
        alda = score.to_alda()
        # Both parts should be present in output
        assert "violin:" in alda
        assert "piano:" in alda
        # Notes should be present
        assert "c" in alda
        assert "d" in alda
        assert "e" in alda
        assert "f" in alda


class TestIntegration:
    """Integration tests for compose -> MIDI pipeline."""

    def test_simple_melody(self, tmp_path):
        """Test a simple melody plays correctly."""
        score = Score.from_elements(
            part("piano"),
            tempo(120),
            note("c", duration=4),
            note("d", duration=4),
            note("e", duration=4),
            note("f", duration=4),
        )

        midi = score.midi
        assert len(midi.notes) == 4

        # Verify pitches
        pitches = [n.pitch for n in midi.notes]
        assert pitches == [60, 62, 64, 65]  # C, D, E, F

    def test_chord_progression(self, tmp_path):
        """Test chord generation."""
        score = Score.from_elements(
            part("piano"),
            chord("c", "e", "g", duration=1),
            chord("f", "a", "c", duration=1),
        )

        midi = score.midi
        assert len(midi.notes) == 6  # 3 notes per chord * 2 chords

    def test_with_attributes(self, tmp_path):
        """Test that attributes affect MIDI generation."""
        score = Score.from_elements(
            part("piano"), tempo(60), volume(80), note("c", duration=4)
        )

        midi = score.midi
        assert len(midi.notes) == 1
        # At tempo 60, quarter note = 1 second
        assert abs(midi.notes[0].duration - 1.0) < 0.2  # Allow for quantization

    def test_repeated_sequence(self, tmp_path):
        """Test sequence repetition."""
        score = Score.from_elements(
            part("piano"), seq(note("c", duration=8), note("d", duration=8)) * 4
        )

        midi = score.midi
        assert len(midi.notes) == 8  # 2 notes * 4 repetitions
