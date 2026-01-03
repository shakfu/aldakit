"""Tests for the aldapy parser."""

import pytest
from aldapy.parser import parse
from aldapy.ast_nodes import (
    RootNode,
    PartNode,
    EventSequenceNode,
    NoteNode,
    RestNode,
    ChordNode,
    NoteLengthNode,
    NoteLengthMsNode,
    NoteLengthSecondsNode,
    BarlineNode,
    OctaveSetNode,
    OctaveUpNode,
    OctaveDownNode,
    LispListNode,
    LispSymbolNode,
    LispNumberNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    MarkerNode,
    AtMarkerNode,
    VoiceGroupNode,
    CramNode,
    RepeatNode,
    OnRepetitionsNode,
    BracketedSequenceNode,
)
from aldapy.errors import AldaSyntaxError


class TestBasicParsing:
    """Test basic parsing functionality."""

    def test_empty_input(self):
        ast = parse("")
        assert isinstance(ast, RootNode)
        assert len(ast.children) == 0

    def test_single_note(self):
        ast = parse("c")
        assert len(ast.children) == 1
        seq = ast.children[0]
        assert isinstance(seq, EventSequenceNode)
        assert len(seq.events) == 1
        note = seq.events[0]
        assert isinstance(note, NoteNode)
        assert note.letter == "c"

    def test_rest(self):
        ast = parse("r")
        seq = ast.children[0]
        rest = seq.events[0]
        assert isinstance(rest, RestNode)


class TestNotes:
    """Test note parsing."""

    def test_note_with_accidentals(self):
        ast = parse("c+")
        note = ast.children[0].events[0]
        assert note.letter == "c"
        assert note.accidentals == ["+"]

    def test_note_with_multiple_accidentals(self):
        ast = parse("c++")
        note = ast.children[0].events[0]
        assert note.accidentals == ["+", "+"]

    def test_note_with_flat(self):
        ast = parse("b-")
        note = ast.children[0].events[0]
        assert note.letter == "b"
        assert note.accidentals == ["-"]

    def test_note_with_natural(self):
        ast = parse("f_")
        note = ast.children[0].events[0]
        assert note.accidentals == ["_"]

    def test_note_with_duration(self):
        ast = parse("c4")
        note = ast.children[0].events[0]
        assert note.letter == "c"
        assert note.duration is not None
        assert isinstance(note.duration.components[0], NoteLengthNode)
        assert note.duration.components[0].denominator == 4

    def test_note_with_dotted_duration(self):
        ast = parse("c4.")
        note = ast.children[0].events[0]
        length = note.duration.components[0]
        assert length.denominator == 4
        assert length.dots == 1

    def test_note_with_double_dot(self):
        ast = parse("c4..")
        note = ast.children[0].events[0]
        length = note.duration.components[0]
        assert length.dots == 2

    def test_note_with_ms_duration(self):
        ast = parse("c500ms")
        note = ast.children[0].events[0]
        ms = note.duration.components[0]
        assert isinstance(ms, NoteLengthMsNode)
        assert ms.ms == 500.0

    def test_note_with_seconds_duration(self):
        ast = parse("c2s")
        note = ast.children[0].events[0]
        sec = note.duration.components[0]
        assert isinstance(sec, NoteLengthSecondsNode)
        assert sec.seconds == 2.0

    def test_slurred_note(self):
        ast = parse("c~d")
        seq = ast.children[0]
        assert seq.events[0].slurred is True
        assert seq.events[1].slurred is False


class TestChords:
    """Test chord parsing."""

    def test_simple_chord(self):
        ast = parse("c/e/g")
        chord = ast.children[0].events[0]
        assert isinstance(chord, ChordNode)
        assert len(chord.notes) == 3
        notes = [n for n in chord.notes if isinstance(n, NoteNode)]
        assert [n.letter for n in notes] == ["c", "e", "g"]

    def test_chord_with_duration(self):
        ast = parse("c1/e/g")
        chord = ast.children[0].events[0]
        first_note = chord.notes[0]
        assert first_note.duration is not None
        assert first_note.duration.components[0].denominator == 1

    def test_chord_with_octave_changes(self):
        ast = parse("c/>e/g")
        chord = ast.children[0].events[0]
        # Should have: NoteNode, OctaveUpNode, NoteNode, NoteNode
        assert isinstance(chord.notes[0], NoteNode)
        assert isinstance(chord.notes[1], OctaveUpNode)
        assert isinstance(chord.notes[2], NoteNode)


class TestOctaves:
    """Test octave control parsing."""

    def test_octave_set(self):
        ast = parse("o4 c")
        seq = ast.children[0]
        assert isinstance(seq.events[0], OctaveSetNode)
        assert seq.events[0].octave == 4

    def test_octave_up(self):
        ast = parse("> c")
        seq = ast.children[0]
        assert isinstance(seq.events[0], OctaveUpNode)

    def test_octave_down(self):
        ast = parse("< c")
        seq = ast.children[0]
        assert isinstance(seq.events[0], OctaveDownNode)


class TestParts:
    """Test part declaration parsing."""

    def test_simple_part(self):
        ast = parse("piano: c d e")
        assert len(ast.children) == 1
        part = ast.children[0]
        assert isinstance(part, PartNode)
        assert part.declaration.names == ["piano"]
        assert len(part.events.events) == 3

    def test_part_with_alias(self):
        ast = parse('violin "v1": c d e')
        part = ast.children[0]
        assert part.declaration.names == ["violin"]
        assert part.declaration.alias == "v1"

    def test_multi_instrument_part(self):
        ast = parse("violin/viola: c d e")
        part = ast.children[0]
        assert part.declaration.names == ["violin", "viola"]

    def test_multiple_parts(self):
        ast = parse("piano: c d e\nviolin: f g a")
        assert len(ast.children) == 2
        assert ast.children[0].declaration.names == ["piano"]
        assert ast.children[1].declaration.names == ["violin"]


class TestBarlines:
    """Test barline parsing."""

    def test_barline(self):
        ast = parse("c d | e f")
        seq = ast.children[0]
        events = seq.events
        assert isinstance(events[2], BarlineNode)


class TestSExpressions:
    """Test S-expression parsing."""

    def test_tempo(self):
        ast = parse("(tempo 120)")
        sexp = ast.children[0].events[0]
        assert isinstance(sexp, LispListNode)
        assert len(sexp.elements) == 2
        assert isinstance(sexp.elements[0], LispSymbolNode)
        assert sexp.elements[0].name == "tempo"
        assert isinstance(sexp.elements[1], LispNumberNode)
        assert sexp.elements[1].value == 120

    def test_volume(self):
        ast = parse("(vol 80)")
        sexp = ast.children[0].events[0]
        assert sexp.elements[0].name == "vol"
        assert sexp.elements[1].value == 80

    def test_nested_sexp(self):
        ast = parse("(foo (bar 42))")
        sexp = ast.children[0].events[0]
        assert isinstance(sexp.elements[1], LispListNode)

    def test_sexp_in_sequence(self):
        ast = parse("(tempo 120) c d e")
        seq = ast.children[0]
        assert isinstance(seq.events[0], LispListNode)
        assert isinstance(seq.events[1], NoteNode)


class TestComplexExamples:
    """Test complete Alda snippets."""

    def test_melody_with_attributes(self):
        source = """
        piano:
          (tempo 120)
          o4 c8 d e f | g4 a b > c
        """
        ast = parse(source)
        part = ast.children[0]
        assert isinstance(part, PartNode)
        events = part.events.events
        # Should have: tempo sexp, octave set, notes, barline, more notes, octave up, note
        assert isinstance(events[0], LispListNode)
        assert isinstance(events[1], OctaveSetNode)

    def test_chord_progression(self):
        source = "c1/e/g | f/a/c | g/b/d | c/e/g"
        ast = parse(source)
        seq = ast.children[0]
        chords = [e for e in seq.events if isinstance(e, ChordNode)]
        assert len(chords) == 4

    def test_tied_notes(self):
        source = "c1~1"
        ast = parse(source)
        seq = ast.children[0]
        # The tie connects durations, so we should have one note with tied duration
        note = seq.events[0]
        assert isinstance(note, NoteNode)
        assert len(note.duration.components) == 2

    def test_multiline_part(self):
        source = """
        piano:
          c d e f
          g a b > c
        """
        ast = parse(source)
        part = ast.children[0]
        assert len(part.events.events) > 0


class TestErrors:
    """Test error handling."""

    def test_name_without_colon_is_not_part(self):
        # 'piano' without colon is not a part declaration
        # It gets skipped (would be a variable reference in full implementation)
        ast = parse("piano c d e")
        # Should parse as implicit events (notes only, piano is skipped)
        seq = ast.children[0]
        notes = [e for e in seq.events if isinstance(e, NoteNode)]
        assert len(notes) == 3

    def test_unclosed_sexp(self):
        with pytest.raises(AldaSyntaxError):
            parse("(tempo 120")


class TestMultiInstrumentParts:
    """Test multi-instrument part declarations with tricky names."""

    def test_instruments_starting_with_note_letters(self):
        # cello, clarinet, flute, etc. start with note letters
        ast = parse('violin/viola/cello "strings": g1')
        part = ast.children[0]
        assert isinstance(part, PartNode)
        assert part.declaration.names == ["violin", "viola", "cello"]
        assert part.declaration.alias == "strings"

    def test_instrument_cello_alone(self):
        ast = parse("cello: c d e")
        part = ast.children[0]
        assert part.declaration.names == ["cello"]

    def test_instrument_flute_alone(self):
        ast = parse("flute: f g a")
        part = ast.children[0]
        assert part.declaration.names == ["flute"]


class TestRest:
    """Test rest parsing."""

    def test_rest_with_duration(self):
        ast = parse("r4")
        rest = ast.children[0].events[0]
        assert isinstance(rest, RestNode)
        assert rest.duration is not None
        assert rest.duration.components[0].denominator == 4

    def test_rest_with_dotted_duration(self):
        ast = parse("r4.")
        rest = ast.children[0].events[0]
        assert rest.duration.components[0].dots == 1


class TestVariables:
    """Test parsing of variables."""

    def test_variable_definition(self):
        ast = parse("myMotif = c d e")
        seq = ast.children[0]
        var_def = seq.events[0]
        assert isinstance(var_def, VariableDefinitionNode)
        assert var_def.name == "myMotif"
        assert len(var_def.events.events) == 3

    def test_variable_reference(self):
        ast = parse("myMotif")
        seq = ast.children[0]
        var_ref = seq.events[0]
        assert isinstance(var_ref, VariableReferenceNode)
        assert var_ref.name == "myMotif"

    def test_variable_definition_and_reference(self):
        ast = parse("theme = c d e\ntheme")
        seq = ast.children[0]
        assert isinstance(seq.events[0], VariableDefinitionNode)
        assert isinstance(seq.events[1], VariableReferenceNode)


class TestMarkers:
    """Test parsing of markers."""

    def test_marker(self):
        ast = parse("%verse")
        seq = ast.children[0]
        marker = seq.events[0]
        assert isinstance(marker, MarkerNode)
        assert marker.name == "verse"

    def test_at_marker(self):
        ast = parse("@chorus")
        seq = ast.children[0]
        at_marker = seq.events[0]
        assert isinstance(at_marker, AtMarkerNode)
        assert at_marker.name == "chorus"

    def test_marker_in_sequence(self):
        ast = parse("c d %bridge e f @bridge")
        seq = ast.children[0]
        events = seq.events
        assert isinstance(events[2], MarkerNode)
        assert isinstance(events[5], AtMarkerNode)


class TestVoices:
    """Test parsing of voice groups."""

    def test_single_voice(self):
        ast = parse("V1: c d e V0:")
        seq = ast.children[0]
        voice_group = seq.events[0]
        assert isinstance(voice_group, VoiceGroupNode)
        assert len(voice_group.voices) == 1
        assert voice_group.voices[0].number == 1

    def test_multiple_voices(self):
        ast = parse("V1: c d e V2: f g a V0:")
        seq = ast.children[0]
        voice_group = seq.events[0]
        assert isinstance(voice_group, VoiceGroupNode)
        assert len(voice_group.voices) == 2
        assert voice_group.voices[0].number == 1
        assert voice_group.voices[1].number == 2

    def test_voice_content(self):
        ast = parse("V1: c d e V0:")
        seq = ast.children[0]
        voice_group = seq.events[0]
        voice1 = voice_group.voices[0]
        assert len(voice1.events.events) == 3
        assert all(isinstance(e, NoteNode) for e in voice1.events.events)


class TestCram:
    """Test parsing of cram expressions."""

    def test_simple_cram(self):
        ast = parse("{c d e}")
        seq = ast.children[0]
        cram = seq.events[0]
        assert isinstance(cram, CramNode)
        assert len(cram.events.events) == 3

    def test_cram_with_duration(self):
        ast = parse("{c d e}2")
        seq = ast.children[0]
        cram = seq.events[0]
        assert isinstance(cram, CramNode)
        assert cram.duration is not None
        assert cram.duration.components[0].denominator == 2

    def test_nested_cram(self):
        ast = parse("{c {d e} f}")
        seq = ast.children[0]
        cram = seq.events[0]
        assert isinstance(cram, CramNode)
        # Should have 3 events: note, cram, note
        assert isinstance(cram.events.events[1], CramNode)


class TestBracketedSequences:
    """Test parsing of bracketed sequences."""

    def test_simple_sequence(self):
        ast = parse("[c d e]")
        seq = ast.children[0]
        bracketed = seq.events[0]
        assert isinstance(bracketed, BracketedSequenceNode)
        assert len(bracketed.events.events) == 3

    def test_sequence_with_repeat(self):
        ast = parse("[c d e]*4")
        seq = ast.children[0]
        repeat = seq.events[0]
        assert isinstance(repeat, RepeatNode)
        assert repeat.times == 4
        assert isinstance(repeat.event, BracketedSequenceNode)


class TestRepeats:
    """Test parsing of repeat expressions."""

    def test_repeat_note(self):
        ast = parse("c*4")
        seq = ast.children[0]
        repeat = seq.events[0]
        assert isinstance(repeat, RepeatNode)
        assert repeat.times == 4
        assert isinstance(repeat.event, NoteNode)

    def test_repeat_sequence(self):
        ast = parse("[c d e]*3")
        seq = ast.children[0]
        repeat = seq.events[0]
        assert isinstance(repeat, RepeatNode)
        assert repeat.times == 3


class TestOnRepetitions:
    """Test parsing of on-repetition expressions."""

    def test_simple_on_repetition(self):
        ast = parse("c'1")
        seq = ast.children[0]
        on_rep = seq.events[0]
        assert isinstance(on_rep, OnRepetitionsNode)
        assert len(on_rep.ranges) == 1
        assert on_rep.ranges[0].first == 1
        assert on_rep.ranges[0].last is None

    def test_range_on_repetition(self):
        ast = parse("c'1-3")
        seq = ast.children[0]
        on_rep = seq.events[0]
        assert isinstance(on_rep, OnRepetitionsNode)
        assert on_rep.ranges[0].first == 1
        assert on_rep.ranges[0].last == 3

    def test_multiple_ranges(self):
        ast = parse("c'1-3,5")
        seq = ast.children[0]
        on_rep = seq.events[0]
        assert isinstance(on_rep, OnRepetitionsNode)
        assert len(on_rep.ranges) == 2
        assert on_rep.ranges[0].first == 1
        assert on_rep.ranges[0].last == 3
        assert on_rep.ranges[1].first == 5


class TestComplexExpressions:
    """Test complex combinations of features."""

    def test_variable_with_cram(self):
        ast = parse("triplet = {c d e}4")
        seq = ast.children[0]
        var_def = seq.events[0]
        assert isinstance(var_def, VariableDefinitionNode)
        cram = var_def.events.events[0]
        assert isinstance(cram, CramNode)

    def test_repeated_variable_reference(self):
        ast = parse("theme = c d e\ntheme*4")
        seq = ast.children[0]
        repeat = seq.events[1]
        assert isinstance(repeat, RepeatNode)
        assert isinstance(repeat.event, VariableReferenceNode)

    def test_voices_with_markers(self):
        ast = parse("V1: %melody c d e V2: %harmony f g a V0:")
        seq = ast.children[0]
        voice_group = seq.events[0]
        # Each voice should have a marker followed by notes
        v1_events = voice_group.voices[0].events.events
        assert isinstance(v1_events[0], MarkerNode)
        assert v1_events[0].name == "melody"

    def test_bracketed_sequence_with_on_repetitions(self):
        ast = parse("[c'1 d'2 e]*4")
        seq = ast.children[0]
        repeat = seq.events[0]
        assert isinstance(repeat, RepeatNode)
        bracketed = repeat.event
        assert isinstance(bracketed, BracketedSequenceNode)
        # First event should be on-repetition
        assert isinstance(bracketed.events.events[0], OnRepetitionsNode)
