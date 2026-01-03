"""Tests for the aldapy scanner."""

import pytest
from aldapy.scanner import Scanner
from aldapy.tokens import TokenType
from aldapy.errors import AldaScanError


class TestBasicTokens:
    """Test basic token recognition."""

    def test_note_letters(self):
        scanner = Scanner("a b c d e f g")
        tokens = scanner.scan()
        note_tokens = [t for t in tokens if t.type == TokenType.NOTE_LETTER]
        assert len(note_tokens) == 7
        assert [t.literal for t in note_tokens] == ["a", "b", "c", "d", "e", "f", "g"]

    def test_rest_letter(self):
        scanner = Scanner("r")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.REST_LETTER
        assert tokens[0].lexeme == "r"

    def test_accidentals(self):
        scanner = Scanner("+ - _")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [TokenType.SHARP, TokenType.FLAT, TokenType.NATURAL]

    def test_octave_controls(self):
        scanner = Scanner("> < o4 o0 o9")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.OCTAVE_UP,
            TokenType.OCTAVE_DOWN,
            TokenType.OCTAVE_SET,
            TokenType.OCTAVE_SET,
            TokenType.OCTAVE_SET,
        ]
        # Check octave values
        octave_tokens = [t for t in tokens if t.type == TokenType.OCTAVE_SET]
        assert [t.literal for t in octave_tokens] == [4, 0, 9]


class TestDurations:
    """Test duration token recognition."""

    def test_simple_durations(self):
        scanner = Scanner("4 8 16 1 2")
        tokens = scanner.scan()
        duration_tokens = [t for t in tokens if t.type == TokenType.NOTE_LENGTH]
        assert len(duration_tokens) == 5
        assert [t.literal for t in duration_tokens] == [4, 8, 16, 1, 2]

    def test_millisecond_duration(self):
        scanner = Scanner("500ms 1000ms")
        tokens = scanner.scan()
        ms_tokens = [t for t in tokens if t.type == TokenType.NOTE_LENGTH_MS]
        assert len(ms_tokens) == 2
        assert [t.literal for t in ms_tokens] == [500.0, 1000.0]

    def test_second_duration(self):
        scanner = Scanner("2s 0.5s")
        tokens = scanner.scan()
        s_tokens = [t for t in tokens if t.type == TokenType.NOTE_LENGTH_SECONDS]
        assert len(s_tokens) == 2
        assert [t.literal for t in s_tokens] == [2.0, 0.5]

    def test_dotted_note(self):
        scanner = Scanner("4.")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.NOTE_LENGTH
        assert tokens[0].literal == 4
        assert tokens[1].type == TokenType.DOT


class TestStructuralTokens:
    """Test structural tokens."""

    def test_barline(self):
        scanner = Scanner("|")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.BARLINE

    def test_tie(self):
        scanner = Scanner("~")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.TIE

    def test_separator(self):
        scanner = Scanner("/")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.SEPARATOR

    def test_colon(self):
        scanner = Scanner(":")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.COLON


class TestPartDeclaration:
    """Test part declaration tokens."""

    def test_simple_part(self):
        scanner = Scanner("piano:")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.NAME
        assert tokens[0].literal == "piano"
        assert tokens[1].type == TokenType.COLON

    def test_part_with_alias(self):
        scanner = Scanner('violin "v1":')
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.NAME
        assert tokens[0].literal == "violin"
        assert tokens[1].type == TokenType.ALIAS
        assert tokens[1].literal == "v1"
        assert tokens[2].type == TokenType.COLON

    def test_multi_instrument(self):
        scanner = Scanner("violin/viola:")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.NAME,
            TokenType.SEPARATOR,
            TokenType.NAME,
            TokenType.COLON,
        ]


class TestSExpressions:
    """Test S-expression (Lisp) tokenization."""

    def test_simple_sexp(self):
        scanner = Scanner("(tempo 120)")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.LEFT_PAREN,
            TokenType.SYMBOL,
            TokenType.NUMBER,
            TokenType.RIGHT_PAREN,
        ]

    def test_nested_sexp(self):
        scanner = Scanner("(foo (bar 42))")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.LEFT_PAREN,
            TokenType.SYMBOL,
            TokenType.LEFT_PAREN,
            TokenType.SYMBOL,
            TokenType.NUMBER,
            TokenType.RIGHT_PAREN,
            TokenType.RIGHT_PAREN,
        ]

    def test_sexp_with_string(self):
        scanner = Scanner('(key-sig "c major")')
        tokens = scanner.scan()
        string_token = [t for t in tokens if t.type == TokenType.STRING][0]
        assert string_token.literal == "c major"

    def test_sexp_with_negative_number(self):
        scanner = Scanner("(pan -50)")
        tokens = scanner.scan()
        num_token = [t for t in tokens if t.type == TokenType.NUMBER][0]
        assert num_token.literal == -50


class TestComments:
    """Test comment handling."""

    def test_comment_skipped(self):
        scanner = Scanner("c4 # this is a comment\nd4")
        tokens = scanner.scan()
        note_tokens = [t for t in tokens if t.type == TokenType.NOTE_LETTER]
        assert len(note_tokens) == 2
        assert [t.literal for t in note_tokens] == ["c", "d"]

    def test_comment_only(self):
        scanner = Scanner("# just a comment")
        tokens = scanner.scan()
        assert len(tokens) == 1  # Just EOF
        assert tokens[0].type == TokenType.EOF


class TestPositionTracking:
    """Test source position tracking."""

    def test_line_tracking(self):
        scanner = Scanner("c\nd\ne")
        tokens = scanner.scan()
        note_tokens = [t for t in tokens if t.type == TokenType.NOTE_LETTER]
        assert note_tokens[0].position.line == 1
        assert note_tokens[1].position.line == 2
        assert note_tokens[2].position.line == 3

    def test_column_tracking(self):
        scanner = Scanner("c d e")
        tokens = scanner.scan()
        note_tokens = [t for t in tokens if t.type == TokenType.NOTE_LETTER]
        assert note_tokens[0].position.column == 1
        assert note_tokens[1].position.column == 3
        assert note_tokens[2].position.column == 5


class TestErrors:
    """Test error handling."""

    def test_unexpected_character(self):
        scanner = Scanner("$")  # $ is not a valid Alda character
        with pytest.raises(AldaScanError) as exc_info:
            scanner.scan()
        assert "Unexpected character" in str(exc_info.value)

    def test_unterminated_string(self):
        scanner = Scanner('"unterminated')
        with pytest.raises(AldaScanError) as exc_info:
            scanner.scan()
        assert "Unterminated" in str(exc_info.value)


class TestComplexExamples:
    """Test complete Alda snippets."""

    def test_simple_melody(self):
        scanner = Scanner("o4 c4 d e f | g2 a b > c")
        tokens = scanner.scan()
        # Should produce valid tokens without errors
        assert any(t.type == TokenType.OCTAVE_SET for t in tokens)
        assert any(t.type == TokenType.BARLINE for t in tokens)
        assert any(t.type == TokenType.OCTAVE_UP for t in tokens)

    def test_chord(self):
        scanner = Scanner("c/e/g")
        tokens = scanner.scan()
        note_tokens = [t for t in tokens if t.type == TokenType.NOTE_LETTER]
        sep_tokens = [t for t in tokens if t.type == TokenType.SEPARATOR]
        assert len(note_tokens) == 3
        assert len(sep_tokens) == 2

    def test_note_with_accidentals(self):
        scanner = Scanner("c+ d- e_")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert types == [
            TokenType.NOTE_LETTER,
            TokenType.SHARP,
            TokenType.NOTE_LETTER,
            TokenType.FLAT,
            TokenType.NOTE_LETTER,
            TokenType.NATURAL,
        ]


class TestVariables:
    """Test scanning of variable-related tokens."""

    def test_equals_token(self):
        scanner = Scanner("foo = c d e")
        tokens = scanner.scan()
        types = [
            t.type
            for t in tokens
            if t.type != TokenType.EOF and t.type != TokenType.NEWLINE
        ]
        assert TokenType.EQUALS in types

    def test_variable_name(self):
        scanner = Scanner("myMotif = c d e")
        tokens = scanner.scan()
        name_tokens = [t for t in tokens if t.type == TokenType.NAME]
        assert len(name_tokens) == 1
        assert name_tokens[0].literal == "myMotif"


class TestMarkers:
    """Test scanning of marker tokens."""

    def test_marker(self):
        scanner = Scanner("%verse")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.MARKER
        assert tokens[0].literal == "verse"

    def test_at_marker(self):
        scanner = Scanner("@chorus")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.AT_MARKER
        assert tokens[0].literal == "chorus"

    def test_marker_with_hyphen(self):
        scanner = Scanner("%my-marker")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.MARKER
        assert tokens[0].literal == "my-marker"


class TestVoices:
    """Test scanning of voice markers."""

    def test_voice_marker(self):
        scanner = Scanner("V1:")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.VOICE_MARKER
        assert tokens[0].literal == 1

    def test_voice_zero(self):
        scanner = Scanner("V0:")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.VOICE_MARKER
        assert tokens[0].literal == 0

    def test_multiple_voices(self):
        scanner = Scanner("V1: c d V2: e f V0:")
        tokens = scanner.scan()
        voice_tokens = [t for t in tokens if t.type == TokenType.VOICE_MARKER]
        assert len(voice_tokens) == 3
        assert [t.literal for t in voice_tokens] == [1, 2, 0]


class TestCramBrackets:
    """Test scanning of cram and bracket tokens."""

    def test_cram_braces(self):
        scanner = Scanner("{c d e}")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.CRAM_OPEN in types
        assert TokenType.CRAM_CLOSE in types

    def test_brackets(self):
        scanner = Scanner("[c d e]")
        tokens = scanner.scan()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.EVENT_SEQ_OPEN in types
        assert TokenType.EVENT_SEQ_CLOSE in types


class TestRepeats:
    """Test scanning of repeat tokens."""

    def test_repeat(self):
        scanner = Scanner("*4")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.REPEAT
        assert tokens[0].literal == 4

    def test_repeat_with_sequence(self):
        scanner = Scanner("[c d e]*3")
        tokens = scanner.scan()
        repeat_tokens = [t for t in tokens if t.type == TokenType.REPEAT]
        assert len(repeat_tokens) == 1
        assert repeat_tokens[0].literal == 3


class TestRepetitions:
    """Test scanning of on-repetition tokens."""

    def test_simple_repetition(self):
        scanner = Scanner("'1")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.REPETITIONS
        assert tokens[0].literal == "1"

    def test_range_repetition(self):
        scanner = Scanner("'1-3")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.REPETITIONS
        assert tokens[0].literal == "1-3"

    def test_multiple_ranges(self):
        scanner = Scanner("'1-3,5,7-9")
        tokens = scanner.scan()
        assert tokens[0].type == TokenType.REPETITIONS
        assert tokens[0].literal == "1-3,5,7-9"
