"""Tests for REPL module components."""

from unittest.mock import MagicMock

# Initialize vendored packages path
from aldakit import ext  # noqa: F401

from prompt_toolkit.document import Document

from aldakit.repl import (
    AldaLexer,
    AldaCompleter,
    create_key_bindings,
    ALDA_STYLE,
)


# =============================================================================
# AldaLexer Tests
# =============================================================================


class TestAldaLexer:
    """Tests for AldaLexer syntax highlighting."""

    def test_highlight_note(self):
        """Highlight single notes."""
        lexer = AldaLexer()
        doc = Document("c d e")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        # Should contain note tokens
        token_classes = [t[0] for t in tokens]
        assert "class:note" in token_classes

    def test_highlight_note_with_accidentals(self):
        """Highlight notes with accidentals."""
        lexer = AldaLexer()
        doc = Document("c+ d- e++")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        # Notes with accidentals
        note_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:note"]
        assert len(note_tokens) == 3

    def test_highlight_note_with_duration(self):
        """Highlight notes with durations."""
        lexer = AldaLexer()
        doc = Document("c4 d8. e2")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        # Should have both note and duration tokens
        token_classes = [t[0] for t in tokens]
        assert "class:note" in token_classes
        assert "class:duration" in token_classes

    def test_highlight_rest(self):
        """Highlight rest."""
        lexer = AldaLexer()
        doc = Document("r4 r8")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        rest_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:rest"]
        assert len(rest_tokens) == 2

    def test_highlight_rest_with_duration(self):
        """Highlight rest with duration."""
        lexer = AldaLexer()
        doc = Document("r4.")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        token_classes = [t[0] for t in tokens]
        assert "class:rest" in token_classes
        assert "class:duration" in token_classes

    def test_highlight_octave_set(self):
        """Highlight octave set."""
        lexer = AldaLexer()
        doc = Document("o4 o5")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        octave_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:octave"]
        assert len(octave_tokens) == 2

    def test_highlight_octave_up_down(self):
        """Highlight octave up/down."""
        lexer = AldaLexer()
        doc = Document("> < > <")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        octave_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:octave"]
        assert len(octave_tokens) == 4

    def test_highlight_instrument(self):
        """Highlight instrument declaration."""
        lexer = AldaLexer()
        doc = Document("piano: c d e")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        inst_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:instrument"]
        assert len(inst_tokens) == 1
        assert inst_tokens[0][1] == "piano:"

    def test_highlight_s_expression(self):
        """Highlight S-expressions (tempo, volume, etc.)."""
        lexer = AldaLexer()
        doc = Document("(tempo 120) c d e")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        attr_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:attribute"]
        assert len(attr_tokens) == 1
        assert "(tempo 120)" in attr_tokens[0][1]

    def test_highlight_nested_s_expression(self):
        """Highlight nested S-expressions."""
        lexer = AldaLexer()
        doc = Document("(key-sig '(a minor))")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        attr_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:attribute"]
        assert len(attr_tokens) == 1

    def test_highlight_comment(self):
        """Highlight comments."""
        lexer = AldaLexer()
        doc = Document("c d e # this is a comment")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        comment_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:comment"]
        assert len(comment_tokens) == 1
        assert "# this is a comment" in comment_tokens[0][1]

    def test_highlight_barline(self):
        """Highlight barlines."""
        lexer = AldaLexer()
        doc = Document("c d | e f")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        bar_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:barline"]
        assert len(bar_tokens) == 1

    def test_highlight_chord_marker(self):
        """Highlight chord markers."""
        lexer = AldaLexer()
        doc = Document("c/e/g")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        note_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:note"]
        # c, /, e, /, g should all be note tokens (/ is chord marker)
        assert len(note_tokens) >= 3

    def test_highlight_ms_duration(self):
        """Highlight millisecond duration."""
        lexer = AldaLexer()
        doc = Document("c500ms")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        dur_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:duration"]
        assert len(dur_tokens) == 1
        assert "500ms" in dur_tokens[0][1]

    def test_highlight_seconds_duration(self):
        """Highlight seconds duration."""
        lexer = AldaLexer()
        doc = Document("c1s")
        get_tokens = lexer.lex_document(doc)
        tokens = get_tokens(0)

        dur_tokens = [(cls, txt) for cls, txt in tokens if cls == "class:duration"]
        assert len(dur_tokens) == 1
        assert "1s" in dur_tokens[0][1]

    def test_multiline_document(self):
        """Handle multiline documents."""
        lexer = AldaLexer()
        doc = Document("piano:\nc d e")
        get_line_tokens = lexer.lex_document(doc)

        # First line has instrument
        tokens_0 = get_line_tokens(0)
        inst_tokens = [(cls, txt) for cls, txt in tokens_0 if cls == "class:instrument"]
        assert len(inst_tokens) == 1

        # Second line has notes
        tokens_1 = get_line_tokens(1)
        note_tokens = [(cls, txt) for cls, txt in tokens_1 if cls == "class:note"]
        assert len(note_tokens) == 3


# =============================================================================
# AldaCompleter Tests
# =============================================================================


class TestAldaCompleter:
    """Tests for AldaCompleter auto-completion."""

    def test_complete_instrument(self):
        """Complete instrument names."""
        completer = AldaCompleter()
        doc = Document("pia")

        completions = list(completer.get_completions(doc, None))

        # Should suggest piano
        labels = [c.text for c in completions]
        assert any("piano:" in label for label in labels)

    def test_complete_instrument_requires_min_length(self):
        """Don't complete short strings that might be notes."""
        completer = AldaCompleter()
        doc = Document("pi")  # Only 2 chars

        completions = list(completer.get_completions(doc, None))

        # Should not suggest anything (too short, might be notes)
        assert len(completions) == 0

    def test_complete_multiple_instruments(self):
        """Complete from multiple matching instruments."""
        completer = AldaCompleter()
        doc = Document("vio")

        completions = list(completer.get_completions(doc, None))

        labels = [c.text for c in completions]
        # Should include violin, viola, etc.
        assert any("violin:" in label for label in labels)

    def test_complete_attribute(self):
        """Complete attributes after opening paren."""
        completer = AldaCompleter()
        doc = Document("(tem")

        completions = list(completer.get_completions(doc, None))

        labels = [c.text for c in completions]
        assert any("tempo" in label for label in labels)

    def test_no_instrument_completion_after_colon(self):
        """Don't complete instruments if colon already present."""
        completer = AldaCompleter()
        doc = Document("piano: pia")

        completions = list(completer.get_completions(doc, None))

        # Should not suggest instruments after colon (already in part)
        assert len(completions) == 0

    def test_instruments_sorted(self):
        """Instruments list is sorted."""
        completer = AldaCompleter()
        assert completer.instruments == sorted(completer.instruments)


# =============================================================================
# Key Bindings Tests
# =============================================================================


class TestCreateKeyBindings:
    """Tests for create_key_bindings."""

    def test_creates_bindings(self):
        """Create key bindings successfully."""
        mock_backend = MagicMock()
        mock_backend.is_playing.return_value = False

        kb = create_key_bindings(mock_backend)

        assert kb is not None
        # Should have some bindings registered
        assert len(kb.bindings) > 0


# =============================================================================
# Style Tests
# =============================================================================


class TestAldaStyle:
    """Tests for ALDA_STYLE."""

    def test_style_exists(self):
        """Style is defined."""
        assert ALDA_STYLE is not None

    def test_style_has_expected_classes(self):
        """Style has expected token classes."""
        # Style should be a prompt_toolkit Style object
        # Just check it exists and is usable
        assert ALDA_STYLE is not None


# =============================================================================
# run_repl Tests (Mocked)
# =============================================================================


class TestRunRepl:
    """Tests for run_repl function with mocking."""

    def test_run_repl_audio_no_tsf(self, monkeypatch):
        """run_repl returns error if TSF not available for audio mode."""
        from aldakit import repl

        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr(repl, "LibremidiBackend", DummyBackend)
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", False)

        result = repl.run_repl(use_audio=True, soundfont="/fake/path.sf2")
        assert result == 1

    def test_run_repl_audio_file_not_found(self, monkeypatch):
        """run_repl returns error if soundfont not found."""
        from aldakit import repl

        class DummyBackend:
            def list_output_ports(self):
                return []

        class MockTsfBackend:
            def __init__(self, soundfont=None):
                raise FileNotFoundError("SoundFont not found")

        monkeypatch.setattr(repl, "LibremidiBackend", DummyBackend)
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", True)
        monkeypatch.setattr("aldakit.midi.backends.TsfBackend", MockTsfBackend)

        result = repl.run_repl(use_audio=True, soundfont="/nonexistent/path.sf2")
        assert result == 1
