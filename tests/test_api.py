"""Tests for the high-level aldakit API."""

from unittest.mock import patch, MagicMock

import pytest

import aldakit
from aldakit import Score
from aldakit.ast_nodes import RootNode
from aldakit.midi.types import MidiSequence
from aldakit.errors import AldaParseError


class TestScore:
    """Test Score class."""

    def test_create_from_string(self):
        score = Score("piano: c d e")
        assert score.source == "piano: c d e"

    def test_create_from_file(self, tmp_path):
        alda_file = tmp_path / "test.alda"
        alda_file.write_text("piano: c d e f g", encoding="utf-8")

        score = Score.from_file(alda_file)
        assert score.source == "piano: c d e f g"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Score.from_file("/nonexistent/file.alda")

    def test_ast_property(self):
        score = Score("piano: c d e")
        ast = score.ast
        assert isinstance(ast, RootNode)
        assert len(ast.children) == 1

    def test_ast_cached(self):
        score = Score("piano: c d e")
        ast1 = score.ast
        ast2 = score.ast
        assert ast1 is ast2  # Same object (cached)

    def test_midi_property(self):
        score = Score("piano: c d e")
        midi = score.midi
        assert isinstance(midi, MidiSequence)
        assert len(midi.notes) == 3

    def test_midi_cached(self):
        score = Score("piano: c d e")
        midi1 = score.midi
        midi2 = score.midi
        assert midi1 is midi2  # Same object (cached)

    def test_duration_property(self):
        score = Score("piano: c4 d4 e4")  # Three quarter notes
        duration = score.duration
        assert duration > 0

    def test_parse_error(self):
        score = Score("piano: c d e (invalid")
        with pytest.raises(AldaParseError):
            _ = score.ast

    def test_repr(self):
        score = Score("piano: c d e")
        repr_str = repr(score)
        assert "Score(" in repr_str
        assert "piano: c d e" in repr_str

    def test_repr_truncates_long_source(self):
        long_source = "piano: " + "c d e f g a b " * 10
        score = Score(long_source)
        repr_str = repr(score)
        assert "..." in repr_str
        assert len(repr_str) < 100

    @patch("aldakit.score.LibremidiBackend")
    def test_play(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.is_playing.return_value = False
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)
        mock_backend_class.return_value = mock_backend

        score = Score("piano: c d e")
        score.play()

        assert mock_backend_class.call_count == 1
        assert mock_backend_class.call_args.kwargs == {"port_name": None}
        assert mock_backend.play.call_count == 1

    @patch("aldakit.score.LibremidiBackend")
    def test_play_with_port(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.is_playing.return_value = False
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)
        mock_backend_class.return_value = mock_backend

        score = Score("piano: c d e")
        score.play(port="TestPort")

        assert mock_backend_class.call_count == 1
        assert mock_backend_class.call_args.kwargs == {"port_name": "TestPort"}
        assert mock_backend.play.call_count == 1

    @patch("aldakit.score.write_midi_file")
    @patch("aldakit.score.LibremidiBackend")
    def test_save(self, mock_backend_class, mock_write):
        score = Score("piano: c d e")
        score.save("output.mid")

        mock_backend_class.assert_not_called()
        mock_write.assert_called_once()
        from pathlib import Path

        assert mock_write.call_args[0][1] == Path("output.mid")


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch("aldakit.api.Score")
    def test_play(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.return_value = mock_score

        aldakit.play("piano: c d e")

        assert mock_score_class.call_count == 1
        assert mock_score_class.call_args.args == ("piano: c d e",)
        assert mock_score.play.call_count == 1
        assert mock_score.play.call_args.kwargs == {"port": None, "wait": True}

    @patch("aldakit.api.Score")
    def test_play_with_options(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.return_value = mock_score

        aldakit.play("piano: c d e", port="TestPort", wait=False)

        assert mock_score.play.call_count == 1
        assert mock_score.play.call_args.kwargs == {"port": "TestPort", "wait": False}

    @patch("aldakit.api.Score")
    def test_play_file(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.from_file.return_value = mock_score

        aldakit.play_file("song.alda")

        assert mock_score_class.from_file.call_count == 1
        assert mock_score.play.call_count == 1
        assert mock_score.play.call_args.kwargs == {"port": None, "wait": True}

    @patch("aldakit.api.Score")
    def test_save(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.return_value = mock_score

        aldakit.save("piano: c d e", "output.mid")

        mock_score_class.assert_called_once_with("piano: c d e")
        mock_score.save.assert_called_once_with("output.mid")

    @patch("aldakit.api.Score")
    def test_save_file(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.from_file.return_value = mock_score

        aldakit.save_file("song.alda", "output.mid")

        assert mock_score_class.from_file.call_count == 1
        assert mock_score.save.call_count == 1
        assert mock_score.save.call_args.args == ("output.mid",)

    @patch("aldakit.api.LibremidiBackend")
    def test_list_ports(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.list_output_ports.return_value = ["Port1", "Port2"]
        mock_backend_class.return_value = mock_backend

        ports = aldakit.list_ports()

        assert ports == ["Port1", "Port2"]


class TestIntegration:
    """Integration tests using real parsing and MIDI generation."""

    def test_score_end_to_end(self, tmp_path):
        """Test full workflow: create score, access properties, save to file."""
        score = Score("""
        piano:
          (tempo 120)
          c4 d e f | g a b > c
        """)

        # Access all properties
        assert score.source.strip().startswith("piano:")
        assert isinstance(score.ast, RootNode)
        assert isinstance(score.midi, MidiSequence)
        assert score.duration > 0

        # Save to file (uses real backend for file writing)
        output_file = tmp_path / "test_output.mid"
        score.save(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_save_function_creates_file(self, tmp_path):
        """Test save() function creates a valid MIDI file."""
        output_file = tmp_path / "test.mid"
        aldakit.save("piano: c d e", output_file)

        assert output_file.exists()
        # MIDI files start with "MThd"
        with open(output_file, "rb") as f:
            assert f.read(4) == b"MThd"


# =============================================================================
# Additional Score Tests
# =============================================================================


class TestScoreFromElements:
    """Tests for Score.from_elements and builder methods."""

    def test_from_elements_basic(self):
        """Create score from compose elements."""
        from aldakit.compose import part, note, tempo

        score = Score.from_elements(
            part("piano"),
            tempo(120),
            note("c", duration=4),
            note("d"),
            note("e"),
        )

        assert score.midi is not None
        assert len(score.midi.notes) == 3

    def test_from_elements_repr(self):
        """Repr for from_elements score."""
        from aldakit.compose import note

        score = Score.from_elements(note("c"), note("d"))
        repr_str = repr(score)

        assert "Score.from_elements" in repr_str
        assert "2 elements" in repr_str

    def test_from_parts(self):
        """Create score from parts."""
        from aldakit.compose import Part

        p = Part("piano")
        score = Score.from_parts(p)

        assert score.midi is not None

    def test_add_elements(self):
        """Add elements to a score."""
        from aldakit.compose import part, note

        score = Score.from_elements(part("piano"))
        score.add(note("c"), note("d"), note("e"))

        assert len(score.midi.notes) == 3

    def test_add_returns_self(self):
        """Add returns self for chaining."""
        from aldakit.compose import part, note

        score = Score.from_elements(part("piano"))
        result = score.add(note("c"))

        assert result is score

    def test_add_invalidates_cache(self):
        """Adding elements invalidates cached properties."""
        from aldakit.compose import part, note

        score = Score.from_elements(part("piano"), note("c"))
        midi1 = score.midi
        assert len(midi1.notes) == 1

        score.add(note("d"))
        midi2 = score.midi

        assert midi2 is not midi1
        assert len(midi2.notes) == 2

    def test_add_to_source_score_raises(self):
        """Cannot add elements to source-based score."""
        from aldakit.compose import note

        score = Score("piano: c d e")

        with pytest.raises(ValueError) as exc_info:
            score.add(note("f"))

        assert "from_elements" in str(exc_info.value)

    def test_with_part(self):
        """Use with_part builder method."""
        score = Score.from_elements().with_part("piano")

        assert score.midi is not None

    def test_with_tempo(self):
        """Use with_tempo builder method."""
        from aldakit.compose import part, note

        score = Score.from_elements(part("piano"), note("c")).with_tempo(100)

        # Should have tempo change
        assert len(score.midi.tempo_changes) >= 1

    def test_with_volume(self):
        """Use with_volume builder method."""
        from aldakit.compose import part, note

        score = Score.from_elements(part("piano"), note("c")).with_volume(80)

        assert score.midi is not None

    def test_chained_builders(self):
        """Chain builder methods."""
        score = Score.from_elements().with_part("piano").with_tempo(120).with_volume(80)

        assert score.midi is not None


class TestScoreToAlda:
    """Tests for Score.to_alda method."""

    def test_to_alda_source_mode(self):
        """to_alda returns original source for source-based score."""
        source = "piano: c d e"
        score = Score(source)
        assert score.to_alda() == source

    def test_to_alda_elements_mode(self):
        """to_alda generates alda from elements."""
        from aldakit.compose import part, note

        score = Score.from_elements(
            part("piano"),
            note("c", duration=4),
            note("d"),
        )
        alda = score.to_alda()

        assert "piano" in alda
        assert "c" in alda


class TestScoreFromMidi:
    """Tests for Score.from_midi_file."""

    def test_from_midi_file(self, tmp_path):
        """Create score from MIDI file."""
        # First create a MIDI file
        source_score = Score("piano: c d e f g")
        midi_path = tmp_path / "test.mid"
        source_score.save(midi_path)

        # Then import it
        imported_score = Score.from_midi_file(midi_path)

        assert imported_score.midi is not None
        assert len(imported_score.midi.notes) >= 3

    def test_from_midi_file_repr(self, tmp_path):
        """Repr for midi-imported score."""
        source_score = Score("piano: c d e")
        midi_path = tmp_path / "test.mid"
        source_score.save(midi_path)

        imported_score = Score.from_midi_file(midi_path)
        repr_str = repr(imported_score)

        assert "Score.from_midi_file" in repr_str

    def test_from_midi_file_to_alda(self, tmp_path):
        """Convert MIDI file to Alda."""
        source_score = Score("piano: c d e")
        midi_path = tmp_path / "test.mid"
        source_score.save(midi_path)

        imported_score = Score.from_midi_file(midi_path)
        alda = imported_score.to_alda()

        # Should generate some Alda code
        assert len(alda) > 0

    def test_from_file_auto_detects_midi(self, tmp_path):
        """from_file auto-detects MIDI files."""
        source_score = Score("piano: c d e")
        midi_path = tmp_path / "test.mid"
        source_score.save(midi_path)

        imported_score = Score.from_file(midi_path)

        assert imported_score.midi is not None

    def test_from_file_unknown_extension(self, tmp_path):
        """from_file handles unknown extensions as Alda."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("piano: c d e", encoding="utf-8")

        score = Score.from_file(test_file)

        assert len(score.midi.notes) == 3


class TestScoreSave:
    """Tests for Score.save method."""

    def test_save_midi(self, tmp_path):
        """Save as MIDI file."""
        score = Score("piano: c d e")
        midi_path = tmp_path / "output.mid"

        score.save(midi_path)

        assert midi_path.exists()
        with open(midi_path, "rb") as f:
            assert f.read(4) == b"MThd"

    def test_save_alda(self, tmp_path):
        """Save as Alda file."""
        source = "piano: c d e"
        score = Score(source)
        alda_path = tmp_path / "output.alda"

        score.save(alda_path)

        assert alda_path.exists()
        content = alda_path.read_text(encoding="utf-8")
        assert "piano" in content

    def test_save_unknown_extension(self, tmp_path):
        """Save with unknown extension defaults to MIDI."""
        score = Score("piano: c d e")
        output_path = tmp_path / "output.xyz"

        score.save(output_path)

        assert output_path.exists()
        with open(output_path, "rb") as f:
            assert f.read(4) == b"MThd"


class TestScorePlay:
    """Tests for Score.play with audio backend."""

    def test_play_audio_backend_not_available(self, monkeypatch):
        """play raises if audio backend not available."""
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", False)

        score = Score("piano: c d e")

        with pytest.raises(RuntimeError) as exc_info:
            score.play(backend="audio")

        assert "not available" in str(exc_info.value)


class TestAstToAlda:
    """Tests for _ast_to_alda helper function."""

    def test_ast_to_alda_notes(self):
        """Convert notes to Alda."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: c d e", "<test>")
        alda = _ast_to_alda(ast)

        assert "c" in alda
        assert "d" in alda
        assert "e" in alda

    def test_ast_to_alda_with_accidentals(self):
        """Convert notes with accidentals."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: c+ d- e", "<test>")
        alda = _ast_to_alda(ast)

        assert "c+" in alda or "c#" in alda
        assert "d-" in alda

    def test_ast_to_alda_with_durations(self):
        """Convert notes with durations."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: c4 d8 e2", "<test>")
        alda = _ast_to_alda(ast)

        assert "4" in alda
        assert "8" in alda
        assert "2" in alda

    def test_ast_to_alda_rest(self):
        """Convert rests."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: c r d", "<test>")
        alda = _ast_to_alda(ast)

        assert "r" in alda

    def test_ast_to_alda_chord(self):
        """Convert chords."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: c/e/g", "<test>")
        alda = _ast_to_alda(ast)

        assert "/" in alda

    def test_ast_to_alda_octave(self):
        """Convert octave changes."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: o5 c > d < e", "<test>")
        alda = _ast_to_alda(ast)

        assert "o5" in alda
        assert ">" in alda
        assert "<" in alda

    def test_ast_to_alda_tempo(self):
        """Convert tempo attributes."""
        from aldakit.score import _ast_to_alda
        from aldakit.parser import parse

        ast = parse("piano: (tempo 140) c d e", "<test>")
        alda = _ast_to_alda(ast)

        assert "tempo" in alda
        assert "140" in alda
