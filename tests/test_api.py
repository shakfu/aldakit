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
        alda_file.write_text("piano: c d e f g")

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

        mock_backend_class.assert_called_once_with(port_name=None)
        mock_backend.play.assert_called_once()

    @patch("aldakit.score.LibremidiBackend")
    def test_play_with_port(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend.is_playing.return_value = False
        mock_backend.__enter__ = MagicMock(return_value=mock_backend)
        mock_backend.__exit__ = MagicMock(return_value=None)
        mock_backend_class.return_value = mock_backend

        score = Score("piano: c d e")
        score.play(port="TestPort")

        mock_backend_class.assert_called_once_with(port_name="TestPort")

    @patch("aldakit.score.LibremidiBackend")
    def test_save(self, mock_backend_class):
        mock_backend = MagicMock()
        mock_backend_class.return_value = mock_backend

        score = Score("piano: c d e")
        score.save("output.mid")

        mock_backend.save.assert_called_once()
        call_args = mock_backend.save.call_args
        assert call_args[0][1] == "output.mid"


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @patch("aldakit.api.Score")
    def test_play(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.return_value = mock_score

        aldakit.play("piano: c d e")

        mock_score_class.assert_called_once_with("piano: c d e")
        mock_score.play.assert_called_once_with(port=None, wait=True)

    @patch("aldakit.api.Score")
    def test_play_with_options(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.return_value = mock_score

        aldakit.play("piano: c d e", port="TestPort", wait=False)

        mock_score.play.assert_called_once_with(port="TestPort", wait=False)

    @patch("aldakit.api.Score")
    def test_play_file(self, mock_score_class):
        mock_score = MagicMock()
        mock_score_class.from_file.return_value = mock_score

        aldakit.play_file("song.alda")

        mock_score_class.from_file.assert_called_once()
        mock_score.play.assert_called_once_with(port=None, wait=True)

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

        mock_score_class.from_file.assert_called_once()
        mock_score.save.assert_called_once_with("output.mid")

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
