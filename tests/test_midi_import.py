"""Tests for MIDI file import functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aldakit import Score
from aldakit.ast_nodes import EventSequenceNode, LispListNode
from aldakit.midi.smf_reader import read_midi_file, MidiParseError
from aldakit.midi.midi_to_ast import (
    midi_pitch_to_note,
    seconds_to_beats,
    beats_to_duration,
    duration_value_to_beats,
    quantize_to_grid,
    midi_to_ast,
)
from aldakit.midi.types import MidiSequence, MidiNote, MidiTempoChange


# =============================================================================
# MIDI Pitch Conversion Tests
# =============================================================================


class TestMidiPitchToNote:
    """Tests for midi_pitch_to_note function."""

    def test_middle_c(self):
        """Middle C (MIDI 60) is C4."""
        letter, octave, accidentals = midi_pitch_to_note(60)
        assert letter == "c"
        assert octave == 4
        assert accidentals == []

    def test_a440(self):
        """A440 (MIDI 69) is A4."""
        letter, octave, accidentals = midi_pitch_to_note(69)
        assert letter == "a"
        assert octave == 4
        assert accidentals == []

    def test_c_sharp(self):
        """MIDI 61 is C#4."""
        letter, octave, accidentals = midi_pitch_to_note(61)
        assert letter == "c"
        assert octave == 4
        assert accidentals == ["+"]

    def test_low_c(self):
        """MIDI 24 is C1."""
        letter, octave, accidentals = midi_pitch_to_note(24)
        assert letter == "c"
        assert octave == 1
        assert accidentals == []

    def test_high_c(self):
        """MIDI 84 is C6."""
        letter, octave, accidentals = midi_pitch_to_note(84)
        assert letter == "c"
        assert octave == 6
        assert accidentals == []


# =============================================================================
# Timing Conversion Tests
# =============================================================================


class TestSecondsToBeats:
    """Tests for seconds_to_beats function."""

    def test_one_second_at_120bpm(self):
        """At 120 BPM, 1 second = 2 beats."""
        result = seconds_to_beats(1.0, 120.0)
        assert result == pytest.approx(2.0)

    def test_half_second_at_120bpm(self):
        """At 120 BPM, 0.5 seconds = 1 beat."""
        result = seconds_to_beats(0.5, 120.0)
        assert result == pytest.approx(1.0)

    def test_one_second_at_60bpm(self):
        """At 60 BPM, 1 second = 1 beat."""
        result = seconds_to_beats(1.0, 60.0)
        assert result == pytest.approx(1.0)


class TestBeatsToDuration:
    """Tests for beats_to_duration function."""

    def test_quarter_note(self):
        """1 beat = quarter note (duration 4)."""
        duration, dots = beats_to_duration(1.0)
        assert duration == 4
        assert dots == 0

    def test_half_note(self):
        """2 beats = half note (duration 2)."""
        duration, dots = beats_to_duration(2.0)
        assert duration == 2
        assert dots == 0

    def test_whole_note(self):
        """4 beats = whole note (duration 1)."""
        duration, dots = beats_to_duration(4.0)
        assert duration == 1
        assert dots == 0

    def test_eighth_note(self):
        """0.5 beats = eighth note (duration 8)."""
        duration, dots = beats_to_duration(0.5)
        assert duration == 8
        assert dots == 0

    def test_dotted_quarter(self):
        """1.5 beats = dotted quarter note."""
        duration, dots = beats_to_duration(1.5)
        assert duration == 4
        assert dots == 1

    def test_triplet_eighth_duration(self):
        """Triplet subdivision uses denominator 12."""
        duration, dots = beats_to_duration(pytest.approx(1 / 3))
        assert duration == 12
        assert dots == 0

    def test_quintuplet_duration(self):
        """Quintuplet subdivision uses denominator 20."""
        duration, dots = beats_to_duration(pytest.approx(0.2))
        assert duration == 20
        assert dots == 0


class TestDurationValueToBeats:
    """Tests for duration_value_to_beats helper."""

    def test_round_trip_quarter(self):
        beats = duration_value_to_beats(4, 0)
        assert beats == pytest.approx(1.0)

    def test_round_trip_dotted_eighth(self):
        beats = duration_value_to_beats(8, 1)
        assert beats == pytest.approx(0.75)

    def test_twelfth_note(self):
        beats = duration_value_to_beats(12, 0)
        assert beats == pytest.approx(1 / 3)


class TestQuantizeToGrid:
    """Tests for quantize_to_grid function."""

    def test_quantize_to_quarter(self):
        """Quantize to quarter note grid."""
        result = quantize_to_grid(1.1, 1.0)
        assert result == pytest.approx(1.0)

    def test_quantize_to_sixteenth(self):
        """Quantize to sixteenth note grid."""
        result = quantize_to_grid(0.3, 0.25)
        assert result == pytest.approx(0.25)

    def test_quantize_rounds_up(self):
        """Quantize rounds to nearest."""
        result = quantize_to_grid(0.6, 0.5)
        assert result == pytest.approx(0.5)

    def test_no_quantize_with_zero_grid(self):
        """Grid of 0 disables quantization."""
        result = quantize_to_grid(1.234, 0)
        assert result == pytest.approx(1.234)


# =============================================================================
# MIDI to AST Conversion Tests
# =============================================================================


class TestMidiToAst:
    """Tests for midi_to_ast function."""

    def test_empty_sequence(self):
        """Empty sequence produces minimal AST."""
        seq = MidiSequence()
        ast = midi_to_ast(seq)
        assert ast is not None

    def test_single_note(self):
        """Single note is converted correctly."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                )
            ]
        )
        ast = midi_to_ast(seq)
        assert ast is not None
        # Should have a part declaration and note

    def test_with_tempo(self):
        """Tempo is included in AST."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                )
            ],
            tempo_changes=[MidiTempoChange(bpm=140.0, time=0.0)],
        )
        ast = midi_to_ast(seq)
        assert ast is not None

    def test_tempo_changes_inserted_in_parts(self):
        """Subsequent tempo changes are emitted as part events."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60,
                    velocity=100,
                    start_time=0.0,
                    duration=0.5,
                    channel=0,
                ),
                MidiNote(
                    pitch=62,
                    velocity=100,
                    start_time=1.0,
                    duration=0.5,
                    channel=0,
                ),
            ],
            tempo_changes=[
                MidiTempoChange(bpm=110.0, time=0.0),
                MidiTempoChange(bpm=90.0, time=0.8),
            ],
        )
        ast = midi_to_ast(seq)
        event_seq = next(
            child for child in ast.children if isinstance(child, EventSequenceNode)
        )
        tempo_nodes = [
            node for node in event_seq.events if isinstance(node, LispListNode)
        ]
        assert any(
            node.elements and getattr(node.elements[0], "name", "") == "tempo"
            for node in tempo_nodes
        )

    def test_multiple_channels(self):
        """Multiple channels become separate parts."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                ),
                MidiNote(
                    pitch=64, velocity=100, start_time=0.0, duration=0.5, channel=1
                ),
            ]
        )
        ast = midi_to_ast(seq)
        assert ast is not None
        # Should have two part declarations


# =============================================================================
# MIDI File Reader Tests
# =============================================================================


class TestMidiFileReader:
    """Tests for reading MIDI files."""

    def test_read_nonexistent_file(self):
        """Reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_midi_file("/nonexistent/file.mid")

    def test_read_invalid_file(self):
        """Reading invalid file raises MidiParseError."""
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(b"not a midi file")
            f.flush()
            with pytest.raises(MidiParseError):
                read_midi_file(f.name)

    def test_read_too_small_file(self):
        """Reading file too small raises MidiParseError."""
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            f.write(b"MThd")  # Just header magic, too short
            f.flush()
            with pytest.raises(MidiParseError):
                read_midi_file(f.name)


# =============================================================================
# Score MIDI Import Integration Tests
# =============================================================================


class TestScoreMidiImport:
    """Integration tests for Score.from_midi_file."""

    def _create_test_midi(self, path: Path) -> None:
        """Create a simple test MIDI file using our writer."""
        from aldakit.midi.smf import write_midi_file

        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                ),
                MidiNote(
                    pitch=62, velocity=100, start_time=0.5, duration=0.5, channel=0
                ),
                MidiNote(
                    pitch=64, velocity=100, start_time=1.0, duration=0.5, channel=0
                ),
            ],
            tempo_changes=[MidiTempoChange(bpm=120.0, time=0.0)],
        )
        write_midi_file(seq, path)

    def test_import_midi_file(self, tmp_path):
        """Import a MIDI file and verify basic properties."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_midi_file(midi_path)
        assert score is not None
        assert score.ast is not None
        assert score.duration > 0

    def test_import_via_from_file(self, tmp_path):
        """Import via Score.from_file with .mid extension."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_file(midi_path)
        assert score is not None

    def test_to_alda_after_import(self, tmp_path):
        """Convert imported MIDI to Alda source."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_midi_file(midi_path)
        alda_source = score.to_alda()

        # Should contain some notes
        assert len(alda_source) > 0
        # Should have piano or instrument name
        assert ":" in alda_source

    def test_repr_for_midi_import(self, tmp_path):
        """Repr shows from_midi_file for imported scores."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_midi_file(midi_path)
        repr_str = repr(score)

        assert "from_midi_file" in repr_str
        assert "test.mid" in repr_str

    def test_play_imported_midi(self, tmp_path):
        """Imported MIDI can be played."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        # Mock the backend to avoid actual MIDI playback
        with patch("aldakit.score.LibremidiBackend") as mock_backend:
            mock_instance = MagicMock()
            mock_backend.return_value.__enter__.return_value = mock_instance
            mock_instance.is_playing.return_value = False

            score = Score.from_midi_file(midi_path)
            score.play()

            assert mock_instance.play.call_count == 1

    def test_save_imported_as_alda(self, tmp_path):
        """Save imported MIDI as Alda file."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_midi_file(midi_path)
        alda_path = tmp_path / "output.alda"
        score.save(alda_path)

        assert alda_path.exists()
        content = alda_path.read_text()
        assert len(content) > 0

    def test_save_imported_as_midi(self, tmp_path):
        """Re-save imported MIDI as MIDI file."""
        midi_path = tmp_path / "test.mid"
        self._create_test_midi(midi_path)

        score = Score.from_midi_file(midi_path)
        output_path = tmp_path / "output.mid"
        score.save(output_path)

        assert output_path.exists()
        # Verify it's a valid MIDI file
        content = output_path.read_bytes()
        assert content[:4] == b"MThd"


class TestRoundTrip:
    """Test round-trip conversion: compose -> MIDI -> import -> Alda."""

    def test_simple_melody_roundtrip(self, tmp_path):
        """Simple melody survives round-trip."""
        from aldakit.compose import part, note, tempo

        # Create a simple score
        original = Score.from_elements(
            part("piano"),
            tempo(120),
            note("c", duration=4),
            note("d", duration=4),
            note("e", duration=4),
        )

        # Save to MIDI
        midi_path = tmp_path / "roundtrip.mid"
        original.save(midi_path)

        # Import back
        imported = Score.from_midi_file(midi_path)

        # Should have notes
        assert imported.duration > 0

        # Should produce Alda output
        alda = imported.to_alda()
        assert "c" in alda.lower() or "d" in alda.lower() or "e" in alda.lower()
