"""Tests for the TinySoundFont backend."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aldakit import Score
from aldakit.midi.types import MidiSequence, MidiNote, MidiProgramChange


# Check if TSF backend is available
try:
    from aldakit import _tsf
    from aldakit.midi.backends import (
        TsfBackend,
        HAS_TSF,
        find_soundfont,
        list_soundfonts,
    )
    from aldakit.midi.backends.tsf_backend import is_available

    TSF_AVAILABLE = HAS_TSF
except ImportError:
    TSF_AVAILABLE = False
    _tsf = None
    TsfBackend = None
    find_soundfont = None
    list_soundfonts = None

    def is_available():  # stub/fallback when TSF backend import fails
        return False


# Skip all tests if TSF not available
pytestmark = pytest.mark.skipif(
    not TSF_AVAILABLE, reason="TinySoundFont backend not available"
)


class TestTsfModule:
    """Test the native _tsf module."""

    def test_module_loads(self):
        assert _tsf is not None
        assert hasattr(_tsf, "TsfPlayer")

    def test_player_creation(self):
        player = _tsf.TsfPlayer()
        assert player is not None
        assert not player.is_loaded()
        assert not player.is_playing()

    def test_player_without_soundfont(self):
        player = _tsf.TsfPlayer()
        assert player.preset_count() == 0
        assert player.preset_name(0) == ""

    def test_player_load_invalid_path(self):
        player = _tsf.TsfPlayer()
        result = player.load_soundfont("/nonexistent/path.sf2")
        assert result is False
        assert not player.is_loaded()


class TestFindSoundFont:
    """Test SoundFont discovery functions."""

    def test_is_available(self):
        assert is_available() is True

    def test_find_soundfont_returns_path_or_none(self):
        result = find_soundfont()
        assert result is None or isinstance(result, Path)

    def test_find_soundfont_with_env_var(self, tmp_path):
        # Create a fake soundfont file
        fake_sf = tmp_path / "test.sf2"
        fake_sf.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal fake

        with patch.dict(os.environ, {"ALDAKIT_SOUNDFONT": str(fake_sf)}):
            result = find_soundfont()
            assert result == fake_sf

    def test_find_soundfont_env_var_missing_file(self):
        with patch.dict(os.environ, {"ALDAKIT_SOUNDFONT": "/nonexistent/sf.sf2"}):
            # Should fall back to searching other locations
            result = find_soundfont()
            # Result depends on system - may be None or found elsewhere
            assert result is None or isinstance(result, Path)

    def test_list_soundfonts_returns_list(self):
        result = list_soundfonts()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, Path)


@pytest.fixture
def soundfont_path():
    """Get a valid SoundFont path, or skip if none available."""
    sf = find_soundfont()
    if sf is None:
        pytest.skip("No SoundFont available for testing")
    return sf


class TestTsfBackend:
    """Test the TsfBackend class."""

    def test_backend_init_with_soundfont(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)
        assert backend.soundfont == soundfont_path
        assert backend.preset_count > 0

    def test_backend_init_auto_find(self, soundfont_path):
        # Should auto-find the SoundFont
        backend = TsfBackend()
        assert backend.soundfont is not None
        assert backend.preset_count > 0

    def test_backend_init_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            TsfBackend(soundfont="/nonexistent/path.sf2")

    def test_backend_preset_names(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)
        # GM SoundFonts should have piano as first preset
        name = backend.preset_name(0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_backend_set_gain(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)
        backend.set_gain(0.5)
        backend.set_gain(1.5)
        # Should not raise, backend still valid
        assert backend.preset_count > 0

    def test_backend_context_manager(self, soundfont_path):
        with TsfBackend(soundfont=soundfont_path) as backend:
            assert backend is not None
            assert not backend.is_playing()

    def test_backend_repr(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)
        repr_str = repr(backend)
        assert "TsfBackend" in repr_str
        assert "presets=" in repr_str


class TestTsfPlayback:
    """Test actual audio playback."""

    def test_play_simple_sequence(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)

        # Create a minimal sequence
        sequence = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                ),
            ],
            program_changes=[
                MidiProgramChange(program=0, time=0.0, channel=0),
            ],
        )

        backend.play(sequence)
        assert backend.is_playing()

        # Wait for completion
        backend.wait()
        assert not backend.is_playing()

    def test_play_multiple_notes(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)

        # C major chord
        sequence = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=80, start_time=0.0, duration=0.2, channel=0
                ),
                MidiNote(
                    pitch=64, velocity=80, start_time=0.0, duration=0.2, channel=0
                ),
                MidiNote(
                    pitch=67, velocity=80, start_time=0.0, duration=0.2, channel=0
                ),
            ],
            program_changes=[
                MidiProgramChange(program=0, time=0.0, channel=0),
            ],
        )

        backend.play(sequence)
        backend.wait()
        assert not backend.is_playing()

    def test_play_and_stop(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)

        # Longer sequence
        sequence = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=80, start_time=0.0, duration=2.0, channel=0
                ),
            ],
        )

        backend.play(sequence)
        time.sleep(0.1)
        assert backend.is_playing()

        backend.stop()
        time.sleep(0.05)
        assert not backend.is_playing()

    def test_current_time_advances(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)

        sequence = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0
                ),
            ],
        )

        backend.play(sequence)

        # Poll for time advancement with timeout (audio thread startup can be slow)
        t = 0.0
        for _ in range(20):  # Up to 1 second total
            time.sleep(0.05)
            t = backend.current_time()
            if t > 0.0:
                break

        assert t > 0.0, "current_time() never advanced from 0.0"

        backend.stop()

    def test_play_empty_sequence(self, soundfont_path):
        backend = TsfBackend(soundfont=soundfont_path)

        sequence = MidiSequence(notes=[], program_changes=[])

        backend.play(sequence)
        # Should complete quickly
        backend.wait()
        assert not backend.is_playing()


class TestScoreAudioBackend:
    """Test Score.play() with backend='audio'."""

    def test_score_play_audio_backend(self, soundfont_path):
        score = Score("piano: c8 d e")
        score.play(backend="audio", wait=True)
        assert score.duration > 0

    def test_score_play_audio_with_soundfont(self, soundfont_path):
        score = Score("piano: c4")
        score.play(backend="audio", soundfont=str(soundfont_path), wait=True)
        assert score.duration > 0

    def test_score_play_audio_chord(self, soundfont_path):
        score = Score("piano: c/e/g")
        score.play(backend="audio", wait=True)
        assert score.duration > 0

    def test_score_play_audio_multiple_parts(self, soundfont_path):
        score = Score("""
            piano: c4 d e
            violin: e4 f g
        """)
        score.play(backend="audio", wait=True)
        assert score.duration > 0

    def test_score_play_audio_with_tempo(self, soundfont_path):
        score = Score("""
            piano:
            (tempo 200)
            c8 d e f g a b > c
        """)
        score.play(backend="audio", wait=True)
        assert score.duration > 0

    def test_score_play_duet_example(self, soundfont_path):
        """Test playing duet.alda with TinySoundFont audio synthesis."""
        # Load and play duet.alda - tests multi-instrument audio synthesis
        duet_path = Path(__file__).parent.parent / "examples" / "duet.alda"
        source = duet_path.read_text(encoding="utf-8")

        score = Score(source)

        # Verify it has notes from both instruments
        midi = score.midi
        channels = set(note.channel for note in midi.notes)
        assert len(channels) == 2, (
            f"Expected 2 channels (violin, cello), got {channels}"
        )

        # Play with audio backend
        score.play(backend="audio", wait=True)
        assert score.duration > 0


class TestTsfBackendSave:
    """Test TsfBackend.save() method."""

    def test_save_writes_midi_file(self, soundfont_path, tmp_path):
        backend = TsfBackend(soundfont=soundfont_path)

        sequence = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0
                ),
            ],
        )

        output_path = tmp_path / "test_output.mid"
        backend.save(sequence, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a valid MIDI file (starts with MThd)
        with open(output_path, "rb") as f:
            header = f.read(4)
            assert header == b"MThd"


class TestTsfBackendNoSoundFont:
    """Test behavior when no SoundFont is available."""

    def test_init_raises_without_soundfont(self):
        # Temporarily hide all SoundFonts
        with patch(
            "aldakit.midi.backends.tsf_backend.find_soundfont", return_value=None
        ):
            with pytest.raises(FileNotFoundError) as exc_info:
                TsfBackend()
            assert "No SoundFont" in str(exc_info.value)

    def test_score_play_audio_raises_without_soundfont(self):
        with patch(
            "aldakit.midi.backends.tsf_backend.find_soundfont", return_value=None
        ):
            score = Score("piano: c")
            with pytest.raises(FileNotFoundError):
                score.play(backend="audio")
