"""Tests for MIDI backend base class."""

import time
from pathlib import Path

import pytest

from aldakit.midi.backends.base import MidiBackend
from aldakit.midi.types import MidiSequence


# =============================================================================
# Concrete Test Backend
# =============================================================================


class ConcreteBackend(MidiBackend):
    """Concrete implementation of MidiBackend for testing."""

    def __init__(self):
        self._playing = False
        self._play_count = 0
        self._stop_count = 0
        self._concurrent = False

    def play(self, sequence: MidiSequence) -> int | None:
        self._play_count += 1
        self._playing = True
        return 1

    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        pass

    def stop(self) -> None:
        self._stop_count += 1
        self._playing = False

    def is_playing(self) -> bool:
        return self._playing

    @property
    def concurrent_mode(self) -> bool:
        return self._concurrent

    @concurrent_mode.setter
    def concurrent_mode(self, value: bool) -> None:
        self._concurrent = value


class TimedPlaybackBackend(MidiBackend):
    """Backend that simulates timed playback for wait() testing."""

    def __init__(self, play_duration: float = 0.1):
        self._playing = False
        self._play_duration = play_duration
        self._play_start = 0.0

    def play(self, sequence: MidiSequence) -> int | None:
        self._playing = True
        self._play_start = time.time()
        return 1

    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        pass

    def stop(self) -> None:
        self._playing = False

    def is_playing(self) -> bool:
        if self._playing:
            elapsed = time.time() - self._play_start
            if elapsed >= self._play_duration:
                self._playing = False
        return self._playing


# =============================================================================
# MidiBackend Abstract Class Tests
# =============================================================================


class TestMidiBackendAbstract:
    """Tests for MidiBackend abstract methods."""

    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate abstract MidiBackend."""
        with pytest.raises(TypeError):
            MidiBackend()


# =============================================================================
# Default Implementation Tests
# =============================================================================


class TestMidiBackendDefaults:
    """Tests for default implementations in MidiBackend."""

    def test_is_playing_default(self):
        """Default is_playing returns False."""

        # Test via base class method called from a subclass that doesn't override
        class MinimalBackend(MidiBackend):
            def play(self, sequence):
                return None

            def save(self, sequence, path):
                pass

            def stop(self):
                pass

        backend = MinimalBackend()
        assert backend.is_playing() is False

    def test_concurrent_mode_default(self):
        """Default concurrent_mode returns False."""

        class MinimalBackend(MidiBackend):
            def play(self, sequence):
                return None

            def save(self, sequence, path):
                pass

            def stop(self):
                pass

        backend = MinimalBackend()
        assert backend.concurrent_mode is False

    def test_concurrent_mode_setter_default(self):
        """Default concurrent_mode setter does nothing."""

        class MinimalBackend(MidiBackend):
            def play(self, sequence):
                return None

            def save(self, sequence, path):
                pass

            def stop(self):
                pass

        backend = MinimalBackend()
        # Setting should not raise, but also not change anything
        backend.concurrent_mode = True
        assert backend.concurrent_mode is False  # Still False because default getter

    def test_wait_blocks_until_done(self):
        """wait() blocks until is_playing() returns False."""
        backend = TimedPlaybackBackend(play_duration=0.05)
        seq = MidiSequence(
            notes=[], program_changes=[], control_changes=[], tempo_changes=[]
        )

        backend.play(seq)
        assert backend.is_playing() is True

        start = time.time()
        backend.wait(poll_interval=0.01)
        elapsed = time.time() - start

        assert backend.is_playing() is False
        assert elapsed >= 0.04  # Should have waited at least ~0.05s

    def test_wait_returns_immediately_if_not_playing(self):
        """wait() returns immediately if not playing."""
        backend = ConcreteBackend()

        start = time.time()
        backend.wait(poll_interval=0.01)
        elapsed = time.time() - start

        # Should return almost immediately
        assert elapsed < 0.05


# =============================================================================
# Concrete Backend Tests
# =============================================================================


class TestConcreteBackend:
    """Tests for concrete backend implementation."""

    def test_play_increments_count(self):
        """play() increments play count."""
        backend = ConcreteBackend()
        seq = MidiSequence(
            notes=[], program_changes=[], control_changes=[], tempo_changes=[]
        )

        backend.play(seq)
        assert backend._play_count == 1

        backend.play(seq)
        assert backend._play_count == 2

    def test_play_returns_slot_id(self):
        """play() returns slot ID."""
        backend = ConcreteBackend()
        seq = MidiSequence(
            notes=[], program_changes=[], control_changes=[], tempo_changes=[]
        )

        slot_id = backend.play(seq)
        assert slot_id == 1

    def test_stop_increments_count(self):
        """stop() increments stop count."""
        backend = ConcreteBackend()

        backend.stop()
        assert backend._stop_count == 1

    def test_is_playing_reflects_state(self):
        """is_playing() reflects backend state."""
        backend = ConcreteBackend()
        seq = MidiSequence(
            notes=[], program_changes=[], control_changes=[], tempo_changes=[]
        )

        assert backend.is_playing() is False

        backend.play(seq)
        assert backend.is_playing() is True

        backend.stop()
        assert backend.is_playing() is False

    def test_concurrent_mode_can_be_set(self):
        """concurrent_mode can be set and read."""
        backend = ConcreteBackend()

        assert backend.concurrent_mode is False

        backend.concurrent_mode = True
        assert backend.concurrent_mode is True

        backend.concurrent_mode = False
        assert backend.concurrent_mode is False
