"""Tests for the TinySoundFont backend's event scheduling.

Regression cover for:

- D12: ``TsfBackend.play`` scheduled programs and notes but ignored
  ``sequence.control_changes``, so ``(panning ...)`` worked over MIDI and was
  silently a no-op in audio mode -- the backend recommended to users who have
  no external synth.
- The audio backend also never selected the General MIDI drum bank for
  channel 9, so percussion parts would have played as a melodic instrument.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aldakit.midi.backends import HAS_TSF
from aldakit.midi.types import (
    MidiControlChange,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
)

pytestmark = pytest.mark.skipif(not HAS_TSF, reason="_tsf module not built")


@pytest.fixture
def backend():
    """A TsfBackend whose native player is replaced by a recorder."""
    from aldakit.midi.backends.tsf_backend import TsfBackend

    instance = TsfBackend.__new__(TsfBackend)
    instance._player = MagicMock()
    instance._soundfont_path = None
    return instance


@pytest.fixture
def sequence():
    return MidiSequence(
        notes=[
            MidiNote(pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0),
            MidiNote(pitch=36, velocity=100, start_time=0.0, duration=0.25, channel=9),
        ],
        program_changes=[MidiProgramChange(program=42, time=0.0, channel=0)],
        control_changes=[
            MidiControlChange(control=10, value=0, time=0.0, channel=0),
            MidiControlChange(control=10, value=127, time=1.5, channel=0),
        ],
    )


class TestNativeApi:
    def test_player_exposes_schedule_control(self):
        from aldakit import _tsf

        assert hasattr(_tsf.TsfPlayer, "schedule_control")

    def test_schedule_control_is_callable(self):
        from aldakit import _tsf

        player = _tsf.TsfPlayer()
        player.schedule_control(0, 10, 64, 0.0)  # must not raise

    def test_clear_schedule_accepts_controls(self):
        from aldakit import _tsf

        player = _tsf.TsfPlayer()
        player.schedule_control(0, 10, 64, 0.0)
        player.clear_schedule()  # must not raise


class TestControlChangeScheduling:
    def test_control_changes_are_scheduled(self, backend, sequence):
        backend.play(sequence)
        assert backend._player.schedule_control.call_count == 2

    def test_control_change_arguments(self, backend, sequence):
        backend.play(sequence)
        calls = [c.args for c in backend._player.schedule_control.call_args_list]
        assert (0, 10, 0, 0.0) in calls
        assert (0, 10, 127, 1.5) in calls

    def test_panning_reaches_the_backend(self, backend):
        """An Alda (panning N) call must not vanish in audio mode."""
        from aldakit import Score

        score = Score("piano: (panning 25) c d e")
        assert score.midi.control_changes, "generator produced no pan event"
        backend.play(score.midi)
        assert backend._player.schedule_control.called

    def test_sequence_without_controls_schedules_none(self, backend):
        backend.play(
            MidiSequence(
                notes=[
                    MidiNote(
                        pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0
                    )
                ]
            )
        )
        assert backend._player.schedule_control.call_count == 0


class TestSoundFontPathExpansion:
    """The config file expanded ~ but the backend did not, so a documented
    ``TsfBackend(soundfont="~/Music/sf2/...")`` raised FileNotFoundError."""

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        from aldakit.midi.backends.tsf_backend import TsfBackend

        monkeypatch.setenv("HOME", str(tmp_path))
        sf = tmp_path / "fake.sf2"
        sf.write_bytes(b"not a real soundfont")

        # Expansion happens before the file is opened, so the path in the
        # error message proves ~ was resolved.
        with pytest.raises((FileNotFoundError, RuntimeError)) as excinfo:
            TsfBackend(soundfont="~/fake.sf2")
        assert "~" not in str(excinfo.value)

    def test_missing_file_reports_the_expanded_path(self, tmp_path, monkeypatch):
        from aldakit.midi.backends.tsf_backend import TsfBackend

        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            TsfBackend(soundfont="~/nope.sf2")
        assert str(tmp_path) in str(excinfo.value)

    def test_environment_variables_are_expanded(self, tmp_path, monkeypatch):
        from aldakit.midi.backends.tsf_backend import TsfBackend

        monkeypatch.setenv("SF_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError) as excinfo:
            TsfBackend(soundfont="$SF_DIR/nope.sf2")
        assert str(tmp_path) in str(excinfo.value)


class TestOtherEventsStillScheduled:
    def test_notes_are_scheduled(self, backend, sequence):
        backend.play(sequence)
        assert backend._player.schedule_note.call_count == 2

    def test_programs_are_scheduled(self, backend, sequence):
        backend.play(sequence)
        assert backend._player.schedule_program.call_count == 1

    def test_schedule_is_cleared_first(self, backend, sequence):
        backend.play(sequence)
        backend._player.clear_schedule.assert_called_once()

    def test_velocity_is_normalized(self, backend, sequence):
        backend.play(sequence)
        for call in backend._player.schedule_note.call_args_list:
            velocity = call.args[2]
            assert 0.0 <= velocity <= 1.0
