"""Tests for non-blocking playback.

Regression cover for D6: ``Score.play(wait=False)`` created the backend inside
a ``with`` block, so the block exited immediately after ``play()`` and closed
the backend -- shutting the playback threads down and broadcasting all-notes-off
before anything could be heard. The flag documented as "don't block" actually
meant "don't play".
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aldakit import Score
from aldakit.score import PlaybackHandle


class FakeBackend:
    """Minimal stand-in that records the calls a backend receives."""

    def __init__(self):
        self.played = []
        self.stopped = False
        self.closed = False
        self.entered = False
        self.exited = False
        self.waited = False
        self._playing = False

    def play(self, sequence):
        self.played.append(sequence)
        self._playing = True
        return 0

    def is_playing(self):
        return self._playing

    def wait(self, poll_interval=0.05):
        self.waited = True
        self._playing = False

    def stop(self):
        self.stopped = True
        self._playing = False

    def close(self):
        self.closed = True
        self.stop()

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        self.close()


@pytest.fixture
def score():
    return Score("piano: c d e")


class TestBlockingPlayback:
    def test_wait_true_returns_none(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            assert score.play() is None

    def test_wait_true_closes_the_backend(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            score.play()
        assert backend.closed
        assert backend.waited

    def test_wait_true_plays_once(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            score.play()
        assert len(backend.played) == 1


class TestNonBlockingPlayback:
    def test_wait_false_returns_a_handle(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            handle = score.play(wait=False)
        assert isinstance(handle, PlaybackHandle)

    def test_wait_false_does_not_close_the_backend(self, score):
        """This is the bug: the backend used to be closed immediately."""
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            score.play(wait=False)
        assert not backend.closed
        assert not backend.stopped

    def test_wait_false_does_not_block(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            score.play(wait=False)
        assert not backend.waited

    def test_playback_is_still_active_after_returning(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            handle = score.play(wait=False)
        assert handle.is_playing()

    def test_handle_stop_closes_the_backend(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            handle = score.play(wait=False)
        handle.stop()
        assert backend.closed
        assert not handle.is_playing()

    def test_score_stop_closes_the_backend(self, score):
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            score.play(wait=False)
        score.stop()
        assert backend.closed

    def test_score_stop_is_safe_without_playback(self, score):
        score.stop()  # must not raise


class TestPlaybackHandle:
    def test_close_is_idempotent(self):
        backend = FakeBackend()
        handle = PlaybackHandle(backend)
        handle.close()
        handle.close()
        assert backend.closed

    def test_stop_after_close_is_safe(self):
        handle = PlaybackHandle(FakeBackend())
        handle.close()
        handle.stop()
        assert not handle.is_playing()

    def test_wait_after_close_returns_immediately(self):
        handle = PlaybackHandle(FakeBackend())
        handle.close()
        handle.wait()  # must not hang or raise

    def test_context_manager_closes(self):
        backend = FakeBackend()
        with PlaybackHandle(backend) as handle:
            assert not backend.closed
            assert handle is not None
        assert backend.closed

    def test_backend_property_exposes_the_backend(self):
        backend = FakeBackend()
        assert PlaybackHandle(backend).backend is backend

    def test_wait_delegates_to_the_backend(self):
        backend = FakeBackend()
        backend.play(None)
        PlaybackHandle(backend).wait()
        assert backend.waited

    def test_falls_back_to_stop_without_close(self):
        """Backends that only implement stop() are still released."""

        class StopOnly:
            def __init__(self):
                self.stopped = False

            def is_playing(self):
                return not self.stopped

            def stop(self):
                self.stopped = True

        backend = StopOnly()
        PlaybackHandle(backend).close()
        assert backend.stopped


class TestRealBackendTiming:
    """End-to-end check that audio does not stop the instant play() returns."""

    def test_playback_continues_after_play_returns(self):
        score = Score("piano: (tempo 60) c1 d1 e1")
        handle = score.play(wait=False)
        try:
            assert handle.is_playing()
            time.sleep(0.25)
            # Without the fix the backend was shut down before this point
            assert handle.is_playing()
        finally:
            handle.stop()
        assert not handle.is_playing()

    def test_blocking_playback_runs_to_completion(self):
        score = Score("piano: (tempo 600) c4")
        start = time.perf_counter()
        assert score.play() is None
        assert time.perf_counter() - start >= 0.05


class TestApiFunctions:
    def test_module_play_returns_handle(self):
        import aldakit

        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            handle = aldakit.play("piano: c d e", wait=False)
        assert isinstance(handle, PlaybackHandle)
        handle.stop()

    def test_module_play_blocking_returns_none(self):
        import aldakit

        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            assert aldakit.play("piano: c d e") is None

    def test_play_file_returns_handle(self, tmp_path):
        import aldakit

        path = tmp_path / "song.alda"
        path.write_text("piano: c d e")
        backend = FakeBackend()
        with patch.object(Score, "_make_backend", return_value=backend):
            handle = aldakit.play_file(path, wait=False)
        assert isinstance(handle, PlaybackHandle)
        handle.stop()


class TestBackendSelection:
    def test_midi_backend_is_the_default(self):
        score = Score("piano: c")
        with patch("aldakit.score.LibremidiBackend") as mock:
            mock.return_value = MagicMock()
            score._make_backend("midi", None, None)
        mock.assert_called_once()

    def test_audio_backend_requires_tsf(self):
        score = Score("piano: c")
        with patch("aldakit.midi.backends.HAS_TSF", False):
            with pytest.raises(RuntimeError, match="Audio backend not available"):
                score._make_backend("audio", None, None)
