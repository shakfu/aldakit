"""Tests for the standard-library REPL frontend and its selection flag."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aldakit import repl_core, repl_terminal
from aldakit.repl_terminal import (
    FRONTEND_ENV_VAR,
    run_terminal_repl,
    stdlib_frontend_requested,
)
from aldakit.terminal.history import History
from tests.terminal_harness import ENTER, FakeInput, FakeOutput, keys


class QuietBackend:
    """A backend that accepts everything and plays nothing."""

    concurrent_mode = True
    active_slots = 0

    def __init__(self, **kwargs) -> None:
        self.played: list[object] = []
        self.stopped = 0
        self.closed = 0

    def play(self, sequence):
        self.played.append(sequence)
        return 1

    def stop(self) -> None:
        self.stopped += 1

    def is_playing(self) -> bool:
        return False

    def close(self) -> None:
        self.closed += 1

    def list_output_ports(self) -> list[str]:
        return ["Fake"]

    def _ensure_port_open(self) -> None:
        pass


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """A REPL whose backend is inert and whose history stays in tmp_path."""
    backend = QuietBackend()
    monkeypatch.setattr(repl_core, "LibremidiBackend", lambda **kwargs: backend)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return backend


def drive(script: bytes, monkeypatch, **kwargs) -> tuple[int, str]:
    """Run the stdlib REPL over a scripted terminal.

    Command output and editor rendering share one stream, exactly as they do
    on a real terminal, so both are returned together.
    """
    output = FakeOutput()
    monkeypatch.setattr(sys, "stdin", FakeInput(script))
    monkeypatch.setattr(sys, "stdout", output)
    status = run_terminal_repl(port_name="Fake", **kwargs)
    return status, output.getvalue()


class TestFrontendSelection:
    def test_unset_environment_uses_prompt_toolkit(self, monkeypatch):
        monkeypatch.delenv(FRONTEND_ENV_VAR, raising=False)
        assert stdlib_frontend_requested() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
    def test_truthy_values_select_the_stdlib_frontend(self, monkeypatch, value):
        monkeypatch.setenv(FRONTEND_ENV_VAR, value)
        assert stdlib_frontend_requested() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  "])
    def test_falsy_values_keep_prompt_toolkit(self, monkeypatch, value):
        monkeypatch.setenv(FRONTEND_ENV_VAR, value)
        assert stdlib_frontend_requested() is False

    def test_cli_selects_the_stdlib_frontend(self, monkeypatch):
        from aldakit.cli import _select_repl

        monkeypatch.setenv(FRONTEND_ENV_VAR, "1")
        assert _select_repl() is repl_terminal.run_terminal_repl

    def test_cli_defaults_to_prompt_toolkit(self, monkeypatch):
        from aldakit.cli import _select_repl
        from aldakit.repl import run_repl

        monkeypatch.delenv(FRONTEND_ENV_VAR, raising=False)
        assert _select_repl() is run_repl


class TestStdlibRepl:
    def test_alda_source_reaches_the_backend(self, stubbed, monkeypatch):
        status, _ = drive(keys(b"piano: c d e", ENTER), monkeypatch)
        assert status == 0
        assert len(stubbed.played) == 1

    def test_quit_command_ends_the_session(self, stubbed, monkeypatch):
        status, _ = drive(keys(b":quit", ENTER, b"piano: c", ENTER), monkeypatch)
        assert status == 0
        assert stubbed.played == []

    def test_commands_are_dispatched_not_played(self, stubbed, monkeypatch):
        _, output = drive(keys(b":help", ENTER), monkeypatch)
        assert "Commands:" in output
        assert stubbed.played == []

    def test_unknown_command_is_reported(self, stubbed, monkeypatch):
        _, output = drive(keys(b":nonsense", ENTER), monkeypatch)
        assert "Unknown command: :nonsense" in output

    def test_load_then_play_uses_the_buffer(self, stubbed, monkeypatch, tmp_path):
        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")
        _, output = drive(keys(f":load {song}".encode(), ENTER, b":play", ENTER), monkeypatch)
        assert "Loaded" in output
        assert len(stubbed.played) == 1

    def test_initial_file_loads_without_playing(self, stubbed, monkeypatch, tmp_path):
        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")
        _, output = drive(keys(b":quit", ENTER), monkeypatch, initial_file=song)
        assert "Loaded" in output
        assert stubbed.played == []

    def test_parse_errors_do_not_end_the_session(self, stubbed, monkeypatch):
        status, output = drive(keys(b"piano: (((", ENTER, b"piano: c", ENTER), monkeypatch)
        assert status == 0
        assert "Error" in output
        assert len(stubbed.played) == 1

    def test_backend_is_closed_on_exit(self, stubbed, monkeypatch):
        drive(keys(b":quit", ENTER), monkeypatch)
        assert stubbed.closed == 1

    def test_history_persists_across_sessions(self, stubbed, monkeypatch, tmp_path):
        drive(keys(b"piano: c", ENTER, b":quit", ENTER), monkeypatch)
        history = History(Path.home() / ".aldakit_history")
        history.load()
        assert "piano: c" in history.entries

    def test_missing_backend_is_reported(self, monkeypatch, capsys):
        def refuse(**kwargs):
            raise repl_core.BackendUnavailable("no audio")

        monkeypatch.setattr(repl_terminal, "open_backend", refuse)
        assert run_terminal_repl(use_audio=True) == 1
        assert "no audio" in capsys.readouterr().out
