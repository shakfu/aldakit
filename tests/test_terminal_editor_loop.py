"""Integration tests driving ``run_editor`` against a fake terminal.

These cover the behaviour the design's "Test Strategy" section lists for the
interactive loop: submission, cancellation, exit, multiline entry, history, and
plain-text output when colour is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aldakit.constants import REPL_HISTORY_FILENAME
from aldakit.terminal.app import run_editor, run_line_mode
from aldakit.terminal.history import History
from tests.terminal_harness import (
    ALT_ENTER,
    BACKSPACE,
    CTRL_C,
    CTRL_D,
    CTRL_J,
    DOWN,
    ENTER,
    ESCAPE,
    IDLE,
    LEFT,
    PAGE_UP,
    TAB,
    UP,
    FakeInput,
    FakeOutput,
    keys,
)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Keep every test off the developer's real history file."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def drive(script, *, history=None, stop_playback=None, poll=None, on_submit=None):
    """Run the editor over a scripted burst and return what it saw."""
    submitted: list[str] = []

    def record(text: str) -> None:
        submitted.append(text)
        if on_submit is not None:
            on_submit(text)

    output = FakeOutput()
    status = run_editor(
        record,
        stop_playback=stop_playback,
        poll=poll,
        history=history,
        input_stream=FakeInput(script),
        output_stream=output,
    )
    return submitted, output.getvalue(), status


def test_submits_typed_alda_exactly_once(isolated_home):
    submitted, _, status = drive(keys(b"piano: c", ENTER))
    assert submitted == ["piano: c"]
    assert status == 0


def test_enter_submits_rather_than_inserting_a_newline(isolated_home):
    """Regression: CR must reach the editor as ENTER, not as a newline."""
    submitted, _, _ = drive(keys(b"c", ENTER, b"d", ENTER))
    assert submitted == ["c", "d"]


def test_alt_enter_and_ctrl_j_build_one_multiline_submission(isolated_home):
    submitted, _, _ = drive(keys(b"piano: c", ALT_ENTER, b"d", CTRL_J, b"e", ENTER))
    assert submitted == ["piano: c\nd\ne"]


def test_ctrl_c_cancels_without_recording(isolated_home):
    history = History(isolated_home / "history")
    submitted, _, _ = drive(keys(b"piano: c", CTRL_C, b"violin: e", ENTER), history=history)
    assert submitted == ["violin: e"]
    assert history.entries == ["violin: e"]


def test_ctrl_c_on_an_empty_buffer_stops_playback(isolated_home):
    stopped: list[int] = []
    submitted, _, _ = drive(keys(CTRL_C), stop_playback=lambda: stopped.append(1))
    assert stopped == [1]
    assert submitted == []


def test_ctrl_d_exits_only_when_the_buffer_is_empty(isolated_home):
    # Ctrl+D with text present deletes forward instead of exiting, so the
    # trailing Enter still submits.
    submitted, _, _ = drive(keys(b"cd", LEFT, CTRL_D, ENTER))
    assert submitted == ["c"]

    submitted, _, status = drive(keys(CTRL_D, b"ignored", ENTER))
    assert submitted == []
    assert status == 0


def test_editing_keys_change_the_submitted_text(isolated_home):
    submitted, _, _ = drive(keys(b"piano: cx", BACKSPACE, b"d", ENTER))
    assert submitted == ["piano: cd"]


def test_history_recall_walks_backwards(isolated_home):
    history = History(isolated_home / "recall")
    history.add("piano: c")
    history.add("violin: e")
    submitted, _, _ = drive(keys(UP, UP, ENTER), history=history)
    assert submitted == ["piano: c"]


def test_history_down_restores_the_unsent_draft(isolated_home):
    history = History(isolated_home / "draft")
    history.add("piano: c")
    history.add("violin: e")
    # Up enters the newest entry; Down past the newest returns to the draft.
    submitted, _, _ = drive(keys(b"draft", UP, DOWN, ENTER), history=history)
    assert submitted == ["draft"]


def test_arrow_up_navigates_a_multiline_buffer_before_history(isolated_home):
    """Regression: history must not claim Up while a line remains above."""
    history = History(isolated_home / "history")
    history.add("piano: c")
    # Up moves onto line one, where Home-like column tracking puts the cursor
    # at the end of "ab"; the inserted "X" therefore lands on the first line.
    submitted, _, _ = drive(keys(b"ab", CTRL_J, b"cd", UP, b"X", ENTER), history=history)
    assert submitted == ["abX\ncd"]


def test_tab_completes_a_unique_command(isolated_home):
    submitted, _, _ = drive(keys(b":por", TAB, ENTER))
    assert submitted == [":ports"]


def test_history_is_persisted_for_the_next_session(isolated_home):
    path = isolated_home / "history"
    drive(keys(b"piano: c", ENTER), history=History(path))
    reloaded = History(path)
    reloaded.load()
    assert reloaded.entries == ["piano: c"]


def test_history_defaults_to_the_shared_repl_history_file(isolated_home):
    """Both frontends read and write the same file under the user's home."""
    drive(keys(b"piano: c", ENTER))
    assert (isolated_home / REPL_HISTORY_FILENAME).exists()


def test_output_is_plain_when_colour_is_unavailable(isolated_home, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    _, output, _ = drive(keys(b"piano: c4", ENTER))
    assert "\x1b[36m" not in output
    assert "piano: c4" in output


def test_output_is_coloured_when_the_terminal_supports_it(isolated_home, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    _, output, _ = drive(keys(b"piano: c4", ENTER))
    assert "\x1b[" in output
    assert "piano: c4" not in output.replace("\x1b[0m", "")


def test_non_ascii_input_survives_the_round_trip(isolated_home):
    """Regression: multi-byte UTF-8 must not decode one byte at a time."""
    submitted, _, _ = drive("# tempo élevé\r".encode())
    assert submitted == ["# tempo élevé"]


def test_a_multiline_paste_burst_is_not_submitted_per_line(isolated_home):
    """A burst of newline bytes builds one buffer; only CR submits it."""
    submitted, _, _ = drive(keys(b"piano: c\nd e\nf g", ENTER))
    assert submitted == ["piano: c\nd e\nf g"]


def test_terminal_mode_is_restored_when_the_loop_raises(isolated_home, monkeypatch):
    """An exception from the callback must not leave raw mode set."""
    calls: list[str] = []

    class Tracking:
        def __init__(self, stream=None) -> None:
            self.stream = stream

        def apply(self) -> None:
            calls.append("apply")

        def restore(self) -> None:
            calls.append("restore")

    import aldakit.terminal.app as app

    monkeypatch.setattr(app, "TerminalMode", Tracking)
    with pytest.raises(RuntimeError):
        run_editor(
            lambda text: (_ for _ in ()).throw(RuntimeError("backend failed")),
            history=History(isolated_home / "history"),
            input_stream=FakeInput(keys(b"piano: c", ENTER)),
            output_stream=FakeOutput(),
        )
    assert calls == ["apply", "restore"]


def test_raw_mode_failure_falls_back_to_line_input(isolated_home, monkeypatch, capsys):
    """A terminal that refuses raw mode must degrade, not crash."""
    import aldakit.terminal.app as app

    class Refusing:
        def __init__(self, stream=None) -> None:
            self.stream = stream

        def apply(self) -> None:
            raise app.RawModeUnavailable("no such device")

        def restore(self) -> None:
            pass

    monkeypatch.setattr(app, "TerminalMode", Refusing)

    class TextInput(FakeOutput):
        """A TTY-reporting stream the line-oriented mode can read from."""

    stream = TextInput()
    stream.write("piano: c\n")
    stream.seek(0)

    submitted: list[str] = []
    output = FakeOutput()
    status = run_editor(
        submitted.append,
        history=History(isolated_home / "history"),
        input_stream=stream,
        output_stream=output,
    )
    assert status == 0
    assert submitted == ["piano: c"]
    assert "falling back to line input" in capsys.readouterr().err


def test_non_tty_input_uses_the_line_oriented_fallback(isolated_home):
    """A non-TTY stdin must never emit ANSI control sequences."""
    from io import StringIO

    submitted: list[str] = []
    output = StringIO()
    status = run_line_mode(
        submitted.append,
        input_stream=StringIO("piano: c\n:help\n"),
        output_stream=output,
        history=History(isolated_home / "history"),
    )
    assert status == 0
    assert submitted == ["piano: c", ":help"]
    assert ANSI.search(output.getvalue()) is None


def test_history_is_saved_when_the_callback_raises(isolated_home):
    """A backend failure must not cost the session its history."""
    path = isolated_home / "history"
    history = History(path)

    def explode(text: str) -> None:
        raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError):
        run_editor(
            explode,
            history=history,
            input_stream=FakeInput(keys(b"piano: c", ENTER)),
            output_stream=FakeOutput(),
        )

    reloaded = History(path)
    reloaded.load()
    assert reloaded.entries == ["piano: c"]


def test_line_mode_saves_history_when_the_callback_raises(isolated_home):
    from io import StringIO

    path = isolated_home / "history"

    def explode(text: str) -> None:
        raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError):
        run_line_mode(
            explode,
            input_stream=StringIO("piano: c\n"),
            output_stream=StringIO(),
            history=History(path),
        )

    reloaded = History(path)
    reloaded.load()
    assert reloaded.entries == ["piano: c"]


def test_unmapped_control_bytes_never_reach_the_buffer(isolated_home):
    """Ctrl+S arrives as a byte now that IXON is cleared; it is not text."""
    submitted, _, _ = drive(keys(b"\x13piano:\x07 c", ENTER))
    assert submitted == ["piano: c"]


def test_tab_applies_a_sole_candidate_without_a_menu(isolated_home):
    """With nothing to choose between, Tab just completes."""
    submitted, output, _ = drive(keys(b":pl", TAB, ENTER))
    assert submitted == [":play"]
    assert "pwd" not in output


def test_tab_opens_a_menu_of_every_candidate(isolated_home):
    _, output, _ = drive(keys(b":p", TAB, ENTER))
    for label in ("play", "pwd", "ports"):
        assert label in output


def test_the_menu_previews_the_highlighted_candidate(isolated_home):
    """The buffer shows the selection, so a choice is visible before it lands."""
    submitted, _, _ = drive([b":p", TAB, IDLE, ENTER, IDLE, ENTER])
    assert submitted == [":play"]


def test_tab_and_arrows_move_through_the_menu(isolated_home):
    # Tab opens on "play"; a second Tab moves to "pwd".
    submitted, _, _ = drive([b":p", TAB, TAB, IDLE, ENTER, IDLE, ENTER])
    assert submitted == [":pwd"]

    # Down does the same, and Up comes back.
    submitted, _, _ = drive([b":p", TAB, DOWN, UP, IDLE, ENTER, IDLE, ENTER])
    assert submitted == [":play"]


def test_the_menu_selection_wraps(isolated_home):
    """Up from the first candidate reaches the last."""
    submitted, _, _ = drive([b":p", TAB, UP, IDLE, ENTER, IDLE, ENTER])
    assert submitted == [":ports"]


def test_enter_accepts_the_candidate_rather_than_submitting(isolated_home):
    """Matching prompt_toolkit: the first Enter closes the menu."""
    submitted, _, _ = drive([b":p", TAB, IDLE, ENTER])
    assert submitted == []


def test_typing_keeps_the_preview_and_continues(isolated_home):
    submitted, _, _ = drive([b"(t", TAB, b"120)", IDLE, ENTER])
    assert submitted == ["(tempo 120)"]


def test_escape_closes_the_candidate_list_without_editing(isolated_home):
    # The IDLE marker is what separates a real Escape press from Alt+Enter:
    # without a pause, ESC followed by CR is one key, not two.
    submitted, _, _ = drive([b":p", TAB, ESCAPE, IDLE, ENTER])
    assert submitted == [":p"]


def test_a_lone_escape_resolves_instead_of_blocking(isolated_home):
    """The escape deadline must deliver Escape rather than wait forever."""
    submitted, _, _ = drive([ESCAPE, IDLE, b"piano: c", ENTER])
    assert submitted == ["piano: c"]


def test_page_up_steps_by_more_than_one_entry(isolated_home):
    history = History(isolated_home / "history")
    for index in range(30):
        history.add(f"entry {index}")
    submitted, _, _ = drive(keys(PAGE_UP, ENTER), history=history)
    assert submitted == ["entry 20"]


def test_page_up_stops_at_the_oldest_entry(isolated_home):
    history = History(isolated_home / "history")
    for index in range(5):
        history.add(f"entry {index}")
    submitted, _, _ = drive(keys(PAGE_UP, PAGE_UP, ENTER), history=history)
    assert submitted == ["entry 0"]


def test_poll_status_is_printed_while_idle(isolated_home):
    """The loop stays responsive to backend status between keystrokes."""
    messages = iter(["playing 3 notes"])
    _, output, _ = drive(
        [b"piano: c", IDLE, ENTER], poll=lambda: next(messages, None)
    )
    assert "playing 3 notes" in output


def test_submitted_line_is_left_before_external_output(isolated_home):
    """Command output must not append to the prompt line."""
    written: list[str] = []
    submitted, output, _ = drive(keys(b":help", ENTER), on_submit=written.append)
    assert submitted == [":help"]
    # The editor region is closed before the callback runs.
    assert output.index("\r\n") < len(output)


def test_ctrl_z_suspends_and_restores_raw_mode(isolated_home, monkeypatch):
    import aldakit.terminal.app as app

    calls: list[str] = []

    class Tracking:
        def __init__(self, stream=None) -> None:
            self.stream = stream

        def apply(self) -> None:
            calls.append("apply")

        def restore(self) -> None:
            calls.append("restore")

    monkeypatch.setattr(app, "TerminalMode", Tracking)
    monkeypatch.setattr(app.os, "kill", lambda pid, sig: calls.append("stop"))
    submitted, _, _ = drive(keys(b"piano: c", b"\x1a", ENTER))
    assert submitted == ["piano: c"]
    # Raw mode is dropped for the stop and re-entered on resume.
    assert calls == ["apply", "restore", "stop", "apply", "restore"]


def test_line_mode_continues_on_a_trailing_backslash(isolated_home):
    from io import StringIO

    submitted: list[str] = []
    run_line_mode(
        submitted.append,
        input_stream=StringIO("piano: c\\\nd e\n"),
        output_stream=StringIO(),
        history=History(isolated_home / "history"),
    )
    assert submitted == ["piano: c\nd e"]


def test_line_mode_stops_when_the_session_asks_to_exit(isolated_home):
    from io import StringIO

    submitted: list[str] = []
    quit_requested: list[bool] = []
    run_line_mode(
        lambda text: (submitted.append(text), quit_requested.append(text == ":quit")),
        should_exit=lambda: bool(quit_requested and quit_requested[-1]),
        input_stream=StringIO(":quit\nnever reached\n"),
        output_stream=StringIO(),
        history=History(isolated_home / "history"),
    )
    assert submitted == [":quit"]


def test_editor_stops_when_the_session_asks_to_exit(isolated_home):
    submitted: list[str] = []
    run_editor(
        submitted.append,
        should_exit=lambda: submitted[-1] == ":quit",
        history=History(isolated_home / "history"),
        input_stream=FakeInput(keys(b":quit", ENTER, b"never reached", ENTER)),
        output_stream=FakeOutput(),
    )
    assert submitted == [":quit"]


def test_windows_uses_the_line_oriented_frontend(isolated_home, monkeypatch):
    """Raw mode is unimplemented there, so degrade rather than misbehave."""
    import aldakit.terminal.app as app

    monkeypatch.setattr(app, "_is_windows", lambda: True)

    class TextInput(FakeOutput):
        pass

    stream = TextInput()
    stream.write("piano: c\n")
    stream.seek(0)

    submitted: list[str] = []
    status = run_editor(
        submitted.append,
        history=History(isolated_home / "history"),
        input_stream=stream,
        output_stream=FakeOutput(),
    )
    assert status == 0
    assert submitted == ["piano: c"]
