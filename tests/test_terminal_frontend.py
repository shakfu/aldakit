"""Tests for the dependency-free terminal frontend."""

from io import StringIO
from pathlib import Path

from aldakit.terminal.app import run_line_mode
from aldakit.terminal.color import tokenize_alda
from aldakit.terminal.completion import ReplCompleter
from aldakit.terminal.editor import EditActionKind, LineEditor
from aldakit.terminal.history import History
from aldakit.terminal.keys import KeyDecoder, KeyEvent, KeyKind, decode_bytes
from aldakit.terminal.render import Renderer, TerminalCapabilities


def event(kind, text=""):
    return KeyEvent(kind, text)


def test_decodes_control_and_ansi_keys():
    events = decode_bytes(b"abc\t\r\n\x1b[A\x1b[3~\x03\x04")
    assert [item.kind for item in events] == [
        KeyKind.CHARACTER,
        KeyKind.CHARACTER,
        KeyKind.CHARACTER,
        KeyKind.TAB,
        KeyKind.ENTER,
        KeyKind.CTRL_J,
        KeyKind.ARROW_UP,
        KeyKind.DELETE,
        KeyKind.CTRL_C,
        KeyKind.CTRL_D,
    ]


def test_incremental_decoder_keeps_escape_pending():
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.feed(b"[A") == [event(KeyKind.ARROW_UP)]
    assert decoder.flush() == []


def test_editor_supports_multiline_and_word_editing():
    editor = LineEditor()
    for character in "piano: c d":
        editor.handle(event(KeyKind.CHARACTER, character))
    editor.handle(event(KeyKind.CTRL_W))
    assert editor.state.text == "piano: c "
    editor.handle(event(KeyKind.CTRL_J))
    assert editor.state.text == "piano: c \n"
    editor.handle(event(KeyKind.CHARACTER, "d"))
    action = editor.handle(event(KeyKind.ENTER))
    assert action.kind is EditActionKind.SUBMIT
    assert action.text == "piano: c \nd"
    assert editor.state.text == ""


def test_editor_ctrl_c_cancels_or_stops():
    editor = LineEditor("piano: c")
    assert editor.handle(event(KeyKind.CTRL_C)).kind is EditActionKind.CANCEL
    assert editor.state.text == ""
    assert editor.handle(event(KeyKind.CTRL_C)).kind is EditActionKind.STOP_PLAYBACK


def test_history_round_trips_multiline(tmp_path):
    history = History(tmp_path / "history", limit=2)
    history.add("piano: c\nd")
    history.add("violin: e")
    history.save()
    loaded = History(tmp_path / "history", limit=2)
    loaded.load()
    assert loaded.entries == ["piano: c\nd", "violin: e"]


def test_completion_handles_commands_and_instruments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    completer = ReplCompleter()
    assert any(item.replacement == "piano: " for item in completer.complete("pian"))
    commands = completer.complete(":lo")
    assert commands[0].replacement == "load "
    (tmp_path / "song.alda").write_text("piano: c", encoding="utf-8")
    paths = completer.complete(":load so")
    assert paths[0].replacement == "song.alda"


def test_color_spans_and_plain_renderer():
    spans = tokenize_alda("piano: c4 # comment")
    assert any(span.style == "note" for span in spans)
    assert any(span.style == "comment" for span in spans)
    renderer = Renderer(TerminalCapabilities(colors=False, cursor_movement=False))
    editor = LineEditor("piano: c4")
    assert renderer.render(editor.state) == "aldakit> piano: c4"


def test_line_mode_submits_without_ansi(tmp_path):
    submitted = []
    output = StringIO()
    result = run_line_mode(
        submitted.append,
        input_stream=StringIO("piano: c\n:help\n"),
        output_stream=output,
        history=History(tmp_path / "history"),
    )
    assert result == 0
    assert submitted == ["piano: c", ":help"]
    assert output.getvalue() == ""


def test_empty_history_is_truthy():
    """An empty History must not be falsy, or callers discard it."""
    assert bool(History("/nonexistent/history")) is True
    assert len(History("/nonexistent/history")) == 0


def test_line_mode_uses_the_supplied_history(tmp_path, monkeypatch):
    """A caller-supplied empty history must not fall back to the home file."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    target = tmp_path / "history"
    run_line_mode(
        lambda text: None,
        input_stream=StringIO("piano: c\n"),
        output_stream=StringIO(),
        history=History(target),
    )

    assert "+piano: c" in target.read_text(encoding="utf-8")
    assert list(home.iterdir()) == []


def test_decodes_ss3_cursor_sequences():
    """xterm sends ESC O A and ESC O H in application cursor mode."""
    assert [item.kind for item in decode_bytes(b"\x1bOA\x1bOH\x1bOF")] == [
        KeyKind.ARROW_UP,
        KeyKind.HOME,
        KeyKind.END,
    ]


def test_csi_scanner_stops_at_the_final_byte():
    """A CSI sequence ends at its first final byte; the next key must survive."""
    assert [item.kind for item in decode_bytes(b"\x1b[BA")] == [
        KeyKind.ARROW_DOWN,
        KeyKind.CHARACTER,
    ]


def test_alt_enter_inserts_a_newline_rather_than_submitting():
    assert decode_bytes(b"\x1b\r") == [event(KeyKind.CTRL_J)]
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.feed(b"\r") == [event(KeyKind.CTRL_J)]


def test_lone_escape_resolves_on_flush():
    """Escape is held only until the caller's timeout expires."""
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.flush() == [event(KeyKind.ESCAPE)]


def test_multibyte_characters_survive_bytewise_decoding():
    decoder = KeyDecoder()
    events = []
    for byte in "cé".encode("utf-8"):
        events.extend(decoder.feed(bytes([byte])))
    assert events == [event(KeyKind.CHARACTER, "c"), event(KeyKind.CHARACTER, "é")]
    assert decoder.pending == b""


def test_an_invalid_byte_does_not_discard_the_character_before_it():
    assert decode_bytes(b"a\xff") == [
        event(KeyKind.CHARACTER, "a"),
        event(KeyKind.UNKNOWN),
    ]


def test_completion_offers_instruments_and_attributes_together():
    completer = ReplCompleter()
    replacements = [item.replacement for item in completer.complete("(tem")]
    assert "(tempo " in replacements


def test_attribute_completion_replaces_from_the_open_paren():
    completer = ReplCompleter()
    candidate = completer.complete("piano: c (t")[0]
    editor = LineEditor("piano: c (t")
    editor.replace_range(candidate.start, candidate.end, candidate.replacement)
    assert editor.state.text == "piano: c (tempo "


def test_attribute_completion_uses_absolute_offsets_on_later_lines():
    completer = ReplCompleter()
    text = "piano: c\nd (tem"
    candidate = completer.complete(text)[0]
    editor = LineEditor(text)
    editor.replace_range(candidate.start, candidate.end, candidate.replacement)
    assert editor.state.text == "piano: c\nd (tempo "


def test_path_completion_preserves_a_typed_home_prefix(tmp_path, monkeypatch):
    (tmp_path / "music").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    completer = ReplCompleter()
    assert [item.replacement for item in completer.complete(":cd ~/mu")] == ["~/music/"]


def test_renderer_uses_one_prompt_per_line():
    renderer = Renderer(TerminalCapabilities(colors=False, cursor_movement=False))
    editor = LineEditor("piano: c\nd e")
    assert renderer.render(editor.state) == "aldakit> piano: c\n  ... d e"


def test_renderer_cursor_column_matches_the_continuation_prompt():
    renderer = Renderer(TerminalCapabilities(colors=False, cursor_movement=True))
    editor = LineEditor("piano: c\nd e")
    editor.state.cursor = len("piano: c\nd")
    output = renderer.render(editor.state)
    # One parameterized move, rather than a run of single-column steps.
    assert output.endswith(f"\r\x1b[{len('  ... ') + 1}C")


def test_history_file_is_readable_by_prompt_toolkit(tmp_path):
    """Both frontends share one file, so the format must round-trip."""
    from aldakit import ext  # noqa: F401 -- puts the vendored copy on the path
    from prompt_toolkit.history import FileHistory

    path = tmp_path / "history"
    history = History(path)
    history.append("piano: c\nd e")
    history.append("violin: g")
    # prompt_toolkit yields newest first; the entries themselves must match.
    assert list(FileHistory(str(path)).load_history_strings()) == [
        "violin: g",
        "piano: c\nd e",
    ]


def test_history_reads_a_prompt_toolkit_file(tmp_path):
    from aldakit import ext  # noqa: F401 -- puts the vendored copy on the path
    from prompt_toolkit.history import FileHistory

    path = tmp_path / "history"
    writer = FileHistory(str(path))
    writer.store_string("piano: c\nd e")
    writer.store_string("violin: g")
    history = History(path)
    history.load()
    assert history.entries == ["piano: c\nd e", "violin: g"]


def test_append_persists_before_the_session_ends(tmp_path):
    """A crash mid-session must not lose entries already submitted."""
    path = tmp_path / "history"
    history = History(path)
    history.append("piano: c")
    reloaded = History(path)
    reloaded.load()
    assert reloaded.entries == ["piano: c"]


def test_append_skips_adjacent_duplicates(tmp_path):
    path = tmp_path / "history"
    history = History(path)
    assert history.append("piano: c") is True
    assert history.append("piano: c") is False
    reloaded = History(path)
    reloaded.load()
    assert reloaded.entries == ["piano: c"]


def test_save_trims_to_the_entry_limit(tmp_path):
    path = tmp_path / "history"
    history = History(path, limit=2)
    for text in ("one", "two", "three"):
        history.append(text)
    history.save()
    reloaded = History(path, limit=10)
    reloaded.load()
    assert reloaded.entries == ["two", "three"]


def test_history_file_is_not_world_readable(tmp_path):
    import stat

    path = tmp_path / "history"
    History(path).append("piano: c")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    history = History(path)
    history.load()
    history.save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_malformed_records_are_ignored(tmp_path):
    path = tmp_path / "history"
    path.write_text(
        "garbage without a marker\n# a comment\n+piano: c\nmore garbage\n+violin: g\n",
        encoding="utf-8",
    )
    history = History(path)
    history.load()
    assert history.entries == ["piano: c", "violin: g"]


def test_load_ignores_a_missing_file(tmp_path):
    history = History(tmp_path / "absent")
    history.load()
    assert history.entries == []


def test_both_frontends_complete_the_same_commands():
    """One command table, so the two completers cannot drift apart."""
    from aldakit.constants import REPL_COMMAND_NAMES, REPL_PATH_COMMANDS
    from aldakit.repl import COMMAND_NAMES as ptk_names, PATH_COMMANDS as ptk_paths
    from aldakit.terminal.completion import COMMAND_NAMES, PATH_COMMANDS

    assert COMMAND_NAMES is REPL_COMMAND_NAMES
    assert ptk_names is REPL_COMMAND_NAMES
    assert PATH_COMMANDS == frozenset(REPL_PATH_COMMANDS)
    assert ptk_paths is REPL_PATH_COMMANDS


def test_menu_opens_and_previews_the_first_candidate():
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor(":p")
    menu = CompletionMenu()
    assert menu.open(editor, ReplCompleter()) is True
    assert menu.is_open
    assert editor.state.text == ":play "
    assert menu.labels == ["play ", "pwd", "ports"]


def test_menu_does_not_open_for_a_sole_candidate():
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor(":lo")
    menu = CompletionMenu()
    assert menu.open(editor, ReplCompleter()) is False
    assert not menu.is_open
    assert editor.state.text == ":load "


def test_menu_does_not_open_when_nothing_matches():
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor(":zzz")
    menu = CompletionMenu()
    assert menu.open(editor, ReplCompleter()) is False
    assert editor.state.text == ":zzz"


def test_menu_cancel_restores_the_original_buffer():
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor(":p")
    editor.state.cursor = 2
    menu = CompletionMenu()
    menu.open(editor, ReplCompleter())
    menu.select(editor, 1)
    menu.cancel(editor)
    assert editor.state.text == ":p"
    assert editor.state.cursor == 2
    assert not menu.is_open


def test_menu_previews_against_the_original_not_the_last_preview():
    """Candidates with different ranges must not compound."""
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor("(t")
    menu = CompletionMenu()
    menu.open(editor, ReplCompleter())
    first = editor.state.text
    menu.select(editor, 1)
    menu.select(editor, -1)
    assert editor.state.text == first
    assert first.count("(") == 1


def test_menu_scrolls_a_long_candidate_list():
    from aldakit.terminal.completion import CompletionMenu

    editor = LineEditor("mid")
    menu = CompletionMenu()
    menu.open(editor, ReplCompleter())
    assert len(menu.labels) > menu.max_visible

    labels, selected = menu.visible()
    assert len(labels) == menu.max_visible
    assert selected == 0

    for _ in range(menu.max_visible):
        menu.select(editor, 1)
    labels, selected = menu.visible()
    assert len(labels) == menu.max_visible
    # The highlighted row stays inside the window as it scrolls.
    assert 0 <= selected < menu.max_visible
    assert labels[selected] == menu.labels[menu.index]


def test_menu_renders_as_an_aligned_vertical_dropdown():
    from aldakit.terminal.render import MenuView

    renderer = Renderer(TerminalCapabilities(colors=False, cursor_movement=False, width=60))
    editor = LineEditor(":play ")
    output = renderer.render(editor.state, menu=MenuView(["play ", "pwd", "ports"], 0, 1))
    lines = output.split("\n")
    assert lines[0] == "aldakit> :play "
    assert [line.strip() for line in lines[1:]] == ["> play", "pwd", "ports"]
    # Aligned under the ":" the candidates replace, not at the margin.
    assert lines[1].index(">") == len("aldakit> ") + 1


def test_menu_highlights_the_selection_with_colour():
    from aldakit.terminal.render import MenuView

    renderer = Renderer(TerminalCapabilities(colors=True, cursor_movement=True, width=60))
    editor = LineEditor(":play ")
    output = renderer.render(editor.state, menu=MenuView(["play ", "pwd"], 1, 1))
    assert output.count("\x1b[46;30m") == 1  # exactly one selected row
    assert output.count("\x1b[47;30m") == 1


def test_menu_stays_on_screen_near_the_right_margin():
    from aldakit.terminal.render import MenuView

    renderer = Renderer(TerminalCapabilities(colors=False, cursor_movement=False, width=24))
    editor = LineEditor("x" * 12)
    output = renderer.render(editor.state, menu=MenuView(["alpha", "beta"], 0, 12))
    for line in output.split("\n"):
        assert len(line) <= 24
