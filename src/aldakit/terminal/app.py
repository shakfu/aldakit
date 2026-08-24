"""Integration loop for the stdlib terminal frontend."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

from ..constants import REPL_CONTINUATION_PROMPT, REPL_HISTORY_FILENAME, REPL_PROMPT
from .completion import CompletionMenu, ReplCompleter
from .editor import EditActionKind, LineEditor
from .history import History, HistorySearch
from .keys import KeyKind
from .platform import (
    RawModeUnavailable,
    TerminalMode,
    TerminalReader,
    capabilities,
    descriptor_of,
)
from .render import MenuView, Renderer, TerminalCapabilities as RenderCapabilities

# How often the loop wakes up to poll for backend status while waiting for a
# key. Short enough that playback state feels live, long enough to stay idle.
POLL_INTERVAL = 0.1

# Page Up/Down step this many history entries at a time.
HISTORY_PAGE = 10


def run_editor(
    submit: Callable[[str], None],
    *,
    stop_playback: Callable[[], None] | None = None,
    poll: Callable[[], str | None] | None = None,
    should_exit: Callable[[], bool] | None = None,
    history: History | None = None,
    input_stream=None,
    output_stream=None,
) -> int:
    """Run the editor around a caller-provided submission callback.

    Command dispatch, score parsing, and playback remain outside this package.
    ``submit`` receives both Alda source and command lines. ``poll`` is called
    whenever the loop is idle and may return a status line to print.
    """
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    if not _is_tty(input_stream) or not _is_tty(output_stream):
        return run_line_mode(
            submit,
            should_exit=should_exit,
            input_stream=input_stream,
            output_stream=output_stream,
            history=history,
        )

    mode = TerminalMode(input_stream)
    try:
        mode.apply()
    except RawModeUnavailable as error:
        print(
            f"aldakit: falling back to line input ({error})",
            file=sys.stderr,
        )
        return run_line_mode(
            submit,
            should_exit=should_exit,
            input_stream=input_stream,
            output_stream=output_stream,
            history=history,
        )

    detected = capabilities(output_stream)
    renderer = Renderer(
        RenderCapabilities(detected.colors, detected.cursor_movement, detected.width)
    )
    editor = LineEditor()
    completer = ReplCompleter()
    if history is None:
        history = History(Path.home() / REPL_HISTORY_FILENAME)
    history.load()
    history_index: int | None = None
    history_draft = ""
    menu = CompletionMenu()
    search = HistorySearch()
    reader = TerminalReader(cast(BinaryIO, _key_source(input_stream, mode)))

    def draw() -> None:
        # Re-read the width so a resize is picked up without a SIGWINCH
        # handler, as the design calls for.
        renderer.capabilities = RenderCapabilities(
            detected.colors, detected.cursor_movement, capabilities(output_stream).width
        )
        view = None
        if menu.is_open:
            labels, selected = menu.visible()
            view = MenuView(labels, selected, menu.anchor)
        prompt = search.prompt() if search.active else REPL_PROMPT
        output_stream.write(renderer.render(editor.state, prompt, menu=view))
        output_stream.flush()

    def leave_editor_line() -> None:
        """End the editor region so external output starts on a fresh line."""
        output_stream.write("\r\n")
        renderer.reset()
        output_stream.flush()

    try:
        draw()
        while True:
            event = reader.read_event(timeout=POLL_INTERVAL if poll else None)
            if event is None:
                message = poll() if poll else None
                if message:
                    leave_editor_line()
                    output_stream.write(message.rstrip("\n") + "\r\n")
                    draw()
                continue

            if search.active:
                if event.kind is KeyKind.CTRL_R:
                    search.again(editor, history)
                    draw()
                    continue
                if event.kind is KeyKind.CHARACTER:
                    search.refine(editor, history, event.text)
                    draw()
                    continue
                if event.kind is KeyKind.BACKSPACE:
                    search.backspace(editor, history)
                    draw()
                    continue
                if event.kind in (KeyKind.ESCAPE, KeyKind.CTRL_C):
                    search.cancel(editor)
                    draw()
                    continue
                # Anything else leaves the search with the match in hand and
                # is then handled as an ordinary key.
                search.accept()
                history_index = None
            elif event.kind is KeyKind.CTRL_R and history.entries:
                menu.close()
                search.start(editor)
                draw()
                continue

            if menu.is_open:
                if event.kind is KeyKind.ESCAPE:
                    menu.cancel(editor)
                    draw()
                    continue
                if event.kind in (KeyKind.ARROW_UP, KeyKind.ARROW_DOWN, KeyKind.TAB):
                    menu.select(editor, -1 if event.kind is KeyKind.ARROW_UP else 1)
                    draw()
                    continue
                if event.kind is KeyKind.ENTER:
                    # Enter takes the highlighted candidate rather than
                    # submitting, as prompt_toolkit's menu does.
                    menu.accept()
                    draw()
                    continue
                # Any other key keeps the preview and is handled normally.
                menu.accept()
            elif event.kind is KeyKind.TAB:
                menu.open(editor, completer)
                draw()
                continue

            on_first_line = "\n" not in editor.state.text[:editor.state.cursor]
            on_last_line = "\n" not in editor.state.text[editor.state.cursor:]
            if event.kind in (KeyKind.ARROW_UP, KeyKind.PAGE_UP) and history.entries and on_first_line:
                step = HISTORY_PAGE if event.kind is KeyKind.PAGE_UP else 1
                if history_index is None:
                    history_draft = editor.state.text
                    history_index = max(0, len(history.entries) - step)
                else:
                    history_index = max(0, history_index - step)
                editor.set_text(history.entries[history_index])
                draw()
                continue
            if event.kind in (KeyKind.ARROW_DOWN, KeyKind.PAGE_DOWN) and history_index is not None and on_last_line:
                step = HISTORY_PAGE if event.kind is KeyKind.PAGE_DOWN else 1
                if history_index + step < len(history.entries):
                    history_index += step
                    editor.set_text(history.entries[history_index])
                else:
                    history_index = None
                    editor.set_text(history_draft)
                draw()
                continue

            action = editor.handle(event)
            if action.kind is EditActionKind.SUBMIT:
                leave_editor_line()
                if action.text.strip():
                    history.append(action.text)
                    submit(action.text.strip())
                history_index = None
                history_draft = ""
                if should_exit and should_exit():
                    break
            elif action.kind is EditActionKind.STOP_PLAYBACK:
                if stop_playback:
                    stop_playback()
            elif action.kind is EditActionKind.SUSPEND:
                leave_editor_line()
                _suspend(mode)
            elif action.kind is EditActionKind.EXIT:
                break
            elif action.kind is EditActionKind.CANCEL:
                leave_editor_line()
            draw()
    except EOFError:
        pass
    finally:
        mode.restore()
        # Entries are already appended as they are made; this trims the file
        # to the entry limit and must run however the loop ended.
        history.save()
    output_stream.write("\r\n")
    return 0


def run_line_mode(
    submit: Callable[[str], None],
    *,
    should_exit: Callable[[], bool] | None = None,
    input_stream=None,
    output_stream=None,
    history: History | None = None,
) -> int:
    """Run a safe, plain-text frontend for pipes and non-TTY streams.

    No ANSI is emitted. A trailing backslash continues the entry onto the next
    line, which is the only multiline convention this mode promises.
    """
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    if history is None:
        history = History(Path.home() / REPL_HISTORY_FILENAME)
    history.load()
    interactive = _is_tty(output_stream)
    pending: list[str] = []
    try:
        while True:
            if interactive:
                output_stream.write(REPL_CONTINUATION_PROMPT if pending else REPL_PROMPT)
                output_stream.flush()
            line = input_stream.readline()
            if not line:
                break
            text = line.rstrip("\r\n")
            if text.endswith("\\"):
                pending.append(text[:-1])
                continue
            pending.append(text)
            entry = "\n".join(pending)
            pending = []
            if not entry.strip():
                continue
            history.append(entry)
            submit(entry)
            if should_exit and should_exit():
                break
    finally:
        history.save()
    return 0


def _suspend(mode: TerminalMode) -> None:
    """Stop the process the way Ctrl+Z normally would, then resume raw mode."""
    if not hasattr(signal, "SIGTSTP"):
        return
    mode.restore()
    try:
        os.kill(os.getpid(), signal.SIGTSTP)
    finally:
        try:
            mode.apply()
        except RawModeUnavailable:
            pass


def _key_source(input_stream, mode: TerminalMode):
    """Return the byte source the reader should decode.

    On Windows keys come from the console rather than a readable descriptor;
    everywhere else the stream's own binary buffer is it. A stream owning no
    descriptor is not the console -- a test double, or an in-memory stream --
    so its own bytes are read there too, rather than the console adapter
    reaching past the stream it was given.
    """
    if _is_windows() and descriptor_of(input_stream) is not None:
        from .windows import ConsoleKeySource

        return ConsoleKeySource(
            virtual_terminal_input=mode.virtual_terminal_input
        )
    return getattr(input_stream, "buffer", input_stream)


def _is_windows() -> bool:
    return os.name == "nt"


def _is_tty(stream) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())
