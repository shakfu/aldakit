"""Small platform adapters for interactive terminal input."""

from __future__ import annotations

import io
import os
import shutil
import sys
from dataclasses import dataclass
import select
from types import TracebackType
from typing import BinaryIO

from .keys import KeyDecoder, KeyEvent

# How long to wait for the rest of an escape sequence before deciding that a
# lone Escape was pressed. Long enough for a terminal to deliver the remaining
# bytes of an arrow key, short enough to feel immediate.
ESCAPE_TIMEOUT = 0.05

_READ_SIZE = 1024


class RawModeUnavailable(RuntimeError):
    """Raised when the terminal cannot be put into raw mode."""


@dataclass(frozen=True)
class TerminalCapabilities:
    colors: bool
    cursor_movement: bool
    width: int


def capabilities(stream=None) -> TerminalCapabilities:
    """Detect conservative terminal capabilities.

    Width is read on every call rather than cached, so a resize is picked up by
    whatever redraw comes next without needing a ``SIGWINCH`` handler.
    """
    stream = stream or sys.stdout
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    no_color = bool(os.environ.get("NO_COLOR"))
    term = os.environ.get("TERM", "")
    return TerminalCapabilities(
        colors=is_tty and not no_color and term != "dumb",
        cursor_movement=is_tty,
        width=max(20, shutil.get_terminal_size((80, 24)).columns),
    )


class TerminalMode:
    """Put a POSIX terminal into raw mode and restore it reliably.

    ``tty.setcbreak`` is not enough. It clears only ``ECHO`` and ``ICANON``,
    leaving ``ICRNL`` to rewrite Enter as a newline, ``IXON`` to swallow Ctrl+S,
    and ``ISIG`` to turn Ctrl+C into a signal the editor never sees. Each flag
    is cleared explicitly here. ``OPOST`` deliberately stays on so a written
    newline still returns the cursor to column zero.
    """

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdin
        self._old: list | None = None
        self._fd: int | None = None
        self._console = None

    def __enter__(self) -> TerminalMode:
        self.apply()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.restore()

    @property
    def active(self) -> bool:
        """Whether raw mode is currently applied."""
        return self._old is not None or self._console is not None

    def apply(self) -> None:
        """Enter raw mode, if the stream is a terminal that supports it."""
        if self.active:
            return
        fd = self._descriptor()
        if fd is None:
            # A stream that reports itself as a terminal but owns no real
            # descriptor, such as a test double. There is nothing to set --
            # on either platform. The Windows path below works on the
            # process's own console handles, so without this check it would
            # reach past the stream it was given and touch the real console.
            return
        if sys.platform == "win32":
            self._apply_windows()
            return
        import termios

        try:
            old = termios.tcgetattr(fd)
            attributes = list(old)
            attributes[0] &= ~(
                termios.ICRNL | termios.INLCR | termios.IGNCR | termios.IXON
            )
            attributes[3] &= ~(
                termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN
            )
            control = list(attributes[6])
            control[termios.VMIN] = 1
            control[termios.VTIME] = 0
            attributes[6] = control
            termios.tcsetattr(fd, termios.TCSADRAIN, attributes)
        except (termios.error, OSError) as error:
            raise RawModeUnavailable(str(error)) from error
        self._old = old
        self._fd = fd

    def _apply_windows(self) -> None:
        from .windows import ConsoleModes, ConsoleUnavailable

        if self._console is not None:
            return
        console = ConsoleModes()
        try:
            console.apply()
        except ConsoleUnavailable as error:
            raise RawModeUnavailable(str(error)) from error
        self._console = console

    @property
    def virtual_terminal_input(self) -> bool:
        """Whether the console delivers keys as escape sequences."""
        return bool(getattr(self._console, "virtual_terminal_input", False))

    def restore(self) -> None:
        """Restore the settings captured by :meth:`apply`."""
        if sys.platform == "win32":
            if self._console is not None:
                self._console.restore()
                self._console = None
            return
        if self._old is None or self._fd is None:
            return
        import termios

        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except (termios.error, OSError):
            pass
        self._old = None

    def _descriptor(self) -> int | None:
        """Return a real terminal descriptor for the stream, if it has one."""
        return descriptor_of(self.stream)


class TerminalReader:
    """Read normalized events from a binary terminal stream.

    Reads go through ``select`` where the stream has a real descriptor, which
    is what lets a caller poll on a timeout and what bounds the wait for the
    rest of an escape sequence.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.decoder = KeyDecoder()
        self._queue: list[KeyEvent] = []
        # select() on Windows only accepts sockets, so the descriptor path
        # is POSIX-only; elsewhere reads fall back to the stream itself.
        self._fd = None if sys.platform == "win32" else descriptor_of(stream)

    def read_event(self, timeout: float | None = None) -> KeyEvent | None:
        """Return the next event, or ``None`` if ``timeout`` elapsed first.

        A read can decode into more than one event -- a paste burst, or a key
        arriving in the same chunk as the one before it -- so decoded events
        are queued rather than discarded. A partial escape sequence is held for
        at most :data:`ESCAPE_TIMEOUT`, after which a lone Escape is delivered
        instead of blocking forever.
        """
        while not self._queue:
            wait = timeout
            if self.decoder.pending:
                wait = ESCAPE_TIMEOUT if wait is None else min(wait, ESCAPE_TIMEOUT)
            if not self._ready(wait):
                if self.decoder.pending:
                    self._queue.extend(self.decoder.flush())
                    if self._queue:
                        break
                if timeout is None:
                    # No deadline was asked for, so keep waiting rather than
                    # handing back a spurious timeout for the caller to spin on.
                    continue
                return None
            data = self._read()
            if data is None:
                continue
            if not data:
                # End of input: resolve anything held back, so a trailing
                # lone Escape is delivered instead of dropped.
                self._queue.extend(self.decoder.flush())
                if not self._queue:
                    raise EOFError
                break
            self._queue.extend(self.decoder.feed(data))
        return self._queue.pop(0)

    def _ready(self, timeout: float | None) -> bool:
        if self._fd is None:
            # A stream with no descriptor. A test double may still model
            # readiness; anything else is treated as always ready.
            waiter = getattr(self.stream, "ready", None)
            return bool(waiter(timeout)) if callable(waiter) else True
        try:
            readable, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            return True
        return bool(readable)

    def _read(self) -> bytes | None:
        """Read available bytes.

        Returns ``b""`` at end of input, or ``None`` when the read was
        interrupted and should simply be retried.
        """
        if self._fd is None:
            return self.stream.read(1) or b""
        try:
            return os.read(self._fd, _READ_SIZE)
        except (BlockingIOError, InterruptedError):
            return None
        except OSError:
            return b""


def descriptor_of(stream) -> int | None:
    """Return the stream's descriptor when it refers to a real terminal."""
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return None
    try:
        return fd if os.isatty(fd) else None
    except OSError:
        return None
