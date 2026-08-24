"""Windows console adapter for the terminal frontend.

Two things differ from POSIX. Console modes are set through ``SetConsoleMode``
rather than ``termios``, and keys arrive through ``msvcrt`` rather than a
readable file descriptor. Both are normalized here so that everything above --
the decoder, the editor, the renderer -- is the same code on every platform.

The ``kernel32`` and ``msvcrt`` dependencies are injectable so this module can
be exercised on any platform.
"""

from __future__ import annotations

import time

STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11

# Input modes. Clearing the first three is what raw mode means here: no line
# buffering, no echo, and Ctrl+C delivered as a byte instead of a signal.
ENABLE_PROCESSED_INPUT = 0x0001
ENABLE_LINE_INPUT = 0x0002
ENABLE_ECHO_INPUT = 0x0004
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

# Output modes, so the renderer's escape sequences are interpreted.
ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

#: How often to look for a keypress while waiting with a timeout. msvcrt has
#: no blocking wait, so readiness is polled.
POLL_GRANULARITY = 0.005

#: Windows delivers cursor and editing keys as a two-character pair led by
#: NUL or 0xE0. Translating them to the sequences a terminal would send keeps
#: one decoder for both platforms.
EXTENDED_KEYS = {
    "H": b"\x1b[A",
    "P": b"\x1b[B",
    "M": b"\x1b[C",
    "K": b"\x1b[D",
    "G": b"\x1b[H",
    "O": b"\x1b[F",
    "S": b"\x1b[3~",
    "I": b"\x1b[5~",
    "Q": b"\x1b[6~",
}

_EXTENDED_LEAD = ("\x00", "\xe0")


class ConsoleUnavailable(RuntimeError):
    """Raised when console modes cannot be read or set."""


def load_kernel32():
    """Return the kernel32 API, or None where it does not exist."""
    try:
        import ctypes

        # Looked up rather than referenced directly: WinDLL exists only on
        # Windows, and a direct attribute breaks type checking elsewhere.
        windll = getattr(ctypes, "WinDLL", None)
        if windll is None:
            return None
        return windll("kernel32", use_last_error=True)
    except (ImportError, OSError):
        return None


def load_msvcrt():
    """Return the msvcrt module, or None where it does not exist."""
    try:
        import msvcrt

        return msvcrt
    except ImportError:
        return None


class ConsoleModes:
    """Put the console into raw mode and put it back afterwards."""

    def __init__(self, kernel32=None) -> None:
        self.kernel32 = kernel32 if kernel32 is not None else load_kernel32()
        self._saved: dict[int, int] = {}
        #: Whether the console accepted virtual-terminal input. When it does,
        #: keys arrive as escape sequences and the shared decoder handles them
        #: unchanged; when it does not, extended-key pairs are translated.
        self.virtual_terminal_input = False

    @property
    def active(self) -> bool:
        return bool(self._saved)

    def apply(self) -> None:
        """Enter raw input mode and enable escape-sequence output."""
        if self.kernel32 is None:
            raise ConsoleUnavailable("kernel32 is not available")
        if self._saved:
            return

        original_input = self._get_mode(STD_INPUT_HANDLE)
        original_output = self._get_mode(STD_OUTPUT_HANDLE)

        raw_input_mode = original_input & ~(
            ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
        )
        # Ask for escape sequences; older consoles reject the flag, in which
        # case the extended-key translation above covers the same keys.
        if self._set_mode(
            STD_INPUT_HANDLE, raw_input_mode | ENABLE_VIRTUAL_TERMINAL_INPUT
        ):
            self.virtual_terminal_input = True
        elif not self._set_mode(STD_INPUT_HANDLE, raw_input_mode):
            raise ConsoleUnavailable("could not set console input mode")
        self._saved[STD_INPUT_HANDLE] = original_input

        self._set_mode(
            STD_OUTPUT_HANDLE,
            original_output
            | ENABLE_PROCESSED_OUTPUT
            | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        )
        self._saved[STD_OUTPUT_HANDLE] = original_output

    def restore(self) -> None:
        """Put back the modes captured by :meth:`apply`."""
        for handle, mode in self._saved.items():
            self._set_mode(handle, mode)
        self._saved.clear()
        self.virtual_terminal_input = False

    def _get_mode(self, handle_id: int) -> int:
        import ctypes

        handle = self.kernel32.GetStdHandle(handle_id)
        mode = ctypes.c_uint32()
        if not self.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            raise ConsoleUnavailable(f"could not read console mode {handle_id}")
        return mode.value

    def _set_mode(self, handle_id: int, mode: int) -> bool:
        handle = self.kernel32.GetStdHandle(handle_id)
        return bool(self.kernel32.SetConsoleMode(handle, mode))


class ConsoleKeySource:
    """A byte source over ``msvcrt``, feeding the shared key decoder.

    Presenting keys as bytes rather than events means the escape parsing,
    UTF-8 handling, and editing logic are identical on every platform.
    """

    def __init__(self, msvcrt_module=None, virtual_terminal_input: bool = True) -> None:
        self.msvcrt = msvcrt_module if msvcrt_module is not None else load_msvcrt()
        self.virtual_terminal_input = virtual_terminal_input
        self._pending = bytearray()

    def ready(self, timeout: float | None = None) -> bool:
        """Whether a key is waiting, polling until ``timeout`` elapses."""
        if self._pending:
            return True
        if self.msvcrt is None:
            # Nothing can ever arrive; report ready so the read reports end of
            # input rather than leaving the caller polling forever.
            return True
        if timeout is None:
            while not self.msvcrt.kbhit():
                time.sleep(POLL_GRANULARITY)
            return True
        deadline = time.monotonic() + timeout
        while True:
            if self.msvcrt.kbhit():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(POLL_GRANULARITY, max(0.0, deadline - time.monotonic())))

    def read(self, size: int = 1) -> bytes:
        """Return up to ``size`` bytes of encoded key input."""
        if not self._pending:
            self._fill()
        if not self._pending:
            return b""
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def _fill(self) -> None:
        if self.msvcrt is None:
            return
        character = self.msvcrt.getwch()
        if not character:
            return
        if character in _EXTENDED_LEAD:
            code = self.msvcrt.getwch()
            # An unrecognized extended key is dropped rather than inserted as
            # a stray character.
            self._pending.extend(EXTENDED_KEYS.get(code, b""))
            return
        if character == "\r":
            # The console reports Enter as a carriage return, which is what
            # the decoder expects for submission.
            self._pending.extend(b"\r")
            return
        self._pending.extend(character.encode("utf-8", errors="replace"))
