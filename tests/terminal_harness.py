"""A fake terminal for driving the stdlib frontend without a real TTY.

The editor loop needs a stream that claims to be a terminal, yields scripted
bytes, and captures what was written back. Owning no real descriptor keeps the
tests free of tty line-discipline behaviour, which is what makes them
deterministic across platforms and CI.

A script is either a bytes object or a list mixing bytes with :data:`IDLE`
markers. An ``IDLE`` marker makes the next readiness check report that nothing
has arrived yet, which is how tests exercise anything driven by a timeout: the
escape-sequence deadline, and backend status polling.
"""

from __future__ import annotations

import io


class _Idle:
    def __repr__(self) -> str:
        return "IDLE"


#: Marks a pause in a scripted input burst.
IDLE = _Idle()


class _ScriptedBytes:
    """The binary half of :class:`FakeInput`."""

    def __init__(self, script) -> None:
        self._items = list(script)
        self._buffer = b""

    def ready(self, timeout: float | None = None) -> bool:
        """Report whether bytes are available, honouring IDLE markers."""
        if self._buffer:
            return True
        self._drop_empty()
        if self._items and self._items[0] is IDLE:
            self._items.pop(0)
            return False
        return True

    def read(self, size: int = 1) -> bytes:
        if not self._buffer:
            self._drop_empty()
            while self._items and self._items[0] is IDLE:
                self._items.pop(0)
                self._drop_empty()
            if not self._items:
                return b""
            self._buffer = bytes(self._items.pop(0))
        chunk, self._buffer = self._buffer[:size], self._buffer[size:]
        return chunk

    def _drop_empty(self) -> None:
        while self._items and self._items[0] is not IDLE and not self._items[0]:
            self._items.pop(0)


class FakeInput:
    """A scripted stdin that reports as a TTY but owns no descriptor."""

    def __init__(self, script) -> None:
        if isinstance(script, (bytes, bytearray)):
            script = [bytes(script)]
        self.buffer = _ScriptedBytes(script)

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fake terminal has no descriptor")


class FakeOutput(io.StringIO):
    """A captured stdout that reports as a TTY."""

    def isatty(self) -> bool:
        return True


def keys(*parts: bytes) -> bytes:
    """Join scripted key bytes, for readable test scripts."""
    return b"".join(parts)


ENTER = b"\r"
CTRL_C = b"\x03"
CTRL_D = b"\x04"
CTRL_J = b"\n"
CTRL_R = b"\x12"
CTRL_T = b"\x14"
CTRL_W = b"\x17"
CTRL_Y = b"\x19"
CTRL_Z = b"\x1a"
CTRL_UNDO = b"\x1f"
ALT_B = b"\x1bb"
ALT_D = b"\x1bd"
ALT_F = b"\x1bf"
ALT_ENTER = b"\x1b\r"
TAB = b"\t"
UP = b"\x1b[A"
DOWN = b"\x1b[B"
LEFT = b"\x1b[D"
RIGHT = b"\x1b[C"
HOME = b"\x1b[H"
END = b"\x1b[F"
DELETE = b"\x1b[3~"
PAGE_UP = b"\x1b[5~"
PAGE_DOWN = b"\x1b[6~"
BACKSPACE = b"\x7f"
ESCAPE = b"\x1b"
