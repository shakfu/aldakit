"""Platform-independent terminal key decoding."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from enum import Enum, auto


class KeyKind(Enum):
    """Normalized keyboard events understood by the editor."""

    CHARACTER = auto()
    ENTER = auto()
    BACKSPACE = auto()
    DELETE = auto()
    TAB = auto()
    ESCAPE = auto()
    CTRL_C = auto()
    CTRL_D = auto()
    CTRL_A = auto()
    CTRL_E = auto()
    CTRL_K = auto()
    CTRL_U = auto()
    CTRL_W = auto()
    CTRL_J = auto()
    CTRL_R = auto()
    CTRL_T = auto()
    CTRL_Y = auto()
    CTRL_Z = auto()
    CTRL_UNDO = auto()
    ALT_B = auto()
    ALT_D = auto()
    ALT_F = auto()
    ARROW_LEFT = auto()
    ARROW_RIGHT = auto()
    ARROW_UP = auto()
    ARROW_DOWN = auto()
    HOME = auto()
    END = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class KeyEvent:
    """A decoded key, with text set only for character events."""

    kind: KeyKind
    text: str = ""


_CONTROL_KEYS = {
    1: KeyKind.CTRL_A,
    3: KeyKind.CTRL_C,
    4: KeyKind.CTRL_D,
    5: KeyKind.CTRL_E,
    9: KeyKind.TAB,
    10: KeyKind.CTRL_J,
    11: KeyKind.CTRL_K,
    18: KeyKind.CTRL_R,
    20: KeyKind.CTRL_T,
    21: KeyKind.CTRL_U,
    23: KeyKind.CTRL_W,
    25: KeyKind.CTRL_Y,
    26: KeyKind.CTRL_Z,
    31: KeyKind.CTRL_UNDO,
    127: KeyKind.BACKSPACE,
}

# Meta (Alt) chords, which terminals send as Escape followed by the key. The
# escape deadline in the reader is what separates these from a lone Escape
# followed by ordinary typing.
_ALT_KEYS = {
    b"b": KeyKind.ALT_B,
    b"d": KeyKind.ALT_D,
    b"f": KeyKind.ALT_F,
}

# Final bytes of a CSI sequence (ESC [ ... final), keyed without parameters.
_CSI_KEYS = {
    b"A": KeyKind.ARROW_UP,
    b"B": KeyKind.ARROW_DOWN,
    b"C": KeyKind.ARROW_RIGHT,
    b"D": KeyKind.ARROW_LEFT,
    b"H": KeyKind.HOME,
    b"F": KeyKind.END,
    b"1~": KeyKind.HOME,
    b"4~": KeyKind.END,
    b"5~": KeyKind.PAGE_UP,
    b"6~": KeyKind.PAGE_DOWN,
    b"3~": KeyKind.DELETE,
}

# Application-cursor-mode sequences (ESC O final), as sent by xterm and friends.
_SS3_KEYS = {
    b"A": KeyKind.ARROW_UP,
    b"B": KeyKind.ARROW_DOWN,
    b"C": KeyKind.ARROW_RIGHT,
    b"D": KeyKind.ARROW_LEFT,
    b"H": KeyKind.HOME,
    b"F": KeyKind.END,
}

_ESCAPE = 0x1B
_CSI_INTRODUCER = 0x5B  # "["
_SS3_INTRODUCER = 0x4F  # "O"


def _utf8_sequence_length(lead: int) -> int | None:
    """Return the byte length of the UTF-8 character starting with ``lead``.

    ``None`` marks a byte that cannot begin a character, including stray
    continuation bytes.
    """
    if lead < 0x80:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return None


class KeyDecoder:
    """Incrementally decode terminal bytes without blocking on Escape.

    Bytes are buffered until they form a complete key. Callers that have
    waited out an escape timeout should call :meth:`flush` to resolve whatever
    remains, which turns a lone Escape into ``ESCAPE`` rather than discarding
    it.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding
        self._utf8 = codecs.lookup(encoding).name == "utf-8"
        self._pending = bytearray()

    def feed(self, data: bytes) -> list[KeyEvent]:
        """Decode whatever bytes are now complete, buffering the rest."""
        self._pending.extend(data)
        return self._drain(final=False)

    def flush(self) -> list[KeyEvent]:
        """Resolve buffered bytes, treating a lone Escape as Escape."""
        return self._drain(final=True)

    @property
    def pending(self) -> bytes:
        """Bytes held back awaiting the rest of a key."""
        return bytes(self._pending)

    def _drain(self, final: bool) -> list[KeyEvent]:
        events: list[KeyEvent] = []
        while self._pending:
            byte = self._pending[0]
            if byte == _ESCAPE:
                decoded = self._decode_escape(final)
                if decoded is None:
                    break
                events.append(decoded)
                continue
            kind = _CONTROL_KEYS.get(byte)
            if kind is not None:
                del self._pending[:1]
                events.append(KeyEvent(kind))
                continue
            if byte == 13:
                del self._pending[:1]
                events.append(KeyEvent(KeyKind.ENTER))
                continue
            if byte < 0x20:
                # An unmapped C0 control byte, such as Ctrl+S now that IXON is
                # cleared. It is not text and must not reach the buffer.
                del self._pending[:1]
                events.append(KeyEvent(KeyKind.UNKNOWN))
                continue
            decoded = self._decode_character(final)
            if decoded is None:
                break
            events.append(decoded)
        return events

    def _decode_escape(self, final: bool) -> KeyEvent | None:
        pending = self._pending
        if len(pending) == 1:
            if not final:
                return None
            del pending[:1]
            return KeyEvent(KeyKind.ESCAPE)

        introducer = pending[1]
        if introducer in (10, 13):
            # Alt+Enter. The design reserves Enter for submission, so this
            # inserts a newline exactly as Ctrl+J does.
            del pending[:2]
            return KeyEvent(KeyKind.CTRL_J)
        if introducer == _CSI_INTRODUCER:
            end = _csi_end(pending)
            if end is None:
                if not final:
                    return None
                del pending[:]
                return KeyEvent(KeyKind.UNKNOWN)
            sequence = bytes(pending[2:end])
            del pending[:end]
            return KeyEvent(_CSI_KEYS.get(sequence, KeyKind.UNKNOWN))
        alt = _ALT_KEYS.get(bytes(pending[1:2]))
        if alt is not None:
            del pending[:2]
            return KeyEvent(alt)
        if introducer == _SS3_INTRODUCER:
            if len(pending) < 3:
                if not final:
                    return None
                del pending[:]
                return KeyEvent(KeyKind.UNKNOWN)
            sequence = bytes(pending[2:3])
            del pending[:3]
            return KeyEvent(_SS3_KEYS.get(sequence, KeyKind.UNKNOWN))

        # Escape followed by anything else is a bare Escape; the next pass
        # decodes the following byte on its own.
        del pending[:1]
        return KeyEvent(KeyKind.ESCAPE)

    def _decode_character(self, final: bool) -> KeyEvent | None:
        pending = self._pending
        length = _utf8_sequence_length(pending[0]) if self._utf8 else 1
        if length is None:
            del pending[:1]
            return KeyEvent(KeyKind.UNKNOWN)
        if len(pending) < length:
            if not final:
                return None
            del pending[:]
            return KeyEvent(KeyKind.UNKNOWN)
        raw = bytes(pending[:length])
        try:
            text = raw.decode(self.encoding)
        except UnicodeDecodeError:
            del pending[:1]
            return KeyEvent(KeyKind.UNKNOWN)
        del pending[:length]
        return KeyEvent(KeyKind.CHARACTER, text)


def _csi_end(data: bytearray) -> int | None:
    """Return the index just past a CSI final byte, or ``None`` if incomplete.

    A CSI sequence is ``ESC [``, parameter bytes, intermediate bytes, then
    exactly one final byte in ``0x40``-``0x7E``. Consuming past that final byte
    would swallow the key that follows.
    """
    for index in range(2, len(data)):
        if 0x40 <= data[index] <= 0x7E:
            return index + 1
    return None


def decode_bytes(data: bytes, encoding: str = "utf-8") -> list[KeyEvent]:
    """Decode a complete burst of input bytes into normalized events.

    Trailing bytes that cannot form a key are resolved rather than buffered,
    so this is only appropriate for input already known to be complete.
    """
    decoder = KeyDecoder(encoding)
    return decoder.feed(data) + decoder.flush()
