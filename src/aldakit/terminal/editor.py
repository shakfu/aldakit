"""TTY-independent line editor state and editing operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .keys import KeyEvent, KeyKind


class EditActionKind(Enum):
    CONTINUE = auto()
    SUBMIT = auto()
    CANCEL = auto()
    STOP_PLAYBACK = auto()
    SUSPEND = auto()
    EXIT = auto()


@dataclass(frozen=True)
class EditAction:
    kind: EditActionKind
    text: str = ""


@dataclass
class EditorState:
    text: str = ""
    cursor: int = 0
    history_index: int | None = None


#: Keys that change the buffer, and so need an undo snapshot first.
_MUTATING_KEYS = frozenset({
    KeyKind.CHARACTER,
    KeyKind.CTRL_J,
    KeyKind.BACKSPACE,
    KeyKind.DELETE,
    KeyKind.CTRL_W,
    KeyKind.CTRL_U,
    KeyKind.CTRL_K,
    KeyKind.CTRL_Y,
    KeyKind.CTRL_T,
    KeyKind.ALT_D,
})


class LineEditor:
    """Small model-only editor supporting Alda's interactive workflows."""

    #: Most undo snapshots kept; older edits fall off the end.
    max_undo = 100

    def __init__(self, text: str = "") -> None:
        self.state = EditorState(text=text, cursor=len(text))
        self._draft = ""
        self.kill_ring = ""
        self._undo: list[tuple[str, int]] = []
        self._coalescing = False

    def snapshot(self, coalesce: bool = False) -> None:
        """Record the buffer so the next Ctrl+_ can restore it.

        Consecutive typed characters coalesce into one snapshot, so undo steps
        back over a word rather than a letter at a time.
        """
        if coalesce and self._coalescing:
            return
        self._undo.append((self.state.text, self.state.cursor))
        del self._undo[:-self.max_undo]
        self._coalescing = coalesce

    def undo(self) -> None:
        """Restore the most recent snapshot."""
        if not self._undo:
            return
        text, cursor = self._undo.pop()
        self.state.text = text
        self.state.cursor = min(cursor, len(text))
        self._coalescing = False

    def handle(self, event: KeyEvent) -> EditAction:
        """Apply one key and return the action requested by that key."""
        kind = event.kind
        if kind in _MUTATING_KEYS:
            self.snapshot(coalesce=kind is KeyKind.CHARACTER)
        elif kind is not KeyKind.CTRL_UNDO:
            self._coalescing = False

        if kind is KeyKind.CHARACTER:
            self._insert(event.text)
        elif kind is KeyKind.ENTER:
            text = self.state.text
            self.clear()
            return EditAction(EditActionKind.SUBMIT, text)
        elif kind is KeyKind.CTRL_J:
            self._insert("\n")
        elif kind is KeyKind.BACKSPACE:
            if self.state.cursor:
                cursor = self.state.cursor
                self.state.text = self.state.text[:cursor - 1] + self.state.text[cursor:]
                self.state.cursor -= 1
        elif kind is KeyKind.DELETE:
            cursor = self.state.cursor
            if cursor < len(self.state.text):
                self.state.text = self.state.text[:cursor] + self.state.text[cursor + 1:]
        elif kind is KeyKind.ARROW_LEFT:
            self.state.cursor = max(0, self.state.cursor - 1)
        elif kind is KeyKind.ARROW_RIGHT:
            self.state.cursor = min(len(self.state.text), self.state.cursor + 1)
        elif kind is KeyKind.HOME or kind is KeyKind.CTRL_A:
            self.state.cursor = self._line_start()
        elif kind is KeyKind.END or kind is KeyKind.CTRL_E:
            self.state.cursor = self._line_end()
        elif kind is KeyKind.ARROW_UP:
            self._move_vertical(-1)
        elif kind is KeyKind.ARROW_DOWN:
            self._move_vertical(1)
        elif kind is KeyKind.CTRL_W:
            self._delete_previous_word()
        elif kind is KeyKind.CTRL_U:
            start = self._line_start()
            self._kill(start, self.state.cursor)
            self.state.cursor = start
        elif kind is KeyKind.CTRL_K:
            self._kill(self.state.cursor, self._line_end())
        elif kind is KeyKind.ALT_B:
            self.state.cursor = self._word_start()
        elif kind is KeyKind.ALT_F:
            self.state.cursor = self._word_end()
        elif kind is KeyKind.ALT_D:
            self._kill(self.state.cursor, self._word_end())
        elif kind is KeyKind.CTRL_Y:
            if self.kill_ring:
                self._insert(self.kill_ring)
        elif kind is KeyKind.CTRL_T:
            self._transpose()
        elif kind is KeyKind.CTRL_UNDO:
            self.undo()
        elif kind is KeyKind.CTRL_C:
            if self.state.text:
                self.clear()
                return EditAction(EditActionKind.CANCEL)
            return EditAction(EditActionKind.STOP_PLAYBACK)
        elif kind is KeyKind.CTRL_Z:
            return EditAction(EditActionKind.SUSPEND)
        elif kind is KeyKind.CTRL_D:
            if not self.state.text:
                return EditAction(EditActionKind.EXIT)
            if self.state.cursor < len(self.state.text):
                self.state.text = self.state.text[:self.state.cursor] + self.state.text[self.state.cursor + 1:]
        return EditAction(EditActionKind.CONTINUE)

    def clear(self) -> None:
        """Clear the buffer and reset cursor state."""
        self.state = EditorState()

    def set_text(self, text: str) -> None:
        """Replace the current buffer."""
        self.state.text = text
        self.state.cursor = len(text)

    def replace_range(self, start: int, end: int, replacement: str) -> None:
        """Replace a buffer range and place the cursor after it."""
        if not 0 <= start <= end <= len(self.state.text):
            raise ValueError("replacement range is outside the editor buffer")
        self.state.text = self.state.text[:start] + replacement + self.state.text[end:]
        self.state.cursor = start + len(replacement)

    def _insert(self, text: str) -> None:
        cursor = self.state.cursor
        self.state.text = self.state.text[:cursor] + text + self.state.text[cursor:]
        self.state.cursor += len(text)

    def _line_start(self) -> int:
        return self.state.text.rfind("\n", 0, self.state.cursor) + 1

    def _line_end(self) -> int:
        end = self.state.text.find("\n", self.state.cursor)
        return len(self.state.text) if end < 0 else end

    def _move_vertical(self, direction: int) -> None:
        start = self._line_start()
        column = self.state.cursor - start
        if direction < 0:
            previous_end = start - 1
            if previous_end < 0:
                return
            previous_start = self.state.text.rfind("\n", 0, previous_end) + 1
            self.state.cursor = min(previous_start + column, previous_end)
        else:
            current_end = self._line_end()
            if current_end == len(self.state.text):
                return
            next_start = current_end + 1
            next_end = self.state.text.find("\n", next_start)
            if next_end < 0:
                next_end = len(self.state.text)
            self.state.cursor = min(next_start + column, next_end)

    def _kill(self, start: int, end: int) -> None:
        """Cut a range into the kill ring, for Ctrl+Y to put back."""
        if start >= end:
            return
        self.kill_ring = self.state.text[start:end]
        self.state.text = self.state.text[:start] + self.state.text[end:]
        self.state.cursor = start

    def _word_end(self) -> int:
        """The offset one word forward, in the readline sense.

        Word movement steps over runs of alphanumerics, which is not the same
        boundary Ctrl+W uses -- that one cuts back to whitespace.
        """
        text = self.state.text
        index = self.state.cursor
        while index < len(text) and not text[index].isalnum():
            index += 1
        while index < len(text) and text[index].isalnum():
            index += 1
        return index

    def _word_start(self) -> int:
        """The offset one word backward."""
        text = self.state.text
        index = self.state.cursor
        while index > 0 and not text[index - 1].isalnum():
            index -= 1
        while index > 0 and text[index - 1].isalnum():
            index -= 1
        return index

    def _transpose(self) -> None:
        """Swap the two characters around the cursor, as Ctrl+T does.

        At the end of a line the last two are swapped instead, which is what
        makes the key useful for fixing a typo just typed.
        """
        text = self.state.text
        cursor = self.state.cursor
        if len(text) < 2:
            return
        if cursor >= len(text) or text[cursor] == "\n":
            cursor = len(text) if cursor > len(text) else cursor
            if cursor < 2:
                return
            left = cursor - 2
        else:
            if cursor < 1:
                return
            left = cursor - 1
            cursor += 1
        self.state.text = (
            text[:left] + text[left + 1] + text[left] + text[left + 2:]
        )
        self.state.cursor = min(cursor, len(self.state.text))

    def _delete_previous_word(self) -> None:
        start = self._line_start()
        cursor = self.state.cursor
        while cursor > start and self.state.text[cursor - 1].isspace():
            cursor -= 1
        while cursor > start and not self.state.text[cursor - 1].isspace():
            cursor -= 1
        self._kill(cursor, self.state.cursor)
