"""Persistent, dependency-free REPL history.

The on-disk format is the one prompt_toolkit's ``FileHistory`` uses: a record
is a ``#`` comment line carrying a timestamp, followed by one ``+``-prefixed
line per line of the entry. Keeping that format lets both frontends share a
single history file while the migration is in progress, and makes each entry an
append rather than a rewrite of the whole file.
"""

from __future__ import annotations

import datetime
import os
import tempfile
from pathlib import Path

_HISTORY_FILE_MODE = 0o600


class History:
    """Bounded history with safe, append-per-entry multiline persistence."""

    def __init__(self, path: str | Path, limit: int = 1000) -> None:
        self.path = Path(path).expanduser()
        self.limit = max(1, limit)
        self.entries: list[str] = []

    def load(self) -> None:
        """Load history oldest-first, ignoring malformed records."""
        try:
            raw = self.path.read_bytes()
        except OSError:
            return

        entries: list[str] = []
        lines: list[str] = []

        def close_record() -> None:
            if not lines:
                return
            text = "".join(lines)
            entries.append(text[:-1] if text.endswith("\n") else text)

        for line in raw.decode("utf-8", errors="replace").splitlines(keepends=True):
            if line.startswith("+"):
                lines.append(line[1:])
                continue
            close_record()
            lines.clear()
        close_record()

        self.entries = [entry for entry in entries if entry][-self.limit:]

    def add(self, text: str) -> bool:
        """Record an entry in memory, deduplicating adjacent repeats.

        Returns whether the entry was new. Callers that also want it on disk
        should use :meth:`append`.
        """
        text = text.strip()
        if not text:
            return False
        if self.entries and self.entries[-1] == text:
            return False
        self.entries.append(text)
        self.entries = self.entries[-self.limit:]
        return True

    def append(self, text: str) -> bool:
        """Record an entry and append it to the file straight away.

        Appending per entry rather than at exit means an interrupted session
        keeps the history it accumulated.
        """
        if not self.add(text):
            return False
        record = _format_record(text.strip())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                _HISTORY_FILE_MODE,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as output:
                output.write(record)
        except OSError:
            # A history file that cannot be written must not end the session.
            return True
        return True

    def save(self) -> None:
        """Atomically rewrite the file, trimming it to the entry limit.

        Entries reach the file through :meth:`append` as they are made; this
        exists to enforce ``limit`` and to persist entries added in memory.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                for entry in self.entries[-self.limit:]:
                    output.write(_format_record(entry))
                output.flush()
                os.fsync(output.fileno())
            os.chmod(name, _HISTORY_FILE_MODE)
            os.replace(name, self.path)
        except Exception:
            try:
                os.unlink(name)
            except OSError:
                pass
            raise

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        """A History is always truthy, even when it holds no entries.

        Without this, ``__len__`` would make an empty history falsy and
        callers testing a caller-supplied history would silently discard
        it in favour of the default file.
        """
        return True


def _format_record(entry: str) -> str:
    """Render one entry as a timestamped, ``+``-prefixed record."""
    body = "".join(f"+{line}\n" for line in entry.split("\n"))
    return f"\n# {datetime.datetime.now()}\n{body}"


class HistorySearch:
    """Incremental reverse search over history, as Ctrl+R does.

    The buffer shows the current match while the search runs, and the original
    line is restored if the search is abandoned.
    """

    def __init__(self) -> None:
        self.query = ""
        self.active = False
        self.failed = False
        self.index: int | None = None
        self._origin: tuple[str, int] | None = None

    def start(self, editor) -> None:
        """Begin a search, remembering the line to come back to."""
        self._origin = (editor.state.text, editor.state.cursor)
        self.query = ""
        self.index = None
        self.failed = False
        self.active = True

    def refine(self, editor, history: History, text: str) -> None:
        """Extend the query, searching on from the current match."""
        self.query += text
        start = len(history.entries) - 1 if self.index is None else self.index
        self._show(editor, history, self._find(history, start))

    def backspace(self, editor, history: History) -> None:
        """Shorten the query and search again from the newest entry."""
        self.query = self.query[:-1]
        self.failed = False
        if not self.query:
            self.index = None
            return
        self._show(editor, history, self._find(history, len(history.entries) - 1))

    def again(self, editor, history: History) -> None:
        """Move to the next older match, if there is one."""
        if not self.query:
            return
        start = len(history.entries) - 1 if self.index is None else self.index - 1
        self._show(editor, history, self._find(history, start))

    def accept(self) -> None:
        """Keep the match in the buffer and leave the search."""
        self.active = False

    def cancel(self, editor) -> None:
        """Restore the line as it was before the search began."""
        if self._origin is not None:
            editor.set_text(self._origin[0])
            editor.state.cursor = self._origin[1]
        self.active = False

    def prompt(self) -> str:
        """The prompt shown in place of the usual one while searching."""
        state = "failed " if self.failed else ""
        return f"({state}reverse-i-search)`{self.query}': "

    def _find(self, history: History, start_at: int) -> int | None:
        needle = self.query.lower()
        for index in range(min(start_at, len(history.entries) - 1), -1, -1):
            if needle in history.entries[index].lower():
                return index
        return None

    def _show(self, editor, history: History, found: int | None) -> None:
        if found is None:
            self.failed = True
            return
        self.failed = False
        self.index = found
        editor.set_text(history.entries[found])
