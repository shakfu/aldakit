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
