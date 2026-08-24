"""Completion independent of prompt-toolkit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..constants import (
    REPL_COMMAND_NAMES,
    REPL_COMPLETION_MIN_WORD_LENGTH,
    REPL_PATH_COMMANDS,
)
from ..midi.types import INSTRUMENT_PROGRAMS


@dataclass(frozen=True)
class Completion:
    """A replacement range in an editor buffer."""

    replacement: str
    start: int
    end: int
    display: str | None = None


# Both frontends complete against the same table, so the two cannot drift.
COMMAND_NAMES = REPL_COMMAND_NAMES
PATH_COMMANDS = frozenset(REPL_PATH_COMMANDS)
ATTRIBUTES = (
    "(tempo ", "(volume ", "(quant ", "(key-sig ", "(pan ",
    "(panning ", "(track-vol ",
)


class ReplCompleter:
    """Complete the same command and Alda contexts as the existing REPL."""

    def __init__(self) -> None:
        self.instruments = tuple(sorted(INSTRUMENT_PROGRAMS))

    def complete(self, text: str, cursor: int | None = None) -> list[Completion]:
        """Return candidates for ``text`` at ``cursor``.

        Instrument and attribute candidates are both offered, matching the
        prompt-toolkit completer: a line may be mid-instrument or inside an
        open attribute expression, and neither context excludes the other.
        """
        cursor = len(text) if cursor is None else cursor
        before = text[:cursor]
        line = before.rsplit("\n", 1)[-1]
        stripped = line.lstrip()
        if stripped.startswith(":"):
            return self._complete_command(text, cursor, stripped)

        line_start = cursor - len(line)
        word_start = cursor
        while word_start > 0 and not _is_word_boundary(text[word_start - 1]):
            word_start -= 1
        word = text[word_start:cursor]

        result: list[Completion] = []
        if ":" not in line.strip() and len(word) >= REPL_COMPLETION_MIN_WORD_LENGTH:
            result.extend(
                Completion(f"{name}: ", word_start, cursor)
                for name in self.instruments
                if name.startswith(word)
            )

        open_paren = line.rfind("(")
        if open_paren >= 0 and ")" not in line[open_paren:]:
            # Replace from the "(" itself, not from the word after it.
            result.extend(
                Completion(attr, line_start + open_paren, cursor)
                for attr in ATTRIBUTES
                if attr.startswith("(" + word)
            )
        return result

    def _complete_command(self, text: str, cursor: int, stripped: str) -> list[Completion]:
        body = stripped[1:]
        offset = cursor - len(stripped)
        if " " not in body:
            return [Completion(name + (" " if name in PATH_COMMANDS else ""), offset + 1, cursor) for name in COMMAND_NAMES if name.startswith(body)]
        command, _, argument = body.partition(" ")
        if command not in PATH_COMMANDS:
            return []
        start = cursor - len(argument)

        # Split on what the user typed rather than on the expanded path, so a
        # "~/" or relative prefix is preserved in the replacement.
        separator = max(argument.rfind("/"), argument.rfind(os.sep))
        typed_directory = argument[:separator + 1] if separator >= 0 else ""
        prefix = argument[separator + 1:]
        if not typed_directory and prefix.startswith("~"):
            # A home-relative path with no separator yet; there is nothing
            # meaningful to list until the user types one.
            return []
        directory = Path(typed_directory).expanduser() if typed_directory else Path(".")
        trailing = argument[separator] if separator >= 0 else os.sep

        try:
            candidates = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            return []
        result: list[Completion] = []
        for candidate in candidates:
            if not candidate.name.startswith(prefix) or candidate.name.startswith("."):
                continue
            if command in ("load", "play") and candidate.is_file() and candidate.suffix.lower() != ".alda":
                continue
            if command == "save" and candidate.is_file() and candidate.suffix.lower() not in (".alda", ".mid", ".midi"):
                continue
            replacement = typed_directory + candidate.name + (trailing if candidate.is_dir() else "")
            result.append(Completion(replacement, start, cursor, candidate.name))
        return result


def _is_word_boundary(character: str) -> bool:
    """Word characters stop at whitespace and at an attribute's open paren."""
    return character.isspace() or character == "("


@dataclass
class _Origin:
    """The buffer as it stood before a completion menu opened."""

    text: str
    cursor: int


class CompletionMenu:
    """A drop-down completion menu over a :class:`LineEditor`.

    Selecting a candidate rewrites the buffer to show it, the way
    prompt_toolkit's menu does, so the effect of a choice is visible before it
    is accepted. Every preview is applied to the buffer as it was when the menu
    opened, which keeps candidates with different replacement ranges -- an
    instrument name and an attribute, say -- from compounding.
    """

    #: Most candidates shown at once; longer lists scroll around the selection.
    max_visible = 10

    def __init__(self) -> None:
        self.candidates: list[Completion] = []
        self.index = 0
        self._origin: _Origin | None = None

    @property
    def is_open(self) -> bool:
        return bool(self.candidates)

    def open(self, editor, completer) -> bool:
        """Complete at the cursor, returning whether a menu is now open.

        A sole candidate is applied outright, since there is nothing to choose
        between.
        """
        self.close()
        found = completer.complete(editor.state.text, editor.state.cursor)
        if not found:
            return False
        self._origin = _Origin(editor.state.text, editor.state.cursor)
        self.candidates = found
        self.index = 0
        self._preview(editor)
        if len(found) == 1:
            self.close()
            return False
        return True

    def select(self, editor, step: int) -> None:
        """Move the selection, wrapping at both ends, and preview it."""
        if not self.candidates:
            return
        self.index = (self.index + step) % len(self.candidates)
        self._preview(editor)

    def accept(self) -> None:
        """Keep the previewed text and close the menu."""
        self.close()

    def cancel(self, editor) -> None:
        """Restore the buffer as it was before the menu opened."""
        if self._origin is not None:
            editor.set_text(self._origin.text)
            editor.state.cursor = self._origin.cursor
        self.close()

    def close(self) -> None:
        self.candidates = []
        self.index = 0
        self._origin = None

    @property
    def labels(self) -> list[str]:
        return [item.display or item.replacement for item in self.candidates]

    @property
    def anchor(self) -> int:
        """Buffer offset the candidates replace, used to align the menu."""
        return min(item.start for item in self.candidates) if self.candidates else 0

    def visible(self) -> tuple[list[str], int]:
        """Return the labels to draw and which of them is selected."""
        labels = self.labels
        if len(labels) <= self.max_visible:
            return labels, self.index
        start = min(
            max(0, self.index - self.max_visible // 2),
            len(labels) - self.max_visible,
        )
        return labels[start:start + self.max_visible], self.index - start

    def _preview(self, editor) -> None:
        origin = self._origin
        if origin is None:
            return
        candidate = self.candidates[self.index]
        editor.set_text(origin.text)
        editor.replace_range(candidate.start, candidate.end, candidate.replacement)
