"""ANSI and plain-text rendering for the model-only editor."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import REPL_CONTINUATION_PROMPT, REPL_PROMPT
from .color import tokenize_alda
from .editor import EditorState


@dataclass(frozen=True)
class TerminalCapabilities:
    colors: bool = True
    cursor_movement: bool = True
    width: int = 80


# Classic 8-colour pairs, so the menu reads correctly on any colour terminal.
_MENU_ITEM = "47;30"
_MENU_SELECTED = "46;30"

_COLORS = {
    "note": "36", "rest": "90", "octave": "35", "duration": "34",
    "attribute": "32", "instrument": "1;35", "barline": "90", "comment": "2;37",
}


@dataclass(frozen=True)
class MenuView:
    """What the renderer needs in order to draw a completion menu."""

    labels: list[str]
    selected: int
    anchor: int = 0


@dataclass(frozen=True)
class RenderedLine:
    """One physical terminal row of the editor region."""

    text: str
    styled: str


class Renderer:
    """Render an editor state into one deterministic terminal write."""

    def __init__(self, capabilities: TerminalCapabilities | None = None) -> None:
        self.capabilities = capabilities or TerminalCapabilities()
        self._previous_rows = 0

    def render_lines(self, state: EditorState, prompt: str = REPL_PROMPT) -> list[RenderedLine]:
        """Return the physical rows the editor occupies at the current width.

        A logical line longer than the terminal is split across rows, because
        the cursor arithmetic below counts rows, not logical lines.
        """
        width = max(1, self.capabilities.width)
        rows: list[RenderedLine] = []
        for index, line in enumerate(state.text.split("\n")):
            prefix = prompt if index == 0 else REPL_CONTINUATION_PROMPT
            content = prefix + line
            styled_prefix = prefix + self._styled(line)
            if len(content) <= width:
                rows.append(RenderedLine(content, styled_prefix))
                continue
            # Wrap on the plain text, then restyle each chunk on its own. The
            # tokenizer is display-only, so per-chunk styling is acceptable and
            # keeps the escape sequences inside the row they belong to.
            for start in range(0, len(content), width):
                chunk = content[start:start + width]
                rows.append(RenderedLine(chunk, self._styled(chunk)))
        return rows

    def cursor_position(self, state: EditorState, prompt: str = REPL_PROMPT) -> tuple[int, int]:
        """Return the cursor's (row, column) within the rendered region."""
        return self.position_of(state, state.cursor, prompt)

    def position_of(
        self, state: EditorState, offset: int, prompt: str = REPL_PROMPT
    ) -> tuple[int, int]:
        """Return the (row, column) a buffer offset renders at."""
        width = max(1, self.capabilities.width)
        row = 0
        for index, line in enumerate(state.text.split("\n")):
            prefix = prompt if index == 0 else REPL_CONTINUATION_PROMPT
            line_start = _line_offset(state.text, index)
            length = len(prefix) + len(line)
            rows_used = max(1, -(-length // width))
            if line_start <= offset <= line_start + len(line):
                column = len(prefix) + offset - line_start
                return row + column // width, column % width
            row += rows_used
        return max(0, row - 1), 0

    def render(
        self,
        state: EditorState,
        prompt: str = REPL_PROMPT,
        menu: MenuView | None = None,
    ) -> str:
        """Return output that redraws the prompt and places the cursor.

        Any candidate list is part of the same region, so it is cleared by the
        next redraw and the cursor is returned to the line the user is editing.
        """
        rows = self.render_lines(state, prompt)
        if menu and menu.labels:
            _, anchor_column = self.position_of(state, menu.anchor, prompt)
            rows.extend(self._menu_rows(menu, anchor_column))
        if not self.capabilities.cursor_movement:
            return "\n".join(row.text for row in rows)

        output = "\r\x1b[J" if self._previous_rows else "\r"
        output += "\n".join(row.styled for row in rows)
        row, column = self.cursor_position(state, prompt)
        remaining = len(rows) - row - 1
        if remaining > 0:
            output += f"\x1b[{remaining}A"
        output += "\r"
        if column:
            output += f"\x1b[{column}C"
        self._previous_rows = len(rows)
        return output

    def _menu_rows(self, menu: MenuView, anchor_column: int) -> list[RenderedLine]:
        """Draw the candidate list as a drop-down beneath the editor line."""
        width = max(1, self.capabilities.width)
        item_width = min(width, max(len(label) for label in menu.labels) + 2)
        # Keep the menu on screen when the completion starts near the margin.
        indent = max(0, min(anchor_column, width - item_width))
        rows: list[RenderedLine] = []
        for index, label in enumerate(menu.labels):
            cell = " " + label.ljust(item_width - 1)
            plain = " " * indent + cell
            if self.capabilities.colors:
                code = _MENU_SELECTED if index == menu.selected else _MENU_ITEM
                styled = " " * indent + f"\x1b[{code}m{cell}\x1b[0m"
            else:
                marker = "> " if index == menu.selected else "  "
                plain = " " * indent + marker + label
                styled = plain
            rows.append(RenderedLine(plain, styled))
        return rows

    def clear(self) -> str:
        """Clear the previous editor region."""
        if not self._previous_rows:
            return ""
        self._previous_rows = 0
        return "\r\x1b[J"

    def reset(self) -> None:
        """Forget the previous region, after output has scrolled past it."""
        self._previous_rows = 0

    def _styled(self, line: str) -> str:
        if not self.capabilities.colors:
            return line
        spans = tokenize_alda(line)
        if not spans:
            return line
        result: list[str] = []
        cursor = 0
        for span in spans:
            result.append(line[cursor:span.start])
            code = _COLORS.get(span.style)
            if code:
                result.append(f"\x1b[{code}m{line[span.start:span.end]}\x1b[0m")
            else:
                result.append(line[span.start:span.end])
            cursor = span.end
        result.append(line[cursor:])
        return "".join(result)


def _line_offset(text: str, index: int) -> int:
    """Return the absolute offset where logical line ``index`` begins."""
    offset = 0
    for _ in range(index):
        offset = text.index("\n", offset) + 1
    return offset
