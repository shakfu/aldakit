"""Error types for the Alda parser."""

from .tokens import SourcePosition


class AldaParseError(Exception):
    """Error during parsing or scanning of Alda source code."""

    def __init__(
        self,
        message: str,
        position: SourcePosition | None = None,
        source_line: str | None = None,
    ):
        self.message = message
        self.position = position
        self.source_line = source_line
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = []

        if self.position:
            parts.append(f"{self.position}: ")

        parts.append(self.message)

        if self.source_line is not None and self.position:
            parts.append(f"\n  {self.source_line}")
            # Add caret pointing to error column
            parts.append(f"\n  {' ' * (self.position.column - 1)}^")

        return "".join(parts)


class AldaScanError(AldaParseError):
    """Error during lexical analysis."""

    pass


class AldaSyntaxError(AldaParseError):
    """Error during parsing (syntax error)."""

    pass
