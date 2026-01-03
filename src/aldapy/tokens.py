"""Token types and Token class for the Alda lexer."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """All token types recognized by the Alda scanner."""

    # Note and rest letters
    NOTE_LETTER = auto()  # a-g
    REST_LETTER = auto()  # r

    # Accidentals
    FLAT = auto()  # -
    SHARP = auto()  # +
    NATURAL = auto()  # _

    # Octave control
    OCTAVE_SET = auto()  # o followed by number
    OCTAVE_UP = auto()  # >
    OCTAVE_DOWN = auto()  # <

    # Duration
    NOTE_LENGTH = auto()  # numeric duration (1, 2, 4, 8, 16, etc.)
    NOTE_LENGTH_MS = auto()  # duration in milliseconds (e.g., 500ms)
    NOTE_LENGTH_SECONDS = auto()  # duration in seconds (e.g., 2s)
    DOT = auto()  # . for dotted notes

    # Connectors
    TIE = auto()  # ~
    BARLINE = auto()  # |
    SEPARATOR = auto()  # / for chords

    # Part declaration
    NAME = auto()  # identifier (instrument name or variable)
    ALIAS = auto()  # quoted string for part alias
    COLON = auto()  # :

    # S-expression tokens (lisp mode)
    LEFT_PAREN = auto()  # (
    RIGHT_PAREN = auto()  # )
    SYMBOL = auto()  # lisp symbol
    NUMBER = auto()  # numeric literal in lisp context
    STRING = auto()  # string literal in lisp context

    # Variables
    EQUALS = auto()  # = for variable definition

    # Markers
    MARKER = auto()  # %name
    AT_MARKER = auto()  # @name

    # Voice markers
    VOICE_MARKER = auto()  # V[number]:

    # Cram expressions
    CRAM_OPEN = auto()  # {
    CRAM_CLOSE = auto()  # }

    # Event sequences
    EVENT_SEQ_OPEN = auto()  # [
    EVENT_SEQ_CLOSE = auto()  # ]

    # Repetition
    REPEAT = auto()  # *[number]
    REPETITIONS = auto()  # '[ranges] for on-repetition (e.g., '1-3,5)

    # Control
    EOF = auto()
    NEWLINE = auto()


@dataclass(frozen=True)
class SourcePosition:
    """Position in source code for error reporting."""

    line: int
    column: int
    filename: str = "<input>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


@dataclass
class Token:
    """A token produced by the scanner."""

    type: TokenType
    lexeme: str
    literal: Any  # Parsed value (e.g., int for NOTE_LENGTH)
    position: SourcePosition

    def __repr__(self) -> str:
        if self.literal is not None and self.literal != self.lexeme:
            return f"Token({self.type.name}, {self.lexeme!r}, {self.literal!r})"
        return f"Token({self.type.name}, {self.lexeme!r})"
