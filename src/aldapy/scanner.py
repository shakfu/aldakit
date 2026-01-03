"""Lexer for the Alda music programming language."""

from .tokens import Token, TokenType, SourcePosition
from .errors import AldaScanError


class Scanner:
    """Tokenizes Alda source code."""

    NOTE_LETTERS = frozenset("abcdefg")
    WHITESPACE = frozenset(" \t\r")

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.tokens: list[Token] = []

        # Position tracking
        self._start = 0  # Start of current token
        self._current = 0  # Current position
        self._line = 1
        self._column = 1
        self._line_start = 0  # Position where current line started

        # Mode tracking
        self._sexp_depth = 0  # Depth of S-expression nesting

    def scan(self) -> list[Token]:
        """Scan the source and return all tokens."""
        self.tokens = []
        self._start = 0
        self._current = 0
        self._line = 1
        self._column = 1
        self._line_start = 0
        self._sexp_depth = 0

        while not self._is_at_end():
            self._start = self._current
            self._scan_token()

        # Add EOF token
        self.tokens.append(
            Token(
                TokenType.EOF,
                "",
                None,
                self._make_position(),
            )
        )
        return self.tokens

    def _scan_token(self) -> None:
        """Scan the next token."""
        c = self._advance()

        # Skip whitespace
        if c in self.WHITESPACE:
            return

        # Newlines
        if c == "\n":
            self._add_token(TokenType.NEWLINE)
            self._line += 1
            self._line_start = self._current
            return

        # Comments
        if c == "#":
            self._skip_comment()
            return

        # S-expression mode changes behavior
        if self._sexp_depth > 0:
            self._scan_lisp_token(c)
        else:
            self._scan_normal_token(c)

    def _scan_normal_token(self, c: str) -> None:
        """Scan a token in normal (non-lisp) mode."""
        # Single character tokens
        if c == ">":
            self._add_token(TokenType.OCTAVE_UP)
        elif c == "<":
            self._add_token(TokenType.OCTAVE_DOWN)
        elif c == "+":
            self._add_token(TokenType.SHARP)
        elif c == "-":
            self._add_token(TokenType.FLAT)
        elif c == "_":
            self._add_token(TokenType.NATURAL)
        elif c == "~":
            self._add_token(TokenType.TIE)
        elif c == "|":
            self._add_token(TokenType.BARLINE)
        elif c == "/":
            self._add_token(TokenType.SEPARATOR)
        elif c == ":":
            self._add_token(TokenType.COLON)
        elif c == ".":
            self._add_token(TokenType.DOT)
        elif c == "=":
            self._add_token(TokenType.EQUALS)
        elif c == "(":
            self._sexp_depth += 1
            self._add_token(TokenType.LEFT_PAREN)
        elif c == ")":
            self._error("Unexpected ')' outside of S-expression")
        elif c == "{":
            self._add_token(TokenType.CRAM_OPEN)
        elif c == "}":
            self._add_token(TokenType.CRAM_CLOSE)
        elif c == "[":
            self._add_token(TokenType.EVENT_SEQ_OPEN)
        elif c == "]":
            self._add_token(TokenType.EVENT_SEQ_CLOSE)
        elif c == "*":
            self._scan_repeat()
        elif c == "%":
            self._scan_marker()
        elif c == "@":
            self._scan_at_marker()
        elif c == "'":
            self._scan_repetitions()
        elif c == '"':
            self._scan_alias()
        elif c == "r" and not self._is_name_continuation(self._peek()):
            # Rest letter (only if not followed by name continuation chars)
            # Note: r followed by a digit is rest + duration, not a name
            self._add_token(TokenType.REST_LETTER)
        elif c in self.NOTE_LETTERS and not self._is_name_continuation(self._peek()):
            # Single note letter (not followed by identifier chars that would make it a name)
            self._add_token(TokenType.NOTE_LETTER, c)
        elif c in self.NOTE_LETTERS:
            # Note letter followed by more identifier chars - treat as name (e.g., 'cello')
            self._scan_name()
        elif c == "V" and self._peek().isdigit():
            # Voice marker: V followed by digits and colon
            self._scan_voice_marker()
        elif c == "o" and self._peek().isdigit():
            # Octave set: o followed by digits
            self._scan_octave_set()
        elif c.isdigit():
            self._scan_duration()
        elif self._is_identifier_start(c):
            self._scan_name()
        else:
            self._error(f"Unexpected character: {c!r}")

    def _scan_lisp_token(self, c: str) -> None:
        """Scan a token in lisp (S-expression) mode."""
        if c == "(":
            self._sexp_depth += 1
            self._add_token(TokenType.LEFT_PAREN)
        elif c == ")":
            self._sexp_depth -= 1
            self._add_token(TokenType.RIGHT_PAREN)
        elif c == '"':
            self._scan_string()
        elif c == "-" and self._peek().isdigit():
            # Negative number
            self._scan_lisp_number()
        elif c.isdigit():
            self._scan_lisp_number()
        elif self._is_symbol_char(c):
            self._scan_symbol()
        else:
            self._error(f"Unexpected character in S-expression: {c!r}")

    def _scan_octave_set(self) -> None:
        """Scan octave set (o followed by digits)."""
        while self._peek().isdigit():
            self._advance()
        lexeme = self.source[self._start : self._current]
        value = int(lexeme[1:])  # Skip the 'o'
        self._add_token(TokenType.OCTAVE_SET, value)

    def _scan_voice_marker(self) -> None:
        """Scan voice marker (V followed by digits and colon)."""
        while self._peek().isdigit():
            self._advance()
        # Expect colon
        if self._peek() == ":":
            self._advance()
            lexeme = self.source[self._start : self._current]
            # Extract number between V and :
            value = int(lexeme[1:-1])
            self._add_token(TokenType.VOICE_MARKER, value)
        else:
            # Not a voice marker, treat V as a name
            self._current = self._start + 1  # Reset to after V
            self._scan_name()

    def _scan_marker(self) -> None:
        """Scan a marker (%name)."""
        # Scan the marker name
        while self._is_marker_char(self._peek()):
            self._advance()
        lexeme = self.source[self._start : self._current]
        name = lexeme[1:]  # Skip the %
        if not name:
            self._error("Expected marker name after '%'")
        self._add_token(TokenType.MARKER, name)

    def _scan_at_marker(self) -> None:
        """Scan a marker reference (@name)."""
        while self._is_marker_char(self._peek()):
            self._advance()
        lexeme = self.source[self._start : self._current]
        name = lexeme[1:]  # Skip the @
        if not name:
            self._error("Expected marker name after '@'")
        self._add_token(TokenType.AT_MARKER, name)

    def _scan_repeat(self) -> None:
        """Scan a repeat operator (*number)."""
        while self._peek().isdigit():
            self._advance()
        lexeme = self.source[self._start : self._current]
        if len(lexeme) == 1:
            # Just *, no number - default to some value or error
            self._error("Expected number after '*'")
        value = int(lexeme[1:])  # Skip the *
        self._add_token(TokenType.REPEAT, value)

    def _scan_repetitions(self) -> None:
        """Scan repetition ranges ('1-3,5)."""
        # Scan the entire repetition specification
        # Format: '[number](-[number])?(,[number](-[number])?)*
        while self._peek().isdigit() or self._peek() in "-,":
            self._advance()
        lexeme = self.source[self._start : self._current]
        ranges_str = lexeme[1:]  # Skip the '
        if not ranges_str:
            self._error("Expected repetition range after apostrophe")
        self._add_token(TokenType.REPETITIONS, ranges_str)

    def _is_marker_char(self, c: str) -> bool:
        """Check if character is valid in a marker name."""
        return c.isalnum() or c in "_-"

    def _scan_duration(self) -> None:
        """Scan a duration (number, possibly followed by ms or s)."""
        while self._peek().isdigit():
            self._advance()

        # Check for decimal point
        if self._peek() == "." and self._peek_next().isdigit():
            self._advance()  # consume the '.'
            while self._peek().isdigit():
                self._advance()

        # Check for ms or s suffix
        lexeme = self.source[self._start : self._current]
        if self._peek() == "m" and self._peek_next() == "s":
            self._advance()  # m
            self._advance()  # s
            value = float(lexeme)
            self._add_token(TokenType.NOTE_LENGTH_MS, value)
        elif self._peek() == "s" and not self._is_identifier_char(self._peek_next()):
            self._advance()  # s
            value = float(lexeme)
            self._add_token(TokenType.NOTE_LENGTH_SECONDS, value)
        else:
            # Regular note length
            value = float(lexeme) if "." in lexeme else int(lexeme)
            self._add_token(TokenType.NOTE_LENGTH, value)

    def _scan_name(self) -> None:
        """Scan an identifier/name."""
        while self._is_identifier_char(self._peek()):
            self._advance()
        lexeme = self.source[self._start : self._current]
        self._add_token(TokenType.NAME, lexeme)

    def _scan_alias(self) -> None:
        """Scan a quoted alias string."""
        while self._peek() != '"' and not self._is_at_end():
            if self._peek() == "\n":
                self._error("Unterminated alias string")
            if self._peek() == "\\":
                self._advance()  # Skip escape char
            self._advance()

        if self._is_at_end():
            self._error("Unterminated alias string")

        self._advance()  # Closing quote
        # Extract string content without quotes
        value = self.source[self._start + 1 : self._current - 1]
        self._add_token(TokenType.ALIAS, value)

    def _scan_string(self) -> None:
        """Scan a string literal in lisp mode."""
        while self._peek() != '"' and not self._is_at_end():
            if self._peek() == "\n":
                self._line += 1
                self._line_start = self._current + 1
            if self._peek() == "\\":
                self._advance()
            self._advance()

        if self._is_at_end():
            self._error("Unterminated string")

        self._advance()  # Closing quote
        value = self.source[self._start + 1 : self._current - 1]
        self._add_token(TokenType.STRING, value)

    def _scan_symbol(self) -> None:
        """Scan a lisp symbol."""
        while self._is_symbol_char(self._peek()):
            self._advance()
        lexeme = self.source[self._start : self._current]
        self._add_token(TokenType.SYMBOL, lexeme)

    def _scan_lisp_number(self) -> None:
        """Scan a number in lisp mode."""
        while self._peek().isdigit():
            self._advance()

        # Check for decimal
        if self._peek() == "." and self._peek_next().isdigit():
            self._advance()
            while self._peek().isdigit():
                self._advance()

        lexeme = self.source[self._start : self._current]
        value = float(lexeme) if "." in lexeme else int(lexeme)
        self._add_token(TokenType.NUMBER, value)

    def _skip_comment(self) -> None:
        """Skip a comment (from # to end of line)."""
        while self._peek() != "\n" and not self._is_at_end():
            self._advance()

    # Helper methods

    def _is_at_end(self) -> bool:
        return self._current >= len(self.source)

    def _advance(self) -> str:
        c = self.source[self._current]
        self._current += 1
        return c

    def _peek(self) -> str:
        if self._is_at_end():
            return "\0"
        return self.source[self._current]

    def _peek_next(self) -> str:
        if self._current + 1 >= len(self.source):
            return "\0"
        return self.source[self._current + 1]

    def _is_identifier_start(self, c: str) -> bool:
        return c.isalpha() or c == "_"

    def _is_identifier_char(self, c: str) -> bool:
        return c.isalnum() or c in "_-"

    def _is_name_continuation(self, c: str) -> bool:
        """Check if character continues a name (making a note letter part of a name like 'cello')."""
        # Only letters continue a name after a note letter
        # - and _ are accidentals (flat, natural) so they don't make a name
        # Digits are durations so they don't make a name
        # e.g., 'cello' -> NAME, but 'c-' -> NOTE + FLAT, 'c4' -> NOTE + DURATION
        return c.isalpha()

    def _is_symbol_char(self, c: str) -> bool:
        """Check if character is valid in a lisp symbol."""
        if c == "\0":
            return False
        return c not in "()\"' \t\n\r"

    def _make_position(self) -> SourcePosition:
        column = self._start - self._line_start + 1
        return SourcePosition(self._line, column, self.filename)

    def _add_token(self, token_type: TokenType, literal: object = None) -> None:
        lexeme = self.source[self._start : self._current]
        self.tokens.append(Token(token_type, lexeme, literal, self._make_position()))

    def _get_current_line(self) -> str:
        """Get the current source line for error reporting."""
        line_end = self.source.find("\n", self._line_start)
        if line_end == -1:
            line_end = len(self.source)
        return self.source[self._line_start : line_end]

    def _error(self, message: str) -> None:
        raise AldaScanError(
            message,
            self._make_position(),
            self._get_current_line(),
        )
