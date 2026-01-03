"""Recursive descent parser for the Alda music programming language."""

from typing import NoReturn

from .tokens import Token, TokenType, SourcePosition
from .scanner import Scanner
from .errors import AldaSyntaxError
from .ast_nodes import (
    ASTNode,
    RootNode,
    PartNode,
    PartDeclarationNode,
    EventSequenceNode,
    NoteNode,
    RestNode,
    ChordNode,
    DurationNode,
    NoteLengthNode,
    NoteLengthMsNode,
    NoteLengthSecondsNode,
    BarlineNode,
    OctaveSetNode,
    OctaveUpNode,
    OctaveDownNode,
    LispListNode,
    LispSymbolNode,
    LispNumberNode,
    LispStringNode,
    # Phase 2 nodes
    VariableDefinitionNode,
    VariableReferenceNode,
    MarkerNode,
    AtMarkerNode,
    VoiceNode,
    VoiceGroupNode,
    CramNode,
    RepeatNode,
    OnRepetitionsNode,
    RepetitionRange,
    BracketedSequenceNode,
)


class Parser:
    """Parses Alda tokens into an AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self._current = 0

    @classmethod
    def from_source(cls, source: str, filename: str = "<input>") -> "Parser":
        """Create a parser from source code."""
        scanner = Scanner(source, filename)
        tokens = scanner.scan()
        return cls(tokens)

    def parse(self) -> RootNode:
        """Parse the tokens and return the AST root."""
        children: list[ASTNode] = []
        self._skip_newlines()

        while not self._is_at_end():
            node = self._parse_top_level()
            if node:
                children.append(node)
            self._skip_newlines()

        return RootNode(children=children, position=SourcePosition(1, 1))

    def _parse_top_level(self) -> ASTNode | None:
        """Parse a top-level construct (part or implicit events)."""
        # Check if this is a part declaration
        if self._is_part_declaration():
            return self._parse_part()

        # Otherwise, parse as implicit events (events without explicit part)
        events = self._parse_event_sequence_content()
        if events:
            return EventSequenceNode(events=events, position=events[0].position)
        return None

    def _is_part_declaration(self) -> bool:
        """Check if current position starts a part declaration."""
        if not self._check(TokenType.NAME):
            return False

        # Look ahead for colon (part separator)
        pos = self._current
        while pos < len(self.tokens):
            token = self.tokens[pos]
            if token.type == TokenType.COLON:
                return True
            if token.type == TokenType.EQUALS:
                # This is a variable definition, not a part
                return False
            if token.type in (TokenType.SEPARATOR, TokenType.ALIAS):
                # Could be part of multi-instrument declaration
                pos += 1
                continue
            if token.type == TokenType.NAME:
                pos += 1
                continue
            break
        return False

    def _is_variable_definition(self) -> bool:
        """Check if current position starts a variable definition."""
        if not self._check(TokenType.NAME):
            return False
        # Look ahead for equals sign
        pos = self._current + 1
        if pos < len(self.tokens) and self.tokens[pos].type == TokenType.EQUALS:
            return True
        return False

    def _parse_part(self) -> PartNode:
        """Parse a part declaration and its events."""
        declaration = self._parse_part_declaration()
        self._skip_newlines()

        # Parse events until we hit another part or EOF
        events = self._parse_event_sequence_content()

        return PartNode(
            declaration=declaration,
            events=EventSequenceNode(events=events, position=declaration.position),
            position=declaration.position,
        )

    def _parse_part_declaration(self) -> PartDeclarationNode:
        """Parse a part declaration (e.g., 'piano:', 'violin/viola "strings":')."""
        position = self._peek().position
        names: list[str] = []

        # Parse first name
        name_token = self._consume(TokenType.NAME, "Expected instrument name")
        names.append(name_token.literal)

        # Parse additional names separated by /
        while self._match(TokenType.SEPARATOR):
            name_token = self._consume(
                TokenType.NAME, "Expected instrument name after '/'"
            )
            names.append(name_token.literal)

        # Optional alias
        alias = None
        if self._match(TokenType.ALIAS):
            alias = self.tokens[self._current - 1].literal

        # Consume colon
        self._consume(TokenType.COLON, "Expected ':' after part declaration")

        return PartDeclarationNode(names=names, alias=alias, position=position)

    def _parse_event_sequence_content(
        self, stop_tokens: set[TokenType] | None = None
    ) -> list[ASTNode]:
        """Parse events until we hit a stop token, part declaration, or EOF."""
        events: list[ASTNode] = []
        stop_tokens = stop_tokens or set()

        while not self._is_at_end() and not self._is_part_declaration():
            # Check for stop tokens
            if self._peek().type in stop_tokens:
                break

            self._skip_newlines()
            if self._is_at_end() or self._is_part_declaration():
                break
            if self._peek().type in stop_tokens:
                break

            event = self._parse_event()
            if event:
                events.append(event)

        return events

    def _parse_event(self) -> ASTNode | None:
        """Parse a single event."""
        # Skip newlines between events
        if self._match(TokenType.NEWLINE):
            return None

        event = self._parse_primary_event()
        if event is None:
            return None

        # Check for postfix operators (repeat, on-repetitions)
        event = self._parse_postfix(event)

        return event

    def _parse_primary_event(self) -> ASTNode | None:
        """Parse a primary event (without postfix operators)."""
        # Barline
        if self._match(TokenType.BARLINE):
            return BarlineNode(position=self.tokens[self._current - 1].position)

        # Octave controls
        if self._match(TokenType.OCTAVE_UP):
            return OctaveUpNode(position=self.tokens[self._current - 1].position)
        if self._match(TokenType.OCTAVE_DOWN):
            return OctaveDownNode(position=self.tokens[self._current - 1].position)
        if self._match(TokenType.OCTAVE_SET):
            token = self.tokens[self._current - 1]
            return OctaveSetNode(octave=token.literal, position=token.position)

        # S-expression
        if self._check(TokenType.LEFT_PAREN):
            return self._parse_sexp()

        # Markers
        if self._match(TokenType.MARKER):
            token = self.tokens[self._current - 1]
            return MarkerNode(name=token.literal, position=token.position)

        if self._match(TokenType.AT_MARKER):
            token = self.tokens[self._current - 1]
            return AtMarkerNode(name=token.literal, position=token.position)

        # Voice marker (starts a voice group)
        if self._check(TokenType.VOICE_MARKER):
            return self._parse_voice_group()

        # Cram expression
        if self._check(TokenType.CRAM_OPEN):
            return self._parse_cram()

        # Bracketed sequence
        if self._check(TokenType.EVENT_SEQ_OPEN):
            return self._parse_bracketed_sequence()

        # Variable definition or reference
        if self._is_variable_definition():
            return self._parse_variable_definition()

        if self._check(TokenType.NAME):
            return self._parse_variable_reference()

        # Rest
        if self._check(TokenType.REST_LETTER):
            return self._parse_rest()

        # Note (might become a chord)
        if self._check(TokenType.NOTE_LETTER):
            return self._parse_note_or_chord()

        # Unexpected token - skip it
        if not self._is_at_end():
            self._advance()
            return None

        return None

    def _parse_postfix(self, event: ASTNode) -> ASTNode:
        """Parse postfix operators (repeat, on-repetitions)."""
        # Check for repeat (*N)
        if self._match(TokenType.REPEAT):
            token = self.tokens[self._current - 1]
            times = token.literal
            event = RepeatNode(event=event, times=times, position=event.position)

        # Check for on-repetitions ('1-3,5)
        if self._match(TokenType.REPETITIONS):
            token = self.tokens[self._current - 1]
            ranges = self._parse_repetition_ranges(token.literal)
            event = OnRepetitionsNode(
                event=event, ranges=ranges, position=event.position
            )

        return event

    def _parse_repetition_ranges(self, ranges_str: str) -> list[RepetitionRange]:
        """Parse a repetition ranges string like '1-3,5' into RepetitionRange objects."""
        ranges = []
        for part in ranges_str.split(","):
            if "-" in part:
                first, last = part.split("-", 1)
                ranges.append(RepetitionRange(first=int(first), last=int(last)))
            else:
                ranges.append(RepetitionRange(first=int(part)))
        return ranges

    def _parse_variable_definition(self) -> VariableDefinitionNode:
        """Parse a variable definition (name = events)."""
        position = self._peek().position
        name_token = self._consume(TokenType.NAME, "Expected variable name")
        name = name_token.literal

        self._consume(TokenType.EQUALS, "Expected '=' after variable name")

        # Parse events on the same line or until newline
        events: list[ASTNode] = []
        while not self._is_at_end() and not self._check(TokenType.NEWLINE):
            if self._is_part_declaration():
                break
            event = self._parse_event()
            if event:
                events.append(event)

        return VariableDefinitionNode(
            name=name,
            events=EventSequenceNode(events=events, position=position),
            position=position,
        )

    def _parse_variable_reference(self) -> VariableReferenceNode:
        """Parse a variable reference."""
        token = self._consume(TokenType.NAME, "Expected variable name")
        return VariableReferenceNode(name=token.literal, position=token.position)

    def _parse_voice_group(self) -> VoiceGroupNode:
        """Parse a voice group (V1: events V2: events V0:)."""
        position = self._peek().position
        voices: list[VoiceNode] = []

        while self._check(TokenType.VOICE_MARKER):
            voice = self._parse_voice()
            if voice.number == 0:
                # V0: ends the voice group
                break
            voices.append(voice)

        return VoiceGroupNode(voices=voices, position=position)

    def _parse_voice(self) -> VoiceNode:
        """Parse a single voice."""
        token = self._consume(TokenType.VOICE_MARKER, "Expected voice marker")
        number = token.literal
        position = token.position

        if number == 0:
            # V0: is just an end marker, no events
            return VoiceNode(
                number=0,
                events=EventSequenceNode(events=[], position=position),
                position=position,
            )

        # Parse events until next voice marker or end of voice group context
        events = self._parse_event_sequence_content(
            stop_tokens={TokenType.VOICE_MARKER}
        )

        return VoiceNode(
            number=number,
            events=EventSequenceNode(events=events, position=position),
            position=position,
        )

    def _parse_cram(self) -> CramNode:
        """Parse a cram expression ({events}duration)."""
        position = self._peek().position
        self._consume(TokenType.CRAM_OPEN, "Expected '{'")

        # Parse events until closing brace
        events = self._parse_event_sequence_content(stop_tokens={TokenType.CRAM_CLOSE})

        self._consume(TokenType.CRAM_CLOSE, "Expected '}'")

        # Optional duration after the cram
        duration = self._try_parse_duration()

        return CramNode(
            events=EventSequenceNode(events=events, position=position),
            duration=duration,
            position=position,
        )

    def _parse_bracketed_sequence(self) -> BracketedSequenceNode:
        """Parse a bracketed event sequence ([events])."""
        position = self._peek().position
        self._consume(TokenType.EVENT_SEQ_OPEN, "Expected '['")

        # Parse events until closing bracket
        events = self._parse_event_sequence_content(
            stop_tokens={TokenType.EVENT_SEQ_CLOSE}
        )

        self._consume(TokenType.EVENT_SEQ_CLOSE, "Expected ']'")

        return BracketedSequenceNode(
            events=EventSequenceNode(events=events, position=position),
            position=position,
        )

    def _parse_note_or_chord(self) -> ASTNode:
        """Parse a note, which might be part of a chord."""
        first_note = self._parse_note()

        # Check if this is a chord (followed by / and more notes)
        if not self._check(TokenType.SEPARATOR):
            return first_note

        # Parse as chord
        chord_elements: list[
            NoteNode
            | RestNode
            | OctaveUpNode
            | OctaveDownNode
            | OctaveSetNode
            | LispListNode
        ] = [first_note]

        while self._match(TokenType.SEPARATOR):
            # Allow octave changes between chord notes
            while (
                self._check(TokenType.OCTAVE_UP)
                or self._check(TokenType.OCTAVE_DOWN)
                or self._check(TokenType.OCTAVE_SET)
                or self._check(TokenType.LEFT_PAREN)
            ):
                if self._match(TokenType.OCTAVE_UP):
                    chord_elements.append(
                        OctaveUpNode(position=self.tokens[self._current - 1].position)
                    )
                elif self._match(TokenType.OCTAVE_DOWN):
                    chord_elements.append(
                        OctaveDownNode(position=self.tokens[self._current - 1].position)
                    )
                elif self._match(TokenType.OCTAVE_SET):
                    token = self.tokens[self._current - 1]
                    chord_elements.append(
                        OctaveSetNode(octave=token.literal, position=token.position)
                    )
                elif self._check(TokenType.LEFT_PAREN):
                    chord_elements.append(self._parse_sexp())

            if self._check(TokenType.NOTE_LETTER):
                note = self._parse_note()
                chord_elements.append(note)
            elif self._check(TokenType.REST_LETTER):
                # Rests can appear in chords too
                rest = self._parse_rest()
                chord_elements.append(rest)

        return ChordNode(notes=chord_elements, position=first_note.position)

    def _parse_note(self) -> NoteNode:
        """Parse a single note with optional accidentals and duration."""
        position = self._peek().position
        letter_token = self._consume(TokenType.NOTE_LETTER, "Expected note letter")
        letter = letter_token.literal

        # Parse accidentals
        accidentals: list[str] = []
        while True:
            if self._match(TokenType.SHARP):
                accidentals.append("+")
            elif self._match(TokenType.FLAT):
                accidentals.append("-")
            elif self._match(TokenType.NATURAL):
                accidentals.append("_")
            else:
                break

        # Parse optional duration
        duration = self._try_parse_duration()

        # Check for slur/tie to next note
        slurred = self._match(TokenType.TIE)

        return NoteNode(
            letter=letter,
            accidentals=accidentals,
            duration=duration,
            slurred=slurred,
            position=position,
        )

    def _parse_rest(self) -> RestNode:
        """Parse a rest with optional duration."""
        position = self._peek().position
        self._consume(TokenType.REST_LETTER, "Expected 'r'")

        duration = self._try_parse_duration()

        return RestNode(duration=duration, position=position)

    def _try_parse_duration(self) -> DurationNode | None:
        """Try to parse a duration. Returns None if no duration present."""
        if not self._is_duration_start():
            return None

        components = []
        position = self._peek().position

        while self._is_duration_start():
            component = self._parse_duration_component()
            components.append(component)

            # Check for tie connecting to another duration component
            if self._check(TokenType.TIE) and self._is_duration_start_at(
                self._current + 1
            ):
                self._advance()  # consume tie
            else:
                break

        if not components:
            return None

        return DurationNode(components=components, position=position)

    def _is_duration_start(self) -> bool:
        """Check if current token starts a duration."""
        return (
            self._check(TokenType.NOTE_LENGTH)
            or self._check(TokenType.NOTE_LENGTH_MS)
            or self._check(TokenType.NOTE_LENGTH_SECONDS)
        )

    def _is_duration_start_at(self, pos: int) -> bool:
        """Check if token at position starts a duration."""
        if pos >= len(self.tokens):
            return False
        token = self.tokens[pos]
        return token.type in (
            TokenType.NOTE_LENGTH,
            TokenType.NOTE_LENGTH_MS,
            TokenType.NOTE_LENGTH_SECONDS,
        )

    def _parse_duration_component(self) -> ASTNode:
        """Parse a single duration component."""
        if self._match(TokenType.NOTE_LENGTH_MS):
            token = self.tokens[self._current - 1]
            return NoteLengthMsNode(ms=token.literal, position=token.position)

        if self._match(TokenType.NOTE_LENGTH_SECONDS):
            token = self.tokens[self._current - 1]
            return NoteLengthSecondsNode(seconds=token.literal, position=token.position)

        if self._match(TokenType.NOTE_LENGTH):
            token = self.tokens[self._current - 1]
            denominator = token.literal

            # Count dots
            dots = 0
            while self._match(TokenType.DOT):
                dots += 1

            return NoteLengthNode(
                denominator=denominator, dots=dots, position=token.position
            )

        self._error("Expected duration component")

    def _parse_sexp(self) -> LispListNode:
        """Parse an S-expression."""
        position = self._peek().position
        self._consume(TokenType.LEFT_PAREN, "Expected '('")

        elements = []
        while not self._check(TokenType.RIGHT_PAREN) and not self._is_at_end():
            element = self._parse_lisp_element()
            if element:
                elements.append(element)

        self._consume(TokenType.RIGHT_PAREN, "Expected ')'")

        return LispListNode(elements=elements, position=position)

    def _parse_lisp_element(self) -> ASTNode | None:
        """Parse a single Lisp element."""
        if self._match(TokenType.SYMBOL):
            token = self.tokens[self._current - 1]
            return LispSymbolNode(name=token.literal, position=token.position)

        if self._match(TokenType.NUMBER):
            token = self.tokens[self._current - 1]
            return LispNumberNode(value=token.literal, position=token.position)

        if self._match(TokenType.STRING):
            token = self.tokens[self._current - 1]
            return LispStringNode(value=token.literal, position=token.position)

        if self._check(TokenType.LEFT_PAREN):
            return self._parse_sexp()

        return None

    # Helper methods

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _peek(self) -> Token:
        return self.tokens[self._current]

    def _peek_next(self) -> Token:
        if self._current + 1 >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self._current + 1]

    def _advance(self) -> Token:
        if not self._is_at_end():
            self._current += 1
        return self.tokens[self._current - 1]

    def _check(self, token_type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == token_type

    def _match(self, token_type: TokenType) -> bool:
        if self._check(token_type):
            self._advance()
            return True
        return False

    def _consume(self, token_type: TokenType, message: str) -> Token:
        if self._check(token_type):
            return self._advance()
        self._error(message)

    def _skip_newlines(self) -> None:
        while self._match(TokenType.NEWLINE):
            pass

    def _error(self, message: str) -> NoReturn:
        token = self._peek()
        raise AldaSyntaxError(message, token.position)


def parse(source: str, filename: str = "<input>") -> RootNode:
    """Convenience function to parse Alda source code."""
    parser = Parser.from_source(source, filename)
    return parser.parse()
