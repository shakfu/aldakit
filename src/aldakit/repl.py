"""Interactive REPL for aldakit with syntax highlighting and completion."""

from pathlib import Path

# Initialize vendored packages path (must be before prompt_toolkit imports)
from . import ext  # noqa: F401

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from .constants import (
    DEFAULT_TEMPO,
    DEFAULT_VIRTUAL_PORT_NAME,
    REPL_COMMAND_NAMES,
    REPL_COMPLETION_MIN_WORD_LENGTH,
    REPL_CONTINUATION_PROMPT,
    REPL_HISTORY_FILENAME,
    REPL_PATH_COMMANDS,
    REPL_PROMPT,
)
from .midi.types import INSTRUMENT_PROGRAMS
from .repl_core import (
    BackendUnavailable,
    ConcurrentBackend,
    ReplBackend,
    ReplContext,
    ReplSession,
    build_context,
    close_backend,
    describe_source,
    dispatch,
    handle_command,
    list_directory,
    load_file,
    load_into_session,
    open_backend,
    print_banner,
    print_help,
)

# Kept as module attributes so existing importers of ``aldakit.repl`` continue
# to work now that the tables live in constants.
COMMAND_NAMES = REPL_COMMAND_NAMES
PATH_COMMANDS = REPL_PATH_COMMANDS

__all__ = [
    "ALDA_STYLE",
    "AldaCompleter",
    "AldaLexer",
    "BackendUnavailable",
    "COMMAND_NAMES",
    "ConcurrentBackend",
    "PATH_COMMANDS",
    "ReplBackend",
    "ReplContext",
    "ReplSession",
    "create_key_bindings",
    "describe_source",
    "handle_command",
    "list_directory",
    "load_file",
    "load_into_session",
    "print_help",
    "run_repl",
]

# Alda token colors - clean scheme
ALDA_STYLE = Style.from_dict(
    {
        "note": "#ffffff",  # white - notes
        "rest": "#888888",  # gray - rests
        "octave": "#cc99ff",  # light purple - octave changes
        "duration": "#66ccff",  # light blue - durations
        "instrument": "#ff99cc bold",  # pink bold - instruments
        "attribute": "#99cc99",  # sage green - attributes
        "barline": "#555555",  # dark gray
        "comment": "#666666 italic",  # comments
    }
)


class AldaLexer(Lexer):
    """Syntax highlighter for alda code."""

    def lex_document(self, document: Document):
        def get_line_tokens(line_number):
            line = document.lines[line_number]
            tokens = []
            i = 0
            while i < len(line):
                ch = line[i]

                # Comments
                if ch == "#":
                    tokens.append(("class:comment", line[i:]))
                    break

                # Instrument/part declaration (word followed by :)
                # Look ahead to check for colon
                if ch.isalpha():
                    j = i
                    while j < len(line) and (line[j].isalnum() or line[j] == "-"):
                        j += 1
                    if j < len(line) and line[j] == ":":
                        # This is an instrument declaration
                        tokens.append(("class:instrument", line[i : j + 1]))
                        i = j + 1
                        continue
                    # Not followed by colon - check if it's a note/rest/octave
                    # (handled below by continuing the loop)

                # S-expressions (tempo, volume, etc.)
                if ch == "(":
                    j = i + 1
                    depth = 1
                    while j < len(line) and depth > 0:
                        if line[j] == "(":
                            depth += 1
                        elif line[j] == ")":
                            depth -= 1
                        j += 1
                    tokens.append(("class:attribute", line[i:j]))
                    i = j
                    continue

                # Notes (with optional accidentals and duration)
                if ch in "abcdefg":
                    j = i + 1
                    # Accidentals
                    while j < len(line) and line[j] in "+-_":
                        j += 1
                    tokens.append(("class:note", line[i:j]))
                    i = j
                    # Duration (separate token)
                    if i < len(line) and (line[i].isdigit() or line[i] == "."):
                        j = i
                        while j < len(line) and (line[j].isdigit() or line[j] == "."):
                            j += 1
                        # ms or s suffix
                        if j + 1 < len(line) and line[j : j + 2] == "ms":
                            j += 2
                        elif (
                            j < len(line)
                            and line[j] == "s"
                            and (j + 1 >= len(line) or not line[j + 1].isalpha())
                        ):
                            j += 1
                        tokens.append(("class:duration", line[i:j]))
                        i = j
                    continue

                # Rest (with optional duration)
                if ch == "r" and (
                    i + 1 >= len(line) or line[i + 1] not in "abcdefghijklmnopqstuvwxyz"
                ):
                    tokens.append(("class:rest", ch))
                    i += 1
                    # Duration (separate token)
                    if i < len(line) and (line[i].isdigit() or line[i] == "."):
                        j = i
                        while j < len(line) and (line[j].isdigit() or line[j] == "."):
                            j += 1
                        tokens.append(("class:duration", line[i:j]))
                        i = j
                    continue

                # Octave set (o followed by number)
                if ch == "o" and i + 1 < len(line) and line[i + 1].isdigit():
                    j = i + 1
                    while j < len(line) and line[j].isdigit():
                        j += 1
                    tokens.append(("class:octave", line[i:j]))
                    i = j
                    continue

                # Octave up/down
                if ch in "<>":
                    tokens.append(("class:octave", ch))
                    i += 1
                    continue

                # Barline
                if ch == "|":
                    tokens.append(("class:barline", ch))
                    i += 1
                    continue

                # Chord markers
                if ch == "/":
                    tokens.append(("class:note", ch))
                    i += 1
                    continue

                # Default (whitespace, etc.)
                tokens.append(("", ch))
                i += 1

            return tokens

        return get_line_tokens


class AldaCompleter(Completer):
    """Auto-completion for alda source, REPL commands and file paths."""

    ATTRIBUTES = [
        "(tempo ",
        "(volume ",
        "(quant ",
        "(key-sig ",
        "(pan ",
        "(panning ",
        "(track-vol ",
    ]

    def __init__(self):
        self.instruments = sorted(INSTRUMENT_PROGRAMS.keys())
        # Directories are offered for :cd; .alda files for :load and :save
        self._paths = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        line = document.current_line_before_cursor
        stripped = line.lstrip()

        # Commands take over the line entirely
        if stripped.startswith(":"):
            yield from self._command_completions(document, complete_event, stripped)
            return

        word = document.get_word_before_cursor()

        # Only complete instruments if:
        # - At start of line (no content yet), OR
        # - Word is at least 3 chars (to avoid matching notes)
        if ":" not in line.strip() and len(word) >= REPL_COMPLETION_MIN_WORD_LENGTH:
            for inst in self.instruments:
                if inst.startswith(word):
                    yield Completion(inst + ": ", start_position=-len(word))

        # Complete attributes after (
        if "(" in line.strip() and ")" not in line.strip()[line.strip().rfind("(") :]:
            for attr in self.ATTRIBUTES:
                if attr.startswith("(" + word):
                    yield Completion(attr, start_position=-len(word) - 1)

    def _command_completions(self, document, complete_event, stripped: str):
        """Complete a ``:command`` and, where relevant, its path argument."""
        body = stripped[1:]

        if " " not in body:
            # Still typing the command name
            for name in COMMAND_NAMES:
                if name.startswith(body):
                    suffix = " " if name in PATH_COMMANDS else ""
                    yield Completion(name + suffix, start_position=-len(body))
            return

        command, _, argument = body.partition(" ")
        if command not in PATH_COMMANDS:
            return

        # Delegate to prompt_toolkit's path completer, re-based onto the
        # argument so its start_position lines up with the real cursor.
        sub_document = Document(argument, cursor_position=len(argument))
        for completion in self._paths.get_completions(sub_document, complete_event):
            yield completion


def create_key_bindings(backend):
    """Create custom key bindings."""
    kb = KeyBindings()

    @kb.add(Keys.Escape, Keys.Enter)
    @kb.add(Keys.ControlJ)  # Ctrl+J as alternative for multi-line
    def _(event):
        """Insert newline for multi-line input."""
        event.current_buffer.insert_text("\n")

    @kb.add(Keys.ControlC)
    def _(event):
        """Stop playback on Ctrl+C."""
        if backend.is_playing():
            backend.stop()
        else:
            event.app.exit(exception=KeyboardInterrupt)

    return kb


def run_repl(
    port_name: str | None = None,
    verbose: bool = False,
    concurrent: bool = True,
    use_audio: bool = False,
    soundfont: str | None = None,
    default_tempo: int = DEFAULT_TEMPO,
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
    initial_file: str | Path | None = None,
) -> int:
    """Run the interactive alda REPL.

    Args:
        port_name: MIDI output port name (None for default/virtual).
        verbose: If True, print note counts and durations.
        concurrent: If True (default), enable concurrent playback mode
            where multiple inputs layer on top of each other.
        use_audio: If True, use TinySoundFont audio backend instead of MIDI.
        soundfont: Path to SoundFont file (for audio backend).
        default_tempo: Default tempo in BPM (default: DEFAULT_TEMPO).
        virtual_port_name: Name for virtual MIDI port (default: DEFAULT_VIRTUAL_PORT_NAME).
        initial_file: Alda file to load and play before the first prompt.
    """
    try:
        backend, backend_name, supports_concurrent = open_backend(
            port_name=port_name,
            concurrent=concurrent,
            use_audio=use_audio,
            soundfont=soundfont,
            virtual_port_name=virtual_port_name,
        )
    except BackendUnavailable as error:
        print(f"Error: {error}")
        return 1

    history_file = Path.home() / REPL_HISTORY_FILENAME

    session = PromptSession(
        history=FileHistory(str(history_file)),
        lexer=AldaLexer(),
        completer=AldaCompleter(),
        style=ALDA_STYLE,
        key_bindings=create_key_bindings(backend),
        multiline=False,
        prompt_continuation=lambda width,
        line_number,
        is_soft_wrap: REPL_CONTINUATION_PROMPT,
    )

    print_banner(backend, backend_name, supports_concurrent)

    ctx = build_context(
        backend,
        supports_concurrent=supports_concurrent,
        verbose=verbose,
        default_tempo=default_tempo,
        virtual_port_name=virtual_port_name,
    )

    # Load a file given on the command line. It is not played: the REPL opens
    # ready to go, and :play starts it.
    if initial_file is not None:
        load_into_session(ctx, str(initial_file))
        print()

    try:
        while True:
            try:
                source = session.prompt(REPL_PROMPT).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                continue

            if not source:
                continue

            dispatch(ctx, source)
            if not ctx.running:
                break

    except KeyboardInterrupt:
        pass

    close_backend(backend, supports_concurrent)
    print("Goodbye!")
    return 0
