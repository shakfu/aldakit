"""Interactive REPL for aldakit with syntax highlighting and completion."""

import time
from pathlib import Path

# Initialize vendored packages path (must be before prompt_toolkit imports)
from . import ext  # noqa: F401

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

from .errors import AldaParseError
from .midi.backends import LibremidiBackend
from .midi.generator import generate_midi
from .midi.types import INSTRUMENT_PROGRAMS
from .parser import parse

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
    """Auto-completion for alda."""

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

    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor()
        line = document.current_line_before_cursor.strip()

        # Only complete instruments if:
        # - At start of line (no content yet), OR
        # - Word is at least 3 chars (to avoid matching notes)
        if ":" not in line and len(word) >= 3:
            for inst in self.instruments:
                if inst.startswith(word):
                    yield Completion(inst + ": ", start_position=-len(word))

        # Complete attributes after (
        if "(" in line and ")" not in line[line.rfind("(") :]:
            for attr in self.ATTRIBUTES:
                if attr.startswith("(" + word):
                    yield Completion(attr, start_position=-len(word) - 1)


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


def run_repl(port_name: str | None = None, verbose: bool = False) -> int:
    """Run the interactive alda REPL."""
    backend = LibremidiBackend(port_name=port_name)
    backend._ensure_port_open()

    history_file = Path.home() / ".alda_history"

    session = PromptSession(
        history=FileHistory(str(history_file)),
        lexer=AldaLexer(),
        completer=AldaCompleter(),
        style=ALDA_STYLE,
        key_bindings=create_key_bindings(backend),
        multiline=False,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "  ... ",
    )

    # State
    default_tempo = 120

    print("aldakit REPL - AldakitMIDI port open")
    print("Enter alda code, press Enter to play. Alt+Enter for multi-line.")
    print("Type :help for commands, Ctrl+D to exit.")
    print()

    try:
        while True:
            try:
                source = session.prompt("aldakit> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                continue

            if not source:
                continue

            # Commands
            if source.startswith(":"):
                parts = source[1:].split(None, 1)
                cmd = parts[0].lower() if parts else ""
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd in ("h", "help", "?"):
                    print("Commands:")
                    print("  :q :quit :exit    - Exit REPL")
                    print("  :help :h :?       - Show this help")
                    print("  :ports            - List MIDI ports")
                    print("  :instruments      - List instruments")
                    print("  :tempo [BPM]      - Show/set default tempo")
                    print("  :stop             - Stop playback")
                    print()
                    print("Shortcuts:")
                    print("  Alt+Enter         - Multi-line input")
                    print("  Ctrl+C            - Stop playback / cancel")
                    print("  Ctrl+D            - Exit")
                    print("  Tab               - Auto-complete")
                    print("  Up/Down           - History")
                elif cmd == "ports":
                    ports = backend.list_output_ports()
                    if ports:
                        for i, p in enumerate(ports):
                            print(f"  {i}: {p}")
                    else:
                        print("  (no ports - using virtual AldakitMIDI)")
                elif cmd == "instruments":
                    insts = sorted(INSTRUMENT_PROGRAMS.keys())
                    # Print in columns
                    cols = 4
                    for i in range(0, len(insts), cols):
                        row = insts[i : i + cols]
                        print("  " + "  ".join(f"{inst:20}" for inst in row))
                elif cmd == "tempo":
                    if arg:
                        try:
                            default_tempo = int(arg)
                            print(f"Default tempo: {default_tempo} BPM")
                        except ValueError:
                            print("Invalid tempo")
                    else:
                        print(f"Default tempo: {default_tempo} BPM")
                elif cmd == "stop":
                    backend.stop()
                    print("Stopped")
                else:
                    print(f"Unknown command: :{cmd}")
                continue

            # Add default tempo if not specified
            if "(tempo" not in source.lower():
                source = f"(tempo {default_tempo}) {source}"

            try:
                ast = parse(source, "<repl>")
                sequence = generate_midi(ast)

                if not sequence.notes:
                    print("(no notes)")
                    continue

                if verbose:
                    print(f"{len(sequence.notes)} notes, {sequence.duration():.2f}s")

                backend.play(sequence)
                while backend.is_playing():
                    time.sleep(0.05)

            except AldaParseError as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        pass

    backend.close()
    print("Goodbye!")
    return 0
