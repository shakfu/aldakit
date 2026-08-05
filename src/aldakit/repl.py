"""Interactive REPL for aldakit with syntax highlighting and completion."""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    MAX_PLAYBACK_SLOTS,
    POLL_INTERVAL_DEFAULT,
    REPL_COMPLETION_MIN_WORD_LENGTH,
    REPL_CONTINUATION_PROMPT,
    REPL_HISTORY_FILENAME,
    REPL_INSTRUMENT_COLUMNS,
    REPL_PROMPT,
)
from .errors import AldaParseError
from .midi.backends import LibremidiBackend
from .midi.generator import generate_midi
from .midi.smf import write_midi_file
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


# REPL commands that take a filesystem path as their argument
PATH_COMMANDS = ("load", "play", "save", "cd")

# Commands offered by name completion at the start of a line
COMMAND_NAMES = (
    "load",
    "play",
    "save",
    "ls",
    "cd",
    "pwd",
    "ports",
    "instruments",
    "tempo",
    "stop",
    "status",
    "concurrent",
    "sequential",
    "help",
    "quit",
)


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


class ReplSession:
    """Tracks the loaded score and everything entered during the session.

    Two distinct things are kept:

    - ``sources``: every piece of Alda accepted -- typed, pasted or loaded --
      which is what ``:save`` writes. This is the session's *source*, not a
      recording of what was heard: in concurrent mode inputs are layered as
      they play, whereas the saved document is one score read top to bottom.
    - ``buffer``: the most recently loaded score, which ``:play`` plays.
      Loading does not play, so a file can be inspected or saved before it is
      heard.
    """

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.buffer: str | None = None
        self.buffer_name: str | None = None

    def add(self, source: str) -> None:
        self.sources.append(source.strip())

    def set_buffer(self, source: str, name: str) -> None:
        """Record ``source`` as the loaded score and add it to the session."""
        self.buffer = source
        self.buffer_name = name
        self.add(source)

    @property
    def has_buffer(self) -> bool:
        return self.buffer is not None

    @property
    def is_empty(self) -> bool:
        return not self.sources

    def to_alda(self) -> str:
        return "\n\n".join(self.sources) + "\n"

    def clear(self) -> None:
        self.sources.clear()
        self.buffer = None
        self.buffer_name = None


def describe_source(source: str) -> str:
    """Summarize a score for the load message.

    Returns:
        A short description such as ``"3 parts, 42 notes, 28.7s"``.

    Raises:
        AldaParseError: If the source does not parse.
    """
    from .ast_nodes import PartNode

    ast = parse(source, "<load>")
    sequence = generate_midi(ast)
    parts = sum(1 for child in ast.children if isinstance(child, PartNode))

    pieces = []
    if parts:
        pieces.append(f"{parts} part{'s' if parts != 1 else ''}")
    pieces.append(f"{len(sequence.notes)} note{'s' if len(sequence.notes) != 1 else ''}")
    pieces.append(f"{sequence.duration():.1f}s")
    return ", ".join(pieces)


def _resolve_path(argument: str) -> Path:
    """Expand ``~`` and make a user-supplied path absolute."""
    return Path(argument).expanduser().resolve()


def load_file(path: Path) -> str:
    """Read Alda source from ``path``.

    Args:
        path: File to read. A missing ``.alda`` suffix is tried as a fallback,
            so ``:load twinkle`` finds ``twinkle.alda``.

    Returns:
        The file's contents.

    Raises:
        FileNotFoundError: If neither the given path nor the ``.alda`` variant
            exists.
        IsADirectoryError: If the path is a directory.
    """
    if not path.exists() and not path.suffix:
        candidate = path.with_suffix(".alda")
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Not a file: {path}")

    return path.read_text(encoding="utf-8")


def list_directory(directory: Path) -> tuple[list[str], list[str]]:
    """Return the subdirectories and Alda files in ``directory``.

    Hidden entries are omitted, as are files that are not Alda sources: the
    listing exists to find something to ``:load``.
    """
    directories: list[str] = []
    files: list[str] = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            directories.append(entry.name + "/")
        elif entry.suffix.lower() == ".alda":
            files.append(entry.name)
    return directories, files


@dataclass
class ReplContext:
    """Mutable state a REPL command may read or change.

    Extracted so that command handling can be exercised without a terminal:
    ``PromptSession`` requires a TTY, which would otherwise make every command
    untestable.
    """

    backend: object
    session: ReplSession
    play: Callable[..., bool]
    supports_concurrent: bool = True
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME
    default_tempo: int = DEFAULT_TEMPO
    running: bool = True


def print_help() -> None:
    """Print the REPL command reference."""
    print("Commands:")
    print("  :q :quit :exit    - Exit REPL")
    print("  :help :h :?       - Show this help")
    print("  :load FILE        - Load an Alda file (does not play)")
    print("  :play [FILE]      - Play the loaded score, or load and play FILE")
    print("  :save FILE        - Save this session (.alda or .mid)")
    print("  :ls [DIR]         - List Alda files and directories")
    print("  :cd [DIR]         - Change directory")
    print("  :pwd              - Show current directory")
    print("  :clear            - Forget the session so far")
    print("  :ports            - List MIDI ports")
    print("  :instruments      - List instruments")
    print("  :tempo [BPM]      - Show/set default tempo")
    print("  :stop             - Stop playback")
    print("  :status           - Show playback status")
    print("  :concurrent       - Enable concurrent mode (layer inputs)")
    print("  :sequential       - Enable sequential mode (wait for each)")
    print()
    print("Shortcuts:")
    print("  Alt+Enter         - Multi-line input")
    print("  Ctrl+C            - Stop playback / cancel")
    print("  Ctrl+D            - Exit")
    print("  Tab               - Auto-complete (commands, files, notes)")
    print("  Up/Down           - History")


def load_into_session(ctx: ReplContext, name: str) -> bool:
    """Read a file into the session buffer without playing it.

    Loading is deliberately silent: the file becomes the score that ``:play``
    plays, and can be saved or inspected first.

    Args:
        ctx: REPL state.
        name: Path as the user typed it.

    Returns:
        True if the file was read and parsed.
    """
    try:
        contents = load_file(_resolve_path(name))
    except (FileNotFoundError, IsADirectoryError, OSError) as e:
        print(f"Error: {e}")
        return False

    try:
        summary = describe_source(contents)
    except AldaParseError as e:
        print(f"Error: {e}")
        return False

    ctx.session.set_buffer(contents, name)
    print(f"Loaded {name} ({summary})")
    print("Type :play to hear it.")
    return True


def _cmd_load(ctx: ReplContext, arg: str) -> None:
    if not arg:
        print("Usage: :load FILE")
        return
    load_into_session(ctx, arg)


def _cmd_play(ctx: ReplContext, arg: str) -> None:
    """Play the loaded score, or load and play the file named by ``arg``."""
    if arg and not load_into_session(ctx, arg):
        return

    if not ctx.session.has_buffer:
        print("Nothing loaded. Use :load FILE first.")
        return

    print(f"Playing {ctx.session.buffer_name}...")
    # A loaded score sets its own tempo, often per part; do not impose the
    # REPL's default on top of it.
    ctx.play(ctx.session.buffer, apply_default_tempo=False, record=False)


def _cmd_save(ctx: ReplContext, arg: str) -> None:
    if not arg:
        print("Usage: :save FILE")
        return
    if ctx.session.is_empty:
        print("Nothing to save yet.")
        return

    target = _resolve_path(arg)
    try:
        if target.suffix.lower() in (".mid", ".midi"):
            sequence = generate_midi(parse(ctx.session.to_alda(), "<session>"))
            write_midi_file(sequence, target)
        else:
            if not target.suffix:
                target = target.with_suffix(".alda")
            target.write_text(ctx.session.to_alda(), encoding="utf-8")
    except AldaParseError as e:
        print(f"Error: session does not parse as a single score: {e}")
    except OSError as e:
        print(f"Error: {e}")
    else:
        print(f"Saved {target}")


def _cmd_ls(ctx: ReplContext, arg: str) -> None:
    directory = _resolve_path(arg) if arg else Path.cwd()
    try:
        directories, files = list_directory(directory)
    except (FileNotFoundError, NotADirectoryError, OSError) as e:
        print(f"Error: {e}")
        return
    if not directories and not files:
        print("  (no directories or .alda files here)")
        return
    for name in directories + files:
        print(f"  {name}")


def _cmd_cd(ctx: ReplContext, arg: str) -> None:
    target = _resolve_path(arg) if arg else Path.home()
    try:
        os.chdir(target)
    except (FileNotFoundError, NotADirectoryError, OSError) as e:
        print(f"Error: {e}")
    else:
        print(Path.cwd())


def _cmd_ports(ctx: ReplContext, arg: str) -> None:
    if not ctx.supports_concurrent:
        print("  (using TinySoundFont audio backend)")
        return
    ports = ctx.backend.list_output_ports()
    if ports:
        for i, name in enumerate(ports):
            print(f"  {i}: {name}")
    else:
        print(f"  (no ports - using virtual {ctx.virtual_port_name})")


def _cmd_instruments(ctx: ReplContext, arg: str) -> None:
    names = sorted(INSTRUMENT_PROGRAMS.keys())
    cols = REPL_INSTRUMENT_COLUMNS
    for i in range(0, len(names), cols):
        print("  " + "  ".join(f"{name:28}" for name in names[i : i + cols]))


def _cmd_tempo(ctx: ReplContext, arg: str) -> None:
    if arg:
        try:
            ctx.default_tempo = int(arg)
        except ValueError:
            print("Invalid tempo")
            return
    print(f"Default tempo: {ctx.default_tempo} BPM")


def _cmd_status(ctx: ReplContext, arg: str) -> None:
    playing = "playing" if ctx.backend.is_playing() else "idle"
    if ctx.supports_concurrent:
        mode = "concurrent" if ctx.backend.concurrent_mode else "sequential"
        print("Backend: MIDI (libremidi)")
        print(f"Mode: {mode}")
        print(f"Status: {playing}")
        print(f"Active slots: {ctx.backend.active_slots}/{MAX_PLAYBACK_SLOTS}")
    else:
        print("Backend: Audio (TinySoundFont)")
        print(f"Status: {playing}")
    loaded = ctx.session.buffer_name or "(nothing)"
    print(f"Loaded: {loaded}")
    print(f"Session: {len(ctx.session.sources)} entries")


def _cmd_concurrent(ctx: ReplContext, arg: str) -> None:
    if ctx.supports_concurrent:
        ctx.backend.concurrent_mode = True
        print("Concurrent mode enabled - inputs will layer on each other")
    else:
        print("Concurrent mode not available with audio backend")


def _cmd_sequential(ctx: ReplContext, arg: str) -> None:
    if ctx.supports_concurrent:
        ctx.backend.concurrent_mode = False
        print("Sequential mode enabled - each input waits for previous")
    else:
        print("Audio backend always uses sequential mode")


def handle_command(ctx: ReplContext, source: str) -> None:
    """Execute a ``:command`` line.

    Args:
        ctx: REPL state. Commands mutate it in place; ``ctx.running`` is set to
            False by the quit commands.
        source: The full input line, including the leading colon.
    """
    parts = source[1:].split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("q", "quit", "exit"):
        ctx.running = False
        return
    if cmd in ("h", "help", "?"):
        print_help()
        return
    if cmd == "pwd":
        print(Path.cwd())
        return
    if cmd == "clear":
        ctx.session.clear()
        print("Session cleared.")
        return
    if cmd == "stop":
        ctx.backend.stop()
        print("Stopped")
        return

    handlers = {
        "load": _cmd_load,
        "play": _cmd_play,
        "save": _cmd_save,
        "ls": _cmd_ls,
        "cd": _cmd_cd,
        "ports": _cmd_ports,
        "instruments": _cmd_instruments,
        "tempo": _cmd_tempo,
        "status": _cmd_status,
        "concurrent": _cmd_concurrent,
        "sequential": _cmd_sequential,
    }
    handler = handlers.get(cmd)
    if handler is None:
        print(f"Unknown command: :{cmd}")
        return
    handler(ctx, arg)


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
    # Check for MIDI ports if not using audio
    if not use_audio and port_name is None:
        test_backend = LibremidiBackend()
        ports = test_backend.list_output_ports()
        if not ports:
            # No MIDI ports - fall back to audio if soundfont is configured,
            # otherwise let the backend create a virtual port (AldakitMIDI)
            if soundfont:
                use_audio = True

    if use_audio:
        from .midi.backends import TsfBackend, HAS_TSF

        if not HAS_TSF:
            print("Error: Audio backend not available. The _tsf module was not built.")
            return 1

        try:
            backend = TsfBackend(soundfont=soundfont)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

        backend_name = "TinySoundFont"
        # TsfBackend doesn't support concurrent mode
        supports_concurrent = False
    else:
        backend = LibremidiBackend(
            port_name=port_name,
            concurrent=concurrent,
            virtual_port_name=virtual_port_name,
        )
        backend._ensure_port_open()
        backend_name = virtual_port_name
        supports_concurrent = True

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

    # State (default_tempo passed as parameter)

    if supports_concurrent:
        mode_str = "concurrent" if backend.concurrent_mode else "sequential"
        print(f"aldakit REPL - {backend_name} port open ({mode_str} mode)")
    else:
        print(f"aldakit REPL - {backend_name} audio backend")
    print("Enter alda code, press Enter to play. Alt+Enter for multi-line.")
    print("Type :help for commands, Ctrl+D to exit.")
    print()

    repl_session = ReplSession()

    def play_source(
        source: str,
        *,
        apply_default_tempo: bool = True,
        record: bool = True,
    ) -> bool:
        """Parse, play and optionally record a piece of Alda source.

        Args:
            source: Alda source code.
            apply_default_tempo: If True, prepend the REPL's default tempo when
                the source does not set one. Loaded scores opt out: they
                normally set their own tempo, often per part, and prefixing one
                would override it.
            record: If True, add the source to the session. Replaying the
                loaded buffer sets this False so :save does not duplicate it.

        Returns:
            True if the source parsed and produced notes.
        """
        to_play = source
        if apply_default_tempo and "(tempo" not in source.lower():
            to_play = f"(tempo {ctx.default_tempo}) {source}"

        try:
            ast = parse(to_play, "<repl>")
            sequence = generate_midi(ast)
        except AldaParseError as e:
            print(f"Error: {e}")
            return False

        if not sequence.notes:
            print("(no notes)")
            if record:
                repl_session.add(source)
            return False

        if verbose:
            print(f"{len(sequence.notes)} notes, {sequence.duration():.2f}s")

        if record:
            repl_session.add(source)
        slot_id = backend.play(sequence)

        if slot_id is None:
            print("(all playback slots busy - use :stop to clear)")
        elif not supports_concurrent or not backend.concurrent_mode:
            # In sequential mode (or audio backend), wait for playback
            while backend.is_playing():
                time.sleep(POLL_INTERVAL_DEFAULT)
        # In concurrent mode, return immediately to accept next input
        return True

    ctx = ReplContext(
        backend=backend,
        session=repl_session,
        play=play_source,
        supports_concurrent=supports_concurrent,
        virtual_port_name=virtual_port_name,
        default_tempo=default_tempo,
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

            if source.startswith(":"):
                handle_command(ctx, source)
                if not ctx.running:
                    break
                continue
            play_source(source)

    except KeyboardInterrupt:
        pass

    # Clean up backend
    if supports_concurrent:
        backend.close()
    else:
        backend.stop()
    print("Goodbye!")
    return 0
