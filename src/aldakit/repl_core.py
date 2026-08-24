"""REPL session state, commands, and backend wiring.

Everything here is free of any terminal frontend. Both the prompt_toolkit REPL
in :mod:`aldakit.repl` and the standard-library REPL in
:mod:`aldakit.repl_terminal` build on this module, so command behaviour and
session semantics cannot drift between them.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, cast

from .constants import (
    DEFAULT_TEMPO,
    DEFAULT_VIRTUAL_PORT_NAME,
    MAX_PLAYBACK_SLOTS,
    POLL_INTERVAL_DEFAULT,
    REPL_INSTRUMENT_COLUMNS,
)
from .errors import AldaParseError
from .midi.backends import LibremidiBackend
from .midi.generator import generate_midi
from .midi.smf import write_midi_file
from .midi.types import INSTRUMENT_PROGRAMS, MidiSequence
from .parser import parse


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
    pieces.append(
        f"{len(sequence.notes)} note{'s' if len(sequence.notes) != 1 else ''}"
    )
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


class ReplBackend(Protocol):
    """What every REPL command can assume about the playback backend.

    Structural rather than nominal: command handling is exercised with a
    stand-in that implements this and nothing else, which is the reason
    ``ReplContext`` was extracted in the first place.
    """

    def play(self, sequence: MidiSequence) -> int | None: ...

    def stop(self) -> None: ...

    def is_playing(self) -> bool: ...


class ConcurrentBackend(ReplBackend, Protocol):
    """The extra surface a MIDI backend has and the audio backend does not.

    ``ReplContext.supports_concurrent`` records which of the two is in hand.
    It is a runtime flag, so a type checker cannot follow it to a type; the
    commands that need this interface go through ``_concurrent`` below.
    """

    def close(self) -> None: ...

    def list_output_ports(self) -> list[str]: ...

    #: Read and written by the :concurrent and :sequential commands.
    concurrent_mode: bool

    @property
    def active_slots(self) -> int: ...


def _concurrent(backend: ReplBackend) -> ConcurrentBackend:
    """Read a backend as a concurrent one, at a site that has checked."""
    return cast(ConcurrentBackend, backend)


@dataclass
class ReplContext:
    """Mutable state a REPL command may read or change.

    Extracted so that command handling can be exercised without a terminal:
    ``PromptSession`` requires a TTY, which would otherwise make every command
    untestable.
    """

    backend: ReplBackend
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
    ports = _concurrent(ctx.backend).list_output_ports()
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
        mode = (
            "concurrent" if _concurrent(ctx.backend).concurrent_mode else "sequential"
        )
        print("Backend: MIDI (libremidi)")
        print(f"Mode: {mode}")
        print(f"Status: {playing}")
        print(
            f"Active slots: {_concurrent(ctx.backend).active_slots}/{MAX_PLAYBACK_SLOTS}"
        )
    else:
        print("Backend: Audio (TinySoundFont)")
        print(f"Status: {playing}")
    loaded = ctx.session.buffer_name or "(nothing)"
    print(f"Loaded: {loaded}")
    print(f"Session: {len(ctx.session.sources)} entries")


def _cmd_concurrent(ctx: ReplContext, arg: str) -> None:
    if ctx.supports_concurrent:
        _concurrent(ctx.backend).concurrent_mode = True
        print("Concurrent mode enabled - inputs will layer on each other")
    else:
        print("Concurrent mode not available with audio backend")


def _cmd_sequential(ctx: ReplContext, arg: str) -> None:
    if ctx.supports_concurrent:
        _concurrent(ctx.backend).concurrent_mode = False
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


class BackendUnavailable(RuntimeError):
    """Raised when no playback backend could be opened."""


def open_backend(
    *,
    port_name: str | None = None,
    concurrent: bool = True,
    use_audio: bool = False,
    soundfont: str | None = None,
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
) -> tuple[ReplBackend, str, bool]:
    """Open the playback backend a REPL session will use.

    Returns:
        The backend, a display name for it, and whether it supports
        concurrent playback.

    Raises:
        BackendUnavailable: If the requested backend cannot be opened.
    """
    if not use_audio and port_name is None:
        if not LibremidiBackend().list_output_ports() and soundfont:
            # No MIDI ports and a soundfont is configured, so use audio rather
            # than opening a virtual port nothing is listening to.
            use_audio = True

    if use_audio:
        from .midi.backends import HAS_TSF, TsfBackend

        if not HAS_TSF:
            raise BackendUnavailable(
                "Audio backend not available. The _tsf module was not built."
            )
        try:
            backend = TsfBackend(soundfont=soundfont)
        except FileNotFoundError as error:
            raise BackendUnavailable(str(error)) from error
        # TsfBackend does not support concurrent mode.
        return backend, "TinySoundFont", False

    backend = LibremidiBackend(
        port_name=port_name,
        concurrent=concurrent,
        virtual_port_name=virtual_port_name,
    )
    backend._ensure_port_open()
    return backend, virtual_port_name, True


def build_context(
    backend: ReplBackend,
    *,
    supports_concurrent: bool,
    verbose: bool = False,
    default_tempo: int = DEFAULT_TEMPO,
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
) -> ReplContext:
    """Assemble the session state and playback callback for a REPL.

    The returned context is what every command handler operates on, and its
    ``play`` callback is what a frontend calls for non-command input.
    """
    session = ReplSession()

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
            to_play = f"(tempo {context.default_tempo}) {source}"

        try:
            ast = parse(to_play, "<repl>")
            sequence = generate_midi(ast)
        except AldaParseError as error:
            print(f"Error: {error}")
            return False

        if not sequence.notes:
            print("(no notes)")
            if record:
                session.add(source)
            return False

        if verbose:
            print(f"{len(sequence.notes)} notes, {sequence.duration():.2f}s")

        if record:
            session.add(source)
        slot_id = backend.play(sequence)

        if slot_id is None:
            print("(all playback slots busy - use :stop to clear)")
        elif not supports_concurrent or not _concurrent(backend).concurrent_mode:
            # In sequential mode (or audio backend), wait for playback.
            while backend.is_playing():
                time.sleep(POLL_INTERVAL_DEFAULT)
        # In concurrent mode, return immediately to accept the next input.
        return True

    context = ReplContext(
        backend=backend,
        session=session,
        play=play_source,
        supports_concurrent=supports_concurrent,
        virtual_port_name=virtual_port_name,
        default_tempo=default_tempo,
    )
    return context


def print_banner(backend: ReplBackend, backend_name: str, supports_concurrent: bool) -> None:
    """Print the greeting a REPL shows before its first prompt."""
    if supports_concurrent:
        mode = "concurrent" if _concurrent(backend).concurrent_mode else "sequential"
        print(f"aldakit REPL - {backend_name} port open ({mode} mode)")
    else:
        print(f"aldakit REPL - {backend_name} audio backend")
    print("Enter alda code, press Enter to play. Alt+Enter for multi-line.")
    print("Type :help for commands, Ctrl+D to exit.")
    print()


def close_backend(backend: ReplBackend, supports_concurrent: bool) -> None:
    """Shut the backend down at the end of a session."""
    if supports_concurrent:
        _concurrent(backend).close()
    else:
        backend.stop()


def dispatch(context: ReplContext, source: str) -> None:
    """Route one submitted line to a command handler or to playback."""
    source = source.strip()
    if not source:
        return
    if source.startswith(":"):
        handle_command(context, source)
        return
    context.play(source)
