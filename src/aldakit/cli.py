"""Command-line interface for Alda."""

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

from . import __version__, generate_midi, parse
from .config import load_config
from .constants import (
    DEFAULT_QUANTIZE_GRID,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_SWING_RATIO,
    DEFAULT_TEMPO,
    DEFAULT_VIRTUAL_PORT_NAME,
    POLL_INTERVAL_PLAYBACK,
    SWING_RATIO_MAX,
    SWING_RATIO_MIN,
)
from .errors import AldaParseError
from .midi import LibremidiBackend
from .midi.generator import MidiGenerator
from .midi.soundfont import DEFAULT_SOUNDFONT


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="aldakit",
        description="Parse and play Alda music files.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------
    # repl subcommand

    repl_parser = subparsers.add_parser(
        "repl",
        help="Interactive REPL with line editing and history",
    )
    repl_parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Alda file to load on startup (use :play to hear it)",
    )
    repl_parser.add_argument(
        "-p",
        "--port",
        metavar="NAME",
        help="MIDI output port name",
    )
    repl_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    repl_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential mode (wait for each input to finish)",
    )
    repl_parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        help="Use built-in audio backend (configured or discovered SoundFont)",
    )
    repl_parser.add_argument(
        "-sf",
        "--soundfont",
        metavar="FILE",
        help="Use TinySoundFont audio backend with specified SoundFont file",
    )
    repl_parser.add_argument(
        "-vp",
        "--virtual-port",
        metavar="NAME",
        default=DEFAULT_VIRTUAL_PORT_NAME,
        help=f"Name for virtual MIDI port (default: {DEFAULT_VIRTUAL_PORT_NAME})",
    )

    # ------------------------------------------------------------
    # ports subcommand

    ports_parser = subparsers.add_parser(
        "ports",
        help="List available MIDI ports",
    )
    ports_parser.add_argument(
        "-i",
        "--inputs",
        action="store_true",
        help="List only MIDI input ports",
    )
    ports_parser.add_argument(
        "-o",
        "--outputs",
        action="store_true",
        help="List only MIDI output ports",
    )

    # ------------------------------------------------------------
    # transcribe subcommand

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Record MIDI input and output Alda code",
    )
    transcribe_parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=DEFAULT_RECORDING_DURATION,
        metavar="SECONDS",
        help=f"Recording duration in seconds (default: {DEFAULT_RECORDING_DURATION:.0f})",
    )
    transcribe_parser.add_argument(
        "-i",
        "--instrument",
        default="piano",
        metavar="NAME",
        help="Instrument name (default: piano)",
    )
    transcribe_parser.add_argument(
        "-t",
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO,
        metavar="BPM",
        help=f"Tempo in BPM for quantization (default: {DEFAULT_TEMPO})",
    )
    transcribe_parser.add_argument(
        "-q",
        "--quantize",
        type=float,
        default=DEFAULT_QUANTIZE_GRID,
        metavar="GRID",
        help=f"Quantize grid in beats (default: {DEFAULT_QUANTIZE_GRID} = 16th notes)",
    )
    transcribe_parser.add_argument(
        "--feel",
        choices=["straight", "swing", "triplet", "quintuplet"],
        default="straight",
        help="Timing feel for quantization (default: straight)",
    )
    transcribe_parser.add_argument(
        "--swing-ratio",
        type=float,
        default=DEFAULT_SWING_RATIO,
        metavar="RATIO",
        help=f"Swing ratio for long vs short notes (default: {DEFAULT_SWING_RATIO:.3f})",
    )
    transcribe_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Save to file (.alda or .mid)",
    )
    transcribe_parser.add_argument(
        "--port",
        metavar="NAME",
        help="MIDI input port name or index (see 'aldakit ports')",
    )
    transcribe_parser.add_argument(
        "--play",
        action="store_true",
        help="Play back the recording after transcription",
    )
    transcribe_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show notes as they are played",
    )
    transcribe_parser.add_argument(
        "--alda-notes",
        action="store_true",
        help="Show notes in Alda notation (requires -v)",
    )

    # ------------------------------------------------------------
    # soundfont subcommand

    soundfont_parser = subparsers.add_parser(
        "soundfont",
        help="Find, download and verify SoundFonts for the audio backend",
    )
    soundfont_actions = soundfont_parser.add_subparsers(dest="soundfont_command")

    soundfont_actions.add_parser(
        "list",
        help="List installed SoundFonts and those available to download",
    )

    install_parser = soundfont_actions.add_parser(
        "install",
        help="Download a SoundFont from the catalog",
    )
    install_parser.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help=f"SoundFont to download (default: {DEFAULT_SOUNDFONT})",
    )
    install_parser.add_argument(
        "--all",
        action="store_true",
        help="Download every SoundFont in the catalog",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the file is already present",
    )

    soundfont_actions.add_parser(
        "verify",
        help="Check the SHA256 checksum of each downloaded SoundFont",
    )

    soundfont_actions.add_parser(
        "path",
        help="Print the SoundFont the audio backend would use",
    )

    # ------------------------------------------------------------
    # info and lint subcommands

    info_parser = subparsers.add_parser(
        "info",
        help="Summarise a score: parts, instruments, channels, duration",
    )
    info_parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Alda file to inspect (use - for stdin)",
    )
    info_parser.add_argument(
        "-e",
        "--eval",
        metavar="CODE",
        help="Inspect Alda code given on the command line",
    )

    lint_parser = subparsers.add_parser(
        "lint",
        help="Report problems in a score without playing it",
    )
    lint_parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Alda file to check (use - for stdin)",
    )
    lint_parser.add_argument(
        "-e",
        "--eval",
        metavar="CODE",
        help="Check Alda code given on the command line",
    )
    lint_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print nothing; report the result through the exit status",
    )
    lint_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as errors",
    )

    # ------------------------------------------------------------
    # play subcommand

    play_parser = subparsers.add_parser(
        "play",
        help="Play an Alda file or code",
    )
    _add_play_arguments(play_parser)

    # ------------------------------------------------------------
    # eval subcommand

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate Alda code directly",
    )
    eval_parser.add_argument(
        "code",
        metavar="CODE",
        help="Alda code to evaluate",
    )
    _add_common_playback_arguments(eval_parser, port_flags=("-p", "--port"))

    return parser


def _add_common_playback_arguments(
    parser: argparse.ArgumentParser,
    port_flags: tuple[str, ...] = ("--port",),
) -> None:
    """Add the options shared by the play and eval subcommands."""
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Save to MIDI file instead of playing",
    )

    parser.add_argument(
        *port_flags,
        metavar="NAME",
        help="MIDI output port name or index (see 'aldakit ports')",
    )

    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse the input and print the AST (don't play)",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Return without waiting for playback to finish. Playback stops "
            "when the command exits."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        help="Use built-in audio backend (configured or discovered SoundFont)",
    )

    parser.add_argument(
        "-sf",
        "--soundfont",
        metavar="FILE",
        help="Use TinySoundFont audio backend with specified SoundFont file",
    )

    parser.add_argument(
        "-vp",
        "--virtual-port",
        metavar="NAME",
        default=DEFAULT_VIRTUAL_PORT_NAME,
        help=f"Name for virtual MIDI port (default: {DEFAULT_VIRTUAL_PORT_NAME})",
    )


def _add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the play subcommand."""
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Alda file to play (use - for stdin)",
    )

    parser.add_argument(
        "-e",
        "--eval",
        metavar="CODE",
        help="Evaluate Alda code directly",
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read alda code from stdin (blank line to play)",
    )

    _add_common_playback_arguments(parser)


class BackendChoice(NamedTuple):
    """The playback backend resolved from CLI arguments and configuration."""

    use_audio: bool
    soundfont: str | None
    error: str | None = None
    #: True when the only thing missing is a SoundFont, which the caller can
    #: offer to download rather than just reporting the error.
    needs_soundfont: bool = False


def resolve_backend(
    args: argparse.Namespace,
    config,
    port: str | None,
    *,
    consider_ports: bool = True,
) -> BackendChoice:
    """Decide whether to play through MIDI or the built-in audio synth.

    Precedence, highest first:

    1. ``-sf/--soundfont`` selects the audio backend with that SoundFont.
    2. ``-a/--audio`` selects the audio backend with a configured or
       discovered SoundFont.
    3. ``backend = audio`` in the config file.
    4. Otherwise MIDI, unless no MIDI output ports exist and a SoundFont can be
       found, in which case audio is used rather than playing into a virtual
       port that nothing is listening to.

    A SoundFont is "available" if one is named by ``-sf``, by the config file,
    or by ``ALDAKIT_SOUNDFONT``, or if one can be discovered in the standard
    locations (see ``aldakit.midi.soundfont.find_soundfont``).

    Args:
        args: Parsed CLI arguments.
        config: Loaded configuration.
        port: The resolved MIDI output port, or None.
        consider_ports: If True, fall back to audio when no MIDI output ports
            are available.

    Returns:
        A BackendChoice. When ``error`` is set the caller should print it and
        exit non-zero.
    """
    from .midi.backends import HAS_TSF
    from .midi.soundfont import find_soundfont

    cli_audio = getattr(args, "audio", False)
    cli_soundfont = getattr(args, "soundfont", None)

    # An explicitly named SoundFont always wins over a discovered one
    soundfont = cli_soundfont or config.soundfont
    audio_requested = (
        cli_audio or cli_soundfont is not None or config.backend == "audio"
    )

    def discovered() -> str | None:
        if not HAS_TSF:
            return None
        found = find_soundfont()
        return str(found) if found else None

    if audio_requested:
        if not HAS_TSF:
            return BackendChoice(
                False,
                None,
                "Audio backend not available. The _tsf module was not built.",
            )
        resolved = soundfont or discovered()
        if resolved is None:
            return BackendChoice(
                False,
                None,
                "No SoundFont found for the audio backend.\n"
                "Install one with 'aldakit soundfont install', set "
                "ALDAKIT_SOUNDFONT, or pass -sf PATH.",
                needs_soundfont=True,
            )
        return BackendChoice(True, resolved)

    # MIDI was not overridden. If there is nowhere to send it, prefer audio
    # over opening a virtual port that produces silence.
    if consider_ports and port is None and not LibremidiBackend().list_output_ports():
        resolved = soundfont or discovered()
        if resolved is not None:
            return BackendChoice(True, resolved)

    return BackendChoice(False, soundfont)


def _resolve_backend_interactively(
    args: argparse.Namespace, config, port: str | None, **kwargs
) -> BackendChoice:
    """Resolve the backend, offering to download a SoundFont if that is all
    that is missing.

    This is what turns the out-of-box experience for a user with no external
    synth from an error message into a working playback.
    """
    choice = resolve_backend(args, config, port, **kwargs)
    if not choice.needs_soundfont:
        return choice

    downloaded = offer_soundfont_download()
    if downloaded is None:
        return choice
    return BackendChoice(True, downloaded)


def list_ports(show_inputs: bool = True, show_outputs: bool = True) -> None:
    """List available MIDI ports."""
    if show_outputs:
        backend = LibremidiBackend()
        ports = backend.list_output_ports()
        if ports:
            print("Available MIDI output ports:")
            for i, port in enumerate(ports):
                print(f"  {i}: {port}")
        else:
            print("No MIDI output ports available.")
            print(
                "You may need to start a software synthesizer or connect a MIDI device."
            )
        if show_inputs:
            print()

    if show_inputs:
        from .midi.transcriber import list_input_ports as get_input_ports

        ports = get_input_ports()

        if ports:
            print("Available MIDI input ports:")
            for i, port in enumerate(ports):
                print(f"  {i}: {port}")
        else:
            print("No MIDI input ports available.")
            print("You may need to connect a MIDI keyboard or controller.")


def soundfont_command(args: argparse.Namespace) -> int:
    """Find, download and verify SoundFonts for the audio backend."""
    from .midi.soundfont import SoundFontManager, print_download_progress

    manager = SoundFontManager()
    action = getattr(args, "soundfont_command", None) or "list"

    if action == "list":
        installed = manager.list_installed()
        if installed:
            print("Installed SoundFonts:")
            for path in installed:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  {path}  ({size_mb:.1f} MB)")
        else:
            print("No SoundFonts installed.")
            print(f"  Searched: {manager.soundfont_dir} and standard locations.")

        print()
        print("Available to download ('aldakit soundfont install NAME'):")
        for name, info in manager.list_available_downloads().items():
            default = " [default]" if name == DEFAULT_SOUNDFONT else ""
            print(f"  {name}{default}")
            print(
                f"      {info.get('description', '')} ({info.get('size_mb', '?')} MB)"
            )
        return 0

    if action == "install":
        try:
            if getattr(args, "all", False):
                manager.setup_all(force=getattr(args, "force", False))
                return 0

            name = getattr(args, "name", None) or DEFAULT_SOUNDFONT
            info = manager.catalog.get(name)
            if info is None:
                available = ", ".join(manager.catalog)
                print(f"Error: Unknown SoundFont: {name}", file=sys.stderr)
                print(f"  Available: {available}", file=sys.stderr)
                return 1

            target = manager.soundfont_dir / str(info["filename"])
            if target.exists() and not getattr(args, "force", False):
                print(f"Already installed: {target}")
                print("  Pass --force to download it again.")
                return 0

            print(f"Downloading {name} ({info.get('size_mb', '?')} MB)...")
            print(f"  {info.get('description', '')}")
            path = manager.download(
                name,
                progress_callback=print_download_progress,
                force=getattr(args, "force", False),
            )
            print()
            print(f"Saved to: {path}")
            return 0
        except (RuntimeError, OSError) as e:
            print(f"Error: Download failed: {e}", file=sys.stderr)
            return 1

    if action == "verify":
        results = manager.verify_checksums()
        present = {
            name: ok
            for name, ok in results.items()
            if _catalog_file_exists(manager, name)
        }
        if not present:
            print("No downloaded SoundFonts from the catalog to verify.")
            print(f"  Looked in: {manager.soundfont_dir}")
            return 0

        failed = [name for name, ok in present.items() if not ok]
        for name, ok in present.items():
            print(f"  {name}: {'ok' if ok else 'CHECKSUM MISMATCH'}")
        if failed:
            print(
                f"\n{len(failed)} of {len(present)} failed verification. "
                "Re-download with 'aldakit soundfont install NAME --force'.",
                file=sys.stderr,
            )
            return 1
        print(f"\nAll {len(present)} verified.")
        return 0

    if action == "path":
        found = manager.find()
        if found is None:
            print("No SoundFont found.", file=sys.stderr)
            print(
                "  Install one with 'aldakit soundfont install', "
                "or set ALDAKIT_SOUNDFONT.",
                file=sys.stderr,
            )
            return 1
        print(found)
        return 0

    print(f"Error: Unknown soundfont action: {action}", file=sys.stderr)
    return 1


def _format_duration(seconds: float) -> str:
    """Seconds as ``m:ss.s`` with the raw value alongside."""
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}:{remainder:04.1f} ({seconds:.2f}s)"


def info_command(args: argparse.Namespace) -> int:
    """Summarise a score without playing it."""
    from .analysis import inspect_score

    source, filename = read_source(args)

    try:
        info = inspect_score(source, filename)
    except AldaParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    print(filename)
    print(f"  parts:    {len(info.parts)}")
    print(f"  notes:    {info.note_count}")
    print(f"  duration: {_format_duration(info.duration)}")

    if info.tempos:
        # Generation always emits the default tempo at time 0, so the tempo a
        # score starts at is the last one set there, and only later changes
        # count as changes.
        starting = [bpm for time, bpm in info.tempos if time == 0.0][-1]
        changes = sum(1 for time, _ in info.tempos if time > 0.0)
        suffix = f", {changes} change{'s' if changes != 1 else ''}" if changes else ""
        print(f"  tempo:    {starting:g} bpm{suffix}")
    if info.control_change_count:
        print(f"  controls: {info.control_change_count}")
    if info.variables:
        print(f"  variables: {', '.join(info.variables)}")
    if info.markers:
        print(f"  markers:  {', '.join(info.markers)}")

    if info.parts:
        print()
        name_width = max(len(p.name) for p in info.parts)
        name_width = max(name_width, len("part"))
        instrument_width = max(len(p.instrument) for p in info.parts)
        instrument_width = max(instrument_width, len("instrument"))
        header = (
            f"  {'part':<{name_width}}  {'instrument':<{instrument_width}}  "
            f"{'prog':>4}  {'chan':>4}  {'notes':>6}"
        )
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for part in info.parts:
            program = "--" if part.percussion else str(part.program)
            print(
                f"  {part.name:<{name_width}}  {part.instrument:<{instrument_width}}  "
                f"{program:>4}  {part.channel:>4}  {part.note_count:>6}"
            )
            details = []
            if part.key_signature:
                spelled = " ".join(
                    f"{letter}{accidental}"
                    for letter, accidental in sorted(part.key_signature.items())
                )
                details.append(f"key {spelled}")
            if part.transpose:
                details.append(f"transposed {part.transpose:+d}")
            if details:
                print(f"  {' ' * name_width}  ({'; '.join(details)})")

    if info.findings:
        errors = sum(1 for f in info.findings if f.severity == "error")
        print()
        print(
            f"  {len(info.findings)} finding(s), {errors} error(s). "
            f"Run 'aldakit lint {filename}' for details."
        )

    return 0


def lint_command(args: argparse.Namespace) -> int:
    """Report problems in a score without playing it.

    Exit status: 0 when clean, 1 when an error was found (or any finding
    under ``--strict``), 2 when the score does not parse.
    """
    from .analysis import ERROR, WARNING, lint_score

    source, filename = read_source(args)
    quiet = getattr(args, "quiet", False)
    strict = getattr(args, "strict", False)

    try:
        findings = lint_score(source, filename)
    except AldaParseError as e:
        if not quiet:
            print(f"Parse error: {e}", file=sys.stderr)
        return 2

    if not quiet:
        for finding in findings:
            print(finding)

    errors = sum(1 for f in findings if f.severity == ERROR)
    warnings = sum(1 for f in findings if f.severity == WARNING)

    if not quiet:
        if findings:
            print()
            print(
                f"{len(findings)} finding(s): {errors} error(s), {warnings} warning(s)."
            )
        else:
            print(f"{filename}: no problems found.")

    if errors or (strict and findings):
        return 1
    return 0


def _catalog_file_exists(manager, name: str) -> bool:
    """Whether the catalog entry ``name`` has been downloaded."""
    info = manager.catalog.get(name)
    if info is None:
        return False
    return (manager.soundfont_dir / str(info["filename"])).exists()


def offer_soundfont_download(name: str | None = None) -> str | None:
    """Offer to download a SoundFont when audio playback has none.

    Only asks when stdin is a terminal, so scripts and CI get the error
    message rather than a prompt that will never be answered.

    Args:
        name: Catalog entry to offer. Defaults to the catalog's default.

    Returns:
        Path to the downloaded SoundFont, or None if it was declined or
        the download failed.
    """
    if not sys.stdin.isatty():
        return None

    from .midi.soundfont import (
        DEFAULT_SOUNDFONT as CATALOG_DEFAULT,
        SoundFontManager,
        print_download_progress,
    )

    name = name or CATALOG_DEFAULT
    manager = SoundFontManager()
    info = manager.catalog.get(name, {})
    size = info.get("size_mb", "?")

    print(f"No SoundFont found. aldakit can download {name} ({size} MB) now.")
    try:
        answer = input("Download it? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if answer not in ("", "y", "yes"):
        return None

    try:
        path = manager.download(name, progress_callback=print_download_progress)
    except (RuntimeError, OSError) as e:
        print(f"\nDownload failed: {e}", file=sys.stderr)
        return None

    print()
    print(f"Saved to: {path}")
    return str(path)


def transcribe_command(args: argparse.Namespace) -> int:
    """Record MIDI input and output Alda code."""
    from .midi.midi_to_ast import midi_pitch_to_note
    from .midi.transcriber import transcribe

    # Validate swing ratio
    if not SWING_RATIO_MIN < args.swing_ratio < SWING_RATIO_MAX:
        print(
            "Error: --swing-ratio must be between 0 and 1 (exclusive).",
            file=sys.stderr,
        )
        return 1

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def on_note(pitch: int, velocity: int, is_on: bool) -> None:
        if args.verbose:
            if args.alda_notes:
                letter, octave, accidentals = midi_pitch_to_note(pitch)
                acc = "".join(accidentals)
                note_str = f"o{octave} {letter}{acc}"
                if is_on:
                    print(f"  {note_str}", file=sys.stderr, flush=True)
            else:
                name = NOTE_NAMES[pitch % 12]
                octave = (pitch // 12) - 1
                if is_on:
                    print(
                        f"  Note ON:  {name}{octave} (vel={velocity})",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"  Note OFF: {name}{octave}", file=sys.stderr, flush=True)

    print(f"Recording for {args.duration} seconds... play some notes!", file=sys.stderr)
    print(file=sys.stderr, flush=True)

    try:
        # Resolve port specifier (can be index like "0" or name)
        port_name, ok = _resolve_input_port(args.port)
        if not ok:
            return 1

        score = transcribe(
            duration=args.duration,
            port_name=port_name,
            instrument=args.instrument,
            quantize_grid=args.quantize,
            tempo=args.tempo,
            feel=args.feel,
            swing_ratio=args.swing_ratio,
            on_note=on_note if args.verbose else None,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(file=sys.stderr, flush=True)
    sys.stderr.flush()
    alda_code = score.to_alda()

    # Handle output
    if args.output:
        score.save(args.output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(alda_code)

    # Play back if requested
    if args.play:
        print("Playing back...", file=sys.stderr)
        score.play()

    return 0


def stdin_mode(
    port_name: str | None,
    verbose: bool,
    virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
) -> int:
    """Read alda code from stdin, blank line to play."""
    if port_name:
        print(
            f"Using MIDI output port '{port_name}'. Paste Alda code, blank line twice to play. Ctrl+C to exit."
        )
    else:
        print(
            f"Opening {virtual_port_name} port... Paste Alda code, blank line twice to play. Ctrl+C to exit."
        )

    with LibremidiBackend(
        port_name=port_name, virtual_port_name=virtual_port_name
    ) as backend:
        try:
            while True:
                lines = []
                try:
                    while True:
                        line = input()
                        if line == "" and lines and lines[-1] == "":
                            break
                        lines.append(line)
                except EOFError:
                    break

                source = "\n".join(lines).strip()
                if not source:
                    continue

                try:
                    ast = parse(source, "<stdin>")
                    sequence = generate_midi(ast)

                    if not sequence.notes:
                        print("(no notes)")
                        continue

                    if verbose:
                        print(
                            f"Playing {len(sequence.notes)} notes...", file=sys.stderr
                        )

                    backend.play(sequence)
                    while backend.is_playing():
                        time.sleep(POLL_INTERVAL_PLAYBACK)

                except AldaParseError as e:
                    print(f"Parse error: {e}", file=sys.stderr)

        except KeyboardInterrupt:
            print()

    return 0


def read_source(args: argparse.Namespace) -> tuple[str, str]:
    """Read Alda source code from file, stdin, or --eval.

    Returns:
        Tuple of (source_code, filename).
    """
    if args.eval:
        return args.eval, "<eval>"

    file_arg = getattr(args, "file", None)
    if file_arg is None:
        print(
            "Error: No input file specified.",
            file=sys.stderr,
        )
        print(
            "  Try: aldakit play <file.alda>",
            file=sys.stderr,
        )
        print(
            "  Or:  aldakit eval -e 'piano: c d e'",
            file=sys.stderr,
        )
        sys.exit(1)

    # file_arg is a Path at this point (type narrowing for the checker)
    assert file_arg is not None

    if str(file_arg) == "-":
        return sys.stdin.read(), "<stdin>"

    if not file_arg.exists():
        print(f"Error: File not found: {file_arg}", file=sys.stderr)
        print(
            "  Check the path and ensure the file has a .alda or .mid extension.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Alda sources are UTF-8. Without this, Windows decodes with the locale
    # codepage and any non-ASCII character raises UnicodeDecodeError.
    return file_arg.read_text(encoding="utf-8"), str(file_arg)


def _resolve_port_specifier(
    specifier: str | None, ports: list[str], kind: str
) -> tuple[str | None, bool]:
    """Resolve a port specifier (index or name) to an actual port name.

    Args:
        specifier: Port index (e.g., "0") or name/partial name.
        ports: List of available port names.
        kind: "input" or "output" for error messages.

    Returns:
        Tuple of (resolved_port_name, success). On failure, prints an error.
    """
    if specifier is None:
        return None, True

    # Check if specifier is a numeric index
    if specifier.isdigit():
        idx = int(specifier)
        if 0 <= idx < len(ports):
            return ports[idx], True
        print(
            f"Error: Port index {idx} out of range. "
            f"Use 'aldakit ports' to see available {kind} ports.",
            file=sys.stderr,
        )
        return None, False

    # Otherwise treat as name (backend will handle partial matching)
    return specifier, True


def _resolve_output_port(port_specifier: str | None) -> tuple[str | None, bool]:
    """Resolve output port specifier (index or name) to port name.

    If no port is specified and exactly one output port exists, it is
    auto-selected for convenience.
    """
    backend = LibremidiBackend()
    ports = backend.list_output_ports()

    if port_specifier is None:
        # Auto-select if exactly one port available
        if len(ports) == 1:
            return ports[0], True
        return None, True

    return _resolve_port_specifier(port_specifier, ports, "output")


def _resolve_input_port(port_specifier: str | None) -> tuple[str | None, bool]:
    """Resolve input port specifier (index or name) to port name.

    If no port is specified and exactly one input port exists, it is
    auto-selected for convenience.
    """
    from .midi.transcriber import list_input_ports as get_input_ports

    ports = get_input_ports()

    if port_specifier is None:
        # Auto-select if exactly one port available
        if len(ports) == 1:
            return ports[0], True
        return None, True

    return _resolve_port_specifier(port_specifier, ports, "input")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load configuration from files
    config = load_config()

    # Handle subcommands
    if args.command == "repl":
        from .repl import run_repl

        # CLI args override config, config overrides defaults
        port_arg = args.port if args.port else config.port
        port, ok = _resolve_output_port(port_arg)
        if not ok:
            return 1
        concurrent = not getattr(args, "sequential", False)
        verbose = args.verbose or config.verbose

        choice = _resolve_backend_interactively(args, config, port)
        if choice.error:
            print(f"Error: {choice.error}", file=sys.stderr)
            return 1

        initial_file = getattr(args, "file", None)
        if initial_file is not None and not initial_file.exists():
            print(f"Error: File not found: {initial_file}", file=sys.stderr)
            return 1

        virtual_port = getattr(args, "virtual_port", DEFAULT_VIRTUAL_PORT_NAME)
        return run_repl(
            port,
            verbose,
            concurrent=concurrent,
            use_audio=choice.use_audio,
            soundfont=choice.soundfont,
            default_tempo=config.tempo,
            virtual_port_name=virtual_port,
            initial_file=initial_file,
        )

    if args.command == "ports":
        show_inputs = args.inputs or not args.outputs
        show_outputs = args.outputs or not args.inputs
        list_ports(show_inputs=show_inputs, show_outputs=show_outputs)
        return 0

    if args.command == "transcribe":
        return transcribe_command(args)

    if args.command == "soundfont":
        return soundfont_command(args)

    if args.command == "info":
        return info_command(args)

    if args.command == "lint":
        return lint_command(args)

    if args.command == "eval":
        # Convert eval command to play with -e. The shared playback options
        # (--parse-only, --no-wait, ...) are already parsed onto args.
        args.eval = args.code
        args.file = None
        args.stdin = False
        # Fall through to play handling

    # Handle play/eval subcommand or default behavior
    # Get optional attributes with defaults, using config as fallback
    stdin_mode_flag = getattr(args, "stdin", False)
    port_arg = getattr(args, "port", None) or config.port
    parse_only = getattr(args, "parse_only", False)
    no_wait = getattr(args, "no_wait", False)
    output = getattr(args, "output", None)
    verbose = getattr(args, "verbose", False) or config.verbose

    # Resolve port specifier (can be index like "0" or name)
    port, ok = _resolve_output_port(port_arg)
    if not ok:
        return 1

    # If no subcommand given, open the REPL
    if args.command is None:
        from .repl import run_repl

        choice = _resolve_backend_interactively(args, config, port)
        if choice.error:
            print(f"Error: {choice.error}", file=sys.stderr)
            return 1
        return run_repl(
            port,
            verbose,
            concurrent=True,
            use_audio=choice.use_audio,
            soundfont=choice.soundfont,
            default_tempo=config.tempo,
            virtual_port_name=DEFAULT_VIRTUAL_PORT_NAME,
        )

    # Get virtual port name for play/eval subcommands
    virtual_port = getattr(args, "virtual_port", DEFAULT_VIRTUAL_PORT_NAME)

    # Handle --stdin (play subcommand only)
    if stdin_mode_flag:
        return stdin_mode(port, verbose, virtual_port)

    # If no file and no -e in play subcommand, show error
    file_arg = getattr(args, "file", None)
    eval_code = getattr(args, "eval", None)
    if args.command == "play" and file_arg is None and eval_code is None:
        print("Error: No input specified.", file=sys.stderr)
        print("  Try: aldakit play song.alda", file=sys.stderr)
        print("  Or:  aldakit eval -e 'piano: c d e'", file=sys.stderr)
        print("  Or:  cat song.alda | aldakit play --stdin", file=sys.stderr)
        return 1

    # Read source
    try:
        source, filename = read_source(args)
    except KeyboardInterrupt:
        return 130

    # Parse
    if verbose:
        print(f"Parsing {filename}...", file=sys.stderr)

    try:
        ast = parse(source, filename)
    except AldaParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    # Handle --parse-only
    if parse_only:
        print(ast)
        return 0

    # Generate MIDI
    if verbose:
        print("Generating MIDI...", file=sys.stderr)

    generator = MidiGenerator()
    sequence = generator.generate(ast)

    # Report problems that do not stop generation but change what is heard
    for diagnostic in generator.diagnostics:
        print(f"Warning: {diagnostic}", file=sys.stderr)

    if not sequence.notes:
        print("Warning: No notes generated.", file=sys.stderr)
        print(
            "  Ensure your score has a part and notes. Example: piano: c d e f g",
            file=sys.stderr,
        )
        return 0

    if verbose:
        print(
            f"Generated {len(sequence.notes)} notes, duration: {sequence.duration():.2f}s",
            file=sys.stderr,
        )

    # Handle --output (save to file)
    if output:
        if verbose:
            print(f"Saving to {output}...", file=sys.stderr)

        backend = LibremidiBackend()
        backend.save(sequence, output)
        print(f"Saved to {output}")
        return 0

    choice = _resolve_backend_interactively(args, config, port)
    if choice.error:
        print(f"Error: {choice.error}", file=sys.stderr)
        return 1
    use_audio, soundfont = choice.use_audio, choice.soundfont

    if not use_audio and port is None and not LibremidiBackend().list_output_ports():
        # Nothing to send MIDI to and no SoundFont to fall back on. A virtual
        # port will be opened, but it is silent unless something connects to it.
        print(
            f"Warning: no MIDI output ports; opening virtual port "
            f"'{virtual_port}'. You will not hear anything unless a synth or "
            f"DAW is connected to it.",
            file=sys.stderr,
        )
        print(
            "Install a SoundFont for built-in audio, or run "
            "'aldakit ports' to check your MIDI setup.",
            file=sys.stderr,
        )

    # Play
    if verbose:
        backend_name = "audio (TinySoundFont)" if use_audio else "MIDI"
        print(f"Playing via {backend_name}...", file=sys.stderr)

    try:
        if use_audio:
            from .midi.backends import TsfBackend, HAS_TSF

            if not HAS_TSF:
                print(
                    "Error: Audio backend not available. The _tsf module was not built.",
                    file=sys.stderr,
                )
                print(
                    "  Reinstall aldakit from source to compile the audio backend,",
                    file=sys.stderr,
                )
                print(
                    "  or use MIDI output instead: aldakit play song.alda",
                    file=sys.stderr,
                )
                return 1

            try:
                with TsfBackend(soundfont=soundfont) as backend:
                    backend.play(sequence)
                    if not no_wait:
                        try:
                            backend.wait()
                        except KeyboardInterrupt:
                            if verbose:
                                print("\nStopping playback...", file=sys.stderr)
                            backend.stop()
                            return 130
            except FileNotFoundError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
        else:
            backend = LibremidiBackend(port_name=port, virtual_port_name=virtual_port)

            with backend:
                backend.play(sequence)

                if not no_wait:
                    try:
                        while backend.is_playing():
                            time.sleep(POLL_INTERVAL_PLAYBACK)
                    except KeyboardInterrupt:
                        if verbose:
                            print("\nStopping playback...", file=sys.stderr)
                        backend.stop()
                        return 130

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Use 'aldakit ports' to see available MIDI ports.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
