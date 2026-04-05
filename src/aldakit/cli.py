"""Command-line interface for Alda."""

import argparse
import sys
import time
from pathlib import Path

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
        help="Use built-in audio backend (with configured soundfont)",
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
    eval_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    eval_parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        help="Use built-in audio backend (with configured soundfont)",
    )
    eval_parser.add_argument(
        "-sf",
        "--soundfont",
        metavar="FILE",
        help="Use TinySoundFont audio backend with specified SoundFont file",
    )
    eval_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Save to MIDI file instead of playing",
    )
    eval_parser.add_argument(
        "-p",
        "--port",
        metavar="NAME",
        help="MIDI output port name or index (see 'aldakit ports')",
    )
    eval_parser.add_argument(
        "-vp",
        "--virtual-port",
        metavar="NAME",
        default=DEFAULT_VIRTUAL_PORT_NAME,
        help=f"Name for virtual MIDI port (default: {DEFAULT_VIRTUAL_PORT_NAME})",
    )

    return parser


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
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Save to MIDI file instead of playing",
    )

    parser.add_argument(
        "--port",
        metavar="NAME",
        help="MIDI output port name or index (see 'aldakit ports')",
    )

    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read alda code from stdin (blank line to play)",
    )

    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse the file and print the AST (don't play)",
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for playback to finish",
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
        help="Use built-in audio backend (with configured soundfont)",
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

    return file_arg.read_text(), str(file_arg)


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

        # CLI -a or -sf explicitly forces audio mode
        cli_audio = getattr(args, "audio", False)
        cli_soundfont = getattr(args, "soundfont", None)
        # Audio mode if: CLI -a passed, CLI -sf passed, or config.backend="audio"
        use_audio = cli_audio or cli_soundfont is not None or config.backend == "audio"
        # Soundfont: CLI overrides config (config.soundfont is fallback)
        soundfont = cli_soundfont or config.soundfont

        # Error if audio mode requested but no soundfont configured
        if use_audio and not soundfont:
            print(
                "Error: No soundfont configured for audio backend.",
                file=sys.stderr,
            )
            print(
                "Set ALDAKIT_SOUNDFONT environment variable or use -sf PATH.",
                file=sys.stderr,
            )
            return 1

        virtual_port = getattr(args, "virtual_port", DEFAULT_VIRTUAL_PORT_NAME)
        return run_repl(
            port,
            verbose,
            concurrent=concurrent,
            use_audio=use_audio,
            soundfont=soundfont,
            default_tempo=config.tempo,
            virtual_port_name=virtual_port,
        )

    if args.command == "ports":
        show_inputs = args.inputs or not args.outputs
        show_outputs = args.outputs or not args.inputs
        list_ports(show_inputs=show_inputs, show_outputs=show_outputs)
        return 0

    if args.command == "transcribe":
        return transcribe_command(args)

    if args.command == "eval":
        # Convert eval command to play with -e
        args.eval = args.code
        args.file = None
        args.stdin = False
        args.parse_only = False
        args.no_wait = False
        # Fall through to play handling

    # Handle play/eval subcommand or default behavior
    # Get optional attributes with defaults, using config as fallback
    stdin_mode_flag = getattr(args, "stdin", False)
    port_arg = getattr(args, "port", None) or config.port
    parse_only = getattr(args, "parse_only", False)
    no_wait = getattr(args, "no_wait", False)
    output = getattr(args, "output", None)
    verbose = getattr(args, "verbose", False) or config.verbose

    # CLI -a or -sf explicitly passed forces audio mode
    cli_audio = getattr(args, "audio", False)
    cli_soundfont = getattr(args, "soundfont", None)

    # Resolve port specifier (can be index like "0" or name)
    port, ok = _resolve_output_port(port_arg)
    if not ok:
        return 1

    # If no subcommand given, open the REPL
    if args.command is None:
        from .repl import run_repl

        # Audio mode if: CLI -sf passed, or config.backend="audio"
        use_audio = cli_soundfont is not None or config.backend == "audio"
        # Soundfont: CLI overrides config
        soundfont = cli_soundfont or config.soundfont
        return run_repl(
            port,
            verbose,
            concurrent=True,
            use_audio=use_audio,
            soundfont=soundfont,
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

    sequence = generate_midi(ast)

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

    # Determine backend:
    # - CLI -a explicitly forces audio mode
    # - CLI -sf explicitly forces audio mode
    # - config.backend="audio" forces audio mode
    # - config.soundfont is just a fallback path when audio is needed
    use_audio = cli_audio or cli_soundfont is not None or config.backend == "audio"
    soundfont = cli_soundfont or config.soundfont

    # Error if audio mode explicitly requested but no soundfont configured
    if use_audio and not soundfont:
        print(
            "Error: No soundfont configured for audio backend.",
            file=sys.stderr,
        )
        print(
            "Set ALDAKIT_SOUNDFONT environment variable or use -sf PATH.",
            file=sys.stderr,
        )
        return 1

    if not use_audio and port is None:
        # Check if any MIDI output ports are available
        ports = LibremidiBackend().list_output_ports()
        if not ports:
            # No MIDI ports - fall back to audio if soundfont is configured,
            # otherwise let the backend create a virtual port (AldakitMIDI)
            if soundfont:
                use_audio = True

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
