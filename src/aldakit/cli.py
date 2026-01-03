"""Command-line interface for Alda."""

import argparse
import sys
import time
from pathlib import Path

from . import generate_midi, parse
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
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command")

    # repl subcommand
    repl_parser = subparsers.add_parser(
        "repl",
        help="Interactive REPL with line editing and history",
    )
    repl_parser.add_argument(
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

    # ports subcommand
    subparsers.add_parser(
        "ports",
        help="List available MIDI output ports",
    )

    # input-ports subcommand
    subparsers.add_parser(
        "input-ports",
        help="List available MIDI input ports",
    )

    # transcribe subcommand
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Record MIDI input and output Alda code",
    )
    transcribe_parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Recording duration in seconds (default: 10)",
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
        default=120.0,
        metavar="BPM",
        help="Tempo in BPM for quantization (default: 120)",
    )
    transcribe_parser.add_argument(
        "-q",
        "--quantize",
        type=float,
        default=0.25,
        metavar="GRID",
        help="Quantize grid in beats (default: 0.25 = 16th notes)",
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
        help="MIDI input port name",
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

    # play subcommand (also the default)
    play_parser = subparsers.add_parser(
        "play",
        help="Play an Alda file or code",
    )
    _add_play_arguments(play_parser)

    # Add play arguments to main parser for default behavior
    _add_play_arguments(parser)

    return parser


def _add_play_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for playing Alda code."""
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
        help="MIDI output port name",
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


def list_ports() -> None:
    """List available MIDI output ports."""
    backend = LibremidiBackend()
    ports = backend.list_output_ports()

    if ports:
        print("Available MIDI output ports:")
        for i, port in enumerate(ports):
            print(f"  {i}: {port}")
    else:
        print("No MIDI output ports available.")
        print("You may need to start a software synthesizer or connect a MIDI device.")


def list_input_ports() -> None:
    """List available MIDI input ports."""
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
                    print(f"  Note ON:  {name}{octave} (vel={velocity})", file=sys.stderr, flush=True)
                else:
                    print(f"  Note OFF: {name}{octave}", file=sys.stderr, flush=True)

    print(f"Recording for {args.duration} seconds... play some notes!", file=sys.stderr)
    print(file=sys.stderr, flush=True)

    try:
        score = transcribe(
            duration=args.duration,
            port_name=args.port,
            instrument=args.instrument,
            quantize_grid=args.quantize,
            tempo=args.tempo,
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


def stdin_mode(port_name: str | None, verbose: bool) -> int:
    """Read alda code from stdin, blank line to play."""
    backend = LibremidiBackend(port_name=port_name)
    backend._ensure_port_open()

    print("AldaPyMIDI port open. Paste alda code, blank line to play. Ctrl+C to exit.")

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
                    print(f"Playing {len(sequence.notes)} notes...", file=sys.stderr)

                backend.play(sequence)
                while backend.is_playing():
                    time.sleep(0.1)

            except AldaParseError as e:
                print(f"Parse error: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print()
        backend.close()

    return 0


def read_source(args: argparse.Namespace) -> tuple[str, str]:
    """Read Alda source code from file, stdin, or --eval.

    Returns:
        Tuple of (source_code, filename).
    """
    if args.eval:
        return args.eval, "<eval>"

    if args.file is None:
        print(
            "Error: No input file specified. Use -e for inline code or provide a file.",
            file=sys.stderr,
        )
        sys.exit(1)

    if str(args.file) == "-":
        return sys.stdin.read(), "<stdin>"

    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    return args.file.read_text(), str(args.file)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle subcommands
    if args.command == "repl":
        from .repl import run_repl

        return run_repl(args.port, args.verbose)

    if args.command == "ports":
        list_ports()
        return 0

    if args.command == "input-ports":
        list_input_ports()
        return 0

    if args.command == "transcribe":
        return transcribe_command(args)

    # Handle play subcommand or default behavior
    # Handle --stdin
    if args.stdin:
        return stdin_mode(args.port, args.verbose)

    # Read source
    try:
        source, filename = read_source(args)
    except KeyboardInterrupt:
        return 130

    # Parse
    if args.verbose:
        print(f"Parsing {filename}...", file=sys.stderr)

    try:
        ast = parse(source, filename)
    except AldaParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        return 1

    # Handle --parse-only
    if args.parse_only:
        print(ast)
        return 0

    # Generate MIDI
    if args.verbose:
        print("Generating MIDI...", file=sys.stderr)

    sequence = generate_midi(ast)

    if not sequence.notes:
        print("Warning: No notes generated.", file=sys.stderr)
        return 0

    if args.verbose:
        print(
            f"Generated {len(sequence.notes)} notes, duration: {sequence.duration():.2f}s",
            file=sys.stderr,
        )

    # Handle --output (save to file)
    if args.output:
        if args.verbose:
            print(f"Saving to {args.output}...", file=sys.stderr)

        backend = LibremidiBackend()
        backend.save(sequence, args.output)
        print(f"Saved to {args.output}")
        return 0

    # Play
    if args.verbose:
        print("Playing...", file=sys.stderr)

    try:
        backend = LibremidiBackend(port_name=args.port)

        with backend:
            backend.play(sequence)

            if not args.no_wait:
                # Wait for playback to finish
                try:
                    while backend.is_playing():
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    if args.verbose:
                        print("\nStopping playback...", file=sys.stderr)
                    backend.stop()
                    return 130

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Use --list-ports to see available MIDI ports.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
