"""Command-line interface for Alda."""

import argparse
import sys
import time
from pathlib import Path

from . import parse, generate_midi
from .midi import LibremidiBackend
from .errors import AldaParseError


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="aldapy",
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
