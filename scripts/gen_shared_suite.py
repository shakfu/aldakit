#!/usr/bin/env python3
"""Regenerate tests/shared_suite/*.expected from Alda's own output.

Each file is written from the committed `alda export` of its score, in
tests/alda_reference/<newest version>/, so the suite records what Alda does
rather than what any implementation of it does. The suite is shared with other
Alda implementations, which copy it.

Run from the project root::

    python scripts/gen_shared_suite.py
    python scripts/gen_shared_suite.py --examples DIR  # also examples/, into DIR

The second form writes ``<example>.expected`` for every score in ``examples/``,
for other implementations that check the examples too.

A score with no reference export is reported; create one with `make alda-diff`,
which needs Alda installed. Comment lines at the top of each file are kept.

Times come from a 128 ticks-per-beat file, so they are within one tick of
Alda's millisecond values; compare them with a tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aldakit.midi.smf_reader import read_midi_file  # noqa: E402
from tests.helpers import PROJECT_ROOT, load_alda_diff  # noqa: E402

SUITE = PROJECT_ROOT / "tests" / "shared_suite"


def header(path: Path) -> list[str]:
    """The leading comment lines of an existing .expected file."""
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            break
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def body(midi: Path) -> list[str]:
    seq = read_midi_file(midi)
    lines = [
        f"PROGRAM {int(p.program)} {p.channel} {p.time:.4f}"
        for p in sorted(seq.program_changes, key=lambda p: (p.time, p.channel))
    ]
    lines += [f"TEMPO {t.bpm:.1f} {t.time:.4f}" for t in seq.tempo_changes]
    lines += [
        f"CC {c.control} {c.value} {c.channel} {c.time:.4f}"
        for c in sorted(seq.control_changes, key=lambda c: (c.time, c.channel, c.control))
    ]
    lines += [
        f"NOTE {n.pitch} {n.start_time:.4f} {n.duration:.4f} {n.velocity} {n.channel}"
        for n in sorted(seq.notes, key=lambda n: (n.start_time, n.channel, n.pitch))
    ]
    return lines


def write_examples(reference: Path, out: Path) -> list[str]:
    """Write <example>.expected into out; returns examples with no export."""
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    provenance = f"# Generated from alda {reference.name} by scripts/gen_shared_suite.py"
    for source in sorted((PROJECT_ROOT / "examples").glob("*.alda")):
        midi = reference / "examples" / source.with_suffix(".mid").name
        if not midi.exists():
            missing.append(source.name)
            continue
        lines = [f"# Expected output for {source.name}", provenance, ""] + body(midi)
        (out / source.with_suffix(".expected").name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--examples", type=Path, metavar="DIR",
                        help="Also write .expected files for examples/ into DIR")
    args = parser.parse_args()

    reference = load_alda_diff().newest_reference()
    if reference is None:
        print("Error: tests/alda_reference/ holds no Alda exports", file=sys.stderr)
        return 1

    missing = []
    for source in sorted(SUITE.glob("*.alda")):
        midi = reference / source.relative_to(PROJECT_ROOT).with_suffix(".mid")
        if not midi.exists():
            missing.append(source.name)
            continue
        expected = source.with_suffix(".expected")
        provenance = f"# Generated from alda {reference.name} by scripts/gen_shared_suite.py"
        kept = [line for line in header(expected) if line != provenance]
        lines = kept + [provenance, ""] + body(midi)
        expected.write_text("\n".join(lines) + "\n", encoding="utf-8")

    count = len(list(SUITE.glob("*.alda"))) - len(missing)
    print(f"Wrote {count} .expected files from alda {reference.name}")
    if args.examples:
        missing += write_examples(reference, args.examples)
        print(f"Wrote examples into {args.examples}")
    if missing:
        print(f"No reference export for: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
