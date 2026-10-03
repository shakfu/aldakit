#!/usr/bin/env python3
"""Compare aldakit's MIDI output with the reference Alda implementation.

Exports each score with ``alda export`` and with aldakit, reads both files back
through ``aldakit.midi.smf_reader``, and reports every difference. Both sides go
through a Standard MIDI File so tick rounding applies to both equally.

Notes are compared on pitch, start, duration and velocity, plus the program and
controllers 7, 10 and 11 in effect on the note's channel when it starts. Channel
numbers and the times of program and control changes are not compared; see
``docs/dev/alda-deviations.md``. Tempo changes are compared as events.

Alda's exports are committed under ``tests/alda_reference/<version>/``: a
``.mid`` per score, or a ``.error`` holding Alda's message for a score it
rejects. With ``alda`` and ``alda-player`` (https://alda.io/install/) on PATH or
in ``--alda DIR``, missing exports are created; without them, the script
compares against the newest committed version. Alda's usage telemetry is
disabled for every call this script makes.

Run from the project root::

    python scripts/alda_diff.py                       # shared suite + examples
    python scripts/alda_diff.py examples/bach_*.alda  # selected files

Exits 0 when every file matches, 1 when any differs, 2 when there is no reference.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from aldakit import generate_midi, parse  # noqa: E402
from aldakit.midi.smf import write_midi_file  # noqa: E402
from aldakit.midi.smf_reader import read_midi_file  # noqa: E402
from aldakit.midi.types import MidiNote, MidiSequence  # noqa: E402

ROOT = Path(__file__).parent.parent
REFERENCE_ROOT = ROOT / "tests" / "alda_reference"
DEFAULT_CORPUS = ("tests/shared_suite/*.alda", "examples/*.alda")
EXPORT_TIMEOUT = 120  # seconds; Alda's own MIDI export timeout is 60
# A note this close to a same-pitch note on the other side counts as moved
# rather than as one missing note plus one extra note.
SHIFT_WINDOW = 0.5  # seconds
# Controllers whose value is compared per note: volume, pan, expression
STATE_CONTROLS = (7, 10, 11)


@dataclass
class Diff:
    """Differences for one score. Each category maps to example lines."""

    error: str | None = None
    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, category: str, line: str, limit: int) -> None:
        self.counts[category] += 1
        if len(self.samples[category]) < limit:
            self.samples[category].append(line)


def find_alda(directory: str | None) -> tuple[str, dict[str, str]] | None:
    """The alda executable and an environment in which it finds alda-player."""
    env = dict(os.environ, ALDA_DISABLE_TELEMETRY="yes")
    if directory:
        env["PATH"] = f"{directory}{os.pathsep}{env.get('PATH', '')}"
    alda = shutil.which("alda", path=env.get("PATH"))
    if alda is None or shutil.which("alda-player", path=env.get("PATH")) is None:
        return None
    return alda, env


def alda_version(alda: str, env: dict[str, str]) -> str:
    out = subprocess.run(
        [alda, "version"], env=env, capture_output=True, text=True, check=True
    ).stdout
    # Output is "alda 2.4.7"
    return out.split()[-1]


def export_reference(
    alda: str, env: dict[str, str], source: Path, target: Path
) -> str | None:
    """Export source with Alda. Returns an error message, or None on success."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Run from the root with a relative path so error messages hold no
        # machine-specific path.
        result = subprocess.run(
            [alda, "export", "-f", str(source), "-o", str(target.resolve())],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=EXPORT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"alda export timed out after {EXPORT_TIMEOUT}s"
    if result.returncode != 0 or not target.exists():
        target.unlink(missing_ok=True)
        # Alda's first line is a generic "Oops! Something went wrong:" header
        lines = [ln.strip() for ln in (result.stderr or result.stdout).splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("Oops!")]
        return lines[0] if lines else "alda export failed"
    return None


def aldakit_sequence(source: Path) -> MidiSequence:
    """aldakit's output for source, after a round trip through an SMF file."""
    sequence = generate_midi(parse(source.read_text(encoding="utf-8"), str(source)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.mid"
        write_midi_file(sequence, path)
        return read_midi_file(path)


Tolerance = Callable[[float], float]


def tick_tolerance(ref: MidiSequence, ours: MidiSequence, ticks: float) -> Tolerance:
    """Seconds spanned by `ticks` of the coarser file's resolution, at the
    reference tempo in effect at a given time.

    Alda writes 128 ticks per beat, so one tick is 3.9 ms at 120 BPM and 9 ms at
    52 BPM; a fixed tolerance in seconds is either too loose or too tight.
    """
    ppq = min(ref.ticks_per_beat, ours.ticks_per_beat)
    changes = sorted((t.time, t.bpm) for t in ref.tempo_changes) or [(0.0, 120.0)]
    times = [t for t, _ in changes]

    def at(time: float) -> float:
        bpm = changes[max(bisect_right(times, time) - 1, 0)][1]
        return ticks * 60.0 / (bpm * ppq)

    return at


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def note_state(
    seq: MidiSequence, tol: Tolerance
) -> Callable[[MidiNote], dict[str, int | None]]:
    """The program and STATE_CONTROLS values a note sounds with."""
    streams: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    for p in seq.program_changes:
        streams[(p.channel, "program")].append((p.time, int(p.program)))
    for c in seq.control_changes:
        if c.control in STATE_CONTROLS:
            streams[(c.channel, f"cc{c.control}")].append((c.time, c.value))
    for events in streams.values():
        events.sort()

    def last(channel: int, name: str, time: float) -> int | None:
        events = streams.get((channel, name), [])
        i = bisect_right([t for t, _ in events], time + tol(time))
        return events[i - 1][1] if i else None

    names = ["program"] + [f"cc{c}" for c in STATE_CONTROLS]
    return lambda n: {k: last(n.channel, k, n.start_time) for k in names}


def compare_notes(
    ref: MidiSequence, ours: MidiSequence, diff: Diff, tol: Tolerance, limit: int
) -> None:
    """Pair notes by pitch and start time, then compare the remaining fields.

    Notes left over are paired again within SHIFT_WINDOW and reported as moved.
    """
    ref_state, our_state = note_state(ref, tol), note_state(ours, tol)
    by_pitch: dict[int, list] = defaultdict(list)
    for note in ours.notes:
        by_pitch[note.pitch].append(note)

    missing = []
    for r in sorted(ref.notes, key=lambda n: (n.start_time, n.channel, n.pitch)):
        candidates = [
            n
            for n in by_pitch[r.pitch]
            if _close(n.start_time, r.start_time, tol(r.start_time))
        ]
        if not candidates:
            missing.append(r)
            continue
        # Unison notes in different parts must pair by part. Channel numbers
        # differ between the two sides, so prefer the same instrument state,
        # then the closest duration.
        theirs = ref_state(r)
        n = min(
            candidates,
            key=lambda c: (
                sum(v != theirs[k] for k, v in our_state(c).items()),
                round(abs(c.duration - r.duration), 6),
                abs(c.start_time - r.start_time),
            ),
        )
        by_pitch[r.pitch].remove(n)
        where = f"{r.start_time:.4f} p{r.pitch}"
        # Start and end are rounded to ticks independently
        if not _close(n.duration, r.duration, 2 * tol(r.start_time)):
            diff.add(
                "note duration",
                f"{where}: alda {r.duration:.4f}, aldakit {n.duration:.4f}",
                limit,
            )
        if n.velocity != r.velocity:
            diff.add(
                "note velocity",
                f"{where}: alda {r.velocity}, aldakit {n.velocity}",
                limit,
            )
        mine = our_state(n)
        for name, value in theirs.items():
            if mine[name] != value:
                diff.add(
                    f"note {name}", f"{where}: alda {value}, aldakit {mine[name]}", limit
                )

    for r in missing:
        moved = [
            n
            for n in by_pitch[r.pitch]
            if _close(n.start_time, r.start_time, SHIFT_WINDOW)
        ]
        if moved:
            n = min(moved, key=lambda c: abs(c.start_time - r.start_time))
            by_pitch[r.pitch].remove(n)
            diff.add(
                "note start",
                f"p{r.pitch}: alda {r.start_time:.4f}, aldakit {n.start_time:.4f}",
                limit,
            )
        else:
            diff.add(
                "note missing",
                f"{r.start_time:.4f} ch{r.channel} p{r.pitch} d{r.duration:.4f} v{r.velocity}",
                limit,
            )

    for notes in by_pitch.values():
        for n in notes:
            diff.add(
                "note extra",
                f"{n.start_time:.4f} ch{n.channel} p{n.pitch} d{n.duration:.4f} v{n.velocity}",
                limit,
            )


def compare_events(
    kind: str,
    ref: list[tuple[float, str]],
    ours: list[tuple[float, str]],
    diff: Diff,
    tol: Tolerance,
    limit: int,
) -> None:
    """Match (time, description) events whose descriptions are equal."""
    remaining = sorted(ours)
    for time, desc in sorted(ref):
        match = next(
            (e for e in remaining if e[1] == desc and _close(e[0], time, tol(time))), None
        )
        if match is None:
            diff.add(f"{kind} missing", f"{time:.4f} {desc}", limit)
        else:
            remaining.remove(match)
    for time, desc in remaining:
        diff.add(f"{kind} extra", f"{time:.4f} {desc}", limit)


def compare(ref: MidiSequence, ours: MidiSequence, ticks: float, limit: int) -> Diff:
    diff = Diff()
    tol = tick_tolerance(ref, ours, ticks)
    compare_notes(ref, ours, diff, tol, limit)
    compare_events(
        "tempo",
        [(t.time, f"{t.bpm:.2f}bpm") for t in ref.tempo_changes],
        [(t.time, f"{t.bpm:.2f}bpm") for t in ours.tempo_changes],
        diff,
        tol,
        limit,
    )
    return diff


def corpus(paths: list[str]) -> list[Path]:
    """Scores to compare, relative to ROOT."""
    if paths:
        return [Path(p).resolve().relative_to(ROOT) for p in paths]
    return [
        p.relative_to(ROOT) for pattern in DEFAULT_CORPUS for p in sorted(ROOT.glob(pattern))
    ]


def newest_reference() -> Path | None:
    """The committed reference directory with the highest Alda version."""
    versions = [d for d in REFERENCE_ROOT.glob("*") if d.is_dir()]
    if not versions:
        return None
    return max(versions, key=lambda d: tuple(int(x) for x in d.name.split(".")))


def compare_file(source: Path, reference: Path, ticks: float, limit: int) -> Diff:
    """Compare aldakit's output for source with Alda's committed export."""
    midi = reference / source.with_suffix(".mid")
    error = reference / source.with_suffix(".error")
    if error.exists():
        return Diff(error=f"alda: {error.read_text(encoding='utf-8').strip()}")
    if not midi.exists():
        return Diff(error="no reference export")
    try:
        ours = aldakit_sequence(ROOT / source)
    except Exception as e:  # any aldakit failure is a finding, not a crash
        return Diff(error=f"aldakit: {type(e).__name__}: {e}")
    return compare(read_midi_file(midi), ours, ticks, limit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", help="Alda files (default: shared suite and examples)")
    parser.add_argument("--alda", help="Directory holding alda and alda-player")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Directory of Alda's exports (default: tests/alda_reference/<version>)",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-export existing references")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Timing tolerance in ticks of the coarser file at the local tempo (default: 1)",
    )
    parser.add_argument("--examples", type=int, default=5, help="Lines shown per category")
    args = parser.parse_args()

    found = find_alda(args.alda)
    if found:
        alda, env = found
        version = alda_version(alda, env)
        reference = args.reference or REFERENCE_ROOT / version
    else:
        reference = args.reference or newest_reference()
        if reference is None or not reference.is_dir():
            print(
                "Error: no committed reference and alda/alda-player not found. "
                "Install them from https://alda.io/install/ or pass --alda DIR.",
                file=sys.stderr,
            )
            return 2
        version = reference.name
    print(f"Reference: alda {version}, exports in {reference}\n")

    results: dict[Path, Diff] = {}
    for source in corpus(args.paths):
        midi = reference / source.with_suffix(".mid")
        error = midi.with_suffix(".error")
        if found and (args.refresh or not (midi.exists() or error.exists())):
            message = export_reference(alda, env, source, midi)
            if message:
                error.write_text(message + "\n", encoding="utf-8")
            else:
                error.unlink(missing_ok=True)
        results[source] = compare_file(source, reference, args.tolerance, args.examples)

    totals: Counter = Counter()
    for name, diff in results.items():
        if diff.error:
            print(f"ERROR  {name}\n       {diff.error.splitlines()[0]}")
            totals["error"] += 1
            continue
        if not diff.counts:
            print(f"MATCH  {name}")
            continue
        print(f"DIFF   {name}")
        for category, count in sorted(diff.counts.items()):
            print(f"       {category}: {count}")
            for line in diff.samples[category]:
                print(f"         {line}")
        totals.update(diff.counts)

    matched = sum(1 for d in results.values() if not d.error and not d.counts)
    print(f"\n{matched}/{len(results)} files match alda {version}")
    for category, count in sorted(totals.items()):
        print(f"  {category}: {count}")
    return 0 if matched == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
