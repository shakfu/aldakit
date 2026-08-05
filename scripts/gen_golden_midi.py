#!/usr/bin/env python3
"""Regenerate the golden MIDI fixtures for the bundled examples.

The fixtures pin the exact notes, channels, programs, timings and velocities
that every file in ``examples/`` generates, so any unintended change to the
scanner, parser or MIDI generator shows up as a test failure rather than as
music that quietly sounds different.

Run from the project root after an intentional behaviour change::

    python scripts/gen_golden_midi.py

Review the resulting diff carefully: a change here means scores that used to
sound one way now sound another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aldakit import generate_midi, parse  # noqa: E402
from tests.helpers import EXAMPLES, GOLDEN_DIR, midi_fingerprint  # noqa: E402


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden: dict[str, dict] = {}

    for path in sorted(EXAMPLES.glob("*.alda")):
        sequence = generate_midi(parse(path.read_text(), str(path)))
        golden[path.name] = midi_fingerprint(sequence)

    out = GOLDEN_DIR / "examples.json"
    out.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")

    note_count = sum(len(v["notes"]) for v in golden.values())
    print(f"Wrote {out}: {len(golden)} examples, {note_count} notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
