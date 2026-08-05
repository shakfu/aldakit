#!/usr/bin/env python3
"""Generate the General MIDI instrument table from the Alda language docs.

Reads ``docs/alda-language/list-of-instruments.md`` -- the canonical list of
Alda instrument names and aliases -- and writes
``src/aldakit/midi/_instruments.py``.

The document lists instruments grouped by General MIDI patch group, eight per
group, in program order. Each bullet has the form::

    * canonical-name (alias, alias, ...)

Run from the project root::

    python scripts/gen_instruments.py

The generated module is committed so that the package has no build-time
dependency on the docs directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOC_PATH = Path("docs/alda-language/list-of-instruments.md")
OUT_PATH = Path("src/aldakit/midi/_instruments.py")

# Patch groups in General MIDI order. Each contributes eight programs.
EXPECTED_GROUPS = 16
PROGRAMS_PER_GROUP = 8

HEADING_RE = re.compile(r"^###\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\*\s+(?P<name>[a-z0-9+\-]+)\s*(?:\((?P<aliases>[^)]*)\)?)?\s*$")


def parse_doc(text: str) -> list[tuple[str, str, list[str]]]:
    """Parse the instrument list into (group, canonical_name, aliases) tuples."""
    entries: list[tuple[str, str, list[str]]] = []
    group = ""
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            group = heading.group(1)
            continue
        bullet = BULLET_RE.match(line.strip())
        if not bullet:
            continue
        name = bullet.group("name")
        raw_aliases = bullet.group("aliases") or ""
        aliases = [a.strip() for a in raw_aliases.split(",") if a.strip()]
        entries.append((group, name, aliases))
    return entries


def build_table(entries: list[tuple[str, str, list[str]]]) -> tuple[list, dict[str, int]]:
    """Assign program numbers and build the name -> program mapping."""
    if len(entries) != EXPECTED_GROUPS * PROGRAMS_PER_GROUP:
        raise SystemExit(
            f"Expected {EXPECTED_GROUPS * PROGRAMS_PER_GROUP} instruments, "
            f"parsed {len(entries)}. The document format may have changed."
        )

    canonical: list[tuple[int, str, str, list[str]]] = []
    table: dict[str, int] = {}

    for program, (group, name, aliases) in enumerate(entries):
        canonical.append((program, group, name, aliases))
        for key in [name, *aliases]:
            if key in table and table[key] != program:
                raise SystemExit(
                    f"Duplicate instrument name {key!r}: program "
                    f"{table[key]} and {program}"
                )
            table[key] = program

    return canonical, table


def render(canonical: list, table: dict[str, int]) -> str:
    lines = [
        '"""General MIDI instrument names for Alda.',
        "",
        "GENERATED FILE - do not edit by hand.",
        "Regenerate with ``python scripts/gen_instruments.py``, which derives this",
        "table from ``docs/alda-language/list-of-instruments.md``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "# The special percussion instrument is not a General MIDI program: it selects",
        "# MIDI channel 10 (9 when zero-indexed), where note numbers pick a drum sound.",
        'PERCUSSION_NAMES: frozenset[str] = frozenset({"midi-percussion", "percussion"})',
        "",
        "# Canonical Alda name for each of the 128 General MIDI programs.",
        "PROGRAM_NAMES: tuple[str, ...] = (",
    ]
    for program, _group, name, _aliases in canonical:
        lines.append(f'    "{name}",  # {program}')
    lines.append(")")
    lines.append("")
    lines.append("# Every accepted instrument name (canonical and alias) to its GM program.")
    lines.append("INSTRUMENT_PROGRAMS: dict[str, int] = {")

    current_group = None
    for program, group, name, aliases in canonical:
        if group != current_group:
            lines.append(f"    # {group}")
            current_group = group
        lines.append(f'    "{name}": {program},')
        for alias in aliases:
            lines.append(f'    "{alias}": {program},')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def canonical_name(program: int) -> str:")
    lines.append('    """Return the canonical Alda name for a GM program number."""')
    lines.append("    if not 0 <= program < len(PROGRAM_NAMES):")
    lines.append('        raise ValueError(f"Program out of range: {program}")')
    lines.append("    return PROGRAM_NAMES[program]")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    if not DOC_PATH.exists():
        print(f"Not found: {DOC_PATH}. Run from the project root.", file=sys.stderr)
        return 1

    entries = parse_doc(DOC_PATH.read_text(encoding="utf-8"))
    canonical, table = build_table(entries)
    OUT_PATH.write_text(render(canonical, table), encoding="utf-8")
    print(f"Wrote {OUT_PATH}: {len(canonical)} programs, {len(table)} names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
