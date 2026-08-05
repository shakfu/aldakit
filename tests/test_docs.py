"""Guards against documentation drifting away from the code.

A doc audit before the 0.2.0 release found README examples using a CLI form that
no longer exists (`aldakit FILE` without the `play` subcommand), a keyword
argument that had been renamed, and a design document importing eight functions
from the wrong module. None of it was caught by anything.

These checks are deliberately narrow and deterministic, covering the three kinds
of claim that can be verified mechanically:

1. every ``from aldakit... import x`` in the docs actually resolves
2. every ```alda block parses and generates MIDI
3. every ``aldakit ...`` command line is accepted by the real argument parser

Prose and illustrative fragments are not checked; those need a human.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import shlex
from pathlib import Path

import pytest

from aldakit import generate_midi, parse
from aldakit.cli import create_parser

ROOT = Path(__file__).parent.parent
FENCE = re.compile(r"^```(\w*)\s*$")

# The upstream Alda language docs describe Alda itself, not aldakit's Python
# API, so only their Alda snippets are aldakit's to guarantee.
UPSTREAM = "alda-language"

# Working documents, not user-facing documentation
SKIP = {"CHANGELOG.md", "REVIEW.md"}


def doc_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").rglob("*.md"))
    files += [ROOT / "TODO.md"]
    return [f for f in files if f.exists() and f.name not in SKIP]


def code_blocks(path: Path):
    """Yield (language, code, line_number) for each fenced block."""
    lang, buf, start = None, [], 0
    for i, line in enumerate(path.read_text().splitlines(), 1):
        match = FENCE.match(line)
        if match and lang is None:
            lang, buf, start = match.group(1) or "text", [], i
        elif match and lang is not None:
            yield lang, "\n".join(buf), start
            lang = None
        elif lang is not None:
            buf.append(line)


def _ids(path: Path) -> str:
    return str(path.relative_to(ROOT))


@pytest.mark.parametrize("path", doc_files(), ids=_ids)
def test_documented_imports_resolve(path: Path):
    """Every aldakit import shown in the docs must actually work."""
    if UPSTREAM in str(path):
        pytest.skip("upstream Alda language docs")

    text = path.read_text()
    broken: list[str] = []

    for match in re.finditer(r"from (aldakit[\w.]*) import ([^\n(]+|\([^)]*\))", text):
        module_name, names = match.group(1), match.group(2)
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            broken.append(f"{module_name} (module does not exist)")
            continue
        # Multi-line import lists often carry comments; those are prose, not
        # names, so strip them before splitting.
        names = re.sub(r"#[^\n]*", "", names)
        for name in re.split(r"[,\s()]+", names):
            name = name.strip()
            if name and name.isidentifier() and not hasattr(module, name):
                broken.append(f"{module_name}.{name}")

    assert broken == [], f"{_ids(path)} documents imports that do not resolve: {broken}"


@pytest.mark.parametrize("path", doc_files(), ids=_ids)
def test_alda_snippets_are_valid(path: Path):
    """Every ```alda block must parse and generate MIDI."""
    failures = []
    for lang, code, line in code_blocks(path):
        if lang != "alda" or not code.strip():
            continue
        try:
            generate_midi(parse(code))
        except Exception as e:
            failures.append(f"line {line}: {type(e).__name__}: {e}")

    assert failures == [], f"{_ids(path)} has invalid Alda:\n" + "\n".join(failures)


@pytest.mark.parametrize("path", doc_files(), ids=_ids)
def test_documented_cli_invocations_are_accepted(path: Path):
    """Every `aldakit ...` command shown must be accepted by the parser."""
    if UPSTREAM in str(path):
        pytest.skip("upstream Alda language docs")

    parser = create_parser()
    rejected = []

    for lang, code, start in code_blocks(path):
        if lang not in ("sh", "bash", "shell"):
            continue
        for offset, raw in enumerate(code.splitlines(), 1):
            line = re.sub(r"^\$\s*", "", raw.strip())
            if not line.startswith("aldakit"):
                continue
            # Skip usage synopses and shell plumbing
            if any(ch in line for ch in "|><[{"):
                continue
            line = re.sub(r"\s+#.*$", "", line)  # trailing comment
            argv = shlex.split(line)[1:]
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    parser.parse_args(argv)
                except SystemExit as e:
                    if e.code not in (0, None):
                        rejected.append(f"line {start + offset}: {line}")

    assert rejected == [], (
        f"{_ids(path)} shows CLI invocations the parser rejects:\n"
        + "\n".join(rejected)
    )


class TestReadmeAccuracy:
    """Specific README claims that were wrong and are worth pinning."""

    README = (ROOT / "README.md").read_text()

    def test_voicing_example_is_accurate(self):
        """README documents voicing(major("c"), [3, 4, 5]) as C3 E4 G5."""
        from aldakit import Score
        from aldakit.compose import major, part, voicing

        assert "C3 E4 G5" in self.README
        score = Score.from_elements(part("piano"), voicing(major("c"), [3, 4, 5]))
        assert sorted(n.pitch for n in score.midi.notes) == [48, 64, 79]

    def test_compose_octave_example_is_accurate(self):
        from aldakit import Score
        from aldakit.compose import note, part

        score = Score.from_elements(part("piano"), note("c", octave=5), note("d"))
        assert score.to_alda() == "piano: o5 c d"
        assert [n.pitch for n in score.midi.notes] == [72, 74]

    def test_diagnostics_example_is_accurate(self):
        from aldakit import Score

        messages = [str(d) for d in Score("bogus-instrument: c d e").diagnostics]
        assert any("Unknown instrument 'bogus-instrument'" in m for m in messages)

    def test_no_stale_subcommandless_invocations(self):
        """`aldakit FILE` without a subcommand no longer works."""
        stale = re.findall(r"^aldakit ([\w./-]+\.alda)", self.README, re.M)
        assert stale == [], f"README uses the removed CLI form: {stale}"
