"""Properties the language reference promises, checked over the whole corpus.

These are deliberately not golden fixtures. A fixture records what the
implementation currently does, so it can only ever agree with the code that
generated it: when a tie stopped carrying its duration across a barline, the
fixtures preserved the wrong sound for four examples without complaint,
because they had been regenerated from the broken parser.

A property states what `docs/reference.md` promises and is true independently
of any implementation. "Barlines are purely visual and have no effect on
timing" is the sentence the tie bug violated, and the first test here is that
sentence made executable. It fails against the old parser on every example
that holds a note across a bar.

Add to this file when the docs make a claim of the form "X has no musical
effect" or "X and Y mean the same thing".
"""

import re
from pathlib import Path

import pytest

from aldakit import generate_midi, parse
from aldakit.parser import Parser
from aldakit.scanner import Scanner
from aldakit.tokens import TokenType

from tests.helpers import EXAMPLES

SHARED_SUITE = Path(__file__).parent / "shared_suite"
CORPUS = sorted(EXAMPLES.glob("*.alda")) + sorted(SHARED_SUITE.glob("*.alda"))


def _music(sequence):
    """Everything about a sequence that can be heard."""
    return (
        [
            (
                round(n.start_time, 9),
                n.pitch,
                n.channel,
                round(n.duration, 9),
                n.velocity,
            )
            for n in sequence.notes
        ],
        [
            (round(c.time, 9), c.channel, c.control, c.value)
            for c in sequence.control_changes
        ],
        [
            (round(p.time, 9), p.channel, int(p.program))
            for p in sequence.program_changes
        ],
    )


def music_of(source: str, name: str = "<test>"):
    """Everything about a score that can be heard."""
    return _music(generate_midi(parse(source, name)))


def music_without_barlines(source: str, name: str):
    """The same score with every barline token removed before parsing.

    Removing barlines from the *AST* would prove nothing: a parser that has
    already dropped a tied duration produces the same wrong music either way.
    The tokens have to go before the parser sees them, so that the two sides
    of the comparison are two independent parses.
    """
    tokens = [
        token
        for token in Scanner(source, name).scan()
        if token.type != TokenType.BARLINE
    ]
    return _music(generate_midi(Parser(tokens).parse()))


def _corpus_with(predicate):
    """Corpus files the property has something to say about."""
    return [p for p in CORPUS if predicate(p.read_text(encoding="utf-8"))]


BARLINE_CORPUS = _corpus_with(lambda text: "|" in text)
COMMENT_CORPUS = _corpus_with(lambda text: "#" in text)
SPACING_CORPUS = _corpus_with(lambda text: re.search(r"[ \t]{2,}", text))


class TestBarlinesAreVisualOnly:
    """docs/reference.md: "Barlines are purely visual and have no effect on
    timing." A tie crossing one used to lose the duration that followed."""

    def test_the_corpus_exercises_this(self):
        assert len(BARLINE_CORPUS) >= 15, (
            f"only {len(BARLINE_CORPUS)} corpus files use barlines; this "
            "property is not being exercised"
        )

    @pytest.mark.parametrize("path", BARLINE_CORPUS, ids=lambda p: p.name)
    def test_removing_every_barline_changes_nothing(self, path):
        source = path.read_text(encoding="utf-8")
        assert music_without_barlines(source, path.name) == music_of(
            source, path.name
        ), f"{path.name} sounds different without its barlines"

    @pytest.mark.parametrize(
        "barred,plain",
        [
            ("piano: c4~|2", "piano: c4~2"),
            ("piano: c4 | ~2", "piano: c4~2"),
            ("piano: c4~|~2", "piano: c4~2"),
            ("piano: c1~|1~|1", "piano: c1~1~1"),
            ("piano: a-8~|2.", "piano: a-8~2."),
        ],
    )
    def test_a_tie_means_the_same_with_or_without_the_bar(self, barred, plain):
        assert music_of(barred) == music_of(plain)


class TestCommentsAreNotMusic:
    """docs/reference.md documents `#` as a comment to end of line."""

    @pytest.mark.parametrize("path", COMMENT_CORPUS, ids=lambda p: p.name)
    def test_removing_comments_changes_nothing(self, path):
        source = path.read_text(encoding="utf-8")
        stripped = "\n".join(
            re.sub(r"#.*$", "", line) for line in source.splitlines()
        )
        assert music_of(stripped, path.name) == music_of(source, path.name), (
            f"{path.name} sounds different with its comments removed"
        )


class TestSpacingIsNotMusic:
    """Whitespace separates tokens; how much of it there is means nothing."""

    @pytest.mark.parametrize("path", SPACING_CORPUS, ids=lambda p: p.name)
    def test_collapsing_runs_of_spaces_changes_nothing(self, path):
        source = path.read_text(encoding="utf-8")
        collapsed = "\n".join(
            re.sub(r"[ \t]+", " ", line) for line in source.splitlines()
        )
        assert music_of(collapsed, path.name) == music_of(source, path.name), (
            f"{path.name} sounds different with its spacing normalised"
        )
