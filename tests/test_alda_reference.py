"""Every corpus score must produce the MIDI the reference Alda produces.

Compares aldakit with the committed `alda export` output in
tests/alda_reference/, using scripts/alda_diff.py. The only differences allowed
are the deliberate deviations in docs/dev/alda-deviations.md, which the
comparison is built not to report. Alda is not needed to run this.

A new score needs a reference export: install Alda and run `make alda-diff`.
"""

from __future__ import annotations

import pytest

from tests.helpers import load_alda_diff

alda_diff = load_alda_diff()
REFERENCE = alda_diff.newest_reference()
CORPUS = alda_diff.corpus([])


def test_a_reference_is_committed():
    assert REFERENCE is not None, "tests/alda_reference/ holds no Alda exports"


def test_the_corpus_is_not_empty():
    # Guards the parametrized test below against passing on nothing
    assert len(CORPUS) >= 60


@pytest.mark.skipif(REFERENCE is None, reason="no committed reference")
@pytest.mark.parametrize("source", CORPUS, ids=str)
def test_matches_alda(source):
    diff = alda_diff.compare_file(source, REFERENCE, ticks=1.0, limit=5)
    assert diff.error is None, diff.error
    report = [
        f"{category}: {count} (e.g. {'; '.join(diff.samples[category])})"
        for category, count in sorted(diff.counts.items())
    ]
    assert not report, f"{source} differs from alda {REFERENCE.name}:\n" + "\n".join(
        report
    )
