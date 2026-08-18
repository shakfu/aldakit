"""Golden-audio tests: every bundled example must keep sounding the same.

The golden MIDI fixtures next door pin what the generator decided. They cannot
tell whether any of it was heard. Every defect this project has had in that
area -- an instrument selected on a channel another part had taken, percussion
routed to a melodic channel, a pan applied to the wrong part -- produced
perfectly well-formed MIDI, and the only way to catch that class of bug is to
render the score and look at the audio.

What is compared is coarse on purpose: loudness per channel over quarter
second windows, plus the peak and the length. That is enough to fail when an
instrument drops out, a part moves in time, a pan flips sides or the mix
changes level, while surviving the last-bit differences between one platform's
floating point and another's.

The fixtures are tied to one checksum-pinned SoundFont, because the samples
belong to the SoundFont rather than to aldakit. Without it there is nothing to
compare against, so these tests skip::

    aldakit soundfont install TimGM6mb

When a change to the generator or the renderer is intentional, regenerate::

    python scripts/gen_golden_audio.py
"""

from __future__ import annotations

import json
import os

import pytest

from aldakit import generate_midi, parse
from tests.helpers import (
    AUDIO_ABSOLUTE_TOLERANCE,
    AUDIO_GAIN,
    AUDIO_RELATIVE_TOLERANCE,
    AUDIO_SOUNDFONT,
    AUDIO_TAIL_SECONDS,
    AUDIO_WINDOW_SECONDS,
    EXAMPLES,
    GOLDEN_DIR,
    audio_fingerprint,
    pinned_soundfont,
)

try:
    from aldakit import _tsf  # noqa: F401
    from aldakit.midi.render import render_pcm

    TSF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the build
    TSF_AVAILABLE = False

GOLDEN_PATH = GOLDEN_DIR / "audio.json"

if os.environ.get("ALDAKIT_REQUIRE_AUDIO_FIXTURES") and not TSF_AVAILABLE:
    raise RuntimeError(
        "ALDAKIT_REQUIRE_AUDIO_FIXTURES is set but the _tsf native module is "
        "not available, so nothing would be compared."
    )

pytestmark = pytest.mark.skipif(
    not TSF_AVAILABLE, reason="TinySoundFont backend not available"
)


#: Set in CI, where a skipped audio comparison is a green tick for a check
#: that never ran. Everywhere else the SoundFont is a 6 MB download that a
#: contributor should not need in order to run the suite.
REQUIRE_FIXTURES = "ALDAKIT_REQUIRE_AUDIO_FIXTURES"


@pytest.fixture(scope="module")
def soundfont():
    path = pinned_soundfont()
    if path is None:
        message = (
            f"The audio fixtures need the pinned {AUDIO_SOUNDFONT} SoundFont, "
            f"which is either not installed or not the pinned version. "
            f"Run: aldakit soundfont install {AUDIO_SOUNDFONT}"
        )
        if os.environ.get(REQUIRE_FIXTURES):
            pytest.fail(f"{REQUIRE_FIXTURES} is set but {message}")
        pytest.skip(message)
    return path


@pytest.fixture(scope="module")
def golden() -> dict:
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. Run: python scripts/gen_golden_audio.py"
    )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _examples():
    return sorted(EXAMPLES.glob("*.alda"))


def compare_windows(actual: list[int], expected: list[int], channel: str) -> None:
    """Compare one channel window by window, and say what went wrong.

    ``pytest.approx`` on a list of several hundred numbers reports that the
    lists differ and prints both. What is needed instead is where the audio
    diverged, by how much, and -- the distinction that matters most -- whether
    one window moved or all of them did. One window is a note that changed;
    every window is a change in level, or a tolerance too tight for the
    platform the fixtures were not generated on.
    """
    assert len(actual) == len(expected), (
        f"{channel}: {len(actual)} windows, fixture has {len(expected)}"
    )

    off = []
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got == pytest.approx(
            want, rel=AUDIO_RELATIVE_TOLERANCE, abs=AUDIO_ABSOLUTE_TOLERANCE
        ):
            continue
        off.append((abs(got - want) / max(want, 1), index, got, want))

    if not off:
        return

    off.sort(reverse=True)
    _, index, got, want = off[0]
    at = index * AUDIO_WINDOW_SECONDS
    pytest.fail(
        f"{channel}: {len(off)} of {len(expected)} windows differ. "
        f"Worst at {at:.2f}s (window {index}): {got} against {want} in the "
        f"fixture, {got / max(want, 1):.2f}x. "
        f"Tolerance is {AUDIO_RELATIVE_TOLERANCE:.0%} or "
        f"{AUDIO_ABSOLUTE_TOLERANCE} absolute. "
        f"If every window is out by about the same factor the render changed "
        f"level; if one is, the score changed there."
    )


def render_fingerprint(path, soundfont) -> dict:
    sequence = generate_midi(parse(path.read_text(encoding="utf-8"), str(path)))
    pcm, sample_rate, peak = render_pcm(
        sequence, soundfont, gain=AUDIO_GAIN, tail=AUDIO_TAIL_SECONDS
    )
    return audio_fingerprint(pcm, sample_rate, peak)


class TestGoldenCoverage:
    def test_every_example_has_a_fixture(self, golden):
        missing = [p.name for p in _examples() if p.name not in golden]
        assert missing == [], (
            f"examples without an audio fixture: {missing}. "
            "Run: python scripts/gen_golden_audio.py"
        )

    def test_no_stale_fixtures(self, golden):
        names = {p.name for p in _examples()}
        stale = sorted(set(golden) - names)
        assert stale == [], f"fixtures for deleted examples: {stale}"

    def test_fixtures_contain_audible_sound(self, golden):
        """A fixture of pure silence would pass any comparison with itself."""
        silent = [
            name
            for name, data in golden.items()
            if max(data["left"] + data["right"], default=0) < 10
        ]
        assert silent == []

    def test_the_same_examples_are_pinned_as_for_midi(self, golden):
        """The two fixture sets must not drift apart in what they cover."""
        midi = json.loads((GOLDEN_DIR / "examples.json").read_text(encoding="utf-8"))
        assert sorted(golden) == sorted(midi)


class TestGoldenAudio:
    @pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
    def test_example_sounds_the_same(self, path, soundfont, golden):
        expected = golden[path.name]
        actual = render_fingerprint(path, soundfont)

        # Length first: everything else is compared window by window, and a
        # file of the wrong length makes those failures hard to read.
        assert actual["sample_rate"] == expected["sample_rate"]
        assert actual["frames"] == expected["frames"]

        assert actual["peak"] == pytest.approx(
            expected["peak"],
            rel=AUDIO_RELATIVE_TOLERANCE,
            abs=AUDIO_ABSOLUTE_TOLERANCE,
        )
        # Channels separately, so a pan that moved says which way it went.
        compare_windows(actual["left"], expected["left"], "left")
        compare_windows(actual["right"], expected["right"], "right")


class TestGoldenInvariants:
    """Properties every fixture must satisfy, independent of its values."""

    def test_channels_are_the_same_length(self, golden):
        for name, data in golden.items():
            assert len(data["left"]) == len(data["right"]), name

    def test_windows_cover_the_frames(self, golden):
        for name, data in golden.items():
            window = data["sample_rate"] * AUDIO_WINDOW_SECONDS
            expected = -(-data["frames"] // window)  # rounded up
            assert len(data["left"]) == expected, name

    def test_nothing_clipped_at_the_fixture_gain(self, golden):
        """Above full scale the renderer clamps, which would flatten a
        difference in level into no difference at all."""
        loud = {name: d["peak"] for name, d in golden.items() if d["peak"] > 1.0}
        assert loud == {}, (
            f"these fixtures clip at gain {AUDIO_GAIN} and cannot detect a "
            f"change in level: {loud}"
        )


class TestPannedExample:
    """The fixture has to be able to tell the channels apart at all."""

    def test_all_instruments_is_panned_across_the_stereo_field(self, golden):
        data = golden["all-instruments.alda"]
        # Every part sets its own panning, so the two channels must differ
        # somewhere. If they did not, a pan regression could not fail here.
        differences = [
            abs(left - right)
            for left, right in zip(data["left"], data["right"])
            if left or right
        ]
        assert max(differences) > 50
