"""Tests for rendering a score to audio without an output device.

Playing a score takes as long as the score lasts and produces nothing a test
can look at. Rendering runs the same synthesis loop with no device attached,
so these tests can assert on the audio itself: that it is the right length,
that it is not silence, that the notes land where the score puts them, and
that the same score twice produces the same bytes.

Everything here needs a SoundFont, which is a large file downloaded on
demand, so the tests skip when none is installed rather than fail.
"""

from __future__ import annotations

import array
import math
import wave
from pathlib import Path

import pytest

from aldakit import Score, parse, render, render_file
from aldakit.midi.generator import MidiGenerator

try:
    from aldakit import _tsf  # noqa: F401
    from aldakit.midi.render import (
        DEFAULT_TAIL_SECONDS,
        render_pcm,
        render_wav,
    )
    from aldakit.midi.soundfont import find_soundfont

    TSF_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the build
    TSF_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TSF_AVAILABLE, reason="TinySoundFont backend not available"
)

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def soundfont():
    path = find_soundfont()
    if path is None:
        pytest.skip("No SoundFont installed")
    return path


def sequence_for(source: str):
    return MidiGenerator().generate(parse(source))


def samples(path: Path) -> tuple[array.array, int]:
    """The left channel of a WAV file, and its sample rate."""
    with wave.open(str(path)) as wav:
        assert wav.getnchannels() == 2
        assert wav.getsampwidth() == 2
        rate = wav.getframerate()
        data = array.array("h")
        data.frombytes(wav.readframes(wav.getnframes()))
    return data[0::2], rate


def rms_windows(channel: array.array, rate: int, seconds: float) -> list[float]:
    """Loudness of each window, for finding where a file has sound in it."""
    size = int(rate * seconds)
    return [
        math.sqrt(sum(s * s for s in channel[i : i + size]) / size)
        for i in range(0, len(channel) - size, size)
    ]


class TestRenderPcm:
    def test_renders_as_long_as_the_score_sounds_plus_a_tail(self, soundfont):
        """The length matches MidiSequence.duration(), which is when the last
        note stops sounding -- not the written length of the last note, which
        at the default quantization is longer than the note is heard for."""
        sequence = sequence_for("piano: (tempo 120) c1")
        pcm, rate, _peak = render_pcm(sequence, soundfont, tail=0.5)
        frames = len(pcm) // 4  # stereo, two bytes a sample
        expected = (sequence.duration() + 0.5) * rate
        assert frames == pytest.approx(expected, rel=0.01)
        assert sequence.duration() == pytest.approx(1.8)  # 2s note at quant 90%

    def test_the_tail_is_configurable(self, soundfont):
        sequence = sequence_for("piano: (tempo 120) c1")
        short, rate, _ = render_pcm(sequence, soundfont, tail=0.0)
        long, _, _ = render_pcm(sequence, soundfont, tail=2.0)
        assert (len(long) - len(short)) // 4 == pytest.approx(rate * 2, rel=0.01)

    def test_a_score_with_no_notes_renders_nothing(self, soundfont):
        pcm, _rate, _peak = render_pcm(sequence_for("piano:"), soundfont, tail=0.0)
        assert pcm == b""

    def test_the_same_score_renders_the_same_bytes(self, soundfont):
        sequence = sequence_for("piano: c d e f")
        first, _, _ = render_pcm(sequence, soundfont)
        second, _, _ = render_pcm(sequence, soundfont)
        assert first == second

    def test_no_soundfont_installed_says_how_to_get_one(self, monkeypatch):
        import aldakit.midi.render as render_module

        monkeypatch.setattr(render_module, "find_soundfont", lambda: None)
        with pytest.raises(FileNotFoundError, match="soundfont install"):
            render_pcm(sequence_for("piano: c"))

    def test_a_missing_soundfont_is_reported(self, tmp_path):
        missing = tmp_path / "nope.sf2"
        with pytest.raises(ValueError, match="Could not load SoundFont"):
            render_pcm(sequence_for("piano: c"), missing)


class TestGain:
    """Gain is a volume factor, where 1.0 is unity."""

    def test_halving_the_gain_halves_the_peak(self, soundfont):
        sequence = sequence_for("piano: (tempo 120) c e g")
        _, _, full = render_pcm(sequence, soundfont, gain=1.0)
        _, _, half = render_pcm(sequence, soundfont, gain=0.5)
        assert half == pytest.approx(full / 2, rel=0.01)

    def test_zero_gain_is_silence(self, soundfont):
        sequence = sequence_for("piano: c e g")
        pcm, _rate, peak = render_pcm(sequence, soundfont, gain=0.0)
        assert peak == 0.0
        assert set(pcm) == {0}

    def test_the_peak_reports_a_mix_that_clipped(self, soundfont):
        sequence = sequence_for("piano: (tempo 120) c e g")
        _, _, peak = render_pcm(sequence, soundfont, gain=2.0)
        quiet = render_pcm(sequence, soundfont, gain=0.05)[2]
        # Whatever the SoundFont's level, the loud render is the louder one,
        # and the quiet one leaves headroom.
        assert peak > quiet
        assert quiet < 1.0


class TestRenderWav:
    def test_writes_a_readable_stereo_wav(self, soundfont, tmp_path):
        sequence = sequence_for("piano: (tempo 120) c d e")
        path, _peak = render_wav(sequence, tmp_path / "out.wav", soundfont)
        with wave.open(str(path)) as wav:
            assert wav.getnchannels() == 2
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 44100
            assert wav.getnframes() > 0

    def test_adds_the_suffix_when_there_is_none(self, soundfont, tmp_path):
        sequence = sequence_for("piano: c")
        path, _peak = render_wav(sequence, tmp_path / "out", soundfont)
        assert path.name == "out.wav"
        assert path.exists()

    def test_the_audio_is_not_silence(self, soundfont, tmp_path):
        sequence = sequence_for("piano: (tempo 120) c d e f g")
        path, _peak = render_wav(sequence, tmp_path / "scale.wav", soundfont)
        left, _rate = samples(path)
        assert max(abs(s) for s in left) > 1000

    def test_the_notes_land_where_the_score_puts_them(self, soundfont, tmp_path):
        """Two notes a bar apart leave a measurable silence between them."""
        sequence = sequence_for("piano: (tempo 120) (quant 50) c4 r2 r4 c4")
        path, _peak = render_wav(sequence, tmp_path / "gap.wav", soundfont, tail=0.1)
        left, rate = samples(path)
        levels = rms_windows(left, rate, 0.25)
        # The first note sounds, the middle two beats are quiet, and the
        # second note sounds again.
        assert levels[0] > 100
        assert min(levels[2:6]) < 50
        assert max(levels[-3:]) > 100


class TestPublicApi:
    def test_score_render(self, soundfont, tmp_path):
        path = Score("piano: c d e").render(tmp_path / "score.wav", soundfont)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_function(self, soundfont, tmp_path):
        path = render("piano: c d e", tmp_path / "api.wav", soundfont)
        assert path.exists()

    def test_render_file_function(self, soundfont, tmp_path):
        path = render_file(
            EXAMPLES / "twinkle.alda", tmp_path / "twinkle.wav", soundfont
        )
        assert path.exists()

    def test_the_default_tail_is_used(self, soundfont, tmp_path):
        score = Score("piano: (tempo 120) c1")
        path = score.render(tmp_path / "tail.wav", soundfont)
        left, rate = samples(path)
        expected = (score.midi.duration() + DEFAULT_TAIL_SECONDS) * rate
        assert len(left) == pytest.approx(expected, rel=0.01)


class TestRenderedExamples:
    """Rendering is the only test that hears what the generator decided."""

    def test_all_instruments_sounds_throughout(self, soundfont, tmp_path):
        """128 instruments through 15 channels: none of them may drop out.

        If a channel were handed on without restating the instrument, or a
        part were left on a channel another part had taken, this score would
        have holes in it.
        """
        source = (EXAMPLES / "all-instruments.alda").read_text(encoding="utf-8")
        path, _peak = render_wav(
            sequence_for(source), tmp_path / "all.wav", soundfont, gain=0.4
        )
        left, rate = samples(path)
        levels = rms_windows(left, rate, 0.25)
        assert len(levels) > 500
        assert min(levels) > 5, "the score falls silent somewhere"

    def test_a_percussion_score_renders(self, soundfont, tmp_path):
        source = "midi-percussion: o2 c8 d c d e f"
        path, _peak = render_wav(
            sequence_for(source), tmp_path / "drums.wav", soundfont
        )
        left, _rate = samples(path)
        assert max(abs(s) for s in left) > 500


class TestRenderCommand:
    """The `aldakit render` subcommand."""

    def test_renders_a_file_next_to_the_source_by_default(
        self, soundfont, tmp_path, capsys
    ):
        from aldakit.cli import main

        source = tmp_path / "song.alda"
        source.write_text("piano: c d e", encoding="utf-8")
        assert main(["render", str(source), "-sf", str(soundfont)]) == 0
        assert (tmp_path / "song.wav").exists()
        assert "Wrote" in capsys.readouterr().out

    def test_renders_code_given_on_the_command_line(self, soundfont, tmp_path, capsys):
        from aldakit.cli import main

        out = tmp_path / "eval.wav"
        assert (
            main(["render", "-e", "piano: c d", "-o", str(out), "-sf", str(soundfont)])
            == 0
        )
        assert out.exists()

    def test_code_without_an_output_file_is_an_error(self, capsys):
        from aldakit.cli import main

        assert main(["render", "-e", "piano: c"]) == 1
        assert "No output file" in capsys.readouterr().err

    def test_a_parse_error_exits_nonzero(self, capsys):
        from aldakit.cli import main

        assert main(["render", "-e", "piano: [c d", "-o", "x.wav"]) == 1
        assert "Parse error" in capsys.readouterr().err

    def test_a_missing_soundfont_exits_nonzero(self, tmp_path, capsys):
        from aldakit.cli import main

        assert (
            main(
                [
                    "render",
                    "-e",
                    "piano: c",
                    "-o",
                    str(tmp_path / "x.wav"),
                    "-sf",
                    str(tmp_path / "nope.sf2"),
                ]
            )
            == 1
        )
        assert "Error" in capsys.readouterr().err

    def test_a_clipped_render_is_reported_with_a_workable_gain(
        self, soundfont, tmp_path, capsys
    ):
        """The suggested gain must actually clear the clipping."""
        from aldakit.cli import main

        loud = "piano: (tempo 120) (vol 100) c/e/g/> c/e/g"
        assert (
            main(
                [
                    "render",
                    "-e",
                    loud,
                    "-o",
                    str(tmp_path / "loud.wav"),
                    "-sf",
                    str(soundfont),
                    "--gain",
                    "2.0",
                ]
            )
            == 0
        )
        warning = capsys.readouterr().err
        assert "clipped" in warning

        suggested = float(warning.split("--gain ")[1].split()[0].rstrip("."))
        _, peak = render_wav(
            sequence_for(loud), tmp_path / "quiet.wav", soundfont, gain=suggested
        )
        assert peak <= 1.0
