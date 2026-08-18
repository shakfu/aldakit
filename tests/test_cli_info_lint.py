"""Tests for the `aldakit info` and `aldakit lint` subcommands.

`lint` is meant to be usable in a build, so its exit status is part of its
contract: 0 clean, 1 when something is wrong, 2 when the score does not parse.
"""

from __future__ import annotations

import pytest

from aldakit.cli import create_parser, main


@pytest.fixture
def score(tmp_path):
    def write(source: str, name: str = "song.alda"):
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        return str(path)

    return write


class TestArgumentParsing:
    @pytest.mark.parametrize("command", ["info", "lint"])
    def test_accepts_a_file(self, command):
        args = create_parser().parse_args([command, "song.alda"])
        assert args.command == command
        assert str(args.file) == "song.alda"

    @pytest.mark.parametrize("command", ["info", "lint"])
    def test_accepts_eval(self, command):
        args = create_parser().parse_args([command, "-e", "piano: c"])
        assert args.eval == "piano: c"

    def test_lint_flags(self):
        args = create_parser().parse_args(["lint", "-e", "piano: c", "-q", "--strict"])
        assert args.quiet is True
        assert args.strict is True


class TestInfo:
    def test_summarises_a_file(self, score, capsys):
        path = score("piano: (tempo 90) c d e\ncello: c2 d2\n")
        assert main(["info", path]) == 0

        out = capsys.readouterr().out
        assert "parts:    2" in out
        assert "notes:    5" in out
        assert "90 bpm" in out
        assert "midi-acoustic-grand-piano" in out
        assert "midi-cello" in out

    def test_summarises_eval_code(self, capsys):
        assert main(["info", "-e", "piano: c d e"]) == 0
        out = capsys.readouterr().out
        assert "<eval>" in out
        assert "notes:    3" in out

    def test_reports_channels_and_note_counts(self, capsys):
        assert main(["info", "-e", "piano: c d\nviolin: e"]) == 0
        lines = [line.split() for line in capsys.readouterr().out.splitlines()]
        rows = {
            parts[0]: parts
            for parts in lines
            if parts and parts[0] in ("piano", "violin")
        }
        assert rows["piano"][-2:] == ["0", "2"]  # channel 0, two notes
        assert rows["violin"][-2:] == ["1", "1"]

    def test_shows_every_channel_a_part_moves_between(self, capsys):
        """A score with more parts than channels hands channels on."""
        instruments = [
            "midi-electric-piano-1",
            "violin",
            "viola",
            "cello",
            "flute",
            "oboe",
            "clarinet",
            "bassoon",
            "trumpet",
            "trombone",
            "tuba",
            "harp",
            "banjo",
            "sitar",
            "koto",
            "organ",
        ]
        source = "\n".join(
            [f"{instruments[0]}: c4 r1*2 c4"]
            + [f"{name}: c4" for name in instruments[1:15]]
            + [f"{instruments[15]}: r2 c1*4"]
        )
        assert main(["info", "-e", source]) == 0
        rows = [
            line.split()
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith(instruments[0])
        ]
        assert rows[0][-2] == "0,1"

    def test_shows_a_dash_for_a_part_that_never_sounds(self, capsys):
        """Once channels are being reused, a silent part is given none."""
        instruments = [
            "midi-electric-piano-1",
            "violin",
            "viola",
            "cello",
            "flute",
            "oboe",
            "clarinet",
            "bassoon",
            "trumpet",
            "trombone",
            "tuba",
            "harp",
            "banjo",
            "sitar",
            "koto",
            "organ",
        ]
        source = "\n".join(
            [f"{name}: {'r1 ' * i}c1" for i, name in enumerate(instruments)]
            + ["midi-kalimba:"]
        )
        assert main(["info", "-e", source]) == 0
        rows = [
            line.split()
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("midi-kalimba")
        ]
        assert rows[0][-2:] == ["-", "0"]

    def test_mentions_findings_and_points_at_lint(self, capsys):
        assert main(["info", "-e", "bogus: c"]) == 0
        out = capsys.readouterr().out
        assert "finding(s)" in out
        assert "aldakit lint" in out

    def test_parse_error_exits_nonzero(self, score, capsys):
        path = score("piano: [c d")
        assert main(["info", path]) == 1
        assert "Parse error" in capsys.readouterr().err

    def test_key_signature_and_transposition_are_shown(self, capsys):
        assert main(["info", "-e", 'piano: (key-sig "f+") (transpose 3) c']) == 0
        out = capsys.readouterr().out
        assert "key f+" in out
        assert "transposed +3" in out


class TestLint:
    def test_clean_score_exits_zero(self, score, capsys):
        path = score("piano: c d e\n")
        assert main(["lint", path]) == 0
        assert "no problems found" in capsys.readouterr().out

    def test_warning_alone_exits_zero(self, capsys):
        assert main(["lint", "-e", "bogus: c"]) == 0
        out = capsys.readouterr().out
        assert "unknown-instrument" in out
        assert "1 error(s)" not in out

    def test_error_exits_one(self, capsys):
        assert main(["lint", "-e", "piano: nosuchvar"]) == 1
        assert "undefined-variable" in capsys.readouterr().out

    def test_strict_makes_warnings_fail(self, capsys):
        assert main(["lint", "-e", "bogus: c", "--strict"]) == 1

    def test_strict_still_passes_a_clean_score(self, capsys):
        assert main(["lint", "-e", "piano: c", "--strict"]) == 0

    def test_quiet_prints_nothing(self, capsys):
        assert main(["lint", "-e", "piano: nosuchvar", "--quiet"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_parse_error_exits_two(self, capsys):
        assert main(["lint", "-e", "piano: [c d"]) == 2
        assert "Parse error" in capsys.readouterr().err

    def test_quiet_parse_error_still_exits_two(self, capsys):
        assert main(["lint", "-e", "piano: [c d", "--quiet"]) == 2
        assert capsys.readouterr().err == ""

    def test_findings_carry_position_and_code(self, score, capsys):
        path = score("piano: c\nbogus: d\n")
        main(["lint", path])
        out = capsys.readouterr().out
        assert ":2:1:" in out
        assert "[unknown-instrument]" in out

    def test_reads_stdin(self, monkeypatch, capsys):
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("piano: c d e\n"))
        assert main(["lint", "-"]) == 0
        assert "no problems found" in capsys.readouterr().out
