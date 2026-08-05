"""Tests for CLI argument parity between the play and eval subcommands.

Regression cover for D8: the README documented ``aldakit eval --parse-only``,
but the eval subparser never registered ``--parse-only`` and main() hardcoded
it to False, so the documented invocation exited with a usage error. ``eval``
was likewise missing ``--no-wait``.
"""

from __future__ import annotations

import pytest

from aldakit.cli import create_parser, main


def parse_args(argv):
    return create_parser().parse_args(argv)


# Options that play and eval must both accept
SHARED_OPTIONS = [
    "--output",
    "--parse-only",
    "--no-wait",
    "--verbose",
    "--audio",
    "--soundfont",
    "--virtual-port",
]


class TestSubcommandParity:
    @pytest.mark.parametrize("option", SHARED_OPTIONS)
    def test_play_accepts(self, option):
        dest = option.lstrip("-").replace("-", "_")
        args = parse_args(["play", "x.alda"])
        assert hasattr(args, dest), f"play is missing {option}"

    @pytest.mark.parametrize("option", SHARED_OPTIONS)
    def test_eval_accepts(self, option):
        dest = option.lstrip("-").replace("-", "_")
        args = parse_args(["eval", "piano: c"])
        assert hasattr(args, dest), f"eval is missing {option}"

    def test_both_accept_a_port(self):
        assert parse_args(["play", "x.alda", "--port", "0"]).port == "0"
        assert parse_args(["eval", "piano: c", "--port", "0"]).port == "0"

    def test_eval_keeps_its_short_port_flag(self):
        assert parse_args(["eval", "piano: c", "-p", "0"]).port == "0"


class TestParseOnly:
    def test_eval_parse_only_is_accepted(self):
        args = parse_args(["eval", "piano: c/e/g", "--parse-only"])
        assert args.parse_only is True

    def test_play_parse_only_is_accepted(self):
        args = parse_args(["play", "x.alda", "--parse-only"])
        assert args.parse_only is True

    def test_eval_parse_only_prints_ast(self, capsys):
        assert main(["eval", "piano: c/e/g", "--parse-only"]) == 0
        out = capsys.readouterr().out
        assert "RootNode" in out
        assert "ChordNode" in out

    def test_eval_parse_only_does_not_play(self, capsys):
        """--parse-only must return before touching a backend."""
        assert main(["eval", "piano: c", "--parse-only"]) == 0
        assert "Error" not in capsys.readouterr().err

    def test_parse_only_defaults_to_false(self):
        assert parse_args(["eval", "piano: c"]).parse_only is False


class TestNoWait:
    def test_eval_no_wait_is_accepted(self):
        assert parse_args(["eval", "piano: c", "--no-wait"]).no_wait is True

    def test_play_no_wait_is_accepted(self):
        assert parse_args(["play", "x.alda", "--no-wait"]).no_wait is True

    def test_no_wait_defaults_to_false(self):
        assert parse_args(["eval", "piano: c"]).no_wait is False

    def test_no_wait_help_mentions_process_exit(self):
        """The flag cannot outlive the process; say so rather than imply it can."""
        help_text = create_parser().format_help()
        parser = create_parser()
        for action in parser._subparsers._group_actions[0].choices["eval"]._actions:
            if "--no-wait" in action.option_strings:
                assert "exits" in (action.help or "")
                return
        pytest.fail("--no-wait not found on the eval subcommand")
        assert help_text


class TestOutputToFile:
    def test_eval_writes_midi(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        assert main(["eval", "piano: c d e", "-o", str(out)]) == 0
        assert out.exists()
        assert out.read_bytes()[:4] == b"MThd"

    def test_play_writes_midi(self, tmp_path):
        source = tmp_path / "s.alda"
        source.write_text("piano: c d e", encoding="utf-8")
        out = tmp_path / "out.mid"
        assert main(["play", str(source), "-o", str(out)]) == 0
        assert out.exists()


class TestDiagnosticsReporting:
    """Problems that change what is heard must be reported, not swallowed."""

    def test_unknown_instrument_warns(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        main(["eval", "bogus-instrument: c d e", "-o", str(out)])
        err = capsys.readouterr().err
        assert "Warning:" in err
        assert "Unknown instrument 'bogus-instrument'" in err

    def test_undefined_variable_warns(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        main(["eval", "piano: c nosuchvar d", "-o", str(out)])
        assert "Undefined variable" in capsys.readouterr().err

    def test_clean_score_produces_no_warnings(self, tmp_path, capsys):
        out = tmp_path / "out.mid"
        main(["eval", "piano: c d e", "-o", str(out)])
        assert "Warning:" not in capsys.readouterr().err
