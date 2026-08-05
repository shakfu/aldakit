"""Tests for the REPL's file commands and startup file argument.

The REPL previously had no way to reach the filesystem: it could not open the
very files `aldakit play` exists to read, nor save an improvisation. This covers
`:load`, `:save`, `:ls`, `:cd`, `:pwd`, `:clear` and the
`aldakit repl FILE` startup argument.

Command handling lives in `handle_command()` rather than inline in the prompt
loop precisely so it can be tested: `PromptSession` requires a TTY, so anything
inside the loop is unreachable from a test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aldakit.repl import (
    COMMAND_NAMES,
    PATH_COMMANDS,
    ReplContext,
    ReplSession,
    handle_command,
    list_directory,
    load_file,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def set_home(monkeypatch, path):
    """Point ``~`` at ``path`` on every platform.

    os.path.expanduser() reads HOME on POSIX but USERPROFILE on Windows, so
    setting only HOME silently does nothing there.
    """
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("USERPROFILE", str(path))


class FakeBackend:
    def __init__(self):
        self.stopped = False
        self.concurrent_mode = True
        self.active_slots = 0
        self.ports = ["Fake Port"]

    def stop(self):
        self.stopped = True

    def is_playing(self):
        return False

    def list_output_ports(self):
        return self.ports


@pytest.fixture
def ctx():
    played: list[tuple[str, bool]] = []

    def play(source, apply_default_tempo=True, record=True):
        played.append((source, apply_default_tempo, record))
        return True

    context = ReplContext(
        backend=FakeBackend(),
        session=ReplSession(),
        play=play,
        supports_concurrent=True,
    )
    context.played = played  # type: ignore[attr-defined]
    return context


@pytest.fixture
def in_tmp_dir(tmp_path):
    """Run inside a temporary directory; :cd mutates the process cwd."""
    original = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(original)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_reads_an_alda_file(self):
        assert "twinkle" in load_file(EXAMPLES / "twinkle.alda").lower()

    def test_suffix_is_optional(self):
        """':load twinkle' should find twinkle.alda."""
        assert load_file(EXAMPLES / "twinkle") == load_file(EXAMPLES / "twinkle.alda")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "nope.alda")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            load_file(tmp_path)


class TestListDirectory:
    def test_lists_alda_files(self):
        directories, files = list_directory(EXAMPLES)
        assert "twinkle.alda" in files
        assert directories == []

    def test_lists_subdirectories_with_a_slash(self, in_tmp_dir):
        (in_tmp_dir / "songs").mkdir()
        directories, _ = list_directory(in_tmp_dir)
        assert directories == ["songs/"]

    def test_skips_non_alda_files(self, in_tmp_dir):
        (in_tmp_dir / "notes.txt").write_text("x", encoding="utf-8")
        (in_tmp_dir / "song.alda").write_text("piano: c", encoding="utf-8")
        _, files = list_directory(in_tmp_dir)
        assert files == ["song.alda"]

    def test_skips_hidden_entries(self, in_tmp_dir):
        (in_tmp_dir / ".hidden.alda").write_text("piano: c", encoding="utf-8")
        _, files = list_directory(in_tmp_dir)
        assert files == []


class TestReplSession:
    def test_starts_empty(self):
        assert ReplSession().is_empty

    def test_records_input(self):
        s = ReplSession()
        s.add("piano: c")
        assert not s.is_empty
        assert "piano: c" in s.to_alda()

    def test_entries_are_separated(self):
        s = ReplSession()
        s.add("piano: c")
        s.add("violin: d")
        assert s.to_alda() == "piano: c\n\nviolin: d\n"

    def test_clear_empties_it(self):
        s = ReplSession()
        s.add("piano: c")
        s.clear()
        assert s.is_empty


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


class TestLoadCommand:
    """Loading must not play. The REPL opens ready, and :play starts it."""

    def test_loads_without_playing(self, ctx, capsys):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        assert ctx.played == [], "loading must not start playback"
        assert ctx.session.has_buffer

    def test_reports_what_was_loaded(self, ctx, capsys):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        out = capsys.readouterr().out
        assert "Loaded" in out
        assert "8 notes" in out
        assert ":play" in out, "the user needs to be told how to hear it"

    def test_records_the_buffer_name(self, ctx):
        target = str(EXAMPLES / "simple.alda")
        handle_command(ctx, f":load {target}")
        assert ctx.session.buffer_name == target

    def test_loading_twice_replaces_the_buffer(self, ctx):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        handle_command(ctx, f":load {EXAMPLES / 'chords.alda'}")
        assert ctx.session.buffer_name.endswith("chords.alda")

    def test_loaded_source_joins_the_session(self, ctx):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        assert not ctx.session.is_empty

    def test_typed_input_still_gets_the_default_tempo(self, ctx):
        ctx.play("piano: c")
        assert ctx.played[0][1] is True

    def test_missing_file_reports_an_error(self, ctx, capsys):
        handle_command(ctx, ":load /no/such/file.alda")
        assert "Error" in capsys.readouterr().out
        assert ctx.played == []
        assert not ctx.session.has_buffer

    def test_unparseable_file_reports_an_error(self, ctx, capsys, in_tmp_dir):
        (in_tmp_dir / "bad.alda").write_text("piano: ((((", encoding="utf-8")
        handle_command(ctx, ":load bad.alda")
        assert "Error" in capsys.readouterr().out
        assert not ctx.session.has_buffer, "a broken file must not become the buffer"

    def test_without_argument_shows_usage(self, ctx, capsys):
        handle_command(ctx, ":load")
        assert "Usage: :load" in capsys.readouterr().out

    def test_expands_user_home(self, ctx, capsys, monkeypatch, tmp_path):
        set_home(monkeypatch, tmp_path)
        (tmp_path / "tune.alda").write_text("piano: c d e", encoding="utf-8")
        handle_command(ctx, ":load ~/tune.alda")
        assert ctx.session.has_buffer


class TestPlayCommand:
    def test_plays_the_loaded_score(self, ctx, capsys):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        handle_command(ctx, ":play")
        assert len(ctx.played) == 1
        assert "Playing" in capsys.readouterr().out

    def test_does_not_impose_the_default_tempo(self, ctx):
        """A loaded score sets its own tempo, often per part."""
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        handle_command(ctx, ":play")
        assert ctx.played[0][1] is False

    def test_nothing_loaded_is_reported(self, ctx, capsys):
        handle_command(ctx, ":play")
        assert "Nothing loaded" in capsys.readouterr().out
        assert ctx.played == []

    def test_with_a_file_loads_then_plays(self, ctx, capsys):
        handle_command(ctx, f":play {EXAMPLES / 'simple.alda'}")
        assert len(ctx.played) == 1
        assert ctx.session.buffer_name.endswith("simple.alda")

    def test_with_a_missing_file_does_not_play(self, ctx, capsys):
        handle_command(ctx, ":play /no/such/file.alda")
        assert ctx.played == []
        assert "Error" in capsys.readouterr().out

    def test_replaying_does_not_duplicate_the_session(self, ctx):
        """:play twice must not save the score twice."""
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        handle_command(ctx, ":play")
        handle_command(ctx, ":play")
        assert len(ctx.session.sources) == 1

    def test_replay_is_possible(self, ctx):
        handle_command(ctx, f":load {EXAMPLES / 'simple.alda'}")
        handle_command(ctx, ":play")
        handle_command(ctx, ":play")
        assert len(ctx.played) == 2


class TestSaveCommand:
    def test_saves_alda(self, ctx, tmp_path, capsys):
        ctx.session.add("piano: c d e")
        target = tmp_path / "out.alda"
        handle_command(ctx, f":save {target}")
        assert target.read_text(encoding="utf-8").strip() == "piano: c d e"
        assert "Saved" in capsys.readouterr().out

    def test_saves_midi(self, ctx, tmp_path):
        ctx.session.add("piano: c d e")
        target = tmp_path / "out.mid"
        handle_command(ctx, f":save {target}")
        assert target.read_bytes()[:4] == b"MThd"

    def test_adds_alda_suffix_when_missing(self, ctx, tmp_path):
        ctx.session.add("piano: c")
        handle_command(ctx, f":save {tmp_path / 'out'}")
        assert (tmp_path / "out.alda").exists()

    def test_empty_session_saves_nothing(self, ctx, tmp_path, capsys):
        target = tmp_path / "out.alda"
        handle_command(ctx, f":save {target}")
        assert not target.exists()
        assert "Nothing to save" in capsys.readouterr().out

    def test_without_argument_shows_usage(self, ctx, capsys):
        ctx.session.add("piano: c")
        handle_command(ctx, ":save")
        assert "Usage: :save" in capsys.readouterr().out

    def test_saved_source_excludes_the_injected_tempo(self, ctx, tmp_path):
        """The session records what was typed, not the tempo-prefixed form."""
        ctx.session.add("piano: c d e")
        target = tmp_path / "out.alda"
        handle_command(ctx, f":save {target}")
        assert "(tempo" not in target.read_text(encoding="utf-8")

    def test_multiple_entries_round_trip(self, ctx, tmp_path):
        from aldakit import Score

        ctx.session.add("piano: c d e")
        ctx.session.add("violin: e f g")
        target = tmp_path / "out.alda"
        handle_command(ctx, f":save {target}")

        score = Score.from_file(target)
        assert len({n.channel for n in score.midi.notes}) == 2

    def test_unwritable_path_reports_an_error(self, ctx, tmp_path, capsys):
        ctx.session.add("piano: c")
        handle_command(ctx, f":save {tmp_path / 'nodir' / 'out.alda'}")
        assert "Error" in capsys.readouterr().out


class TestNavigationCommands:
    def test_pwd_prints_cwd(self, ctx, capsys, in_tmp_dir):
        handle_command(ctx, ":pwd")
        assert str(in_tmp_dir) in capsys.readouterr().out

    def test_cd_changes_directory(self, ctx, capsys, in_tmp_dir):
        (in_tmp_dir / "songs").mkdir()
        handle_command(ctx, ":cd songs")
        assert Path.cwd().name == "songs"
        assert "songs" in capsys.readouterr().out

    def test_cd_to_missing_directory_errors(self, ctx, capsys, in_tmp_dir):
        handle_command(ctx, ":cd nowhere")
        assert "Error" in capsys.readouterr().out
        assert Path.cwd() == in_tmp_dir

    def test_ls_lists_current_directory(self, ctx, capsys, in_tmp_dir):
        (in_tmp_dir / "song.alda").write_text("piano: c", encoding="utf-8")
        handle_command(ctx, ":ls")
        assert "song.alda" in capsys.readouterr().out

    def test_ls_takes_a_directory(self, ctx, capsys):
        handle_command(ctx, f":ls {EXAMPLES}")
        assert "twinkle.alda" in capsys.readouterr().out

    def test_ls_on_empty_directory(self, ctx, capsys, in_tmp_dir):
        handle_command(ctx, ":ls")
        assert "no directories or .alda files" in capsys.readouterr().out

    def test_load_after_cd_uses_the_new_directory(self, ctx, in_tmp_dir):
        (in_tmp_dir / "songs").mkdir()
        (in_tmp_dir / "songs" / "tune.alda").write_text("piano: c d e", encoding="utf-8")
        handle_command(ctx, ":cd songs")
        handle_command(ctx, ":load tune.alda")
        assert ctx.session.has_buffer
        assert ctx.session.buffer_name == "tune.alda"


class TestSessionCommands:
    def test_clear_empties_the_session(self, ctx, capsys):
        ctx.session.add("piano: c")
        handle_command(ctx, ":clear")
        assert ctx.session.is_empty
        assert "cleared" in capsys.readouterr().out.lower()

    def test_status_reports_session_size(self, ctx, capsys):
        ctx.session.add("piano: c")
        handle_command(ctx, ":status")
        assert "1 entries" in capsys.readouterr().out


class TestExistingCommandsStillWork:
    def test_quit_stops_the_loop(self, ctx):
        handle_command(ctx, ":quit")
        assert ctx.running is False

    @pytest.mark.parametrize("alias", [":q", ":quit", ":exit"])
    def test_quit_aliases(self, ctx, alias):
        handle_command(ctx, alias)
        assert ctx.running is False

    def test_help_lists_the_new_commands(self, ctx, capsys):
        handle_command(ctx, ":help")
        out = capsys.readouterr().out
        for command in (":load", ":save", ":ls", ":cd", ":pwd"):
            assert command in out

    def test_stop_stops_the_backend(self, ctx):
        handle_command(ctx, ":stop")
        assert ctx.backend.stopped

    def test_tempo_sets_the_default(self, ctx, capsys):
        handle_command(ctx, ":tempo 90")
        assert ctx.default_tempo == 90
        assert "90" in capsys.readouterr().out

    def test_invalid_tempo_is_rejected(self, ctx, capsys):
        handle_command(ctx, ":tempo fast")
        assert ctx.default_tempo == 120
        assert "Invalid" in capsys.readouterr().out

    def test_ports_lists_ports(self, ctx, capsys):
        handle_command(ctx, ":ports")
        assert "Fake Port" in capsys.readouterr().out

    def test_instruments_lists_instruments(self, ctx, capsys):
        handle_command(ctx, ":instruments")
        assert "midi-acoustic-grand-piano" in capsys.readouterr().out

    def test_concurrent_and_sequential(self, ctx):
        handle_command(ctx, ":sequential")
        assert ctx.backend.concurrent_mode is False
        handle_command(ctx, ":concurrent")
        assert ctx.backend.concurrent_mode is True

    def test_unknown_command(self, ctx, capsys):
        handle_command(ctx, ":nonsense")
        assert "Unknown command" in capsys.readouterr().out

    def test_bare_colon_is_unknown(self, ctx, capsys):
        handle_command(ctx, ":")
        assert "Unknown command" in capsys.readouterr().out


class TestCompletion:
    def test_path_commands_are_all_known_commands(self):
        assert set(PATH_COMMANDS) <= set(COMMAND_NAMES)

    def _completions(self, text):
        from prompt_toolkit.document import Document

        from aldakit.repl import AldaCompleter

        document = Document(text, cursor_position=len(text))
        return list(AldaCompleter().get_completions(document, None))

    def _applied(self, text):
        """The lines that result from accepting each offered completion.

        Completions carry a replacement span, not the finished word, so this
        applies them the way the prompt would.
        """
        results = []
        for completion in self._completions(text):
            cut = len(text) + completion.start_position
            results.append(text[:cut] + completion.text)
        return results

    def test_completes_command_names(self):
        assert any(r.startswith(":load") for r in self._applied(":lo"))

    def test_path_command_completion_appends_a_space(self):
        assert ":load " in self._applied(":load")

    def test_completes_paths_after_load(self, in_tmp_dir):
        (in_tmp_dir / "tune.alda").write_text("piano: c", encoding="utf-8")
        assert ":load tune.alda" in self._applied(":load tu")

    def test_completes_paths_after_save(self, in_tmp_dir):
        (in_tmp_dir / "tune.alda").write_text("piano: c", encoding="utf-8")
        assert ":save tune.alda" in self._applied(":save tu")

    def test_completes_directories_after_cd(self, in_tmp_dir):
        (in_tmp_dir / "songs").mkdir()
        assert any(r.startswith(":cd songs") for r in self._applied(":cd so"))

    def test_no_instrument_completion_inside_a_command(self):
        """':load pia' must offer files, not the piano instrument."""
        texts = [c.text for c in self._completions(":load pia")]
        assert not any(t.startswith("piano:") for t in texts)

    def test_non_path_command_offers_no_paths(self, in_tmp_dir):
        (in_tmp_dir / "tune.alda").write_text("piano: c", encoding="utf-8")
        assert self._completions(":tempo tu") == []

    def test_instrument_completion_still_works(self):
        texts = [c.text for c in self._completions("pia")]
        assert any(t.startswith("piano") for t in texts)


class TestCliArgument:
    def test_repl_accepts_a_file(self):
        from aldakit.cli import create_parser

        args = create_parser().parse_args(["repl", "examples/twinkle.alda"])
        assert args.file == Path("examples/twinkle.alda")

    def test_repl_file_is_optional(self):
        from aldakit.cli import create_parser

        assert create_parser().parse_args(["repl"]).file is None

    def test_missing_file_exits_non_zero(self, capsys):
        from aldakit.cli import main

        assert main(["repl", "/no/such/song.alda"]) == 1
        assert "File not found" in capsys.readouterr().err

    def test_file_is_passed_to_run_repl(self, monkeypatch, tmp_path):
        from aldakit import cli

        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")
        captured = {}

        def fake_run_repl(*args, **kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr("aldakit.repl.run_repl", fake_run_repl)
        assert cli.main(["repl", str(song)]) == 0
        assert captured["initial_file"] == song


class TestRunReplStartup:
    """End-to-end through run_repl with the prompt stubbed out.

    PromptSession needs a TTY, so the prompt is replaced by one that exits
    immediately. Everything before the first prompt -- backend setup and the
    startup file load -- runs for real.
    """

    @pytest.fixture
    def stubbed_repl(self, monkeypatch):
        played: list = []

        class InstantEof:
            def __init__(self, *a, **k):
                pass

            def prompt(self, *a, **k):
                raise EOFError

        class QuietBackend(FakeBackend):
            def play(self, sequence):
                played.append(sequence)
                return 0

            def close(self):
                pass

            def _ensure_port_open(self):
                pass

        monkeypatch.setattr("aldakit.repl.PromptSession", InstantEof)
        monkeypatch.setattr(
            "aldakit.repl.LibremidiBackend", lambda **kwargs: QuietBackend()
        )
        monkeypatch.setattr(
            "aldakit.repl.LibremidiBackend.list_output_ports",
            lambda self: ["Fake"],
            raising=False,
        )
        return played

    def test_starts_without_a_file(self, stubbed_repl, capsys):
        from aldakit.repl import run_repl

        assert run_repl(port_name="Fake") == 0
        assert stubbed_repl == []

    def test_startup_file_loads_without_playing(self, stubbed_repl, capsys, tmp_path):
        """The reported problem: the REPL blocked, playing the song."""
        from aldakit.repl import run_repl

        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")

        assert run_repl(port_name="Fake", initial_file=song) == 0
        assert stubbed_repl == [], "startup file must not play automatically"

        out = capsys.readouterr().out
        assert "Loaded" in out
        assert ":play" in out

    def test_startup_file_reports_a_summary(self, stubbed_repl, capsys, tmp_path):
        from aldakit.repl import run_repl

        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")

        run_repl(port_name="Fake", initial_file=song)
        assert "3 notes" in capsys.readouterr().out

    def test_broken_startup_file_does_not_crash(self, stubbed_repl, capsys, tmp_path):
        from aldakit.repl import run_repl

        song = tmp_path / "bad.alda"
        song.write_text("piano: ((((", encoding="utf-8")

        assert run_repl(port_name="Fake", initial_file=song) == 0
        assert stubbed_repl == []

    def test_missing_startup_file_does_not_crash(self, stubbed_repl, capsys):
        from aldakit.repl import run_repl

        assert run_repl(port_name="Fake", initial_file="/no/such/file.alda") == 0
        assert "Error" in capsys.readouterr().out

    def test_startup_file_enters_the_session(self, stubbed_repl, tmp_path, monkeypatch):
        """A loaded file must be savable, and playable, afterwards."""
        from aldakit import repl as repl_module

        song = tmp_path / "song.alda"
        song.write_text("piano: c d e", encoding="utf-8")

        captured = {}
        original = repl_module.ReplSession

        class Recording(original):
            def __init__(self):
                super().__init__()
                captured["session"] = self

        monkeypatch.setattr(repl_module, "ReplSession", Recording)
        repl_module.run_repl(port_name="Fake", initial_file=song)

        assert "piano: c d e" in captured["session"].to_alda()
