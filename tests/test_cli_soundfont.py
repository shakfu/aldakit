"""Tests for the `aldakit soundfont` subcommand.

`midi/soundfont.py` implemented discovery, download with SHA256 verification,
and a catalog, but nothing in the CLI reached it: a user with no external synth
was told to set ALDAKIT_SOUNDFONT with no hint that aldakit could fetch one.
These tests cover the commands that close that gap, and the prompt that offers
the download when audio playback finds no SoundFont.

No test here touches the network: the catalog is replaced with an entry whose
checksum matches what the stubbed downloader writes, so the real verification
path still runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aldakit import cli
from aldakit.config import Config
from aldakit.midi import soundfont as sf_module
from aldakit.midi.soundfont import SoundFontManager

CONTENT = b"not really a soundfont, but it hashes"
DIGEST = hashlib.sha256(CONTENT).hexdigest()

FAKE_CATALOG = {
    "TestSF": {
        "url": "https://example.invalid/test.sf2",
        "filename": "TestSF.sf2",
        "size_mb": 0.1,
        "description": "Stand-in for a real SoundFont",
        "sha256": DIGEST,
    },
}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An isolated home directory, an empty catalog and no downloads."""
    monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(sf_module, "SOUNDFONT_CATALOG", FAKE_CATALOG)
    monkeypatch.setattr(sf_module, "DEFAULT_SOUNDFONT", "TestSF")
    monkeypatch.setattr(cli, "DEFAULT_SOUNDFONT", "TestSF")
    # Only this directory is searched, so SoundFonts installed on the machine
    # running the tests cannot make a test pass or fail.
    monkeypatch.setattr(
        SoundFontManager,
        "get_search_paths",
        lambda self: [self.soundfont_dir],
    )
    # The developer's own ~/.aldakit/config.ini must not decide these tests.
    monkeypatch.setattr(cli, "load_config", Config)
    # find_soundfont() and friends delegate to a manager built at import time,
    # which captured the real home directory before the patch above.
    monkeypatch.setattr(sf_module, "_default_manager", SoundFontManager())
    return tmp_path


@pytest.fixture
def offline_download(monkeypatch):
    """Stub the network transfer, keeping checksum verification real."""
    calls: list[str] = []

    def fake_download_file(url, target, progress_callback=None):
        calls.append(url)
        Path(target).write_bytes(CONTENT)
        if progress_callback is not None:
            progress_callback(len(CONTENT), len(CONTENT))

    monkeypatch.setattr(
        SoundFontManager, "_download_file", staticmethod(fake_download_file)
    )
    return calls


class TestArgumentParsing:
    def test_actions_are_registered(self):
        parser = cli.create_parser()
        for action in ("list", "install", "verify", "path"):
            args = parser.parse_args(["soundfont", action])
            assert args.command == "soundfont"
            assert args.soundfont_command == action

    def test_install_takes_an_optional_name(self):
        parser = cli.create_parser()
        assert parser.parse_args(["soundfont", "install"]).name is None
        assert (
            parser.parse_args(["soundfont", "install", "TimGM6mb"]).name == "TimGM6mb"
        )

    def test_install_flags(self):
        parser = cli.create_parser()
        args = parser.parse_args(["soundfont", "install", "--all", "--force"])
        assert args.all is True
        assert args.force is True

    def test_bare_soundfont_defaults_to_list(self, sandbox, capsys):
        assert cli.main(["soundfont"]) == 0
        assert "Available to download" in capsys.readouterr().out


class TestList:
    def test_reports_nothing_installed(self, sandbox, capsys):
        assert cli.main(["soundfont", "list"]) == 0
        out = capsys.readouterr().out
        assert "No SoundFonts installed." in out
        assert "TestSF" in out  # the catalog is still shown

    def test_lists_installed_files(self, sandbox, capsys):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(CONTENT)

        assert cli.main(["soundfont", "list"]) == 0
        out = capsys.readouterr().out
        assert "TestSF.sf2" in out
        assert "No SoundFonts installed." not in out


class TestInstall:
    def test_downloads_the_default(self, sandbox, offline_download, capsys):
        assert cli.main(["soundfont", "install"]) == 0
        installed = sandbox / ".aldakit" / "soundfonts" / "TestSF.sf2"
        assert installed.read_bytes() == CONTENT
        assert offline_download == ["https://example.invalid/test.sf2"]
        assert "Saved to" in capsys.readouterr().out

    def test_skips_an_existing_file(self, sandbox, offline_download, capsys):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(b"already here")

        assert cli.main(["soundfont", "install", "TestSF"]) == 0
        assert offline_download == []  # nothing downloaded
        assert "Already installed" in capsys.readouterr().out

    def test_force_redownloads(self, sandbox, offline_download):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(b"stale")

        assert cli.main(["soundfont", "install", "TestSF", "--force"]) == 0
        assert (directory / "TestSF.sf2").read_bytes() == CONTENT
        assert len(offline_download) == 1

    def test_all_downloads_the_catalog(self, sandbox, offline_download):
        assert cli.main(["soundfont", "install", "--all"]) == 0
        assert len(offline_download) == len(FAKE_CATALOG)

    def test_unknown_name_is_an_error(self, sandbox, offline_download, capsys):
        assert cli.main(["soundfont", "install", "Nonexistent"]) == 1
        err = capsys.readouterr().err
        assert "Unknown SoundFont" in err
        assert "TestSF" in err  # the error names what is available
        assert offline_download == []

    def test_checksum_mismatch_fails_the_install(self, sandbox, monkeypatch, capsys):
        def bad_download(url, target, progress_callback=None):
            Path(target).write_bytes(b"corrupted")

        monkeypatch.setattr(
            SoundFontManager, "_download_file", staticmethod(bad_download)
        )
        assert cli.main(["soundfont", "install"]) == 1
        assert "Download failed" in capsys.readouterr().err
        assert not (sandbox / ".aldakit" / "soundfonts" / "TestSF.sf2").exists()


class TestVerify:
    def test_nothing_downloaded(self, sandbox, capsys):
        assert cli.main(["soundfont", "verify"]) == 0
        assert "No downloaded SoundFonts" in capsys.readouterr().out

    def test_good_checksum(self, sandbox, capsys):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(CONTENT)

        assert cli.main(["soundfont", "verify"]) == 0
        assert "TestSF: ok" in capsys.readouterr().out

    def test_bad_checksum_exits_nonzero(self, sandbox, capsys):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(b"tampered with")

        assert cli.main(["soundfont", "verify"]) == 1
        captured = capsys.readouterr()
        assert "CHECKSUM MISMATCH" in captured.out
        assert "--force" in captured.err  # tells the user how to recover


class TestPath:
    def test_prints_the_resolved_soundfont(self, sandbox, capsys):
        directory = sandbox / ".aldakit" / "soundfonts"
        directory.mkdir(parents=True)
        (directory / "TestSF.sf2").write_bytes(CONTENT)

        assert cli.main(["soundfont", "path"]) == 0
        assert capsys.readouterr().out.strip() == str(directory / "TestSF.sf2")

    def test_exits_nonzero_when_none_found(self, sandbox, capsys):
        assert cli.main(["soundfont", "path"]) == 1
        assert "No SoundFont found" in capsys.readouterr().err


class TestDownloadOffer:
    """Audio playback with no SoundFont offers to fetch one."""

    def test_declines_to_prompt_when_not_a_terminal(self, sandbox, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert cli.offer_soundfont_download() is None

    def test_prompt_accepted_downloads(self, sandbox, offline_download, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        path = cli.offer_soundfont_download("TestSF")
        assert path is not None
        assert Path(path).read_bytes() == CONTENT

    def test_empty_answer_accepts(self, sandbox, offline_download, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "")

        assert cli.offer_soundfont_download("TestSF") is not None

    def test_prompt_declined_downloads_nothing(
        self, sandbox, offline_download, monkeypatch
    ):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "n")

        assert cli.offer_soundfont_download("TestSF") is None
        assert offline_download == []

    def test_interrupted_prompt_downloads_nothing(
        self, sandbox, offline_download, monkeypatch
    ):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def interrupt(_):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", interrupt)
        assert cli.offer_soundfont_download("TestSF") is None
        assert offline_download == []

    def test_failed_download_is_reported_not_raised(self, sandbox, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        def fail(url, target, progress_callback=None):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(SoundFontManager, "_download_file", staticmethod(fail))
        assert cli.offer_soundfont_download("TestSF") is None
        assert "Download failed" in capsys.readouterr().err

    def test_resolve_backend_flags_the_missing_soundfont(self, sandbox, monkeypatch):
        # The flag is what tells the caller a download would fix this, rather
        # than having to match on the message text.
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", True)
        args = cli.create_parser().parse_args(["play", "x.alda", "--audio"])
        choice = cli.resolve_backend(args, Config(), None)

        assert choice.error is not None
        assert choice.needs_soundfont is True
        assert "aldakit soundfont install" in choice.error

    def test_play_offers_the_download_and_then_plays(
        self, sandbox, offline_download, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", True)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _: "y")

        played: list[str | None] = []

        class FakeBackend:
            def __init__(self, soundfont=None, **kwargs):
                played.append(soundfont)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def play(self, sequence):
                pass

            def wait(self, *args):
                pass

            def stop(self):
                pass

        monkeypatch.setattr("aldakit.midi.backends.TsfBackend", FakeBackend)

        score = tmp_path / "song.alda"
        score.write_text("piano: c d e\n")
        assert cli.main(["play", str(score), "--audio"]) == 0

        # The declined path would have exited 1 with the error; instead the
        # SoundFont was fetched and handed to the audio backend.
        assert len(offline_download) == 1
        assert played == [str(sandbox / ".aldakit" / "soundfonts" / "TestSF.sf2")]
