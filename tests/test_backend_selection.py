"""Tests for how the CLI chooses between MIDI and built-in audio.

Regression cover for D14: on a machine with no MIDI output ports and no
`ALDAKIT_SOUNDFONT`/config entry, `aldakit play file.alda` opened a virtual MIDI
port and played into it. Nothing was listening, so the command appeared to work
and produced silence.

The cause was that the CLI decided whether a SoundFont was available by looking
only at explicit configuration, never calling `find_soundfont()`. So `-a` also
failed with "No soundfont configured" while SoundFonts sat in
`~/.aldakit/soundfonts/`, and the no-ports fallback to audio never triggered.

The logic also existed in three near-copies that disagreed with each other; it
is now one function, `resolve_backend()`.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from aldakit.cli import create_parser, resolve_backend
from aldakit.config import Config


def args(**overrides) -> Namespace:
    base = dict(
        audio=False,
        soundfont=None,
        port=None,
        verbose=False,
        virtual_port="AldakitMIDI",
    )
    base.update(overrides)
    return Namespace(**base)


def config(**overrides) -> Config:
    return Config(**overrides)


def native(posix_path: str) -> str:
    """Render a POSIX-style test path the way discovery would return it.

    resolve_backend() treats the two sources of a SoundFont path differently:

    - a *discovered* path comes from ``find_soundfont()`` and is returned as
      ``str(Path(...))``, so its separators are platform-native (backslashes on
      Windows);
    - a path from ``-sf`` or the config file is passed through exactly as the
      user wrote it.

    Use this helper only for the discovered case; comparing a configured path
    against it fails on Windows.
    """
    return str(Path(posix_path))


class Env:
    """Patches the three things resolve_backend() consults."""

    def __init__(self, *, ports=(), discovered=None, has_tsf=True):
        self.ports = list(ports)
        self.discovered = discovered
        self.has_tsf = has_tsf

    def __enter__(self):
        self._patches = [
            patch(
                "aldakit.midi.soundfont.find_soundfont",
                return_value=Path(self.discovered) if self.discovered else None,
            ),
            patch("aldakit.midi.backends.HAS_TSF", self.has_tsf),
            patch(
                "aldakit.cli.LibremidiBackend.list_output_ports",
                return_value=self.ports,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


class TestNoMidiPorts:
    """The reported symptom: silence when nothing is listening."""

    def test_discovered_soundfont_is_used(self):
        """A SoundFont on disk is enough; it need not be configured."""
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(), None)
        assert choice.use_audio
        assert choice.soundfont == native("/sf/found.sf2")
        assert choice.error is None

    def test_configured_soundfont_wins_over_discovery(self):
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(soundfont="/sf/configured.sf2"), None)
        assert choice.use_audio
        # A configured path is passed through verbatim, not normalised
        assert choice.soundfont == "/sf/configured.sf2"

    def test_no_soundfont_falls_back_to_midi(self):
        """With nothing to synthesize with, the virtual port is all there is."""
        with Env(ports=[], discovered=None):
            choice = resolve_backend(args(), config(), None)
        assert not choice.use_audio
        assert choice.error is None

    def test_no_tsf_module_falls_back_to_midi(self):
        with Env(ports=[], discovered="/sf/found.sf2", has_tsf=False):
            choice = resolve_backend(args(), config(), None)
        assert not choice.use_audio


class TestMidiPortsAvailable:
    def test_midi_is_preferred_when_ports_exist(self):
        """A discovered SoundFont must not hijack a working MIDI setup."""
        with Env(ports=["IAC Driver Bus 1"], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(), None)
        assert not choice.use_audio

    def test_explicit_port_skips_the_fallback(self):
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(), "SomePort")
        assert not choice.use_audio

    def test_consider_ports_false_skips_the_probe(self):
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(), None, consider_ports=False)
        assert not choice.use_audio


class TestExplicitAudioRequest:
    def test_dash_a_uses_a_discovered_soundfont(self):
        """-a used to error out even with SoundFonts installed."""
        with Env(ports=["IAC Driver Bus 1"], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(audio=True), config(), None)
        assert choice.use_audio
        assert choice.soundfont == native("/sf/found.sf2")
        assert choice.error is None

    def test_dash_sf_selects_audio_with_that_file(self):
        with Env(ports=["IAC Driver Bus 1"], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(soundfont="/sf/explicit.sf2"), config(), None)
        assert choice.use_audio
        assert choice.soundfont == "/sf/explicit.sf2"

    def test_config_backend_audio_selects_audio(self):
        with Env(ports=["IAC Driver Bus 1"], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(backend="audio"), None)
        assert choice.use_audio

    def test_audio_without_any_soundfont_is_an_error(self):
        with Env(ports=[], discovered=None):
            choice = resolve_backend(args(audio=True), config(), None)
        assert choice.error is not None
        assert "No SoundFont found" in choice.error

    def test_audio_without_tsf_module_is_an_error(self):
        with Env(ports=[], discovered="/sf/found.sf2", has_tsf=False):
            choice = resolve_backend(args(audio=True), config(), None)
        assert choice.error is not None
        assert "_tsf module" in choice.error


class TestPathNormalisation:
    """Where a SoundFont path came from decides whether it is normalised.

    This is invisible on POSIX, where both forms are the same string, and it
    broke a Windows CI run when the two were conflated.
    """

    def test_discovered_paths_are_platform_native(self):
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(args(), config(), None)
        assert choice.soundfont == str(Path("/sf/found.sf2"))

    def test_configured_paths_are_passed_through_verbatim(self):
        with Env(ports=[], discovered=None):
            choice = resolve_backend(args(), config(soundfont="/sf/as-written.sf2"), None)
        assert choice.soundfont == "/sf/as-written.sf2"

    def test_cli_paths_are_passed_through_verbatim(self):
        with Env(ports=[], discovered=None):
            choice = resolve_backend(args(soundfont="/sf/as-typed.sf2"), config(), None)
        assert choice.soundfont == "/sf/as-typed.sf2"

    def test_both_forms_load_the_same_file(self, tmp_path):
        """The two spellings must still resolve to one file on disk."""
        sf = tmp_path / "sound.sf2"
        sf.write_bytes(b"stub")
        as_posix = sf.as_posix()
        assert Path(as_posix).resolve() == Path(str(sf)).resolve()


class TestPrecedence:
    def test_cli_soundfont_beats_config_soundfont(self):
        with Env(ports=[], discovered=None):
            choice = resolve_backend(
                args(soundfont="/cli.sf2"), config(soundfont="/config.sf2"), None
            )
        assert choice.soundfont == "/cli.sf2"

    def test_config_soundfont_beats_discovery(self):
        with Env(ports=[], discovered="/discovered.sf2"):
            choice = resolve_backend(args(audio=True), config(soundfont="/config.sf2"), None)
        assert choice.soundfont == "/config.sf2"


class TestAllSubcommandsAgree:
    """The three call sites used to be near-copies that disagreed."""

    @pytest.mark.parametrize("argv", [["play", "x.alda"], ["eval", "piano: c"], ["repl"]])
    def test_same_decision_for_every_subcommand(self, argv):
        parsed = create_parser().parse_args(argv)
        with Env(ports=[], discovered="/sf/found.sf2"):
            choice = resolve_backend(parsed, config(), None)
        assert choice.use_audio
        assert choice.soundfont == native("/sf/found.sf2")

    @pytest.mark.parametrize("argv", [["play", "x.alda", "-a"], ["eval", "piano: c", "-a"], ["repl", "-a"]])
    def test_dash_a_works_for_every_subcommand(self, argv):
        parsed = create_parser().parse_args(argv)
        with Env(ports=["Port"], discovered="/sf/found.sf2"):
            choice = resolve_backend(parsed, config(), None)
        assert choice.use_audio
        assert choice.error is None


class TestSilentVirtualPortWarning:
    """Silence must be announced, not just happen."""

    @staticmethod
    def _silent_midi_backend():
        """A LibremidiBackend mock whose playback finishes immediately.

        The CLI polls ``backend.is_playing()`` on the instance itself, so that
        is what has to report False; a bare MagicMock is truthy and would hang.
        """
        mock = patch("aldakit.cli.LibremidiBackend").start()
        mock.return_value.list_output_ports.return_value = []
        mock.return_value.is_playing.return_value = False
        mock.return_value.__enter__.return_value = mock.return_value
        return mock

    def test_warns_when_nothing_can_hear_the_output(self, tmp_path, capsys):
        from aldakit.cli import main

        source = tmp_path / "s.alda"
        source.write_text("piano: c", encoding="utf-8")

        with Env(ports=[], discovered=None):
            self._silent_midi_backend()
            try:
                assert main(["play", str(source)]) == 0
            finally:
                patch.stopall()

        err = capsys.readouterr().err
        assert "no MIDI output ports" in err
        assert "virtual port" in err

    def test_no_warning_when_audio_is_used(self, tmp_path, capsys):
        from aldakit.cli import main

        source = tmp_path / "s.alda"
        source.write_text("piano: c", encoding="utf-8")

        with Env(ports=[], discovered="/sf/found.sf2"):
            tsf = patch("aldakit.midi.backends.TsfBackend").start()
            tsf.return_value.__enter__.return_value = tsf.return_value
            tsf.return_value.is_playing.return_value = False
            try:
                assert main(["play", str(source)]) == 0
            finally:
                patch.stopall()

        assert "no MIDI output ports" not in capsys.readouterr().err
