"""CLI regression tests."""

import argparse
import builtins
from pathlib import Path

import pytest

from aldakit import __version__
from aldakit.cli import (
    create_parser,
    stdin_mode,
    list_ports,
    read_source,
    transcribe_command,
    main,
    _resolve_port_specifier,
    _resolve_output_port,
    _resolve_input_port,
)


def test_cli_version_matches_package(capsys):
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])

    out = capsys.readouterr().out.strip()
    assert out.endswith(__version__)


def test_stdin_mode_uses_backend_context(monkeypatch):
    entered = False
    exited = False

    class DummyBackend:
        def __init__(self, port_name=None, virtual_port_name=None):
            pass

        def __enter__(self):
            nonlocal entered
            entered = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            nonlocal exited
            exited = True

        def play(self, sequence):
            return None

        def is_playing(self):
            return False

    def fake_input(prompt: str | None = None):
        raise EOFError

    monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
    monkeypatch.setattr(builtins, "input", fake_input)

    assert (
        stdin_mode(port_name=None, verbose=False, virtual_port_name="AldakitMIDI") == 0
    )
    assert entered and exited


class TestResolvePortSpecifier:
    """Tests for _resolve_port_specifier."""

    def test_none_returns_none_with_multiple_ports(self):
        port, ok = _resolve_port_specifier(None, ["PortA", "PortB"], "output")
        assert port is None
        assert ok is True

    def test_none_returns_none_with_no_ports(self):
        port, ok = _resolve_port_specifier(None, [], "output")
        assert port is None
        assert ok is True

    def test_index_resolves_to_name(self):
        port, ok = _resolve_port_specifier("0", ["FluidSynth", "IAC"], "output")
        assert port == "FluidSynth"
        assert ok is True

    def test_index_second_port(self):
        port, ok = _resolve_port_specifier("1", ["FluidSynth", "IAC"], "output")
        assert port == "IAC"
        assert ok is True

    def test_index_out_of_range_fails(self, capsys):
        port, ok = _resolve_port_specifier("5", ["A", "B"], "output")
        assert port is None
        assert ok is False
        err = capsys.readouterr().err
        assert "out of range" in err

    def test_name_passed_through(self):
        port, ok = _resolve_port_specifier("FluidSynth", ["FluidSynth"], "output")
        assert port == "FluidSynth"
        assert ok is True

    def test_partial_name_passed_through(self):
        # Backend handles partial matching, so we just pass it through
        port, ok = _resolve_port_specifier("Fluid", ["FluidSynth"], "output")
        assert port == "Fluid"
        assert ok is True


class TestResolveOutputPort:
    """Tests for _resolve_output_port auto-selection."""

    def test_auto_selects_single_port(self, monkeypatch):
        class DummyBackend:
            def list_output_ports(self):
                return ["OnlyPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        port, ok = _resolve_output_port(None)
        assert port == "OnlyPort"
        assert ok is True

    def test_no_auto_select_with_multiple_ports(self, monkeypatch):
        class DummyBackend:
            def list_output_ports(self):
                return ["PortA", "PortB"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        port, ok = _resolve_output_port(None)
        assert port is None
        assert ok is True

    def test_no_auto_select_with_no_ports(self, monkeypatch):
        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        port, ok = _resolve_output_port(None)
        assert port is None
        assert ok is True


class TestResolveInputPort:
    """Tests for _resolve_input_port auto-selection."""

    def test_auto_selects_single_port(self, monkeypatch):
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["OnlyInputPort"]
        )
        port, ok = _resolve_input_port(None)
        assert port == "OnlyInputPort"
        assert ok is True

    def test_no_auto_select_with_multiple_ports(self, monkeypatch):
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["InputA", "InputB"]
        )
        port, ok = _resolve_input_port(None)
        assert port is None
        assert ok is True


# =============================================================================
# Parser Tests
# =============================================================================


class TestCreateParser:
    """Tests for create_parser."""

    def test_creates_parser(self):
        """Parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "aldakit"

    def test_has_version_argument(self):
        """Parser has --version argument."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_has_repl_subcommand(self):
        """Parser has repl subcommand."""
        parser = create_parser()
        args = parser.parse_args(["repl"])
        assert args.command == "repl"

    def test_has_ports_subcommand(self):
        """Parser has ports subcommand."""
        parser = create_parser()
        args = parser.parse_args(["ports"])
        assert args.command == "ports"

    def test_has_transcribe_subcommand(self):
        """Parser has transcribe subcommand."""
        parser = create_parser()
        args = parser.parse_args(["transcribe"])
        assert args.command == "transcribe"

    def test_has_play_subcommand(self):
        """Parser has play subcommand."""
        parser = create_parser()
        args = parser.parse_args(["play", "test.alda"])
        assert args.command == "play"

    def test_has_eval_subcommand(self):
        """Parser has eval subcommand."""
        parser = create_parser()
        args = parser.parse_args(["eval", "piano: c d e"])
        assert args.command == "eval"
        assert args.code == "piano: c d e"

    def test_repl_arguments(self):
        """REPL subcommand has all expected arguments."""
        parser = create_parser()
        args = parser.parse_args(
            ["repl", "-p", "TestPort", "-v", "--sequential", "-a", "-sf", "test.sf2"]
        )
        assert args.port == "TestPort"
        assert args.verbose is True
        assert args.sequential is True
        assert args.audio is True
        assert args.soundfont == "test.sf2"

    def test_transcribe_arguments(self):
        """Transcribe subcommand has all expected arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "transcribe",
                "-d",
                "30",
                "-i",
                "guitar",
                "-t",
                "100",
                "-q",
                "0.5",
                "--feel",
                "swing",
                "--swing-ratio",
                "0.6",
                "-o",
                "output.alda",
                "--port",
                "MyPort",
                "--play",
                "-v",
                "--alda-notes",
            ]
        )
        assert args.duration == 30.0
        assert args.instrument == "guitar"
        assert args.tempo == 100.0
        assert args.quantize == 0.5
        assert args.feel == "swing"
        assert args.swing_ratio == 0.6
        assert args.output == Path("output.alda")
        assert args.port == "MyPort"
        assert args.play is True
        assert args.verbose is True
        assert args.alda_notes is True

    def test_play_arguments(self):
        """Play subcommand has all expected arguments."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "play",
                "test.alda",
                "-e",
                "c d e",
                "-o",
                "output.mid",
                "--port",
                "TestPort",
                "--parse-only",
                "--no-wait",
                "-v",
                "-a",
                "-sf",
                "test.sf2",
            ]
        )
        assert args.file == Path("test.alda")
        assert args.eval == "c d e"
        assert args.output == Path("output.mid")
        assert args.port == "TestPort"
        assert args.parse_only is True
        assert args.no_wait is True
        assert args.verbose is True
        assert args.audio is True
        assert args.soundfont == "test.sf2"

    def test_ports_arguments(self):
        """Ports subcommand has all expected arguments."""
        parser = create_parser()
        args = parser.parse_args(["ports", "-i"])
        assert args.inputs is True
        assert args.outputs is False

        args = parser.parse_args(["ports", "-o"])
        assert args.inputs is False
        assert args.outputs is True


# =============================================================================
# list_ports Tests
# =============================================================================


class TestListPorts:
    """Tests for list_ports function."""

    def test_list_outputs_only(self, monkeypatch, capsys):
        """List only output ports."""

        class DummyBackend:
            def list_output_ports(self):
                return ["Port1", "Port2"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        list_ports(show_inputs=False, show_outputs=True)

        out = capsys.readouterr().out
        assert "MIDI output ports" in out
        assert "Port1" in out
        assert "Port2" in out
        assert "0: Port1" in out
        assert "1: Port2" in out

    def test_list_inputs_only(self, monkeypatch, capsys):
        """List only input ports."""

        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        def mock_get_input_ports():
            return ["InputA", "InputB"]

        # Patch the import at the source module
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", mock_get_input_ports
        )

        list_ports(show_inputs=True, show_outputs=False)

        out = capsys.readouterr().out
        assert "MIDI input ports" in out
        assert "InputA" in out
        assert "InputB" in out

    def test_list_no_output_ports(self, monkeypatch, capsys):
        """Show message when no output ports available."""

        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        list_ports(show_inputs=False, show_outputs=True)

        out = capsys.readouterr().out
        assert "No MIDI output ports available" in out

    def test_list_no_input_ports(self, monkeypatch, capsys):
        """Show message when no input ports available."""

        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr("aldakit.midi.transcriber.list_input_ports", lambda: [])

        list_ports(show_inputs=True, show_outputs=False)

        out = capsys.readouterr().out
        assert "No MIDI input ports available" in out


# =============================================================================
# read_source Tests
# =============================================================================


class TestReadSource:
    """Tests for read_source function."""

    def test_read_from_eval(self):
        """Read source from --eval argument."""
        args = argparse.Namespace(eval="piano: c d e", file=None)
        source, filename = read_source(args)
        assert source == "piano: c d e"
        assert filename == "<eval>"

    def test_read_from_file(self, tmp_path):
        """Read source from file."""
        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e f g")

        args = argparse.Namespace(eval=None, file=test_file)
        source, filename = read_source(args)
        assert source == "piano: c d e f g"
        assert filename == str(test_file)

    def test_read_from_stdin(self, monkeypatch):
        """Read source from stdin when file is '-'."""
        monkeypatch.setattr("sys.stdin.read", lambda: "piano: c")

        args = argparse.Namespace(eval=None, file=Path("-"))
        source, filename = read_source(args)
        assert source == "piano: c"
        assert filename == "<stdin>"

    def test_file_not_found(self, tmp_path):
        """Exit with error when file not found."""
        args = argparse.Namespace(eval=None, file=tmp_path / "nonexistent.alda")

        with pytest.raises(SystemExit) as exc_info:
            read_source(args)
        assert exc_info.value.code == 1

    def test_no_input_specified(self):
        """Exit with error when no input specified."""
        args = argparse.Namespace(eval=None, file=None)

        with pytest.raises(SystemExit) as exc_info:
            read_source(args)
        assert exc_info.value.code == 1


# =============================================================================
# stdin_mode Tests
# =============================================================================


class TestStdinMode:
    """Tests for stdin_mode function."""

    def test_handles_keyboard_interrupt(self, monkeypatch, capsys):
        """Handle Ctrl+C gracefully."""
        call_count = 0

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def play(self, sequence):
                return None

            def is_playing(self):
                return False

        def fake_input(prompt=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr(builtins, "input", fake_input)

        result = stdin_mode(port_name=None, verbose=False)
        assert result == 0

    def test_plays_valid_input(self, monkeypatch, capsys):
        """Parse and play valid Alda input."""
        input_calls = 0

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                self.played = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def play(self, sequence):
                self.played = True
                return None

            def is_playing(self):
                return False

        def fake_input(prompt=None):
            nonlocal input_calls
            input_calls += 1
            if input_calls == 1:
                return "piano: c d e"
            elif input_calls == 2:
                return ""  # First blank line
            elif input_calls == 3:
                return ""  # Second blank line triggers play
            raise EOFError

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr(builtins, "input", fake_input)

        result = stdin_mode(port_name=None, verbose=False)
        assert result == 0

    def test_handles_parse_error(self, monkeypatch, capsys):
        """Show parse error for invalid input."""
        input_calls = 0

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def play(self, sequence):
                return None

            def is_playing(self):
                return False

        def fake_input(prompt=None):
            nonlocal input_calls
            input_calls += 1
            if input_calls == 1:
                return "piano: ((((invalid"
            elif input_calls == 2:
                return ""
            elif input_calls == 3:
                return ""
            raise EOFError

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr(builtins, "input", fake_input)

        result = stdin_mode(port_name=None, verbose=False)
        assert result == 0

        err = capsys.readouterr().err
        assert "Parse error" in err

    def test_verbose_mode(self, monkeypatch, capsys):
        """Verbose mode prints note count."""
        input_calls = 0

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def play(self, sequence):
                return None

            def is_playing(self):
                return False

        def fake_input(prompt=None):
            nonlocal input_calls
            input_calls += 1
            if input_calls == 1:
                return "piano: c d e"
            elif input_calls == 2:
                return ""
            elif input_calls == 3:
                return ""
            raise EOFError

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr(builtins, "input", fake_input)

        result = stdin_mode(port_name=None, verbose=True)
        assert result == 0

        err = capsys.readouterr().err
        assert "Playing" in err
        assert "notes" in err


# =============================================================================
# main() Tests
# =============================================================================


class TestMain:
    """Tests for main function."""

    def test_ports_command(self, monkeypatch, capsys):
        """Test 'aldakit ports' command."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr("aldakit.midi.transcriber.list_input_ports", lambda: [])

        result = main(["ports", "-o"])
        assert result == 0

        out = capsys.readouterr().out
        assert "TestPort" in out

    def test_eval_command(self, monkeypatch, tmp_path):
        """Test 'aldakit eval' command with output file."""

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def save(self, sequence, path):
                Path(path).write_bytes(b"MIDI")

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        output_file = tmp_path / "output.mid"
        result = main(["eval", "piano: c d e", "-o", str(output_file)])
        assert result == 0
        assert output_file.exists()

    def test_play_command_parse_only(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play --parse-only'."""

        class DummyBackend:
            def list_output_ports(self):
                return []

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        result = main(["play", str(test_file), "--parse-only"])
        assert result == 0

        out = capsys.readouterr().out
        assert "RootNode" in out or "PartNode" in out

    def test_play_command_with_output(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play -o output.mid'."""
        saved_files = []

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def save(self, sequence, path):
                saved_files.append(path)
                Path(path).write_bytes(b"MIDI")

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")
        output_file = tmp_path / "output.mid"

        result = main(["play", str(test_file), "-o", str(output_file)])
        assert result == 0
        assert len(saved_files) == 1
        assert output_file.exists()

    def test_play_command_no_notes(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play' with file that produces no notes."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        # Empty file produces no notes
        test_file = tmp_path / "empty.alda"
        test_file.write_text("# Just a comment")

        result = main(["play", str(test_file)])
        assert result == 0

        err = capsys.readouterr().err
        assert "No notes generated" in err

    def test_play_command_parse_error(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play' with invalid syntax."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        test_file = tmp_path / "invalid.alda"
        test_file.write_text("piano: ((((invalid")

        result = main(["play", str(test_file)])
        assert result == 1

        err = capsys.readouterr().err
        assert "Parse error" in err

    def test_play_command_verbose(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play -v' verbose mode."""
        played = False

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def play(self, sequence):
                nonlocal played
                played = True
                return None

            def is_playing(self):
                return False

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        result = main(["play", str(test_file), "-v"])
        assert result == 0
        assert played

        err = capsys.readouterr().err
        assert "Parsing" in err
        assert "Generating MIDI" in err
        assert "Generated" in err

    def test_play_no_file_no_eval(self, monkeypatch, capsys):
        """Test 'aldakit play' without file or -e."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        result = main(["play"])
        assert result == 1

        err = capsys.readouterr().err
        assert "No input specified" in err

    def test_play_with_eval(self, monkeypatch, tmp_path, capsys):
        """Test 'aldakit play -e CODE'."""
        played = False

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def play(self, sequence):
                nonlocal played
                played = True
                return None

            def is_playing(self):
                return False

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        result = main(["play", "-e", "piano: c d e"])
        assert result == 0
        assert played

    def test_play_stdin_mode(self, monkeypatch, capsys):
        """Test 'aldakit play --stdin'."""

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def play(self, sequence):
                return None

            def is_playing(self):
                return False

        def fake_input(prompt=None):
            raise EOFError

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.setattr(builtins, "input", fake_input)

        result = main(["play", "--stdin"])
        assert result == 0

    def test_play_no_wait(self, monkeypatch, tmp_path):
        """Test 'aldakit play --no-wait'."""
        played = False
        waited = False

        class DummyBackend:
            def __init__(self, port_name=None, virtual_port_name=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def list_output_ports(self):
                return ["TestPort"]

            def play(self, sequence):
                nonlocal played
                played = True
                return None

            def is_playing(self):
                nonlocal waited
                waited = True
                return False

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        result = main(["play", str(test_file), "--no-wait"])
        assert result == 0
        assert played
        # With --no-wait, is_playing should not be called for wait loop
        assert not waited


class TestMainAudioBackend:
    """Tests for main function with audio backend."""

    def test_audio_no_soundfont_error(self, monkeypatch, tmp_path, capsys):
        """Audio requested with no SoundFont configured *or* discoverable."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        # Clear any environment soundfont
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)
        # ...and any SoundFont installed on the machine running the tests,
        # otherwise this assertion depends on the developer's setup.
        monkeypatch.setattr(
            "aldakit.midi.soundfont.find_soundfont", lambda: None
        )

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        result = main(["play", str(test_file), "-a"])
        assert result == 1

        err = capsys.readouterr().err
        assert "No SoundFont found" in err

    def test_audio_uses_discovered_soundfont(self, monkeypatch, tmp_path, capsys):
        """-a succeeds when a SoundFont can be found, without configuration."""
        from pathlib import Path as _Path

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        class DummyTsfBackend:
            def __init__(self, soundfont=None, **kwargs):
                DummyTsfBackend.soundfont_used = soundfont

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def play(self, sequence):
                return 0

            def wait(self, *a, **k):
                return None

            def stop(self):
                return None

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)
        monkeypatch.setattr(
            "aldakit.midi.soundfont.find_soundfont", lambda: _Path("/sf/found.sf2")
        )
        monkeypatch.setattr("aldakit.midi.backends.TsfBackend", DummyTsfBackend)
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", True)

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        assert main(["play", str(test_file), "-a"]) == 0
        assert DummyTsfBackend.soundfont_used == "/sf/found.sf2"

    def test_audio_with_soundfont(self, monkeypatch, tmp_path, capsys):
        """Test audio playback with soundfont."""
        played = False

        class DummyBackend:
            def list_output_ports(self):
                return []

        class DummyTsfBackend:
            def __init__(self, soundfont=None):
                self.soundfont = soundfont

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def play(self, sequence):
                nonlocal played
                played = True
                return None

            def wait(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)

        # Create fake soundfont file
        sf_file = tmp_path / "test.sf2"
        sf_file.write_bytes(b"fake sf2")

        test_file = tmp_path / "test.alda"
        test_file.write_text("piano: c d e")

        # Patch TsfBackend and HAS_TSF at the backends module level
        monkeypatch.setattr("aldakit.midi.backends.TsfBackend", DummyTsfBackend)
        monkeypatch.setattr("aldakit.midi.backends.HAS_TSF", True)

        result = main(["play", str(test_file), "-sf", str(sf_file)])
        assert result == 0
        assert played is True


class TestMainRepl:
    """Tests for main function REPL command."""

    def test_repl_command_calls_run_repl(self, monkeypatch):
        """Test 'aldakit repl' calls run_repl."""
        run_repl_called = False

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        def mock_run_repl(*args, **kwargs):
            nonlocal run_repl_called
            run_repl_called = True
            return 0

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        # Patch at the source module since it's imported inside the function
        monkeypatch.setattr("aldakit.repl.run_repl", mock_run_repl)

        result = main(["repl"])
        assert result == 0
        assert run_repl_called

    def test_repl_audio_no_soundfont_error(self, monkeypatch, capsys):
        """REPL with audio but no SoundFont configured or discoverable."""

        class DummyBackend:
            def list_output_ports(self):
                return ["TestPort"]

        monkeypatch.setattr("aldakit.cli.LibremidiBackend", DummyBackend)
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)
        monkeypatch.setattr(
            "aldakit.midi.soundfont.find_soundfont", lambda: None
        )

        result = main(["repl", "-a"])
        assert result == 1

        err = capsys.readouterr().err
        assert "No SoundFont found" in err


class TestTranscribeCommand:
    """Tests for transcribe_command function."""

    def test_invalid_swing_ratio(self, capsys):
        """Test error for invalid swing ratio."""
        args = argparse.Namespace(
            swing_ratio=1.5,  # Invalid - must be between 0 and 1
            duration=10,
            instrument="piano",
            tempo=120,
            quantize=0.25,
            feel="swing",
            output=None,
            port=None,
            play=False,
            verbose=False,
            alda_notes=False,
        )

        result = transcribe_command(args)
        assert result == 1

        err = capsys.readouterr().err
        assert "swing-ratio" in err

    def test_transcribe_success(self, monkeypatch, capsys):
        """Test successful transcription."""
        from aldakit import Score

        def mock_transcribe(**kwargs):
            return Score("piano: c d e")

        # Patch at source module since it's imported inside transcribe_command
        monkeypatch.setattr("aldakit.midi.transcriber.transcribe", mock_transcribe)
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["InputPort"]
        )

        args = argparse.Namespace(
            swing_ratio=0.67,
            duration=5,
            instrument="piano",
            tempo=120,
            quantize=0.25,
            feel="straight",
            output=None,
            port=None,
            play=False,
            verbose=False,
            alda_notes=False,
        )

        result = transcribe_command(args)
        assert result == 0

        out = capsys.readouterr().out
        assert "piano" in out

    def test_transcribe_with_output(self, monkeypatch, tmp_path, capsys):
        """Test transcription with output file."""
        from aldakit import Score

        def mock_transcribe(**kwargs):
            return Score("piano: c d e")

        monkeypatch.setattr("aldakit.midi.transcriber.transcribe", mock_transcribe)
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["InputPort"]
        )

        output_file = tmp_path / "output.alda"
        args = argparse.Namespace(
            swing_ratio=0.67,
            duration=5,
            instrument="piano",
            tempo=120,
            quantize=0.25,
            feel="straight",
            output=output_file,
            port=None,
            play=False,
            verbose=False,
            alda_notes=False,
        )

        result = transcribe_command(args)
        assert result == 0

        err = capsys.readouterr().err
        assert "Saved to" in err

    def test_transcribe_verbose_alda_notes(self, monkeypatch, capsys):
        """Test transcription with verbose alda notes."""
        from aldakit import Score

        def mock_transcribe(**kwargs):
            # Call the on_note callback if provided
            on_note = kwargs.get("on_note")
            if on_note:
                on_note(60, 100, True)  # Note on
                on_note(60, 0, False)  # Note off
            return Score("piano: c")

        monkeypatch.setattr("aldakit.midi.transcriber.transcribe", mock_transcribe)
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["InputPort"]
        )

        args = argparse.Namespace(
            swing_ratio=0.67,
            duration=5,
            instrument="piano",
            tempo=120,
            quantize=0.25,
            feel="straight",
            output=None,
            port=None,
            play=False,
            verbose=True,
            alda_notes=True,
        )

        result = transcribe_command(args)
        assert result == 0

    def test_transcribe_runtime_error(self, monkeypatch, capsys):
        """Test transcription with runtime error."""

        def mock_transcribe(**kwargs):
            raise RuntimeError("No MIDI input available")

        monkeypatch.setattr("aldakit.midi.transcriber.transcribe", mock_transcribe)
        monkeypatch.setattr(
            "aldakit.midi.transcriber.list_input_ports", lambda: ["InputPort"]
        )

        args = argparse.Namespace(
            swing_ratio=0.67,
            duration=5,
            instrument="piano",
            tempo=120,
            quantize=0.25,
            feel="straight",
            output=None,
            port=None,
            play=False,
            verbose=False,
            alda_notes=False,
        )

        result = transcribe_command(args)
        assert result == 1

        err = capsys.readouterr().err
        assert "Error" in err
