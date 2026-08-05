"""Tests for configuration file handling."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from aldakit.config import (
    Config,
    get_config_paths,
    load_config,
    _expand_path,
    _load_file,
)


class TestConfig:
    """Test the Config dataclass."""

    def test_default_values(self):
        config = Config()
        assert config.soundfont is None
        assert config.backend == "midi"
        assert config.port is None
        assert config.tempo == 120
        assert config.verbose is False

    def test_custom_values(self):
        config = Config(
            soundfont="/path/to/sf2",
            backend="audio",
            port="FluidSynth",
            tempo=140,
            verbose=True,
        )
        assert config.soundfont == "/path/to/sf2"
        assert config.backend == "audio"
        assert config.port == "FluidSynth"
        assert config.tempo == 140
        assert config.verbose is True

    def test_sources_tracking(self):
        config = Config()
        config._sources["soundfont"] = "/path/to/config.ini"
        assert "soundfont" in config._sources


class TestExpandPath:
    """Test path expansion."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_PATH_EXPANSION_TESTS") == "1",
        reason="Path expansion tests skipped via SKIP_PATH_EXPANSION_TESTS",
    )
    def test_expands_tilde(self):
        result = _expand_path("~/Music/sf2/test.sf2")
        # Use Path for cross-platform comparison (handles slash differences)
        result_path = Path(result)
        assert result_path.is_relative_to(Path.home())
        assert result_path.parts[-3:] == ("Music", "sf2", "test.sf2")

    @pytest.mark.skipif(
        os.environ.get("SKIP_PATH_EXPANSION_TESTS") == "1",
        reason="Path expansion tests skipped via SKIP_PATH_EXPANSION_TESTS",
    )
    @pytest.mark.skipif(
        sys.platform == "win32", reason="Unix-style paths on non-Windows"
    )
    def test_expands_env_var(self):
        with patch.dict(os.environ, {"MY_PATH": "/custom/path"}):
            result = _expand_path("$MY_PATH/soundfont.sf2")
            assert result == "/custom/path/soundfont.sf2"

    @pytest.mark.skipif(
        os.environ.get("SKIP_PATH_EXPANSION_TESTS") == "1",
        reason="Path expansion tests skipped via SKIP_PATH_EXPANSION_TESTS",
    )
    @pytest.mark.skipif(
        sys.platform == "win32", reason="Unix-style paths on non-Windows"
    )
    def test_absolute_path_unchanged(self):
        result = _expand_path("/absolute/path/to/file.sf2")
        assert result == "/absolute/path/to/file.sf2"

    @pytest.mark.skipif(
        os.environ.get("SKIP_PATH_EXPANSION_TESTS") == "1",
        reason="Path expansion tests skipped via SKIP_PATH_EXPANSION_TESTS",
    )
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-style paths only")
    def test_expands_env_var_windows(self):
        with patch.dict(os.environ, {"MY_PATH": "C:\\custom\\path"}):
            result = _expand_path("$MY_PATH/soundfont.sf2")
            result_path = Path(result)
            assert result_path.parts[-1] == "soundfont.sf2"
            assert "custom" in result_path.parts

    @pytest.mark.skipif(
        os.environ.get("SKIP_PATH_EXPANSION_TESTS") == "1",
        reason="Path expansion tests skipped via SKIP_PATH_EXPANSION_TESTS",
    )
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-style paths only")
    def test_absolute_path_unchanged_windows(self):
        result = _expand_path("C:\\absolute\\path\\to\\file.sf2")
        result_path = Path(result)
        assert result_path.parts[-1] == "file.sf2"
        assert result_path.is_absolute()


class TestGetConfigPaths:
    """Test config file path discovery."""

    def test_returns_list(self):
        result = get_config_paths()
        assert isinstance(result, list)

    def test_finds_local_config(self, tmp_path, monkeypatch):
        # Create local config
        config_file = tmp_path / "aldakit.ini"
        config_file.write_text("[aldakit]\ntempo = 100\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        paths = get_config_paths()
        assert config_file in paths

    def test_finds_user_config(self, tmp_path, monkeypatch):
        # Create user config directory
        aldakit_dir = tmp_path / ".aldakit"
        aldakit_dir.mkdir()
        config_file = aldakit_dir / "config.ini"
        config_file.write_text("[aldakit]\nport = TestPort\n", encoding="utf-8")

        # Mock home directory
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Change to a directory without local config
        monkeypatch.chdir(tmp_path)

        paths = get_config_paths()
        assert config_file in paths

    def test_priority_order(self, tmp_path, monkeypatch):
        # Create both local and user config
        aldakit_dir = tmp_path / ".aldakit"
        aldakit_dir.mkdir()
        user_config = aldakit_dir / "config.ini"
        user_config.write_text("[aldakit]\nport = UserPort\n", encoding="utf-8")

        local_config = tmp_path / "aldakit.ini"
        local_config.write_text("[aldakit]\nport = LocalPort\n", encoding="utf-8")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)

        paths = get_config_paths()
        # Local should be first (higher priority)
        assert paths[0] == local_config
        assert paths[1] == user_config


class TestLoadFile:
    """Test loading settings from INI file."""

    def test_loads_all_settings(self, tmp_path):
        config_file = tmp_path / "config.ini"
        config_file.write_text(
            """
[aldakit]
soundfont = ~/Music/sf2/test.sf2
backend = audio
port = TestPort
tempo = 140
verbose = true
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)

        assert config.soundfont.endswith("Music/sf2/test.sf2")
        assert config.backend == "audio"
        assert config.port == "TestPort"
        assert config.tempo == 140
        assert config.verbose is True

    def test_partial_settings(self, tmp_path):
        config_file = tmp_path / "config.ini"
        config_file.write_text(
            """
[aldakit]
tempo = 100
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)

        # Only tempo should be changed
        assert config.tempo == 100
        # Others remain default
        assert config.soundfont is None
        assert config.backend == "midi"
        assert config.port is None
        assert config.verbose is False

    def test_tracks_sources(self, tmp_path):
        config_file = tmp_path / "config.ini"
        config_file.write_text(
            """
[aldakit]
tempo = 150
port = MyPort
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)

        assert config._sources["tempo"] == str(config_file)
        assert config._sources["port"] == str(config_file)

    def test_ignores_missing_section(self, tmp_path):
        config_file = tmp_path / "config.ini"
        config_file.write_text(
            """
[other]
foo = bar
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)

        # Should remain default
        assert config.tempo == 120

    def test_boolean_values(self, tmp_path):
        config_file = tmp_path / "config.ini"
        # Test various boolean representations
        config_file.write_text(
            """
[aldakit]
verbose = yes
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)
        assert config.verbose is True

        config_file.write_text(
            """
[aldakit]
verbose = no
""",
            encoding="utf-8",
        )
        config = Config()
        _load_file(config, config_file)
        assert config.verbose is False


class TestLoadConfig:
    """Test the full config loading process."""

    def test_default_config(self, tmp_path, monkeypatch):
        # Empty environment with no config files
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        config = load_config()

        assert config.soundfont is None
        assert config.backend == "midi"
        assert config.port is None
        assert config.tempo == 120
        assert config.verbose is False

    def test_env_var_overrides_config(self, tmp_path, monkeypatch):
        # Create config with soundfont
        aldakit_dir = tmp_path / ".aldakit"
        aldakit_dir.mkdir()
        config_file = aldakit_dir / "config.ini"
        config_file.write_text(
            """
[aldakit]
soundfont = /config/soundfont.sf2
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)

        # Env var should override
        monkeypatch.setenv("ALDAKIT_SOUNDFONT", "/env/soundfont.sf2")

        config = load_config()
        assert config.soundfont == "/env/soundfont.sf2"
        assert config._sources["soundfont"] == "env:ALDAKIT_SOUNDFONT"

    def test_local_overrides_user(self, tmp_path, monkeypatch):
        # Create user config
        aldakit_dir = tmp_path / ".aldakit"
        aldakit_dir.mkdir()
        user_config = aldakit_dir / "config.ini"
        user_config.write_text(
            """
[aldakit]
tempo = 100
port = UserPort
""",
            encoding="utf-8",
        )

        # Create local config (should override)
        local_config = tmp_path / "aldakit.ini"
        local_config.write_text(
            """
[aldakit]
tempo = 200
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        config = load_config()

        # tempo from local config
        assert config.tempo == 200
        # port from user config (not overridden)
        assert config.port == "UserPort"


class TestConfigIntegration:
    """Integration tests for config with CLI."""

    def test_config_import(self):
        """Ensure config module can be imported."""
        from aldakit.config import Config, load_config, get_config_paths

        assert Config is not None
        assert load_config is not None
        assert get_config_paths is not None
