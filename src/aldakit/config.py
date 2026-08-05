"""Configuration file handling for aldakit.

Supports INI-format configuration files in two locations (priority order):
1. ./aldakit.ini - Project-local config (current working directory)
2. ~/.aldakit/config.ini - User config (home directory)

CLI arguments always override config file settings.
Environment variables (e.g., ALDAKIT_SOUNDFONT) override config files but not CLI args.

Example config file (~/.aldakit/config.ini):

    [aldakit]
    soundfont = ~/Music/sf2/FluidR3_GM.sf2
    backend = midi
    port = FluidSynth
    tempo = 120
    verbose = false

Backend options:
    - "midi": Use libremidi for MIDI output (external synths, DAWs, virtual port)
    - "audio": Use built-in TinySoundFont for direct audio output (requires soundfont)
"""

import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path

from .constants import DEFAULT_BACKEND, DEFAULT_TEMPO, SOUNDFONT_ENV_VAR


@dataclass
class Config:
    """aldakit configuration."""

    soundfont: str | None = None
    backend: str = DEFAULT_BACKEND  # "midi" or "audio"
    port: str | None = None
    tempo: int = DEFAULT_TEMPO
    verbose: bool = False

    # Source tracking (for debugging)
    _sources: dict = field(default_factory=dict, repr=False)


def get_config_paths() -> list[Path]:
    """Return config file paths in priority order (highest first)."""
    paths = []
    # Local project config (highest priority)
    local = Path.cwd() / "aldakit.ini"
    if local.exists():
        paths.append(local)
    # User config
    user = Path.home() / ".aldakit" / "config.ini"
    if user.exists():
        paths.append(user)
    return paths


def load_config() -> Config:
    """Load configuration from files and environment.

    Priority order (highest to lowest):
    1. CLI arguments (handled by caller)
    2. Environment variables
    3. Local config (./aldakit.ini)
    4. User config (~/.aldakit/config.ini)
    5. Built-in defaults
    """
    config = Config()

    # Load from config files (lower priority first, so higher priority overwrites)
    for path in reversed(get_config_paths()):
        _load_file(config, path)

    # Environment variables override config files
    if sf := os.environ.get(SOUNDFONT_ENV_VAR):
        config.soundfont = _expand_path(sf)
        config._sources["soundfont"] = f"env:{SOUNDFONT_ENV_VAR}"

    return config


def _expand_path(path: str) -> str:
    """Expand ~ and environment variables in a path."""
    return os.path.expandvars(os.path.expanduser(path))


def _load_file(config: Config, path: Path) -> None:
    """Load settings from an INI file into config."""
    parser = ConfigParser()
    parser.read(path)

    if not parser.has_section("aldakit"):
        return

    source = str(path)

    # String options
    for key in ("soundfont", "backend", "port"):
        if parser.has_option("aldakit", key):
            value = parser.get("aldakit", key)
            if key == "soundfont":
                value = _expand_path(value)
            setattr(config, key, value)
            config._sources[key] = source

    # Integer options
    if parser.has_option("aldakit", "tempo"):
        config.tempo = parser.getint("aldakit", "tempo")
        config._sources["tempo"] = source

    # Boolean options
    if parser.has_option("aldakit", "verbose"):
        config.verbose = parser.getboolean("aldakit", "verbose")
        config._sources["verbose"] = source
