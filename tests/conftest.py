"""Shared fixtures.

The suite must not read or write the developer's real home directory. Config
discovery looks for ``~/.aldakit/config.ini`` and the REPL writes
``~/.aldakit_history``, so without isolation a developer's own settings change
what the tests see, and a test that saves history overwrites their file.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _fake_home(tmp_path_factory):
    """One throwaway home directory shared by the whole session."""
    return tmp_path_factory.mktemp("home")


@pytest.fixture(autouse=True)
def isolated_user_home(_fake_home, monkeypatch):
    """Point every way of reaching ``~`` at the throwaway home.

    ``Path.home()`` and ``os.path.expanduser`` resolve ``~`` differently --
    the latter reads HOME on POSIX and USERPROFILE on Windows -- so all three
    are set, or the two disagree and tests pass for the wrong reason.
    """
    monkeypatch.setenv("HOME", str(_fake_home))
    monkeypatch.setenv("USERPROFILE", str(_fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: _fake_home))
    return _fake_home
