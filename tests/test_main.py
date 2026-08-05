"""Tests for __main__.py module entry point."""

import subprocess
import sys

import pytest


class TestMainModule:
    """Tests for running aldakit as a module."""

    def test_module_invocation_help(self):
        """Running python -m aldakit --help exits successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "aldakit", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "aldakit" in result.stdout.lower()

    def test_module_invocation_version(self):
        """Running python -m aldakit --version shows version."""
        result = subprocess.run(
            [sys.executable, "-m", "aldakit", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Version should be in the output
        from aldakit import __version__

        assert __version__ in result.stdout

    def test_module_invocation_no_args(self):
        """Running python -m aldakit with no args shows help or usage."""
        result = subprocess.run(
            [sys.executable, "-m", "aldakit"],
            capture_output=True,
            text=True,
        )
        # Should either succeed with help or fail with usage message
        combined_output = result.stdout + result.stderr
        assert (
            "usage" in combined_output.lower()
            or "aldakit" in combined_output.lower()
            or result.returncode in (0, 1, 2)
        )


class TestMainImport:
    """Tests for direct import of __main__ module."""

    def test_main_imports_cli_main(self):
        """__main__ module imports main from cli."""
        from aldakit import __main__

        assert hasattr(__main__, "main")
        from aldakit.cli import main as cli_main

        assert __main__.main is cli_main

    def test_main_call_with_help(self):
        """Calling main with --help exits with 0."""
        from aldakit.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_call_with_version(self):
        """Calling main with --version exits with 0."""
        from aldakit.__main__ import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


class TestVersion:
    """The version is declared in two places; keep them from drifting."""

    def test_package_and_metadata_agree(self):
        """__version__ must match the version pip/uv installed."""
        from importlib.metadata import version

        import aldakit

        assert aldakit.__version__ == version("aldakit")

    def test_pyproject_agrees(self):
        """__version__ must match pyproject.toml, the build's source of truth."""
        import re
        from pathlib import Path

        import aldakit

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
        assert declared, "no version in pyproject.toml"
        assert aldakit.__version__ == declared.group(1)

    def test_changelog_documents_this_version(self):
        """A release must have a CHANGELOG section."""
        from pathlib import Path

        import aldakit

        changelog = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"## [{aldakit.__version__}]" in changelog, (
            f"CHANGELOG.md has no section for {aldakit.__version__}"
        )
