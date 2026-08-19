"""Tests for SoundFont management utilities."""

import hashlib
from pathlib import Path

import pytest

from aldakit.midi.soundfont import (
    SoundFontManager,
    SOUNDFONT_CATALOG,
    DEFAULT_SOUNDFONT,
    SOUNDFONT_NAMES,
    get_soundfont_dir,
    find_soundfont,
    list_soundfonts,
    list_available_downloads,
    print_download_progress,
)


# =============================================================================
# SoundFontManager Tests
# =============================================================================


class TestSoundFontManager:
    """Tests for SoundFontManager class."""

    def test_init_defaults(self):
        """Manager initializes with default directory and catalog."""
        manager = SoundFontManager()
        assert manager.soundfont_dir == Path.home() / ".aldakit" / "soundfonts"
        assert manager.catalog == SOUNDFONT_CATALOG

    def test_init_custom_directory(self, tmp_path):
        """Manager accepts custom soundfont directory."""
        custom_dir = tmp_path / "custom_soundfonts"
        manager = SoundFontManager(soundfont_dir=custom_dir)
        assert manager.soundfont_dir == custom_dir

    def test_init_custom_catalog(self, tmp_path):
        """Manager accepts custom catalog."""
        custom_catalog = {
            "TestSF": {
                "url": "https://example.com/test.sf2",
                "filename": "test.sf2",
                "size_mb": 1,
                "description": "Test SoundFont",
            }
        }
        manager = SoundFontManager(
            soundfont_dir=tmp_path,
            catalog=custom_catalog,
        )
        assert "TestSF" in manager.catalog
        assert "FluidR3_GM" not in manager.catalog

    def test_catalog_returns_copy(self, tmp_path):
        """Catalog property returns a copy, not the original."""
        manager = SoundFontManager(soundfont_dir=tmp_path)
        catalog = manager.catalog
        catalog["NewSF"] = {}
        assert "NewSF" not in manager.catalog

    def test_get_search_paths(self, tmp_path):
        """Search paths include expected directories."""
        manager = SoundFontManager(soundfont_dir=tmp_path)
        paths = manager.get_search_paths()

        # Should include the custom soundfont dir
        assert tmp_path in paths

        # Should include common locations
        home = Path.home()
        assert home / "Music" / "sf2" in paths
        assert home / "Music" / "SoundFonts" in paths

    def test_find_returns_none_when_no_soundfonts(self, tmp_path, monkeypatch):
        """Find returns None when no SoundFont files exist."""
        # Clear environment variable
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.find()
        # Result may be None or an actual soundfont if one exists on the system
        assert result is None or result.suffix == ".sf2"

    def test_find_uses_env_variable(self, tmp_path, monkeypatch):
        """Find checks ALDAKIT_SOUNDFONT environment variable first."""
        sf_path = tmp_path / "env_test.sf2"
        sf_path.write_bytes(b"fake soundfont data")

        monkeypatch.setenv("ALDAKIT_SOUNDFONT", str(sf_path))

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.find()
        assert result == sf_path

    def test_find_searches_known_names(self, tmp_path, monkeypatch):
        """Find searches for known SoundFont filenames."""
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        # Create a known soundfont file in the soundfont dir
        sf_path = tmp_path / "TimGM6mb.sf2"
        sf_path.write_bytes(b"fake soundfont data")

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.find()
        assert result == sf_path

    def test_find_falls_back_to_any_sf2(self, tmp_path, monkeypatch):
        """Find falls back to any .sf2 file if known names not found."""
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        # Create an unknown soundfont file
        sf_path = tmp_path / "random_soundfont.sf2"
        sf_path.write_bytes(b"fake soundfont data")

        manager = SoundFontManager(soundfont_dir=tmp_path)

        # Mock get_search_paths to only return tmp_path (avoid system soundfonts)
        monkeypatch.setattr(manager, "get_search_paths", lambda: [tmp_path])

        result = manager.find()
        assert result == sf_path

    def test_list_returns_all_soundfonts(self, tmp_path, monkeypatch):
        """List returns all SoundFont files in search paths."""
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        # Create multiple soundfonts
        sf1 = tmp_path / "font1.sf2"
        sf2 = tmp_path / "font2.sf2"
        sf1.write_bytes(b"data1")
        sf2.write_bytes(b"data2")

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.list()

        # Should contain our test files
        assert sf1 in result
        assert sf2 in result

    def test_list_includes_env_soundfont(self, tmp_path, monkeypatch):
        """List includes soundfont from environment variable."""
        sf_path = tmp_path / "env_font.sf2"
        sf_path.write_bytes(b"env data")
        monkeypatch.setenv("ALDAKIT_SOUNDFONT", str(sf_path))

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.list()

        assert sf_path in result

    def test_list_deduplicates(self, tmp_path, monkeypatch):
        """List doesn't include duplicate paths."""
        sf_path = tmp_path / "font.sf2"
        sf_path.write_bytes(b"data")

        # Set env to same path
        monkeypatch.setenv("ALDAKIT_SOUNDFONT", str(sf_path))

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.list()

        # Should only appear once
        assert result.count(sf_path) == 1

    def test_list_available_downloads(self, tmp_path):
        """List available downloads returns catalog copy."""
        manager = SoundFontManager(soundfont_dir=tmp_path)
        downloads = manager.list_available_downloads()

        assert "TimGM6mb" in downloads
        assert "FluidR3_GM" in downloads
        assert downloads == SOUNDFONT_CATALOG

    def test_download_unknown_soundfont_raises(self, tmp_path):
        """Download raises ValueError for unknown SoundFont."""
        manager = SoundFontManager(soundfont_dir=tmp_path)

        with pytest.raises(ValueError) as exc_info:
            manager.download("NonExistent")

        assert "Unknown SoundFont" in str(exc_info.value)
        assert "NonExistent" in str(exc_info.value)

    def test_download_skips_existing_file(self, tmp_path):
        """Download skips if file already exists."""
        # Create target file
        target = tmp_path / "TimGM6mb.sf2"
        target.write_bytes(b"existing data")

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.download("TimGM6mb")

        assert result == target
        # File should not be modified
        assert target.read_bytes() == b"existing data"

    def test_download_force_overwrites(self, tmp_path, monkeypatch):
        """Download with force=True re-downloads even if file exists."""
        # Create target file
        target = tmp_path / "TimGM6mb.sf2"
        target.write_bytes(b"old data")

        # Mock the download to avoid actual network call
        def mock_download(url, path, callback):
            path.write_bytes(b"new data")

        # Also need to mock _file_sha256 to return expected hash
        def mock_sha256(path):
            return SOUNDFONT_CATALOG["TimGM6mb"]["sha256"]

        monkeypatch.setattr(
            SoundFontManager, "_download_file", staticmethod(mock_download)
        )
        monkeypatch.setattr(SoundFontManager, "_file_sha256", staticmethod(mock_sha256))

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.download("TimGM6mb", force=True)

        assert result == target
        assert target.read_bytes() == b"new data"

    def test_a_refused_download_says_how_to_get_the_file(self, tmp_path, monkeypatch):
        """A host behind a bot challenge answers every retry with 403.

        Reporting the status code alone leaves the reader with nothing to do,
        so the error names the URL a browser can still fetch and the path to
        save it to.
        """
        import urllib.error

        def refuse(url, path, callback):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

        monkeypatch.setattr(SoundFontManager, "_download_file", staticmethod(refuse))
        manager = SoundFontManager(soundfont_dir=tmp_path)

        with pytest.raises(RuntimeError) as excinfo:
            manager.download("TimGM6mb")

        message = str(excinfo.value)
        assert "retrying will not help" in message
        assert SOUNDFONT_CATALOG["TimGM6mb"]["url"] in message
        assert str(tmp_path / "TimGM6mb.sf2") in message

    def test_an_ordinary_download_failure_is_reported_plainly(
        self, tmp_path, monkeypatch
    ):
        """A 404 is not a bot challenge, so it gets no browser advice."""
        import urllib.error

        def missing(url, path, callback):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        monkeypatch.setattr(SoundFontManager, "_download_file", staticmethod(missing))
        manager = SoundFontManager(soundfont_dir=tmp_path)

        with pytest.raises(RuntimeError, match="404") as excinfo:
            manager.download("TimGM6mb")
        assert "retrying will not help" not in str(excinfo.value)

    def test_ensure_returns_existing(self, tmp_path, monkeypatch):
        """Ensure returns existing SoundFont if available."""
        monkeypatch.delenv("ALDAKIT_SOUNDFONT", raising=False)

        # Create existing soundfont
        sf_path = tmp_path / "TimGM6mb.sf2"
        sf_path.write_bytes(b"data")

        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.ensure()

        assert result == sf_path

    def test_verify_checksums_missing_file(self, tmp_path):
        """Verify checksums returns False for missing files."""
        manager = SoundFontManager(soundfont_dir=tmp_path)
        result = manager.verify_checksums()

        # All catalog entries should be False (files don't exist)
        for name in SOUNDFONT_CATALOG:
            assert result[name] is False

    def test_verify_checksums_valid_file(self, tmp_path):
        """Verify checksums returns True for valid checksums."""
        # Create file with known content
        sf_path = tmp_path / "TimGM6mb.sf2"
        content = b"test content for hashing"
        sf_path.write_bytes(content)

        # Calculate actual hash
        actual_hash = hashlib.sha256(content).hexdigest()

        # Create custom catalog with our hash
        custom_catalog = {
            "TimGM6mb": {
                "url": "https://example.com/test.sf2",
                "filename": "TimGM6mb.sf2",
                "size_mb": 1,
                "sha256": actual_hash,
            }
        }

        manager = SoundFontManager(soundfont_dir=tmp_path, catalog=custom_catalog)
        result = manager.verify_checksums()

        assert result["TimGM6mb"] is True

    def test_verify_checksums_invalid_file(self, tmp_path):
        """Verify checksums returns False for invalid checksums."""
        # Create file with some content
        sf_path = tmp_path / "TimGM6mb.sf2"
        sf_path.write_bytes(b"different content")

        # Create catalog with different hash
        custom_catalog = {
            "TimGM6mb": {
                "url": "https://example.com/test.sf2",
                "filename": "TimGM6mb.sf2",
                "size_mb": 1,
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            }
        }

        manager = SoundFontManager(soundfont_dir=tmp_path, catalog=custom_catalog)
        result = manager.verify_checksums()

        assert result["TimGM6mb"] is False

    def test_verify_checksums_no_hash_in_catalog(self, tmp_path):
        """Verify checksums returns True for files without hash in catalog."""
        # Create file
        sf_path = tmp_path / "test.sf2"
        sf_path.write_bytes(b"content")

        # Catalog entry without sha256
        custom_catalog = {
            "TestSF": {
                "url": "https://example.com/test.sf2",
                "filename": "test.sf2",
                "size_mb": 1,
                # No sha256 key
            }
        }

        manager = SoundFontManager(soundfont_dir=tmp_path, catalog=custom_catalog)
        result = manager.verify_checksums()

        assert result["TestSF"] is True


class TestSoundFontManagerFileHash:
    """Tests for SoundFontManager._file_sha256."""

    def test_file_sha256(self, tmp_path):
        """File SHA256 calculates correct hash."""
        test_file = tmp_path / "test.bin"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        actual = SoundFontManager._file_sha256(test_file)

        assert actual == expected

    def test_file_sha256_empty_file(self, tmp_path):
        """File SHA256 handles empty files."""
        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        actual = SoundFontManager._file_sha256(test_file)

        assert actual == expected


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_soundfont_dir_creates_directory(self, tmp_path, monkeypatch):
        """get_soundfont_dir creates the directory if needed."""
        # Patch the default manager's soundfont_dir
        from aldakit.midi import soundfont as sf_module

        original_manager = sf_module._default_manager
        try:
            sf_module._default_manager = SoundFontManager(
                soundfont_dir=tmp_path / "new_dir"
            )
            result = get_soundfont_dir()
            assert result.exists()
            assert result == tmp_path / "new_dir"
        finally:
            sf_module._default_manager = original_manager

    def test_find_soundfont_delegates(self, tmp_path, monkeypatch):
        """find_soundfont delegates to manager.find()."""
        sf_path = tmp_path / "test.sf2"
        sf_path.write_bytes(b"data")
        monkeypatch.setenv("ALDAKIT_SOUNDFONT", str(sf_path))

        result = find_soundfont()
        assert result == sf_path

    def test_list_soundfonts_delegates(self, tmp_path, monkeypatch):
        """list_soundfonts delegates to manager.list()."""
        result = list_soundfonts()
        assert isinstance(result, list)

    def test_list_available_downloads_delegates(self):
        """list_available_downloads delegates to manager."""
        result = list_available_downloads()
        assert "TimGM6mb" in result


class TestPrintDownloadProgress:
    """Tests for print_download_progress function."""

    def test_progress_with_total(self, capsys):
        """Progress prints percentage when total is known."""
        print_download_progress(512 * 1024, 1024 * 1024)  # 512KB of 1MB
        captured = capsys.readouterr()
        assert "50%" in captured.out or "0.5" in captured.out

    def test_progress_without_total(self, capsys):
        """Progress prints only downloaded when total is unknown."""
        print_download_progress(1024 * 1024, 0)  # 1MB downloaded, unknown total
        captured = capsys.readouterr()
        assert "1.0" in captured.out  # 1.0 MB
        assert "%" not in captured.out


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_soundfont_in_catalog(self):
        """DEFAULT_SOUNDFONT exists in SOUNDFONT_CATALOG."""
        assert DEFAULT_SOUNDFONT in SOUNDFONT_CATALOG

    def test_catalog_entries_have_required_fields(self):
        """All catalog entries have required fields."""
        required_fields = ["url", "filename", "size_mb", "description"]

        for name, info in SOUNDFONT_CATALOG.items():
            for field in required_fields:
                assert field in info, f"{name} missing {field}"

    def test_catalog_entries_have_sha256(self):
        """All catalog entries have SHA256 checksums."""
        for name, info in SOUNDFONT_CATALOG.items():
            assert "sha256" in info, f"{name} missing sha256"
            assert len(info["sha256"]) == 64, f"{name} has invalid sha256 length"

    def test_soundfont_names_are_sf2(self):
        """All SOUNDFONT_NAMES end with .sf2."""
        for name in SOUNDFONT_NAMES:
            assert name.endswith(".sf2"), f"{name} should end with .sf2"
