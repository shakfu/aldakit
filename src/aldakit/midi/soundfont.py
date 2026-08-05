"""SoundFont management utilities.

This module provides tools for finding, downloading, and managing
SoundFont files for use with the TinySoundFont backend.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ..constants import SOUNDFONT_ENV_VAR

# Available SoundFonts for download (public domain / freely distributable)
SOUNDFONT_CATALOG: dict[str, dict] = {
    "FluidR3_GM": {
        "url": "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2",
        "filename": "FluidR3_GM.sf2",
        "size_mb": 141,
        "description": "High-quality GM SoundFont (large)",
        "sha256": "74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0",
    },
    "GeneralUser_GS": {
        "url": "https://musical-artifacts.com/artifacts/6789/GeneralUser-GS.sf2",
        "filename": "GeneralUser-GS.sf2",
        "size_mb": 31,
        "description": "Well-balanced GM/GS SoundFont",
        "sha256": "c278464b823daf9c52106c0957f752817da0e52964817ff682fe3a8d2f8446ce",
    },
    "TimGM6mb": {
        "url": "https://musical-artifacts.com/artifacts/7293/TimGM6mb.sf2",
        "filename": "TimGM6mb.sf2",
        "size_mb": 5.8,
        "description": "Compact GM SoundFont, good quality for size",
        "sha256": "82475b91a76de15cb28a104707d3247ba932e228bada3f47bba63c6b31aaf7a1",
    },
}

# Default SoundFont to download
DEFAULT_SOUNDFONT = "TimGM6mb"

# Common SoundFont filenames (ordered by preference)
SOUNDFONT_NAMES = [
    "FluidR3_GM.sf2",
    "FluidR3_GS.sf2",
    "GeneralUser_GS.sf2",
    "TimGM6mb.sf2",
    "GeneralUser-GS.sf2",
    "SGM-V2.01.sf2",
    "Arachno.sf2",
    "default.sf2",
]


class SoundFontManager:
    """Manages SoundFont discovery, downloading, and setup.

    This class provides methods for finding existing SoundFont files,
    downloading new ones from a catalog, and ensuring a SoundFont is
    available for playback.

    Example:
        >>> manager = SoundFontManager()
        >>> sf = manager.find()  # Find existing SoundFont
        >>> if sf is None:
        ...     sf = manager.download("TimGM6mb")  # Download one

        >>> # Or use ensure() to auto-download if needed
        >>> sf = manager.ensure()
    """

    def __init__(
        self,
        soundfont_dir: Path | None = None,
        catalog: dict[str, dict] | None = None,
    ):
        """Initialize the SoundFont manager.

        Args:
            soundfont_dir: Directory for storing downloaded SoundFonts.
                Defaults to ~/.aldakit/soundfonts/
            catalog: Custom catalog of available SoundFonts. Defaults to
                the built-in SOUNDFONT_CATALOG.
        """
        self._soundfont_dir = soundfont_dir or (Path.home() / ".aldakit" / "soundfonts")
        self._catalog = catalog if catalog is not None else SOUNDFONT_CATALOG

    @property
    def soundfont_dir(self) -> Path:
        """The directory where SoundFonts are stored."""
        return self._soundfont_dir

    @property
    def catalog(self) -> dict[str, dict]:
        """The catalog of available SoundFonts for download."""
        return self._catalog.copy()

    def get_search_paths(self) -> list[Path]:
        """Get the list of paths searched for SoundFont files.

        Returns:
            List of directory paths to search.
        """
        search_paths: list[Path] = []
        home = Path.home()

        # aldakit data directory (highest priority after env var)
        search_paths.append(self._soundfont_dir)

        # User locations (all platforms)
        search_paths.extend(
            [
                home / "Music" / "sf2",
                home / "Music" / "SoundFonts",
                home / ".local" / "share" / "soundfonts",
                home / ".local" / "share" / "sounds" / "sf2",
            ]
        )

        # Linux system locations
        search_paths.extend(
            [
                Path("/usr/share/soundfonts"),
                Path("/usr/share/sounds/sf2"),
                Path("/usr/local/share/soundfonts"),
            ]
        )

        # macOS locations
        search_paths.extend(
            [
                Path("/Library/Audio/Sounds/Banks"),
                home / "Library" / "Audio" / "Sounds" / "Banks",
            ]
        )

        # Windows locations
        search_paths.extend(
            [
                Path("C:/soundfonts"),
                home / "Documents" / "SoundFonts",
            ]
        )

        return search_paths

    def find(self) -> Path | None:
        """Search common locations for a General MIDI SoundFont.

        Checks the ALDAKIT_SOUNDFONT environment variable first, then
        searches standard locations for known SoundFont filenames.

        Returns:
            Path to a SoundFont file, or None if not found.
        """
        # Check environment variable first
        env_path = os.environ.get(SOUNDFONT_ENV_VAR)
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p

        search_paths = self.get_search_paths()

        # Search for specific names first
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for name in SOUNDFONT_NAMES:
                sf_path = search_path / name
                if sf_path.exists():
                    return sf_path

        # Fall back to any .sf2 file
        for search_path in search_paths:
            if not search_path.exists():
                continue
            sf2_files = sorted(search_path.glob("*.sf2"))
            if sf2_files:
                return sf2_files[0]

        return None

    def list_installed(self) -> list[Path]:
        """List all SoundFont files found in common locations.

        Returns:
            Sorted list of paths to SoundFont files.
        """
        found: list[Path] = []
        seen: set[Path] = set()

        # Check environment variable
        env_path = os.environ.get(SOUNDFONT_ENV_VAR)
        if env_path:
            p = Path(env_path)
            if p.exists() and p not in seen:
                found.append(p)
                seen.add(p)

        # Search all paths
        for search_path in self.get_search_paths():
            if not search_path.exists():
                continue
            for sf_path in search_path.glob("*.sf2"):
                if sf_path not in seen:
                    found.append(sf_path)
                    seen.add(sf_path)

        return sorted(found)

    def list_available_downloads(self) -> dict[str, dict]:
        """List SoundFonts available for download.

        Returns:
            Dictionary of SoundFont names to their metadata.
        """
        return self._catalog.copy()

    def download(
        self,
        name: str = DEFAULT_SOUNDFONT,
        target_dir: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        force: bool = False,
    ) -> Path:
        """Download a SoundFont file.

        Args:
            name: Name of the SoundFont (see list_available_downloads()).
            target_dir: Directory to save to. Defaults to soundfont_dir.
            progress_callback: Optional callback(bytes_downloaded, total_bytes).
            force: If True, re-download even if file exists.

        Returns:
            Path to the downloaded SoundFont file.

        Raises:
            ValueError: If SoundFont name is not in catalog.
            RuntimeError: If download fails or hash verification fails.
        """
        if name not in self._catalog:
            available = ", ".join(self._catalog.keys())
            raise ValueError(f"Unknown SoundFont: {name}. Available: {available}")

        info = self._catalog[name]
        target_dir = target_dir or self._soundfont_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = str(info["filename"])
        target_path = target_dir / filename

        # Skip if exists (unless force)
        if target_path.exists() and not force:
            return target_path

        url = str(info["url"])

        # Download to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sf2") as tmp:
            tmp_path = Path(tmp.name)

        try:
            self._download_file(url, tmp_path, progress_callback)

            # Verify hash if provided
            if info.get("sha256"):
                actual_hash = self._file_sha256(tmp_path)
                if actual_hash != info["sha256"]:
                    raise RuntimeError(
                        f"Hash mismatch for {name}: expected {info['sha256']}, got {actual_hash}"
                    )

            # Move to target location
            shutil.move(str(tmp_path), str(target_path))
            return target_path

        except Exception:
            # Clean up temp file on error
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def ensure(
        self,
        name: str = DEFAULT_SOUNDFONT,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Ensure a SoundFont is available, downloading if necessary.

        This method:
        1. Checks if any SoundFont already exists
        2. Downloads the specified one if not found

        Args:
            name: Name of the SoundFont to download if needed.
            progress_callback: Optional callback(bytes_downloaded, total_bytes).

        Returns:
            Path to the SoundFont file.
        """
        existing = self.find()
        if existing is not None:
            return existing

        return self.download(name, progress_callback=progress_callback)

    def setup(self, name: str = DEFAULT_SOUNDFONT) -> Path:
        """Interactive SoundFont setup with progress display.

        Args:
            name: Name of SoundFont to download.

        Returns:
            Path to the SoundFont file.
        """
        existing = self.find()
        if existing:
            print(f"SoundFont already available: {existing}")
            return existing

        info = self._catalog.get(name, {})
        print(f"Downloading {name} ({info.get('size_mb', '?')} MB)...")
        print(f"  {info.get('description', '')}")

        path = self.download(name, progress_callback=print_download_progress)
        print()  # Newline after progress
        print(f"Saved to: {path}")
        return path

    def setup_all(self, force: bool = False) -> list[Path]:
        """Download all SoundFonts from the catalog with progress display.

        Downloads each SoundFont in the catalog, verifying SHA256 checksums.
        Skips files that already exist unless force=True.

        Args:
            force: If True, re-download even if files exist.

        Returns:
            List of paths to all downloaded SoundFont files.

        Raises:
            RuntimeError: If any download fails or checksum verification fails.
        """
        downloaded_paths: list[Path] = []
        total_items = len(self._catalog)

        for idx, (name, info) in enumerate(self._catalog.items(), 1):
            target_path = self._soundfont_dir / str(info["filename"])

            if target_path.exists() and not force:
                print(f"[{idx}/{total_items}] {name}: already exists, skipping")
                downloaded_paths.append(target_path)
                continue

            print(
                f"[{idx}/{total_items}] Downloading {name} ({info.get('size_mb', '?')} MB)..."
            )
            print(f"  {info.get('description', '')}")

            if not info.get("sha256"):
                print(f"  WARNING: No SHA256 checksum defined for {name}")

            path = self.download(
                name, progress_callback=print_download_progress, force=force
            )
            print()  # Newline after progress
            print(f"  Saved to: {path}")
            print(f"  SHA256 verified: {info.get('sha256', 'N/A')[:16]}...")
            downloaded_paths.append(path)

        print(
            f"\nDownloaded {len(downloaded_paths)} SoundFont(s) to {self._soundfont_dir}"
        )
        return downloaded_paths

    def verify_checksums(self) -> dict[str, bool]:
        """Verify SHA256 checksums for all downloaded SoundFonts.

        Checks each file in the catalog that exists in the soundfont directory.

        Returns:
            Dictionary mapping SoundFont names to verification status.
            True if checksum matches, False if mismatch or file missing.
        """
        results: dict[str, bool] = {}

        for name, info in self._catalog.items():
            target_path = self._soundfont_dir / str(info["filename"])

            if not target_path.exists():
                results[name] = False
                continue

            expected_hash = info.get("sha256")
            if not expected_hash:
                # No hash to verify, assume ok if file exists
                results[name] = True
                continue

            actual_hash = self._file_sha256(target_path)
            results[name] = actual_hash == expected_hash

        return results

    @staticmethod
    def _download_file(
        url: str,
        target: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Download a file with optional progress callback."""
        request = urllib.request.Request(
            url, headers={"User-Agent": "aldakit/0.1 (SoundFont downloader)"}
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536  # 64KB chunks

            with open(target, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# Deprecated alias for SoundFontManager.list_installed(). Attached after the
# class body: binding the name ``list`` inside it would shadow the builtin for
# every ``list[...]`` annotation in the class.
SoundFontManager.list = SoundFontManager.list_installed  # type: ignore[attr-defined]


# Default manager instance
_default_manager = SoundFontManager()


# Module-level convenience functions (backwards compatibility)


def get_soundfont_dir() -> Path:
    """Get the aldakit SoundFont directory, creating it if needed."""
    _default_manager.soundfont_dir.mkdir(parents=True, exist_ok=True)
    return _default_manager.soundfont_dir


def find_soundfont() -> Path | None:
    """Search common locations for a General MIDI SoundFont.

    Returns:
        Path to a SoundFont file, or None if not found.
    """
    return _default_manager.find()


def list_soundfonts() -> list[Path]:
    """List all SoundFont files found in common locations.

    Returns:
        List of paths to SoundFont files.
    """
    return _default_manager.list_installed()


def list_available_downloads() -> dict[str, dict]:
    """List SoundFonts available for download.

    Returns:
        Dictionary of SoundFont names to their metadata.
    """
    return _default_manager.list_available_downloads()


def download_soundfont(
    name: str = DEFAULT_SOUNDFONT,
    target_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """Download a SoundFont file.

    Args:
        name: Name of the SoundFont (see list_available_downloads()).
        target_dir: Directory to save to (default: ~/.aldakit/soundfonts/).
        progress_callback: Optional callback(bytes_downloaded, total_bytes).
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded SoundFont file.

    Raises:
        ValueError: If SoundFont name is not in catalog.
        RuntimeError: If download fails.
    """
    return _default_manager.download(name, target_dir, progress_callback, force)


def ensure_soundfont(
    name: str = DEFAULT_SOUNDFONT,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Ensure a SoundFont is available, downloading if necessary.

    This is a convenience function that:
    1. Checks if the SoundFont already exists
    2. Downloads it if not found

    Args:
        name: Name of the SoundFont.
        progress_callback: Optional callback(bytes_downloaded, total_bytes).

    Returns:
        Path to the SoundFont file.
    """
    return _default_manager.ensure(name, progress_callback)


def print_download_progress(downloaded: int, total: int) -> None:
    """Simple console progress printer."""
    if total > 0:
        pct = (downloaded / total) * 100
        mb_down = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        print(f"\rDownloading: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="")
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\rDownloading: {mb_down:.1f} MB", end="")


def setup_soundfont(name: str = DEFAULT_SOUNDFONT) -> Path:
    """Interactive SoundFont setup with progress display.

    Args:
        name: Name of SoundFont to download.

    Returns:
        Path to the SoundFont file.
    """
    return _default_manager.setup(name)


def setup_all_soundfonts(force: bool = False) -> list[Path]:
    """Download all SoundFonts from the catalog with progress display.

    Downloads each SoundFont in the catalog, verifying SHA256 checksums.
    Skips files that already exist unless force=True.

    Args:
        force: If True, re-download even if files exist.

    Returns:
        List of paths to all downloaded SoundFont files.

    Raises:
        RuntimeError: If any download fails or checksum verification fails.
    """
    return _default_manager.setup_all(force)


def verify_soundfont_checksums() -> dict[str, bool]:
    """Verify SHA256 checksums for all downloaded SoundFonts.

    Checks each file in the catalog that exists in the soundfont directory.

    Returns:
        Dictionary mapping SoundFont names to verification status.
        True if checksum matches, False if mismatch or file missing.
    """
    return _default_manager.verify_checksums()
