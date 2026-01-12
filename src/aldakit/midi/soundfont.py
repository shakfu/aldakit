"""SoundFont management utilities.

This module provides tools for finding, downloading, and managing
SoundFont files for use with the TinySoundFont backend.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Default SoundFont directory
SOUNDFONT_DIR = Path.home() / ".aldakit" / "soundfonts"

# Available SoundFonts for download (public domain / freely distributable)
SOUNDFONT_CATALOG = {
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


def get_soundfont_dir() -> Path:
    """Get the aldakit SoundFont directory, creating it if needed."""
    SOUNDFONT_DIR.mkdir(parents=True, exist_ok=True)
    return SOUNDFONT_DIR


def list_available_downloads() -> dict[str, dict]:
    """List SoundFonts available for download.

    Returns:
        Dictionary of SoundFont names to their metadata.
    """
    return SOUNDFONT_CATALOG.copy()


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
    if name not in SOUNDFONT_CATALOG:
        available = ", ".join(SOUNDFONT_CATALOG.keys())
        raise ValueError(f"Unknown SoundFont: {name}. Available: {available}")

    info = SOUNDFONT_CATALOG[name]
    target_dir = target_dir or get_soundfont_dir()
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
        _download_file(url, tmp_path, progress_callback)

        # Verify hash if provided
        if info.get("sha256"):
            actual_hash = _file_sha256(tmp_path)
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


def _download_file(
    url: str, target: Path, progress_callback: Callable[[int, int], None] | None = None
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


def _file_sha256(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


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
    from .backends.tsf_backend import find_soundfont

    # First check if any SoundFont is already available
    existing = find_soundfont()
    if existing is not None:
        return existing

    # Download the requested one
    return download_soundfont(name, progress_callback=progress_callback)


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
    from .backends.tsf_backend import find_soundfont

    existing = find_soundfont()
    if existing:
        print(f"SoundFont already available: {existing}")
        return existing

    info = SOUNDFONT_CATALOG.get(name, {})
    print(f"Downloading {name} ({info.get('size_mb', '?')} MB)...")
    print(f"  {info.get('description', '')}")

    path = download_soundfont(name, progress_callback=print_download_progress)
    print()  # Newline after progress
    print(f"Saved to: {path}")
    return path
