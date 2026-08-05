"""TinySoundFont backend for direct audio synthesis.

This backend renders MIDI to audio directly using SoundFont files,
without requiring an external synthesizer like FluidSynth.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ...constants import DEFAULT_SOUNDFONT_GAIN
from .base import MidiBackend
from ..smf import write_midi_file
from ..soundfont import find_soundfont, list_soundfonts
from ..types import MidiSequence

# Try to import the native module
try:
    from ... import _tsf  # type: ignore[attr-defined]

    HAS_TSF = True
except ImportError:
    HAS_TSF = False
    _tsf = None


def is_available() -> bool:
    """Check if the TinySoundFont backend is available."""
    return HAS_TSF


# Re-export for backwards compatibility (functions now live in soundfont.py)
__all__ = ["TsfBackend", "is_available", "find_soundfont", "list_soundfonts"]


class TsfBackend(MidiBackend):
    """MIDI backend using TinySoundFont for direct audio output.

    This backend synthesizes audio directly from MIDI events using a
    SoundFont file. No external synthesizer is required.

    Example:
        >>> backend = TsfBackend()  # Auto-finds SoundFont
        >>> score = Score("piano: c d e f g")
        >>> backend.play(score.midi)
        >>> while backend.is_playing():
        ...     time.sleep(0.1)

        >>> # Specify SoundFont explicitly
        >>> backend = TsfBackend(soundfont="/path/to/FluidR3_GM.sf2")

    Environment:
        ALDAKIT_SOUNDFONT: Path to default SoundFont file.
    """

    def __init__(
        self,
        soundfont: str | Path | None = None,
        gain: float = DEFAULT_SOUNDFONT_GAIN,
    ):
        """Initialize the TinySoundFont backend.

        Args:
            soundfont: Path to a SoundFont file (.sf2). If None, searches
                common locations for a GM SoundFont.
            gain: Global volume gain (0.0 - 2.0, default 1.0).

        Raises:
            RuntimeError: If TinySoundFont module is not available.
            FileNotFoundError: If no SoundFont is found.
        """
        if not HAS_TSF or _tsf is None:
            raise RuntimeError(
                "TinySoundFont backend not available. "
                "The _tsf native module was not built or failed to load."
            )

        self._player = _tsf.TsfPlayer()  # type: ignore[union-attr]
        self._soundfont_path: Path | None = None

        # Find SoundFont. Paths may come from a config file, an environment
        # variable or the command line, so "~" and $VARS are expanded here as
        # they are for the config file.
        if soundfont is not None:
            sf_path = Path(os.path.expandvars(os.path.expanduser(str(soundfont))))
        else:
            sf_path = find_soundfont()

        if sf_path is None:
            raise FileNotFoundError(
                "No SoundFont file found. Please either:\n"
                "  1. Set ALDAKIT_SOUNDFONT=/path/to/soundfont.sf2\n"
                "  2. Place a .sf2 file in ~/Music/sf2/\n"
                "  3. Pass soundfont='/path/to/file.sf2' to TsfBackend()"
            )

        if not sf_path.exists():
            raise FileNotFoundError(f"SoundFont file not found: {sf_path}")

        if not self._player.load_soundfont(str(sf_path)):
            raise RuntimeError(f"Failed to load SoundFont: {sf_path}")

        self._soundfont_path = sf_path
        self._player.set_gain(gain)

    @property
    def soundfont(self) -> Path | None:
        """The currently loaded SoundFont path."""
        return self._soundfont_path

    @property
    def preset_count(self) -> int:
        """Number of presets in the loaded SoundFont."""
        return self._player.preset_count()

    def preset_name(self, index: int) -> str:
        """Get the name of a preset by index."""
        return self._player.preset_name(index)

    def set_gain(self, gain: float) -> None:
        """Set global volume gain (0.0 - 2.0)."""
        self._player.set_gain(gain)

    def play(self, sequence: MidiSequence) -> int | None:
        """Play a MIDI sequence through the audio output.

        Note: TsfBackend does not currently support concurrent playback.
        Calling play() will stop any currently playing sequence.

        Args:
            sequence: The MIDI sequence to play.

        Returns:
            0 to indicate playback started (single slot).
        """
        self._player.clear_schedule()

        # Schedule program changes
        for pc in sequence.program_changes:
            self._player.schedule_program(pc.channel, pc.program, pc.time)

        # Schedule control changes (panning, expression, ...)
        for cc in sequence.control_changes:
            self._player.schedule_control(cc.channel, cc.control, cc.value, cc.time)

        # Schedule notes
        for note in sequence.notes:
            self._player.schedule_note(
                note.channel,
                note.pitch,
                note.velocity / 127.0,  # Convert to 0.0-1.0
                note.start_time,
                note.duration,
            )

        self._player.play()
        return 0

    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        """Save a MIDI sequence to a file.

        Note: This saves as a standard MIDI file, not audio.

        Args:
            sequence: The MIDI sequence to save.
            path: Output file path.
        """
        write_midi_file(sequence, path)

    def stop(self) -> None:
        """Stop playback."""
        self._player.stop()

    def is_playing(self) -> bool:
        """Check if playback is in progress."""
        return self._player.is_playing()

    def current_time(self) -> float:
        """Get current playback position in seconds."""
        return self._player.current_time()

    def wait(self, poll_interval: float = 0.05) -> None:
        """Block until playback completes.

        Args:
            poll_interval: Seconds between status checks.
        """
        while self.is_playing():
            time.sleep(poll_interval)

    def __enter__(self) -> "TsfBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False

    def __repr__(self) -> str:
        sf = self._soundfont_path.name if self._soundfont_path else "None"
        return f"TsfBackend(soundfont={sf!r}, presets={self.preset_count})"
