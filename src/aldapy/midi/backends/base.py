"""Abstract base class for MIDI backends."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import MidiSequence


class MidiBackend(ABC):
    """Abstract base class for MIDI output backends."""

    @abstractmethod
    def play(self, sequence: MidiSequence) -> None:
        """Play a MIDI sequence in realtime.

        Args:
            sequence: The MIDI sequence to play.
        """
        pass

    @abstractmethod
    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        """Save a MIDI sequence to a file.

        Args:
            sequence: The MIDI sequence to save.
            path: The output file path.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop any currently playing sequence."""
        pass

    def is_playing(self) -> bool:
        """Check if a sequence is currently playing.

        Returns:
            True if playing, False otherwise.
        """
        return False
