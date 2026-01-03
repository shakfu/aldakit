"""MIDI generation and playback for Alda."""

from .types import (
    MidiSequence,
    MidiNote,
    MidiProgramChange,
    MidiControlChange,
    MidiTempoChange,
    GeneralMidiProgram,
    INSTRUMENT_PROGRAMS,
    note_to_midi,
)
from .generator import MidiGenerator, generate_midi
from .backends import MidiBackend, LibremidiBackend

__all__ = [
    # Types
    "MidiSequence",
    "MidiNote",
    "MidiProgramChange",
    "MidiControlChange",
    "MidiTempoChange",
    "GeneralMidiProgram",
    "INSTRUMENT_PROGRAMS",
    "note_to_midi",
    # Generator
    "MidiGenerator",
    "generate_midi",
    # Backends
    "MidiBackend",
    "LibremidiBackend",
]
