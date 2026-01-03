"""MIDI generation and playback for Alda."""

from .backends import LibremidiBackend, MidiBackend
from .generator import MidiGenerator, generate_midi
from .types import (
    INSTRUMENT_PROGRAMS,
    GeneralMidiProgram,
    MidiControlChange,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
    note_to_midi,
)

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
