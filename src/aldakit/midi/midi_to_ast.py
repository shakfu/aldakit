"""Convert MIDI sequences to AST nodes for import."""

from __future__ import annotations

from dataclasses import dataclass

from ..ast_nodes import (
    ChordNode,
    DurationNode,
    EventSequenceNode,
    LispListNode,
    LispNumberNode,
    LispSymbolNode,
    NoteLengthNode,
    NoteNode,
    OctaveSetNode,
    PartDeclarationNode,
    PartNode,
    RestNode,
    RootNode,
)
from ..constants import MIDI_DRUM_CHANNEL
from .types import MidiNote, MidiSequence, canonical_name

# MIDI pitch to note letter and accidental
# We use sharps for black keys
PITCH_CLASS_TO_NOTE: list[tuple[str, list[str]]] = [
    ("c", []),  # 0
    ("c", ["+"]),  # 1 (C#)
    ("d", []),  # 2
    ("d", ["+"]),  # 3 (D#)
    ("e", []),  # 4
    ("f", []),  # 5
    ("f", ["+"]),  # 6 (F#)
    ("g", []),  # 7
    ("g", ["+"]),  # 8 (G#)
    ("a", []),  # 9
    ("a", ["+"]),  # 10 (A#)
    ("b", []),  # 11
]


# Standard note duration values and their lengths in quarter notes
# (duration_value, length_in_quarters)
DURATION_VALUES: list[tuple[int, float]] = [
    (1, 4.0),  # whole note
    (2, 2.0),  # half note
    (4, 1.0),  # quarter note
    (6, 0.6666666667),  # two-thirds beat (used for swing)
    (8, 0.5),  # eighth note
    (12, 0.3333333333),  # triplet eighth (12th note)
    (16, 0.25),  # sixteenth note
    (20, 0.2),  # quintuplet eighth (20th note)
    (24, 0.1666666667),  # triplet sixteenth (24th note)
    (32, 0.125),  # thirty-second note
    (40, 0.1),  # quintuplet sixteenth
    (48, 0.0833333333),  # triplet thirty-second
    (64, 0.0625),  # sixty-fourth note
    (80, 0.05),  # quintuplet thirty-second
]

# Dotted versions add 50% more
DOTTED_DURATION_VALUES: list[tuple[int, int, float]] = [
    (1, 1, 6.0),  # dotted whole
    (2, 1, 3.0),  # dotted half
    (4, 1, 1.5),  # dotted quarter
    (8, 1, 0.75),  # dotted eighth
    (12, 1, 0.5),  # dotted triplet eighth
    (16, 1, 0.375),  # dotted sixteenth
    (24, 1, 0.25),  # dotted triplet sixteenth
]


# Reverse mapping from GM program to its canonical Alda instrument name
PROGRAM_TO_INSTRUMENT: dict[int, str] = {
    program: canonical_name(program) for program in range(128)
}


@dataclass
class _QuantizedNote:
    """A note with quantized timing and duration."""

    pitch: int
    velocity: int
    start_beat: float
    duration_beats: float
    channel: int
    start_seconds: float
    duration_seconds: float


def midi_pitch_to_note(pitch: int) -> tuple[str, int, list[str]]:
    """Convert a MIDI pitch number to note name, octave, and accidentals.

    Args:
        pitch: MIDI pitch number (0-127, where 60 = middle C = C4).

    Returns:
        Tuple of (letter, octave, accidentals).
    """
    # MIDI note 60 = C4
    # octave = pitch // 12 - 1
    # C0 = MIDI 12
    octave = (pitch // 12) - 1
    pitch_class = pitch % 12

    letter, accidentals = PITCH_CLASS_TO_NOTE[pitch_class]
    return letter, octave, accidentals


def seconds_to_beats(seconds: float, bpm: float) -> float:
    """Convert seconds to beats at a given tempo."""
    return seconds * bpm / 60.0


def _make_duration_node(denominator: int, dots: int = 0) -> DurationNode:
    """Create a DurationNode with the given note length and dots."""
    return DurationNode(
        components=[NoteLengthNode(denominator=denominator, dots=dots, position=None)],
        position=None,
    )


def _make_tempo_node(bpm: float, global_: bool = False) -> LispListNode:
    """Create a tempo or tempo! lisp node."""
    symbol = "tempo!" if global_ else "tempo"
    return LispListNode(
        elements=[
            LispSymbolNode(name=symbol, position=None),
            LispNumberNode(value=int(round(bpm)), position=None),
        ],
        position=None,
    )


def beats_to_duration(beats: float) -> tuple[int, int]:
    """Convert beats to the closest duration value and dots.

    Args:
        beats: Duration in beats (quarter notes).

    Returns:
        Tuple of (duration_value, dots). Duration value is Alda's notation
        where 4 = quarter, 8 = eighth, etc.
    """
    beats = float(beats)

    if beats <= 0:
        return 4, 0  # Default to quarter note

    # Try exact matches first
    for duration_value, length in DURATION_VALUES:
        if abs(beats - length) < 0.01:
            return duration_value, 0

    # Try dotted values
    for duration_value, dots, length in DOTTED_DURATION_VALUES:
        if abs(beats - length) < 0.01:
            return duration_value, dots

    # Find closest match
    best_duration = 4
    best_dots = 0
    best_diff = float("inf")

    for duration_value, length in DURATION_VALUES:
        diff = abs(beats - length)
        if diff < best_diff:
            best_diff = diff
            best_duration = duration_value
            best_dots = 0

    for duration_value, dots, length in DOTTED_DURATION_VALUES:
        diff = abs(beats - length)
        if diff < best_diff:
            best_diff = diff
            best_duration = duration_value
            best_dots = dots

    return best_duration, best_dots


def duration_value_to_beats(denominator: int, dots: int = 0) -> float:
    """Convert a duration value back to beats."""
    if denominator <= 0:
        return 0.0
    base = 4.0 / denominator
    total = base
    addition = base / 2.0
    for _ in range(dots):
        total += addition
        addition /= 2.0
    return total


def quantize_to_grid(value: float, grid: float) -> float:
    """Quantize a value to the nearest grid point."""
    if grid <= 0:
        return value
    return round(value / grid) * grid


def midi_to_ast(
    sequence: MidiSequence,
    *,
    quantize_grid: float = 0.25,  # Quantize to sixteenth notes by default
    default_bpm: float = 120.0,
) -> RootNode:
    """Convert a MidiSequence to an AST RootNode.

    Args:
        sequence: The MIDI sequence to convert.
        quantize_grid: Grid size in beats for quantization (0.25 = 16th notes).
        default_bpm: Default tempo if none specified in the sequence.

    Returns:
        A RootNode representing the imported MIDI.
    """
    # Determine tempo
    tempo_changes = sorted(sequence.tempo_changes, key=lambda t: t.time)
    bpm = tempo_changes[0].bpm if tempo_changes else default_bpm
    per_part_tempos = tempo_changes[1:] if tempo_changes else []
    tempo_events = [(tc.time, tc.bpm) for tc in per_part_tempos]

    # Group notes by channel
    channels: dict[int, list[MidiNote]] = {}
    for note in sequence.notes:
        if note.channel not in channels:
            channels[note.channel] = []
        channels[note.channel].append(note)

    # Get program changes to determine instruments
    channel_programs: dict[int, int] = {}
    for pc in sequence.program_changes:
        if pc.channel not in channel_programs:
            channel_programs[pc.channel] = pc.program

    # Build AST for each channel as a part
    children: list = []

    # Add tempo if not default
    if abs(bpm - 120.0) > 0.1:
        children.append(_make_tempo_node(bpm, global_=True))

    for channel in sorted(channels.keys()):
        notes = channels[channel]
        if not notes:
            continue

        # Determine instrument. Channel 9 is the General MIDI drum channel:
        # its note numbers name drums, not pitches.
        if channel == MIDI_DRUM_CHANNEL:
            instrument_name = "midi-percussion"
        else:
            program = channel_programs.get(channel, 0)
            instrument_name = canonical_name(program)

        # Create part declaration
        declaration = PartDeclarationNode(
            names=[instrument_name],
            alias=None,
            position=None,
        )

        # Convert notes to quantized form
        quantized = _quantize_notes(notes, bpm, quantize_grid)

        # Convert to AST events
        events = _notes_to_events(quantized, bpm, tempo_events)

        # Wrap declaration and events in a PartNode. A bare PartDeclarationNode
        # is not an executable event, so emitting the two as siblings would
        # leave the generator with no part to attach the notes to.
        children.append(
            PartNode(
                declaration=declaration,
                events=EventSequenceNode(events=events, position=None),
                position=None,
            )
        )

    return RootNode(children=children, position=None)


def _quantize_notes(
    notes: list[MidiNote], bpm: float, grid: float
) -> list[_QuantizedNote]:
    """Quantize notes to a grid."""
    result = []

    for note in notes:
        start_beats = seconds_to_beats(note.start_time, bpm)
        duration_beats = seconds_to_beats(note.duration, bpm)

        # Quantize start time
        start_beats = quantize_to_grid(start_beats, grid)

        # Quantize duration to nearest standard value
        duration_beats = max(grid, quantize_to_grid(duration_beats, grid))

        result.append(
            _QuantizedNote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_beat=start_beats,
                duration_beats=duration_beats,
                channel=note.channel,
                start_seconds=note.start_time,
                duration_seconds=note.duration,
            )
        )

    # Sort by start time, then by pitch
    result.sort(key=lambda n: (n.start_beat, n.pitch))
    return result


def _notes_to_events(
    notes: list[_QuantizedNote], bpm: float, tempo_events: list[tuple[float, float]]
) -> list:
    """Convert quantized notes to AST events."""
    if not notes:
        return []

    events: list = []
    current_beat = 0.0
    current_octave = 4  # Default octave
    tempo_index = 0

    i = 0
    while i < len(notes):
        note = notes[i]
        tempo_index = _emit_due_tempos(
            tempo_events, tempo_index, note.start_seconds, events
        )

        # Insert rest if there's a gap
        gap = note.start_beat - current_beat
        if gap > 0.01:
            rest_duration, rest_dots = beats_to_duration(gap)
            rest_node = RestNode(
                duration=_make_duration_node(rest_duration, rest_dots),
                position=None,
            )
            events.append(rest_node)
            current_beat = note.start_beat

        # Check for simultaneous notes (chord)
        chord_notes = [note]
        j = i + 1
        while j < len(notes) and abs(notes[j].start_beat - note.start_beat) < 0.01:
            chord_notes.append(notes[j])
            j += 1

        if len(chord_notes) > 1:
            # Create chord
            chord_elements = []
            chord_duration = chord_notes[0].duration_beats
            duration_val, dots = beats_to_duration(chord_duration)

            # Set octave for first note of chord if needed
            first_letter, first_octave, first_acc = midi_pitch_to_note(
                chord_notes[0].pitch
            )
            if first_octave != current_octave:
                events.append(OctaveSetNode(octave=first_octave, position=None))
                current_octave = first_octave

            for idx, cn in enumerate(chord_notes):
                letter, octave, accidentals = midi_pitch_to_note(cn.pitch)
                # Duration goes on first note only (Alda convention)
                note_duration = (
                    _make_duration_node(duration_val, dots) if idx == 0 else None
                )
                chord_elements.append(
                    NoteNode(
                        letter=letter,
                        accidentals=accidentals,
                        duration=note_duration,
                        slurred=False,
                        position=None,
                    )
                )

            chord_node = ChordNode(
                notes=chord_elements,
                position=None,
            )
            events.append(chord_node)
            current_beat = note.start_beat + chord_duration
            i = j

        else:
            # Single note
            letter, octave, accidentals = midi_pitch_to_note(note.pitch)
            duration_val, dots = beats_to_duration(note.duration_beats)

            # Set octave if changed
            if octave != current_octave:
                events.append(OctaveSetNode(octave=octave, position=None))
                current_octave = octave

            note_node = NoteNode(
                letter=letter,
                accidentals=accidentals,
                duration=_make_duration_node(duration_val, dots),
                slurred=False,
                position=None,
            )
            events.append(note_node)
            current_beat = note.start_beat + note.duration_beats
            i += 1

    _emit_due_tempos(tempo_events, tempo_index, float("inf"), events)
    return events


def _emit_due_tempos(
    tempo_events: list[tuple[float, float]],
    index: int,
    target_time: float,
    events: list,
) -> int:
    while index < len(tempo_events) and tempo_events[index][0] <= target_time + 1e-4:
        events.append(_make_tempo_node(tempo_events[index][1], global_=False))
        index += 1
    return index
