"""Scale and mode helpers for music theory operations.

This module provides utilities for working with musical scales and modes,
making it easy to generate melodies within specific tonal contexts.
"""

from __future__ import annotations

# The interval tables and pitch spellings live in aldakit.theory, which the
# MIDI generator shares. They are re-exported here under the names this module
# has always used.
from ..theory import PITCH_SEMITONES as PITCH_TO_OFFSET
from ..theory import SCALE_INTERVALS as SCALE_INTERVALS
from ..theory import SEMITONE_PITCHES as OFFSET_TO_PITCH
from .core import Seq, note

# =============================================================================
# Scale Functions
# =============================================================================


def scale(
    root: str,
    scale_type: str = "major",
    *,
    octave: int = 4,
    duration: int | None = None,
) -> list[str]:
    """Get the pitch names for a scale.

    Args:
        root: Root note of the scale (e.g., "c", "f", "b").
        scale_type: Type of scale (e.g., "major", "minor", "pentatonic").
        octave: Starting octave (used for calculating pitches).
        duration: Not used, kept for API consistency.

    Returns:
        List of pitch names in the scale.

    Examples:
        >>> scale("c", "major")
        ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        >>> scale("a", "minor")
        ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> scale("c", "pentatonic")
        ['c', 'd', 'e', 'g', 'a']
    """
    if scale_type not in SCALE_INTERVALS:
        available = ", ".join(sorted(SCALE_INTERVALS.keys()))
        raise ValueError(f"Unknown scale type: {scale_type}. Available: {available}")

    root_lower = root.lower()
    if root_lower not in PITCH_TO_OFFSET:
        raise ValueError(f"Invalid root note: {root}")

    root_offset = PITCH_TO_OFFSET[root_lower]
    intervals = SCALE_INTERVALS[scale_type]

    pitches = []
    for interval in intervals:
        pitch_offset = (root_offset + interval) % 12
        pitch_name, accidental = OFFSET_TO_PITCH[pitch_offset]
        if accidental:
            pitches.append(f"{pitch_name}{accidental}")
        else:
            pitches.append(pitch_name)

    return pitches


def scale_notes(
    root: str,
    scale_type: str = "major",
    *,
    octave: int = 4,
    duration: int | None = None,
    ascending: bool = True,
) -> Seq:
    """Generate a Seq of notes for a scale.

    Args:
        root: Root note of the scale.
        scale_type: Type of scale.
        octave: Starting octave.
        duration: Duration for each note.
        ascending: If True, ascending scale; if False, descending.

    Returns:
        A Seq containing the scale notes.

    Examples:
        >>> scale_notes("c", "major", duration=8)
        >>> scale_notes("a", "minor", octave=5, ascending=False)
    """
    pitches = scale(root, scale_type)

    notes = []
    current_octave = octave

    for i, pitch in enumerate(pitches):
        # Handle accidentals in pitch name
        if len(pitch) > 1 and pitch[1] in "+-":
            base_pitch = pitch[0]
            accidental = pitch[1]
        else:
            base_pitch = pitch
            accidental = None

        # Check if we need to go up an octave
        if i > 0:
            prev_pitch = pitches[i - 1][0].lower()
            curr_pitch = base_pitch.lower()
            if PITCH_TO_OFFSET.get(curr_pitch, 0) < PITCH_TO_OFFSET.get(prev_pitch, 0):
                current_octave += 1

        notes.append(
            note(
                base_pitch,
                duration=duration,
                octave=current_octave,
                accidental=accidental,
            )
        )

    if not ascending:
        notes = list(reversed(notes))

    return Seq(elements=notes)


def scale_degree(
    root: str,
    scale_type: str,
    degree: int,
    *,
    octave: int = 4,
) -> tuple[str, str | None, int]:
    """Get the pitch for a specific scale degree.

    Args:
        root: Root note of the scale.
        scale_type: Type of scale.
        degree: Scale degree (1-based, can exceed scale length for higher octaves).
        octave: Base octave.

    Returns:
        Tuple of (pitch_name, accidental, octave).

    Examples:
        >>> scale_degree("c", "major", 1)  # Root
        ('c', None, 4)
        >>> scale_degree("c", "major", 5)  # Fifth
        ('g', None, 4)
        >>> scale_degree("c", "major", 8)  # Octave
        ('c', None, 5)
    """
    if degree < 1:
        raise ValueError("Scale degree must be >= 1")

    pitches = scale(root, scale_type)
    scale_len = len(pitches)

    # Calculate octave offset and index
    octave_offset = (degree - 1) // scale_len
    index = (degree - 1) % scale_len

    pitch = pitches[index]
    if len(pitch) > 1 and pitch[1] in "+-":
        base_pitch = pitch[0]
        accidental: str | None = pitch[1]
    else:
        base_pitch = pitch
        accidental = None

    return base_pitch, accidental, octave + octave_offset


def mode(
    root: str,
    mode_name: str,
    *,
    octave: int = 4,
    duration: int | None = None,
) -> list[str]:
    """Get the pitch names for a mode.

    This is an alias for `scale()` - modes are just different names for scales.

    Args:
        root: Root note of the mode.
        mode_name: Name of the mode (ionian, dorian, phrygian, etc.).
        octave: Starting octave.
        duration: Not used.

    Returns:
        List of pitch names in the mode.

    Examples:
        >>> mode("d", "dorian")
        ['d', 'e', 'f', 'g', 'a', 'b', 'c']
    """
    return scale(root, mode_name, octave=octave, duration=duration)


def relative_minor(major_root: str) -> str:
    """Get the relative minor root for a major key.

    Args:
        major_root: Root of the major key.

    Returns:
        Root note of the relative minor.

    Examples:
        >>> relative_minor("c")  # C major -> A minor
        'a'
        >>> relative_minor("g")  # G major -> E minor
        'e'
    """
    root_offset = PITCH_TO_OFFSET[major_root.lower()]
    minor_offset = (root_offset - 3) % 12
    pitch, accidental = OFFSET_TO_PITCH[minor_offset]
    if accidental:
        return f"{pitch}{accidental}"
    return pitch


def relative_major(minor_root: str) -> str:
    """Get the relative major root for a minor key.

    Args:
        minor_root: Root of the minor key.

    Returns:
        Root note of the relative major.

    Examples:
        >>> relative_major("a")  # A minor -> C major
        'c'
        >>> relative_major("e")  # E minor -> G major
        'g'
    """
    # Handle accidentals
    if len(minor_root) > 1 and minor_root[1] in "+-":
        base = minor_root[0].lower()
        acc = minor_root[1]
        root_offset = PITCH_TO_OFFSET[base]
        if acc == "+":
            root_offset += 1
        elif acc == "-":
            root_offset -= 1
        root_offset = root_offset % 12
    else:
        root_offset = PITCH_TO_OFFSET[minor_root.lower()]

    major_offset = (root_offset + 3) % 12
    pitch, accidental = OFFSET_TO_PITCH[major_offset]
    if accidental:
        return f"{pitch}{accidental}"
    return pitch


def parallel_minor(major_root: str) -> str:
    """Get the parallel minor root (same root, different mode).

    Args:
        major_root: Root of the major key.

    Returns:
        Same root (parallel minor has the same root as parallel major).

    Examples:
        >>> parallel_minor("c")  # C major -> C minor
        'c'
    """
    return major_root.lower()


def parallel_major(minor_root: str) -> str:
    """Get the parallel major root (same root, different mode).

    Args:
        minor_root: Root of the minor key.

    Returns:
        Same root.

    Examples:
        >>> parallel_major("c")  # C minor -> C major
        'c'
    """
    return minor_root.lower()


def transpose_scale(
    pitches: list[str],
    semitones: int,
) -> list[str]:
    """Transpose a list of pitches by a number of semitones.

    Args:
        pitches: List of pitch names.
        semitones: Number of semitones to transpose.

    Returns:
        Transposed list of pitch names.

    Examples:
        >>> transpose_scale(["c", "d", "e"], 5)  # Up a fourth
        ['f', 'g', 'a']
    """
    result = []
    for pitch in pitches:
        # Parse pitch
        if len(pitch) > 1 and pitch[1] in "+-":
            base = pitch[0].lower()
            acc = pitch[1]
            offset = PITCH_TO_OFFSET[base]
            if acc == "+":
                offset += 1
            elif acc == "-":
                offset -= 1
        else:
            offset = PITCH_TO_OFFSET[pitch.lower()]

        # Transpose
        new_offset = (offset + semitones) % 12
        new_pitch, new_acc = OFFSET_TO_PITCH[new_offset]
        if new_acc:
            result.append(f"{new_pitch}{new_acc}")
        else:
            result.append(new_pitch)

    return result


def interval_name(semitones: int) -> str:
    """Get the name of an interval given its size in semitones.

    Args:
        semitones: Number of semitones (0-12).

    Returns:
        Interval name.

    Examples:
        >>> interval_name(0)
        'unison'
        >>> interval_name(7)
        'perfect fifth'
    """
    names = {
        0: "unison",
        1: "minor second",
        2: "major second",
        3: "minor third",
        4: "major third",
        5: "perfect fourth",
        6: "tritone",
        7: "perfect fifth",
        8: "minor sixth",
        9: "major sixth",
        10: "minor seventh",
        11: "major seventh",
        12: "octave",
    }
    semitones = semitones % 13
    return names.get(semitones, f"{semitones} semitones")


def list_scales() -> list[str]:
    """Get a list of all available scale types.

    Returns:
        Sorted list of scale type names.

    Examples:
        >>> "pentatonic" in list_scales()
        True
    """
    return sorted(SCALE_INTERVALS.keys())
