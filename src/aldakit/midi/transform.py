"""MIDI-level transformers for timing and velocity manipulation.

These transformers operate on MidiSequence objects with absolute timing (seconds)
and velocity values. Unlike AST-level transformers, these cannot be converted
back to Alda notation because the symbolic information is lost.

For AST-level transformers (pitch, structure), see aldakit.compose.transform.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from .types import MidiNote, MidiSequence


# =============================================================================
# Timing Transformers
# =============================================================================


def quantize(
    sequence: MidiSequence, grid: float, strength: float = 1.0
) -> MidiSequence:
    """Quantize note start times to a grid.

    Args:
        sequence: The MIDI sequence to quantize.
        grid: Grid size in seconds (e.g., 0.25 for 16th notes at 60 BPM).
        strength: How strongly to quantize (0.0 = no change, 1.0 = full snap).

    Returns:
        A new MidiSequence with quantized timing.

    Examples:
        >>> # Quantize to 8th notes at 120 BPM (0.25 seconds)
        >>> quantized = quantize(midi_seq, 0.25)
        >>> # Partial quantize (50% toward grid)
        >>> soft_quantize = quantize(midi_seq, 0.25, strength=0.5)
    """
    if grid <= 0:
        raise ValueError("Grid size must be positive")

    new_notes = []
    for note in sequence.notes:
        # Find nearest grid point
        grid_point = round(note.start_time / grid) * grid
        # Interpolate between original and grid point
        new_start = note.start_time + (grid_point - note.start_time) * strength
        new_notes.append(replace(note, start_time=max(0.0, new_start)))

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def humanize(
    sequence: MidiSequence,
    timing: float = 0.0,
    velocity: float = 0.0,
    duration: float = 0.0,
    seed: int | None = None,
) -> MidiSequence:
    """Add random variations to timing, velocity, and duration.

    This makes mechanical MIDI sound more human by introducing subtle
    variations in timing and dynamics.

    Args:
        sequence: The MIDI sequence to humanize.
        timing: Maximum timing deviation in seconds (e.g., 0.02 for +/- 20ms).
        velocity: Maximum velocity deviation (0-127 scale, e.g., 10 for +/- 10).
        duration: Maximum duration deviation in seconds.
        seed: Optional random seed for reproducibility.

    Returns:
        A new MidiSequence with humanized values.

    Examples:
        >>> humanized = humanize(midi_seq, timing=0.02, velocity=10)
    """
    rng = random.Random(seed)

    new_notes = []
    for note in sequence.notes:
        new_start = note.start_time
        new_vel = note.velocity
        new_dur = note.duration

        if timing > 0:
            new_start += rng.uniform(-timing, timing)
            new_start = max(0.0, new_start)

        if velocity > 0:
            new_vel += int(rng.uniform(-velocity, velocity))
            new_vel = max(1, min(127, new_vel))

        if duration > 0:
            new_dur += rng.uniform(-duration, duration)
            new_dur = max(0.01, new_dur)  # Minimum 10ms duration

        new_notes.append(
            replace(note, start_time=new_start, velocity=new_vel, duration=new_dur)
        )

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def swing(
    sequence: MidiSequence, amount: float = 0.33, grid: float = 0.5
) -> MidiSequence:
    """Apply swing feel by delaying offbeat notes.

    Swing delays notes that fall on the "and" of the beat (offbeats),
    creating a lilting, jazz-like feel.

    Args:
        sequence: The MIDI sequence to swing.
        amount: How much to delay offbeats (0.0 = straight, 0.5 = triplet feel).
                0.33 is a typical light swing.
        grid: The beat division to swing on in seconds (default 0.5 = half notes
              at 60 BPM, which swings 8th notes).

    Returns:
        A new MidiSequence with swing applied.

    Examples:
        >>> # Light swing on 8th notes at 120 BPM
        >>> swung = swing(midi_seq, amount=0.33, grid=0.25)
        >>> # Heavy shuffle
        >>> shuffled = swing(midi_seq, amount=0.5, grid=0.25)
    """
    if grid <= 0:
        raise ValueError("Grid size must be positive")

    new_notes = []
    for note in sequence.notes:
        # Determine position within the grid cycle
        grid_position = (note.start_time % grid) / grid

        # If we're on an offbeat (roughly 0.5 through the grid cycle)
        # delay it by the swing amount
        new_start = note.start_time
        if 0.4 < grid_position < 0.6:
            # This note is on the offbeat, delay it
            delay = grid * amount * 0.5
            new_start += delay

        new_notes.append(replace(note, start_time=new_start))

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def stretch(sequence: MidiSequence, factor: float) -> MidiSequence:
    """Time-stretch the entire sequence.

    Args:
        sequence: The MIDI sequence to stretch.
        factor: Stretch factor (2.0 = twice as long/half speed,
                0.5 = half as long/double speed).

    Returns:
        A new MidiSequence with stretched timing.

    Examples:
        >>> # Slow down to half speed
        >>> slower = stretch(midi_seq, 2.0)
        >>> # Speed up to double speed
        >>> faster = stretch(midi_seq, 0.5)
    """
    if factor <= 0:
        raise ValueError("Stretch factor must be positive")

    new_notes = [
        replace(
            note, start_time=note.start_time * factor, duration=note.duration * factor
        )
        for note in sequence.notes
    ]

    new_program_changes = [
        replace(pc, time=pc.time * factor) for pc in sequence.program_changes
    ]

    new_control_changes = [
        replace(cc, time=cc.time * factor) for cc in sequence.control_changes
    ]

    new_tempo_changes = [
        replace(tc, time=tc.time * factor) for tc in sequence.tempo_changes
    ]

    return MidiSequence(
        notes=new_notes,
        program_changes=new_program_changes,
        control_changes=new_control_changes,
        tempo_changes=new_tempo_changes,
        ticks_per_beat=sequence.ticks_per_beat,
    )


def shift(sequence: MidiSequence, offset: float) -> MidiSequence:
    """Shift all events in time by an offset.

    Args:
        sequence: The MIDI sequence to shift.
        offset: Time offset in seconds (positive = later, negative = earlier).

    Returns:
        A new MidiSequence with shifted timing.

    Examples:
        >>> # Delay everything by 1 second
        >>> delayed = shift(midi_seq, 1.0)
        >>> # Start earlier (removes notes before time 0)
        >>> earlier = shift(midi_seq, -0.5)
    """
    new_notes = []
    for note in sequence.notes:
        new_start = note.start_time + offset
        if new_start >= 0:
            new_notes.append(replace(note, start_time=new_start))
        elif new_start + note.duration > 0:
            # Note starts before 0 but extends past it - truncate
            new_dur = note.duration + new_start  # new_start is negative
            new_notes.append(replace(note, start_time=0.0, duration=new_dur))

    new_program_changes = [
        replace(pc, time=max(0.0, pc.time + offset))
        for pc in sequence.program_changes
        if pc.time + offset >= 0
    ]

    new_control_changes = [
        replace(cc, time=max(0.0, cc.time + offset))
        for cc in sequence.control_changes
        if cc.time + offset >= 0
    ]

    new_tempo_changes = [
        replace(tc, time=max(0.0, tc.time + offset))
        for tc in sequence.tempo_changes
        if tc.time + offset >= 0
    ]

    return MidiSequence(
        notes=new_notes,
        program_changes=new_program_changes,
        control_changes=new_control_changes,
        tempo_changes=new_tempo_changes,
        ticks_per_beat=sequence.ticks_per_beat,
    )


# =============================================================================
# Velocity Transformers
# =============================================================================


def accent(
    sequence: MidiSequence,
    pattern: list[float],
    base_velocity: int | None = None,
) -> MidiSequence:
    """Apply an accent pattern to note velocities.

    The pattern repeats across the sequence, with each value being a
    multiplier for the base velocity.

    Args:
        sequence: The MIDI sequence to accent.
        pattern: List of velocity multipliers (e.g., [1.2, 0.8, 1.0, 0.8]
                 for accenting beat 1).
        base_velocity: Optional base velocity to use. If None, uses each
                       note's original velocity as the base.

    Returns:
        A new MidiSequence with accented velocities.

    Examples:
        >>> # Accent every 4th note (beat 1 of 4/4)
        >>> accented = accent(midi_seq, [1.3, 0.9, 1.0, 0.9])
        >>> # Strong/weak alternation
        >>> alternating = accent(midi_seq, [1.2, 0.8])
    """
    if not pattern:
        return sequence

    new_notes = []
    for i, note in enumerate(sequence.notes):
        multiplier = pattern[i % len(pattern)]
        base = base_velocity if base_velocity is not None else note.velocity
        new_vel = int(base * multiplier)
        new_vel = max(1, min(127, new_vel))
        new_notes.append(replace(note, velocity=new_vel))

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def crescendo(
    sequence: MidiSequence,
    start_velocity: int,
    end_velocity: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> MidiSequence:
    """Apply a crescendo (gradual velocity increase) over a time range.

    Args:
        sequence: The MIDI sequence.
        start_velocity: Starting velocity.
        end_velocity: Ending velocity.
        start_time: Start time for the crescendo (default: sequence start).
        end_time: End time for the crescendo (default: sequence end).

    Returns:
        A new MidiSequence with crescendo applied.

    Examples:
        >>> # Crescendo from pp to ff over the whole sequence
        >>> cresc = crescendo(midi_seq, 40, 110)
        >>> # Crescendo in a specific section
        >>> cresc = crescendo(midi_seq, 60, 100, start_time=2.0, end_time=6.0)
    """
    if not sequence.notes:
        return sequence

    seq_start = min(n.start_time for n in sequence.notes)
    seq_end = max(n.start_time for n in sequence.notes)

    if start_time is None:
        start_time = seq_start
    if end_time is None:
        end_time = seq_end

    time_range = end_time - start_time
    if time_range <= 0:
        return sequence

    new_notes = []
    for note in sequence.notes:
        if note.start_time < start_time:
            new_notes.append(replace(note, velocity=max(1, min(127, start_velocity))))
        elif note.start_time > end_time:
            new_notes.append(replace(note, velocity=max(1, min(127, end_velocity))))
        else:
            # Linear interpolation
            progress = (note.start_time - start_time) / time_range
            new_vel = int(start_velocity + (end_velocity - start_velocity) * progress)
            new_vel = max(1, min(127, new_vel))
            new_notes.append(replace(note, velocity=new_vel))

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def diminuendo(
    sequence: MidiSequence,
    start_velocity: int,
    end_velocity: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> MidiSequence:
    """Apply a diminuendo (gradual velocity decrease) over a time range.

    This is equivalent to crescendo with swapped start/end velocities,
    but provided for semantic clarity.

    Args:
        sequence: The MIDI sequence.
        start_velocity: Starting velocity (usually higher).
        end_velocity: Ending velocity (usually lower).
        start_time: Start time for the diminuendo (default: sequence start).
        end_time: End time for the diminuendo (default: sequence end).

    Returns:
        A new MidiSequence with diminuendo applied.

    Examples:
        >>> # Diminuendo from ff to pp
        >>> dim = diminuendo(midi_seq, 110, 40)
    """
    return crescendo(sequence, start_velocity, end_velocity, start_time, end_time)


def normalize(sequence: MidiSequence, target: int = 100) -> MidiSequence:
    """Normalize all velocities to a target value.

    Scales velocities so the maximum velocity becomes the target.

    Args:
        sequence: The MIDI sequence.
        target: Target maximum velocity (default 100).

    Returns:
        A new MidiSequence with normalized velocities.

    Examples:
        >>> # Normalize to velocity 100
        >>> normalized = normalize(midi_seq, 100)
    """
    if not sequence.notes:
        return sequence

    max_vel = max(note.velocity for note in sequence.notes)
    if max_vel == 0:
        return sequence

    scale = target / max_vel

    new_notes = [
        replace(note, velocity=max(1, min(127, int(note.velocity * scale))))
        for note in sequence.notes
    ]

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def velocity_curve(sequence: MidiSequence, curve: Callable[[int], int]) -> MidiSequence:
    """Apply a custom velocity curve function.

    Args:
        sequence: The MIDI sequence.
        curve: A function that takes a velocity (0-127) and returns a new
               velocity (0-127).

    Returns:
        A new MidiSequence with transformed velocities.

    Examples:
        >>> # Compress dynamics (reduce range)
        >>> compressed = velocity_curve(midi_seq, lambda v: 64 + (v - 64) // 2)
        >>> # Expand dynamics
        >>> expanded = velocity_curve(midi_seq, lambda v: max(1, min(127, v * 1.5)))
    """
    new_notes = [
        replace(note, velocity=max(1, min(127, curve(note.velocity))))
        for note in sequence.notes
    ]

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def compress(
    sequence: MidiSequence, threshold: int = 80, ratio: float = 2.0
) -> MidiSequence:
    """Compress velocities above a threshold.

    Reduces the dynamic range by attenuating loud notes.

    Args:
        sequence: The MIDI sequence.
        threshold: Velocity above which compression is applied.
        ratio: Compression ratio (2.0 = 2:1 compression above threshold).

    Returns:
        A new MidiSequence with compressed velocities.

    Examples:
        >>> # Light compression
        >>> compressed = compress(midi_seq, threshold=80, ratio=2.0)
        >>> # Heavy compression (limiting)
        >>> limited = compress(midi_seq, threshold=100, ratio=10.0)
    """
    if ratio <= 0:
        raise ValueError("Ratio must be positive")

    def compress_velocity(v: int) -> int:
        if v <= threshold:
            return v
        excess = v - threshold
        compressed_excess = excess / ratio
        return int(threshold + compressed_excess)

    return velocity_curve(sequence, compress_velocity)


# =============================================================================
# Filtering
# =============================================================================


def filter_notes(
    sequence: MidiSequence,
    predicate: Callable[[MidiNote], bool],
) -> MidiSequence:
    """Filter notes based on a predicate function.

    Args:
        sequence: The MIDI sequence.
        predicate: A function that takes a MidiNote and returns True to keep it.

    Returns:
        A new MidiSequence with only matching notes.

    Examples:
        >>> # Keep only loud notes
        >>> loud = filter_notes(midi_seq, lambda n: n.velocity > 80)
        >>> # Keep notes in a time range
        >>> section = filter_notes(midi_seq, lambda n: 2.0 <= n.start_time <= 6.0)
        >>> # Keep only middle register
        >>> middle = filter_notes(midi_seq, lambda n: 48 <= n.pitch <= 72)
    """
    new_notes = [note for note in sequence.notes if predicate(note)]

    return MidiSequence(
        notes=new_notes,
        program_changes=list(sequence.program_changes),
        control_changes=list(sequence.control_changes),
        tempo_changes=list(sequence.tempo_changes),
        ticks_per_beat=sequence.ticks_per_beat,
    )


def trim(
    sequence: MidiSequence, start: float = 0.0, end: float | None = None
) -> MidiSequence:
    """Trim the sequence to a time range.

    Notes that start within the range are kept, and the sequence is
    shifted so the new start time is 0.

    Args:
        sequence: The MIDI sequence.
        start: Start time of the range to keep.
        end: End time of the range to keep (default: sequence end).

    Returns:
        A new MidiSequence trimmed to the range.

    Examples:
        >>> # Keep only first 4 seconds
        >>> intro = trim(midi_seq, 0, 4.0)
        >>> # Extract middle section
        >>> middle = trim(midi_seq, 4.0, 8.0)
    """
    if end is None:
        end = sequence.duration()

    # Filter notes within range and shift to start at 0
    new_notes = []
    for note in sequence.notes:
        if start <= note.start_time < end:
            new_notes.append(replace(note, start_time=note.start_time - start))

    new_program_changes = [
        replace(pc, time=pc.time - start)
        for pc in sequence.program_changes
        if start <= pc.time < end
    ]

    new_control_changes = [
        replace(cc, time=cc.time - start)
        for cc in sequence.control_changes
        if start <= cc.time < end
    ]

    new_tempo_changes = [
        replace(tc, time=tc.time - start)
        for tc in sequence.tempo_changes
        if start <= tc.time < end
    ]

    return MidiSequence(
        notes=new_notes,
        program_changes=new_program_changes,
        control_changes=new_control_changes,
        tempo_changes=new_tempo_changes,
        ticks_per_beat=sequence.ticks_per_beat,
    )


# =============================================================================
# Combining
# =============================================================================


def merge(*sequences: MidiSequence) -> MidiSequence:
    """Merge multiple MIDI sequences into one.

    All notes play simultaneously (useful for layering parts).

    Args:
        *sequences: MIDI sequences to merge.

    Returns:
        A new MidiSequence with all events combined.

    Examples:
        >>> # Layer bass and melody
        >>> combined = merge(bass_seq, melody_seq)
    """
    if not sequences:
        return MidiSequence()

    all_notes = []
    all_program_changes = []
    all_control_changes = []
    all_tempo_changes = []

    for seq in sequences:
        all_notes.extend(seq.notes)
        all_program_changes.extend(seq.program_changes)
        all_control_changes.extend(seq.control_changes)
        all_tempo_changes.extend(seq.tempo_changes)

    # Sort by time
    all_notes.sort(key=lambda n: n.start_time)
    all_program_changes.sort(key=lambda pc: pc.time)
    all_control_changes.sort(key=lambda cc: cc.time)
    all_tempo_changes.sort(key=lambda tc: tc.time)

    return MidiSequence(
        notes=all_notes,
        program_changes=all_program_changes,
        control_changes=all_control_changes,
        tempo_changes=all_tempo_changes,
        ticks_per_beat=sequences[0].ticks_per_beat,
    )


def concatenate(*sequences: MidiSequence, gap: float = 0.0) -> MidiSequence:
    """Concatenate MIDI sequences end-to-end.

    Args:
        *sequences: MIDI sequences to concatenate.
        gap: Time gap between sequences in seconds.

    Returns:
        A new MidiSequence with sequences played in order.

    Examples:
        >>> # Play verse then chorus
        >>> song = concatenate(verse, chorus)
        >>> # With a 1-second gap
        >>> song = concatenate(verse, chorus, gap=1.0)
    """
    if not sequences:
        return MidiSequence()

    result_notes = []
    result_program_changes = []
    result_control_changes = []
    result_tempo_changes = []

    current_time = 0.0

    for seq in sequences:
        # Shift all events by current_time
        for note in seq.notes:
            result_notes.append(
                replace(note, start_time=note.start_time + current_time)
            )

        for pc in seq.program_changes:
            result_program_changes.append(replace(pc, time=pc.time + current_time))

        for cc in seq.control_changes:
            result_control_changes.append(replace(cc, time=cc.time + current_time))

        for tc in seq.tempo_changes:
            result_tempo_changes.append(replace(tc, time=tc.time + current_time))

        # Move current time to end of this sequence plus gap
        current_time += seq.duration() + gap

    return MidiSequence(
        notes=result_notes,
        program_changes=result_program_changes,
        control_changes=result_control_changes,
        tempo_changes=result_tempo_changes,
        ticks_per_beat=sequences[0].ticks_per_beat,
    )
