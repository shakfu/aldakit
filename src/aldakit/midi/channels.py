"""Assignment of MIDI channels to parts, including reuse over time.

The MIDI spec gives a synthesizer 16 channels, one of which (channel 10, index
9) is reserved for percussion. A score with more than 15 melodic parts
therefore cannot give every part a channel of its own for the length of the
score -- but it does not have to. A channel is only in use while the part on it
is sounding, so a part that has finished can hand its channel to a part that is
about to start. Alda does this, which is why ``all-instruments.alda`` plays 128
instruments through 15 channels.

The generator does not do this while it walks the AST, because whether reuse is
needed is not known until the last part has been declared. Instead it hands
every melodic part a *virtual* channel -- an integer at or above
:data:`VIRTUAL_CHANNEL_BASE`, allocated without limit -- and this module
rewrites those virtual channels to real ones once the whole score is known.

Two policies keep the output predictable:

- Reuse is a fallback, not the default. A score with 15 or fewer melodic parts
  gets one channel per part in declaration order, exactly as if this pass did
  not exist.
- Within reuse, a part prefers the channel it used last, so a part with rests
  in it keeps one channel rather than hopping between whichever are free.

When a channel does change hands, the part taking it over cannot inherit the
previous occupant's instrument or its pan and volume, so this pass re-emits
that state at the point of the handover.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Callable

from ..constants import (
    MIDI_CC_PAN,
    MIDI_CC_VOLUME,
    MIDI_DRUM_CHANNEL,
    MIDI_MAX_CHANNELS,
)
from .types import MidiControlChange, MidiProgramChange, MidiSequence

# Channels available to pitched instruments. Channel 9 (MIDI channel 10) is
# reserved by the General MIDI spec for percussion, where the note number
# selects a drum sound rather than a pitch.
MELODIC_CHANNELS: tuple[int, ...] = tuple(
    c for c in range(MIDI_MAX_CHANNELS) if c != MIDI_DRUM_CHANNEL
)

#: Channel numbers at or above this are placeholders standing in for a part
#: until this module works out which real channel it should sound on.
VIRTUAL_CHANNEL_BASE = MIDI_MAX_CHANNELS

#: Values a controller returns to when a channel changes hands and the part
#: taking it over does not set that controller itself. Without this the new
#: instrument would inherit the previous one's pan or level. Both are the
#: General MIDI power-on defaults.
CONTROL_DEFAULTS: dict[int, int] = {
    MIDI_CC_PAN: 64,  # centre
    MIDI_CC_VOLUME: 100,
}

# Times are floats in seconds, so comparisons need a tolerance. A nanosecond is
# far below any musically meaningful difference and far above float noise.
EPSILON = 1e-9


def is_virtual(channel: int) -> bool:
    """True if this channel is a placeholder rather than a real MIDI channel."""
    return channel >= VIRTUAL_CHANNEL_BASE


@dataclass
class Run:
    """A stretch of time during which one part is sounding without a break."""

    owner: int  # The channel the generator emitted, virtual or pinned
    start: float
    end: float
    channel: int = -1  # The real channel, filled in by the assignment sweep


@dataclass
class ChannelAssignment:
    """What :func:`assign_channels` decided, for tools that report on it."""

    #: Channel the generator emitted -> the real channels it was rewritten to,
    #: in the order they were used. A part that never sounds maps to no
    #: channels at all.
    channels: dict[int, list[int]] = field(default_factory=dict)
    #: (real channel, owner, owner) triples for parts whose notes overlap in
    #: time on one channel. Only possible when a score asks for more channels
    #: than exist, or pins two parts to the same channel with (midi-channel).
    conflicts: list[tuple[int, int, int]] = field(default_factory=list)
    #: The largest number of melodic parts sounding at the same moment. This,
    #: not the number of parts declared, is what has to fit in 15 channels.
    max_concurrent: int = 0
    #: Notes each part sounded, keyed by the channel the generator emitted.
    #: Rewriting loses this, because two parts can share one real channel.
    note_counts: dict[int, int] = field(default_factory=dict)
    #: True if any channel was handed from one part to another.
    reused: bool = False
    #: True if the score did not fit and parts had to share a channel.
    overflowed: bool = False


def assign_channels(
    sequence: MidiSequence,
    allocated: list[int],
    warn: Callable[..., None] | None = None,
) -> ChannelAssignment:
    """Rewrite the virtual channels in ``sequence`` to real MIDI channels.

    Args:
        sequence: The generated sequence, modified in place.
        allocated: The virtual channels the generator handed out, in the order
            it handed them out. Parts that never sounded are included, so that
            the channel a part gets does not depend on whether an earlier part
            happened to be silent.
        warn: Called with ``(message, code=...)`` when the score does not fit.

    Returns:
        A description of the assignment, for the linter and for tests.
    """
    runs = _build_runs(sequence)
    assignment = ChannelAssignment(
        max_concurrent=_peak_concurrency(runs),
        note_counts=_note_counts(sequence),
    )

    if not allocated:
        # Nothing virtual to rewrite, but pinned parts can still collide.
        _place_runs(runs, {})
        assignment.conflicts = _find_conflicts(runs)
        return assignment

    if len(allocated) <= len(MELODIC_CHANNELS):
        mapping = {v: MELODIC_CHANNELS[i] for i, v in enumerate(allocated)}
        assignment.channels = {v: [c] for v, c in mapping.items()}
        _place_runs(runs, mapping)
        assignment.conflicts = _find_conflicts(runs)
        _rewrite_static(sequence, mapping)
        return assignment

    assignment.reused = True
    _assign_over_time(runs, allocated, sequence, assignment)
    assignment.conflicts = _find_conflicts(runs)
    _rewrite_reused(sequence, runs, allocated)

    if assignment.overflowed and warn is not None:
        warn(
            f"{assignment.max_concurrent} melodic parts sound at the same time "
            f"but only {len(MELODIC_CHANNELS)} MIDI channels are available; "
            "parts are sharing channels and their instruments will collide.",
            code="channel-exhaustion",
        )
    return assignment


# ----------------------------------------------------------------------------
# Runs: when each part is actually sounding


def _build_runs(sequence: MidiSequence) -> list[Run]:
    """Group each part's notes into the stretches where it is sounding.

    Notes that touch or overlap belong to the same run; a rest long enough to
    leave a gap ends one. Percussion is left out because every percussion part
    legitimately shares channel 10.
    """
    intervals: dict[int, list[tuple[float, float]]] = {}
    for note in sequence.notes:
        if note.channel == MIDI_DRUM_CHANNEL:
            continue
        intervals.setdefault(note.channel, []).append(
            (note.start_time, note.start_time + note.duration)
        )

    runs: list[Run] = []
    for owner, spans in intervals.items():
        spans.sort()
        start, end = spans[0]
        for next_start, next_end in spans[1:]:
            if next_start <= end + EPSILON:
                end = max(end, next_end)
            else:
                runs.append(Run(owner=owner, start=start, end=end))
                start, end = next_start, next_end
        runs.append(Run(owner=owner, start=start, end=end))

    # Chronological, with a deterministic tie-break so the same score always
    # produces the same channel layout.
    runs.sort(key=lambda r: (r.start, r.owner, r.end))
    return runs


def _note_counts(sequence: MidiSequence) -> dict[int, int]:
    """How many notes each part sounded, keyed by the channel it was on."""
    counts: dict[int, int] = {}
    for note in sequence.notes:
        counts[note.channel] = counts.get(note.channel, 0) + 1
    return counts


def _peak_concurrency(runs: list[Run]) -> int:
    """The largest number of runs overlapping at any one moment."""
    edges: list[tuple[float, int]] = []
    for run in runs:
        edges.append((run.start, 1))
        edges.append((run.end, -1))
    # Releases sort before claims at the same instant: a part can hand its
    # channel to another part that starts exactly where it stopped.
    edges.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in edges:
        current += delta
        peak = max(peak, current)
    return peak


def _place_runs(runs: list[Run], mapping: dict[int, int]) -> None:
    """Fill in each run's real channel from a fixed owner -> channel mapping."""
    for run in runs:
        run.channel = mapping.get(run.owner, run.owner)


def _find_conflicts(runs: list[Run]) -> list[tuple[int, int, int]]:
    """Owners whose runs overlap in time on the same real channel."""
    by_channel: dict[int, list[Run]] = {}
    for run in runs:
        by_channel.setdefault(run.channel, []).append(run)

    seen: set[tuple[int, int, int]] = set()
    conflicts: list[tuple[int, int, int]] = []
    for channel, channel_runs in sorted(by_channel.items()):
        ordered = sorted(channel_runs, key=lambda r: (r.start, r.owner))
        for index, run in enumerate(ordered):
            for other in ordered[index + 1 :]:
                if other.start >= run.end - EPSILON:
                    break
                # Two runs of one part never overlap: they were merged when
                # they were built, so any overlap here is between two parts.
                first, second = sorted((run.owner, other.owner))
                pair = (channel, first, second)
                if pair not in seen:
                    seen.add(pair)
                    conflicts.append(pair)
    return conflicts


# ----------------------------------------------------------------------------
# Assignment


def _assign_over_time(
    runs: list[Run],
    allocated: list[int],
    sequence: MidiSequence,
    assignment: ChannelAssignment,
) -> None:
    """Hand out real channels run by run, reusing them once they are free."""
    virtual = set(allocated)
    # A channel a part was pinned to with (midi-channel) belongs to that part
    # for the whole score; handing it out would undo the pin.
    pinned = {
        channel
        for channel in _event_channels(sequence)
        if not is_virtual(channel) and channel != MIDI_DRUM_CHANNEL
    }
    pool = [c for c in MELODIC_CHANNELS if c not in pinned]
    if not pool:
        # Every channel is pinned. Reuse the whole melodic set anyway: the
        # alternative is leaving virtual channels in the output.
        pool = list(MELODIC_CHANNELS)

    free_from = dict.fromkeys(pool, float("-inf"))
    owner_of: dict[int, int] = {}
    last_used: dict[int, int] = {}
    used: dict[int, list[int]] = {v: [] for v in allocated}

    for run in runs:
        if run.owner not in virtual:
            run.channel = run.owner  # pinned; leave it where the score put it
            continue

        available = [c for c in pool if free_from[c] <= run.start + EPSILON]
        preferred = last_used.get(run.owner)
        if preferred is not None and preferred in available:
            choice = preferred
        elif available:
            # Prefer a channel that has been free the longest, so a part that
            # stops briefly is unlikely to find its channel taken when it
            # comes back. Never-used channels sort first, which hands out
            # 0, 1, 2 ... before anything is reused.
            choice = min(available, key=lambda c: (free_from[c], c))
        else:
            choice = min(pool, key=lambda c: (free_from[c], c))
            assignment.overflowed = True

        run.channel = choice
        free_from[choice] = max(free_from[choice], run.end)
        last_used[run.owner] = choice
        owner_of[choice] = run.owner
        if not used[run.owner] or used[run.owner][-1] != choice:
            used[run.owner].append(choice)

    assignment.channels = used


def _event_channels(sequence: MidiSequence) -> set[int]:
    """Every channel any event in the sequence is on."""
    channels = {n.channel for n in sequence.notes}
    channels |= {p.channel for p in sequence.program_changes}
    channels |= {c.channel for c in sequence.control_changes}
    return channels


# ----------------------------------------------------------------------------
# Rewriting the sequence


def _rewrite_static(sequence: MidiSequence, mapping: dict[int, int]) -> None:
    """Rewrite every event through a fixed owner -> channel mapping."""
    for note in sequence.notes:
        note.channel = mapping.get(note.channel, note.channel)
    for program in sequence.program_changes:
        program.channel = mapping.get(program.channel, program.channel)
    for control in sequence.control_changes:
        control.channel = mapping.get(control.channel, control.channel)


def _rewrite_reused(
    sequence: MidiSequence, runs: list[Run], allocated: list[int]
) -> None:
    """Rewrite events onto reused channels, re-emitting state on handover.

    A part's program change and its pan and volume were emitted once, on the
    assumption that the channel stayed its own. Once channels are shared, that
    state has to be restated whenever a part takes a channel over, and dropped
    where it would otherwise land on a channel another part is using.
    """
    virtual = set(allocated)
    runs_by_owner: dict[int, list[Run]] = {}
    for run in runs:
        runs_by_owner.setdefault(run.owner, []).append(run)

    programs_by_owner: dict[int, list[MidiProgramChange]] = {}
    for program in sequence.program_changes:
        programs_by_owner.setdefault(program.channel, []).append(program)
    controls_by_owner: dict[int, list[MidiControlChange]] = {}
    for control in sequence.control_changes:
        controls_by_owner.setdefault(control.channel, []).append(control)
    for events in programs_by_owner.values():
        events.sort(key=lambda e: e.time)
    for events in controls_by_owner.values():
        events.sort(key=lambda e: e.time)

    # Notes: each one falls inside exactly one of its part's runs.
    for note in sequence.notes:
        if note.channel not in virtual:
            continue
        run = _run_at(runs_by_owner.get(note.channel, []), note.start_time)
        if run is not None:
            note.channel = run.channel

    # Everything not on a virtual channel is already where it belongs.
    programs = [p for p in sequence.program_changes if p.channel not in virtual]
    controls = [c for c in sequence.control_changes if c.channel not in virtual]

    # State each real channel is known to be in, so nothing is restated
    # needlessly: channel -> (owner, program, {control: value}).
    channel_owner: dict[int, int] = {}
    channel_program: dict[int, int] = {}
    channel_controls: dict[int, dict[int, int]] = {}

    for run in runs:
        if run.owner not in virtual:
            channel_owner[run.channel] = run.owner
            continue

        owner_programs = programs_by_owner.get(run.owner, [])
        owner_controls = controls_by_owner.get(run.owner, [])
        taking_over = channel_owner.get(run.channel) != run.owner
        state = channel_controls.setdefault(run.channel, {})

        program = _program_at(owner_programs, run.start)
        if program is not None and (
            taking_over or channel_program.get(run.channel) != program
        ):
            programs.append(
                MidiProgramChange(program=program, time=run.start, channel=run.channel)
            )
            channel_program[run.channel] = program

        wanted = _controls_at(owner_controls, run.start)
        if taking_over:
            # Controllers the incoming part does not set would otherwise keep
            # the previous part's values.
            for control in state:
                if control not in wanted:
                    wanted[control] = CONTROL_DEFAULTS.get(control, state[control])
        for control, value in sorted(wanted.items()):
            if taking_over or state.get(control) != value:
                controls.append(
                    MidiControlChange(
                        control=control,
                        value=value,
                        time=run.start,
                        channel=run.channel,
                    )
                )
                state[control] = value

        # Changes made while the part is sounding keep their own time.
        for control in owner_controls:
            if run.start + EPSILON < control.time <= run.end + EPSILON:
                controls.append(
                    MidiControlChange(
                        control=control.control,
                        value=control.value,
                        time=control.time,
                        channel=run.channel,
                    )
                )
                state[control.control] = control.value

        channel_owner[run.channel] = run.owner

    # Program changes and controls for a part that never sounds are dropped
    # with it: there is no channel they could be put on.
    sequence.program_changes = programs
    sequence.control_changes = controls


def _run_at(runs: list[Run], time: float) -> Run | None:
    """The run covering ``time``, or the nearest one if it falls in a gap."""
    if not runs:
        return None
    starts = [run.start for run in runs]
    index = bisect_right(starts, time + EPSILON) - 1
    if index < 0:
        return runs[0]
    return runs[index]


def _program_at(programs: list[MidiProgramChange], time: float) -> int | None:
    """The program in effect at ``time``, or the first one set after it."""
    if not programs:
        return None
    chosen = programs[0]
    for program in programs:
        if program.time > time + EPSILON:
            break
        chosen = program
    return chosen.program


def _controls_at(controls: list[MidiControlChange], time: float) -> dict[int, int]:
    """The value of every controller the part has set by ``time``."""
    values: dict[int, int] = {}
    for control in controls:
        if control.time > time + EPSILON:
            break
        values[control.control] = control.value
    return values
