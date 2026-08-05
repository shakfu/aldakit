"""Standard MIDI File (SMF) writer."""

import struct
from pathlib import Path

from ..constants import (
    MIDI_CHANNEL_MASK,
    MIDI_DATA_MASK,
    MIDI_STATUS_CONTROL_CHANGE,
    MIDI_STATUS_NOTE_OFF,
    MIDI_STATUS_NOTE_ON,
    MIDI_STATUS_PROGRAM_CHANGE,
)
from .types import MidiSequence


def _write_variable_length(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    if value < 0:
        raise ValueError("Variable-length value must be non-negative")

    if value == 0:
        return b"\x00"

    result = []
    while value:
        result.append(value & 0x7F)
        value >>= 7

    # Reverse and set continuation bits
    result.reverse()
    for i in range(len(result) - 1):
        result[i] |= 0x80

    return bytes(result)


def _seconds_to_ticks_simple(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    """Convert seconds to MIDI ticks assuming constant tempo.

    Args:
        seconds: Time in seconds.
        ticks_per_beat: MIDI ticks per beat.
        tempo_us: Tempo in microseconds per beat.

    Returns:
        Time in MIDI ticks.
    """
    # tempo_us is microseconds per beat
    # seconds * 1_000_000 = microseconds
    # microseconds / tempo_us = beats
    # beats * ticks_per_beat = ticks
    beats = (seconds * 1_000_000) / tempo_us
    return int(beats * ticks_per_beat)


class TempoMap:
    """Maps absolute time (seconds) to MIDI ticks, accounting for tempo changes."""

    def __init__(self, sequence: "MidiSequence") -> None:
        self.ticks_per_beat = sequence.ticks_per_beat
        self.default_tempo_us = 500000  # 120 BPM

        # Build sorted list of (time_seconds, tempo_us, tick_at_this_time)
        # We precompute tick positions for each tempo change
        self._tempo_points: list[tuple[float, int, int]] = []

        if not sequence.tempo_changes:
            # No tempo changes - everything at default tempo
            self._tempo_points = [(0.0, self.default_tempo_us, 0)]
            return

        sorted_changes = sorted(sequence.tempo_changes, key=lambda t: t.time)

        # First tempo change may not be at t=0, so handle initial segment
        current_tick = 0
        current_time = 0.0
        current_tempo_us = self.default_tempo_us

        for tc in sorted_changes:
            if tc.time > current_time:
                # Compute ticks elapsed during this segment
                segment_duration = tc.time - current_time
                segment_ticks = _seconds_to_ticks_simple(
                    segment_duration, self.ticks_per_beat, current_tempo_us
                )
                current_tick += segment_ticks

            # Record this tempo change point
            new_tempo_us = _bpm_to_tempo(tc.bpm)
            self._tempo_points.append((tc.time, new_tempo_us, current_tick))
            current_time = tc.time
            current_tempo_us = new_tempo_us

        # If first tempo change wasn't at t=0, insert the initial segment
        if self._tempo_points[0][0] > 0:
            self._tempo_points.insert(0, (0.0, self.default_tempo_us, 0))

    def seconds_to_ticks(self, seconds: float) -> int:
        """Convert absolute time in seconds to MIDI ticks."""
        if seconds <= 0:
            return 0

        # Find the tempo segment containing this timestamp
        # Walk through tempo points to find the last one <= seconds
        last_point_time = 0.0
        last_point_tempo_us = self.default_tempo_us
        last_point_tick = 0

        for point_time, point_tempo_us, point_tick in self._tempo_points:
            if point_time > seconds:
                break
            last_point_time = point_time
            last_point_tempo_us = point_tempo_us
            last_point_tick = point_tick

        # Compute ticks from last tempo change point to target time
        remaining_duration = seconds - last_point_time
        remaining_ticks = _seconds_to_ticks_simple(
            remaining_duration, self.ticks_per_beat, last_point_tempo_us
        )

        return last_point_tick + remaining_ticks


def _bpm_to_tempo(bpm: float) -> int:
    """Convert BPM to microseconds per beat."""
    return int(60_000_000 / bpm)


def write_midi_file(sequence: MidiSequence, path: Path | str) -> None:
    """Write a MidiSequence to a Standard MIDI File.

    Args:
        sequence: The MIDI sequence to write.
        path: Output file path.
    """
    # Build tempo map for accurate time-to-tick conversion
    tempo_map = TempoMap(sequence)

    # Group notes by channel
    channels: dict[int, list] = {}
    for note in sequence.notes:
        if note.channel not in channels:
            channels[note.channel] = []
        channels[note.channel].append(note)

    # Add program changes to their channels
    for pc in sequence.program_changes:
        if pc.channel not in channels:
            channels[pc.channel] = []

    tracks: list[bytes] = []

    # Track 0: tempo track
    tempo_track_data = _build_tempo_track(sequence, tempo_map)
    tracks.append(tempo_track_data)

    # One track per channel
    for channel in sorted(channels.keys()):
        track_data = _build_channel_track(sequence, channel, tempo_map)
        tracks.append(track_data)

    # Build the complete file
    output = _build_header(len(tracks), sequence.ticks_per_beat)
    for track_data in tracks:
        output += _build_track_chunk(track_data)

    # Write to file
    Path(path).write_bytes(output)


def _build_header(num_tracks: int, ticks_per_beat: int) -> bytes:
    """Build the MIDI file header chunk."""
    # MThd chunk
    # Format 1: multiple tracks, synchronous
    # Format type (2 bytes) + num tracks (2 bytes) + time division (2 bytes)
    header_data = struct.pack(">HHH", 1, num_tracks, ticks_per_beat)
    return b"MThd" + struct.pack(">I", len(header_data)) + header_data


def _build_track_chunk(track_data: bytes) -> bytes:
    """Wrap track data in an MTrk chunk."""
    return b"MTrk" + struct.pack(">I", len(track_data)) + track_data


def _build_tempo_track(sequence: MidiSequence, tempo_map: TempoMap) -> bytes:
    """Build the tempo track (track 0)."""
    events: list[tuple[int, bytes]] = []

    default_tempo_us = 500000  # 120 BPM

    if sequence.tempo_changes:
        for tc in sorted(sequence.tempo_changes, key=lambda t: t.time):
            tempo_us = _bpm_to_tempo(tc.bpm)
            tick = tempo_map.seconds_to_ticks(tc.time)
            # Meta event: FF 51 03 tt tt tt (set tempo)
            tempo_bytes = struct.pack(">I", tempo_us)[1:]  # 3 bytes, big-endian
            events.append((tick, b"\xff\x51\x03" + tempo_bytes))
    else:
        # Add default tempo at time 0
        tempo_bytes = struct.pack(">I", default_tempo_us)[1:]
        events.append((0, b"\xff\x51\x03" + tempo_bytes))

    # End of track
    if events:
        last_tick = max(e[0] for e in events)
    else:
        last_tick = 0
    events.append((last_tick, b"\xff\x2f\x00"))

    return _encode_track_events(events)


def _build_channel_track(
    sequence: MidiSequence, channel: int, tempo_map: TempoMap
) -> bytes:
    """Build a track for a specific MIDI channel."""
    events: list[tuple[int, bytes]] = []

    # Add program changes
    for pc in sequence.program_changes:
        if pc.channel == channel:
            tick = tempo_map.seconds_to_ticks(pc.time)
            # Program change: Cn pp
            msg = bytes(
                [
                    MIDI_STATUS_PROGRAM_CHANGE | (channel & MIDI_CHANNEL_MASK),
                    pc.program & MIDI_DATA_MASK,
                ]
            )
            events.append((tick, msg))

    # Add control changes
    for cc in sequence.control_changes:
        if cc.channel == channel:
            tick = tempo_map.seconds_to_ticks(cc.time)
            # Control change: Bn cc vv
            msg = bytes(
                [
                    MIDI_STATUS_CONTROL_CHANGE | (channel & MIDI_CHANNEL_MASK),
                    cc.control & MIDI_DATA_MASK,
                    cc.value & MIDI_DATA_MASK,
                ]
            )
            events.append((tick, msg))

    # Add note on/off events
    for note in sequence.notes:
        if note.channel == channel:
            start_tick = tempo_map.seconds_to_ticks(note.start_time)
            end_tick = tempo_map.seconds_to_ticks(note.start_time + note.duration)

            # Note on: 9n kk vv
            note_on = bytes(
                [
                    MIDI_STATUS_NOTE_ON | (channel & MIDI_CHANNEL_MASK),
                    note.pitch & MIDI_DATA_MASK,
                    note.velocity & MIDI_DATA_MASK,
                ]
            )
            # Note off: 8n kk vv
            note_off = bytes(
                [
                    MIDI_STATUS_NOTE_OFF | (channel & MIDI_CHANNEL_MASK),
                    note.pitch & MIDI_DATA_MASK,
                    0,
                ]
            )

            events.append((start_tick, note_on))
            events.append((end_tick, note_off))

    # Sort events: by tick, then note_off before note_on at same tick
    events.sort(key=lambda e: (e[0], e[1][0] & 0xF0 != MIDI_STATUS_NOTE_OFF))

    # End of track
    if events:
        last_tick = max(e[0] for e in events)
    else:
        last_tick = 0
    events.append((last_tick, b"\xff\x2f\x00"))

    return _encode_track_events(events)


def _encode_track_events(events: list[tuple[int, bytes]]) -> bytes:
    """Encode a list of (absolute_tick, event_bytes) to track data with delta times."""
    result = bytearray()
    last_tick = 0

    for tick, event_data in events:
        delta = max(0, tick - last_tick)
        result.extend(_write_variable_length(delta))
        result.extend(event_data)
        last_tick = tick

    return bytes(result)
