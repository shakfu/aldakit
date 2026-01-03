"""Standard MIDI File (SMF) reader."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .types import (
    MidiControlChange,
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    MidiTempoChange,
)


class MidiParseError(Exception):
    """Error parsing a MIDI file."""

    pass


@dataclass
class _PendingNote:
    """A note that has started but not yet ended."""

    pitch: int
    velocity: int
    start_tick: int
    channel: int


def _read_variable_length(data: bytes, offset: int) -> tuple[int, int]:
    """Read a MIDI variable-length quantity.

    Returns:
        Tuple of (value, bytes_consumed).
    """
    value = 0
    bytes_consumed = 0

    while True:
        if offset + bytes_consumed >= len(data):
            raise MidiParseError("Unexpected end of data reading variable-length value")

        byte = data[offset + bytes_consumed]
        bytes_consumed += 1
        value = (value << 7) | (byte & 0x7F)

        if not (byte & 0x80):
            break

        if bytes_consumed > 4:
            raise MidiParseError("Variable-length value too long")

    return value, bytes_consumed


def _ticks_to_seconds(tick: int, ticks_per_beat: int, tempo_us: int = 500000) -> float:
    """Convert MIDI ticks to seconds.

    Args:
        tick: Time in MIDI ticks.
        ticks_per_beat: MIDI ticks per beat.
        tempo_us: Tempo in microseconds per beat (default 500000 = 120 BPM).

    Returns:
        Time in seconds.
    """
    # tick / ticks_per_beat = beats
    # beats * tempo_us = microseconds
    # microseconds / 1_000_000 = seconds
    beats = tick / ticks_per_beat
    microseconds = beats * tempo_us
    return microseconds / 1_000_000


def _tempo_us_to_bpm(tempo_us: int) -> float:
    """Convert microseconds per beat to BPM."""
    return 60_000_000 / tempo_us


def read_midi_file(path: Path | str) -> MidiSequence:
    """Read a Standard MIDI File and return a MidiSequence.

    Args:
        path: Path to the MIDI file.

    Returns:
        A MidiSequence containing all notes, tempo changes, etc.

    Raises:
        MidiParseError: If the file cannot be parsed.
    """
    data = Path(path).read_bytes()

    if len(data) < 14:
        raise MidiParseError("File too small to be a valid MIDI file")

    # Parse header chunk
    header_type = data[0:4]
    if header_type != b"MThd":
        raise MidiParseError(f"Invalid MIDI file: expected MThd, got {header_type!r}")

    header_length = struct.unpack(">I", data[4:8])[0]
    if header_length < 6:
        raise MidiParseError(f"Invalid header length: {header_length}")

    format_type, num_tracks, time_division = struct.unpack(">HHH", data[8:14])

    # Check for SMPTE time division (not supported)
    if time_division & 0x8000:
        raise MidiParseError("SMPTE time division not supported")

    ticks_per_beat = time_division

    # Parse tracks
    offset = 8 + header_length
    tracks_data: list[list[tuple[int, bytes]]] = []

    for _ in range(num_tracks):
        if offset + 8 > len(data):
            raise MidiParseError("Unexpected end of file reading track header")

        track_type = data[offset : offset + 4]
        if track_type != b"MTrk":
            raise MidiParseError(
                f"Invalid track chunk: expected MTrk, got {track_type!r}"
            )

        track_length = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        track_data = data[offset + 8 : offset + 8 + track_length]

        events = _parse_track_events(track_data)
        tracks_data.append(events)

        offset += 8 + track_length

    # Build tempo map from all tracks
    tempo_map = _build_tempo_map(tracks_data, ticks_per_beat)

    # Convert all tracks to absolute time and collect events
    sequence = MidiSequence(ticks_per_beat=ticks_per_beat)

    for track_events in tracks_data:
        _process_track_events(track_events, tempo_map, ticks_per_beat, sequence)

    # Sort notes by start time
    sequence.notes.sort(key=lambda n: n.start_time)

    return sequence


def _parse_track_events(track_data: bytes) -> list[tuple[int, bytes]]:
    """Parse a track chunk into a list of (absolute_tick, event_bytes) tuples."""
    events: list[tuple[int, bytes]] = []
    offset = 0
    absolute_tick = 0
    running_status: int | None = None

    while offset < len(track_data):
        # Read delta time
        delta, consumed = _read_variable_length(track_data, offset)
        offset += consumed
        absolute_tick += delta

        if offset >= len(track_data):
            break

        # Read event
        status_byte = track_data[offset]

        # Check for running status
        if status_byte < 0x80:
            # Running status: use previous status byte
            if running_status is None:
                raise MidiParseError("Running status without previous status byte")
            status_byte = running_status
        else:
            offset += 1
            if status_byte < 0xF0:
                running_status = status_byte

        # Parse based on status byte
        event_data: bytes

        if status_byte == 0xFF:
            # Meta event
            if offset + 1 >= len(track_data):
                break
            meta_type = track_data[offset]
            offset += 1
            length, consumed = _read_variable_length(track_data, offset)
            offset += consumed
            meta_data = track_data[offset : offset + length]
            offset += length
            event_data = bytes([0xFF, meta_type]) + bytes([length]) + meta_data

        elif status_byte == 0xF0 or status_byte == 0xF7:
            # SysEx event
            length, consumed = _read_variable_length(track_data, offset)
            offset += consumed
            sysex_data = track_data[offset : offset + length]
            offset += length
            event_data = bytes([status_byte]) + bytes([length]) + sysex_data

        elif 0x80 <= status_byte <= 0xEF:
            # Channel message
            msg_type = status_byte & 0xF0

            if msg_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                # Two data bytes
                if offset + 1 >= len(track_data):
                    break
                data1 = track_data[offset]
                data2 = track_data[offset + 1]
                offset += 2
                event_data = bytes([status_byte, data1, data2])

            elif msg_type in (0xC0, 0xD0):
                # One data byte
                if offset >= len(track_data):
                    break
                data1 = track_data[offset]
                offset += 1
                event_data = bytes([status_byte, data1])

            else:
                # Unknown, skip
                event_data = bytes([status_byte])

        else:
            # System common message (F1-F6) or realtime (F8-FE) - skip
            event_data = bytes([status_byte])

        events.append((absolute_tick, event_data))

    return events


def _build_tempo_map(
    tracks_data: list[list[tuple[int, bytes]]], ticks_per_beat: int
) -> list[tuple[int, int]]:
    """Build a tempo map from all tracks.

    Returns:
        List of (tick, tempo_us) tuples sorted by tick.
    """
    tempo_events: list[tuple[int, int]] = []

    for track_events in tracks_data:
        for tick, event_data in track_events:
            if len(event_data) >= 6 and event_data[0] == 0xFF and event_data[1] == 0x51:
                # Set tempo meta event
                tempo_us = (event_data[3] << 16) | (event_data[4] << 8) | event_data[5]
                tempo_events.append((tick, tempo_us))

    # Sort by tick
    tempo_events.sort(key=lambda x: x[0])

    # If no tempo events, use default (120 BPM)
    if not tempo_events:
        tempo_events = [(0, 500000)]

    return tempo_events


def _tick_to_seconds_with_tempo_map(
    tick: int, tempo_map: list[tuple[int, int]], ticks_per_beat: int
) -> float:
    """Convert a tick to seconds using a tempo map."""
    if not tempo_map:
        return _ticks_to_seconds(tick, ticks_per_beat, 500000)

    seconds = 0.0
    last_tick = 0
    last_tempo_us = 500000  # Default 120 BPM

    for tempo_tick, tempo_us in tempo_map:
        if tempo_tick >= tick:
            break

        # Add time from last_tick to tempo_tick at last_tempo_us
        delta_ticks = tempo_tick - last_tick
        seconds += _ticks_to_seconds(delta_ticks, ticks_per_beat, last_tempo_us)
        last_tick = tempo_tick
        last_tempo_us = tempo_us

    # Add remaining time
    delta_ticks = tick - last_tick
    seconds += _ticks_to_seconds(delta_ticks, ticks_per_beat, last_tempo_us)

    return seconds


def _process_track_events(
    events: list[tuple[int, bytes]],
    tempo_map: list[tuple[int, int]],
    ticks_per_beat: int,
    sequence: MidiSequence,
) -> None:
    """Process track events and add them to the sequence."""
    pending_notes: dict[tuple[int, int], _PendingNote] = {}  # (channel, pitch) -> note

    for tick, event_data in events:
        if not event_data:
            continue

        status_byte = event_data[0]

        if status_byte == 0xFF:
            # Meta event
            if len(event_data) >= 6 and event_data[1] == 0x51:
                # Set tempo
                tempo_us = (event_data[3] << 16) | (event_data[4] << 8) | event_data[5]
                bpm = _tempo_us_to_bpm(tempo_us)
                time_seconds = _tick_to_seconds_with_tempo_map(
                    tick, tempo_map, ticks_per_beat
                )
                sequence.tempo_changes.append(
                    MidiTempoChange(bpm=bpm, time=time_seconds)
                )

        elif 0x80 <= status_byte <= 0xEF:
            channel = status_byte & 0x0F
            msg_type = status_byte & 0xF0

            if msg_type == 0x90 and len(event_data) >= 3:
                # Note on
                pitch = event_data[1]
                velocity = event_data[2]

                if velocity == 0:
                    # Note on with velocity 0 = note off
                    key = (channel, pitch)
                    if key in pending_notes:
                        pending = pending_notes.pop(key)
                        start_seconds = _tick_to_seconds_with_tempo_map(
                            pending.start_tick, tempo_map, ticks_per_beat
                        )
                        end_seconds = _tick_to_seconds_with_tempo_map(
                            tick, tempo_map, ticks_per_beat
                        )
                        duration = end_seconds - start_seconds
                        sequence.notes.append(
                            MidiNote(
                                pitch=pending.pitch,
                                velocity=pending.velocity,
                                start_time=start_seconds,
                                duration=max(0.001, duration),
                                channel=pending.channel,
                            )
                        )
                else:
                    # Start a new note
                    key = (channel, pitch)
                    pending_notes[key] = _PendingNote(
                        pitch=pitch,
                        velocity=velocity,
                        start_tick=tick,
                        channel=channel,
                    )

            elif msg_type == 0x80 and len(event_data) >= 3:
                # Note off
                pitch = event_data[1]
                key = (channel, pitch)
                if key in pending_notes:
                    pending = pending_notes.pop(key)
                    start_seconds = _tick_to_seconds_with_tempo_map(
                        pending.start_tick, tempo_map, ticks_per_beat
                    )
                    end_seconds = _tick_to_seconds_with_tempo_map(
                        tick, tempo_map, ticks_per_beat
                    )
                    duration = end_seconds - start_seconds
                    sequence.notes.append(
                        MidiNote(
                            pitch=pending.pitch,
                            velocity=pending.velocity,
                            start_time=start_seconds,
                            duration=max(0.001, duration),
                            channel=pending.channel,
                        )
                    )

            elif msg_type == 0xC0 and len(event_data) >= 2:
                # Program change
                program = event_data[1]
                time_seconds = _tick_to_seconds_with_tempo_map(
                    tick, tempo_map, ticks_per_beat
                )
                sequence.program_changes.append(
                    MidiProgramChange(
                        program=program, time=time_seconds, channel=channel
                    )
                )

            elif msg_type == 0xB0 and len(event_data) >= 3:
                # Control change
                control = event_data[1]
                value = event_data[2]
                time_seconds = _tick_to_seconds_with_tempo_map(
                    tick, tempo_map, ticks_per_beat
                )
                sequence.control_changes.append(
                    MidiControlChange(
                        control=control, value=value, time=time_seconds, channel=channel
                    )
                )
