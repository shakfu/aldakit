"""Standard MIDI File (SMF) writer."""

from pathlib import Path
import struct

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


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    """Convert seconds to MIDI ticks.

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


def _bpm_to_tempo(bpm: float) -> int:
    """Convert BPM to microseconds per beat."""
    return int(60_000_000 / bpm)


def write_midi_file(sequence: MidiSequence, path: Path | str) -> None:
    """Write a MidiSequence to a Standard MIDI File.

    Args:
        sequence: The MIDI sequence to write.
        path: Output file path.
    """
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
    tempo_track_data = _build_tempo_track(sequence)
    tracks.append(tempo_track_data)

    # Default tempo for tick calculations
    default_tempo_us = 500000  # 120 BPM
    if sequence.tempo_changes:
        default_tempo_us = _bpm_to_tempo(sequence.tempo_changes[0].bpm)

    # One track per channel
    for channel in sorted(channels.keys()):
        track_data = _build_channel_track(sequence, channel, default_tempo_us)
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


def _build_tempo_track(sequence: MidiSequence) -> bytes:
    """Build the tempo track (track 0)."""
    events: list[tuple[int, bytes]] = []

    # Add tempo changes
    default_tempo_us = 500000  # 120 BPM
    current_tempo_us = default_tempo_us

    if sequence.tempo_changes:
        for tc in sorted(sequence.tempo_changes, key=lambda t: t.time):
            tempo_us = _bpm_to_tempo(tc.bpm)
            tick = _seconds_to_ticks(tc.time, sequence.ticks_per_beat, current_tempo_us)
            # Meta event: FF 51 03 tt tt tt (set tempo)
            tempo_bytes = struct.pack(">I", tempo_us)[1:]  # 3 bytes, big-endian
            events.append((tick, b"\xff\x51\x03" + tempo_bytes))
            current_tempo_us = tempo_us
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
    sequence: MidiSequence, channel: int, default_tempo_us: int
) -> bytes:
    """Build a track for a specific MIDI channel."""
    events: list[tuple[int, bytes]] = []

    ticks_per_beat = sequence.ticks_per_beat

    # Add program changes
    for pc in sequence.program_changes:
        if pc.channel == channel:
            tick = _seconds_to_ticks(pc.time, ticks_per_beat, default_tempo_us)
            # Program change: Cn pp
            msg = bytes([0xC0 | (channel & 0x0F), pc.program & 0x7F])
            events.append((tick, msg))

    # Add control changes
    for cc in sequence.control_changes:
        if cc.channel == channel:
            tick = _seconds_to_ticks(cc.time, ticks_per_beat, default_tempo_us)
            # Control change: Bn cc vv
            msg = bytes([0xB0 | (channel & 0x0F), cc.control & 0x7F, cc.value & 0x7F])
            events.append((tick, msg))

    # Add note on/off events
    for note in sequence.notes:
        if note.channel == channel:
            start_tick = _seconds_to_ticks(
                note.start_time, ticks_per_beat, default_tempo_us
            )
            end_tick = _seconds_to_ticks(
                note.start_time + note.duration, ticks_per_beat, default_tempo_us
            )

            # Note on: 9n kk vv
            note_on = bytes(
                [0x90 | (channel & 0x0F), note.pitch & 0x7F, note.velocity & 0x7F]
            )
            # Note off: 8n kk vv
            note_off = bytes([0x80 | (channel & 0x0F), note.pitch & 0x7F, 0])

            events.append((start_tick, note_on))
            events.append((end_tick, note_off))

    # Sort events: by tick, then note_off before note_on at same tick
    events.sort(key=lambda e: (e[0], e[1][0] & 0xF0 != 0x80))

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
