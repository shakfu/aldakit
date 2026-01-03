"""Real-time MIDI input transcription."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .._libremidi import (  # type: ignore[import-not-found]
    MidiIn,
    MidiMessage,
    Observer,
)
from ..compose.attributes import Tempo
from ..compose.core import Note, Rest, Seq
from ..compose.part import Part
from ..score import Score
from .midi_to_ast import midi_pitch_to_note

if TYPE_CHECKING:
    from ..score import Score


@dataclass
class PendingNote:
    """A note that has been started but not yet released."""

    pitch: int
    velocity: int
    start_time: float  # In seconds


@dataclass
class RecordedNote:
    """A completed note with start time and duration."""

    pitch: int
    velocity: int
    start_time: float
    duration: float


@dataclass
class TranscribeSession:
    """A real-time MIDI transcription session.

    Records incoming MIDI notes and converts them to aldakit compose elements.

    Example:
        >>> session = TranscribeSession()
        >>> session.start()  # Opens MIDI input
        >>> # Play some notes on your MIDI keyboard...
        >>> time.sleep(5)
        >>> score = session.stop()  # Returns a Score with recorded notes
        >>> print(score.to_alda())
    """

    port_name: str | None = None
    quantize_grid: float = 0.25  # Quantize to 16th notes
    default_tempo: float = 120.0

    # Internal state
    _midi_in: MidiIn | None = field(default=None, repr=False)
    _pending_notes: dict[int, PendingNote] = field(default_factory=dict, repr=False)
    _recorded_notes: list[RecordedNote] = field(default_factory=list, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _running: bool = field(default=False, repr=False)
    _on_note: Callable[[int, int, bool], None] | None = field(default=None, repr=False)

    def list_input_ports(self) -> list[str]:
        """List available MIDI input ports."""
        observer = Observer()
        return [p.display_name for p in observer.get_input_ports()]

    def start(self, port_name: str | None = None) -> None:
        """Start recording MIDI input.

        Args:
            port_name: MIDI input port name. If None, uses the first available port.

        Raises:
            RuntimeError: If no MIDI input ports are available.
        """
        if self._running:
            return

        self._midi_in = MidiIn()
        observer = Observer()
        input_ports = observer.get_input_ports()

        if not input_ports:
            raise RuntimeError("No MIDI input ports available")

        # Find the requested port or use the first one
        target_port = None
        if port_name:
            for port in input_ports:
                if port_name.lower() in port.display_name.lower():
                    target_port = port
                    break
            if target_port is None:
                raise RuntimeError(f"MIDI input port '{port_name}' not found")
        else:
            target_port = input_ports[0]

        err = self._midi_in.open_port(target_port)
        if err:
            raise RuntimeError(f"Failed to open MIDI port: {err}")

        self._pending_notes = {}
        self._recorded_notes = []
        self._start_time = time.time()
        self._running = True

    def stop(self) -> Seq:
        """Stop recording and return the recorded notes as a Seq.

        Returns:
            A Seq containing the recorded notes.
        """
        if not self._running:
            return Seq()

        # Process any remaining messages
        self.poll()

        # Close any pending notes
        end_time = time.time() - self._start_time
        for pitch, pending in self._pending_notes.items():
            duration = end_time - pending.start_time
            self._recorded_notes.append(
                RecordedNote(
                    pitch=pending.pitch,
                    velocity=pending.velocity,
                    start_time=pending.start_time,
                    duration=max(0.1, duration),
                )
            )

        self._running = False
        if self._midi_in:
            self._midi_in.close_port()
            self._midi_in = None

        # Convert recorded notes to Seq
        return self._notes_to_seq()

    def poll(self) -> None:
        """Poll for incoming MIDI messages. Call this periodically."""
        if not self._running or not self._midi_in:
            return

        current_time = time.time() - self._start_time
        messages = self._midi_in.poll()

        for msg in messages:
            self._process_message(msg, current_time)

    def _process_message(self, msg: MidiMessage, current_time: float) -> None:
        """Process a single MIDI message."""
        if len(msg.bytes) < 2:
            return

        status = msg.bytes[0]
        msg_type = status & 0xF0

        if msg_type == 0x90 and len(msg.bytes) >= 3:
            # Note On
            pitch = msg.bytes[1]
            velocity = msg.bytes[2]

            if velocity == 0:
                # Note On with velocity 0 = Note Off
                self._note_off(pitch, current_time)
            else:
                self._note_on(pitch, velocity, current_time)

        elif msg_type == 0x80 and len(msg.bytes) >= 3:
            # Note Off
            pitch = msg.bytes[1]
            self._note_off(pitch, current_time)

    def _note_on(self, pitch: int, velocity: int, time: float) -> None:
        """Handle a note on event."""
        # If there's already a pending note at this pitch, end it first
        if pitch in self._pending_notes:
            self._note_off(pitch, time)

        self._pending_notes[pitch] = PendingNote(
            pitch=pitch, velocity=velocity, start_time=time
        )

        if self._on_note:
            self._on_note(pitch, velocity, True)

    def _note_off(self, pitch: int, time: float) -> None:
        """Handle a note off event."""
        if pitch not in self._pending_notes:
            return

        pending = self._pending_notes.pop(pitch)
        duration = time - pending.start_time

        self._recorded_notes.append(
            RecordedNote(
                pitch=pending.pitch,
                velocity=pending.velocity,
                start_time=pending.start_time,
                duration=max(0.01, duration),  # Minimum duration
            )
        )

        if self._on_note:
            self._on_note(pitch, 0, False)

    def _notes_to_seq(self) -> Seq:
        """Convert recorded notes to a Seq."""
        if not self._recorded_notes:
            return Seq()

        # Sort by start time
        notes = sorted(self._recorded_notes, key=lambda n: n.start_time)

        # Convert to compose elements
        elements = []
        current_time = 0.0

        for note in notes:
            # Insert rest if there's a gap
            gap = note.start_time - current_time
            if gap > 0.1:  # Only insert rest for gaps > 100ms
                rest_duration = self._seconds_to_duration(gap)
                if rest_duration:
                    elements.append(Rest(duration=rest_duration))

            # Convert pitch to note
            letter, octave, accidentals = midi_pitch_to_note(note.pitch)
            accidental = accidentals[0] if accidentals else None
            note_duration = self._seconds_to_duration(note.duration)

            elements.append(
                Note(
                    pitch=letter,
                    duration=note_duration,
                    octave=octave,
                    accidental=accidental,
                )
            )
            current_time = note.start_time + note.duration

        return Seq(elements=elements)

    def _seconds_to_duration(self, seconds: float) -> int | None:
        """Convert seconds to the closest standard duration value."""
        # At default tempo (120 BPM), one beat = 0.5 seconds
        beats = seconds * self.default_tempo / 60.0

        # Quantize to grid
        if self.quantize_grid > 0:
            beats = round(beats / self.quantize_grid) * self.quantize_grid

        # Standard durations: 1=whole, 2=half, 4=quarter, 8=eighth, 16=sixteenth
        duration_map = [
            (4.0, 1),  # whole note
            (2.0, 2),  # half note
            (1.5, 4),  # dotted quarter (approximate as quarter)
            (1.0, 4),  # quarter note
            (0.75, 8),  # dotted eighth (approximate as eighth)
            (0.5, 8),  # eighth note
            (0.25, 16),  # sixteenth note
            (0.125, 32),  # thirty-second note
        ]

        # Find closest match
        best_duration = 4  # Default to quarter note
        best_diff = float("inf")

        for length, duration in duration_map:
            diff = abs(beats - length)
            if diff < best_diff:
                best_diff = diff
                best_duration = duration

        return best_duration

    def on_note(self, callback: Callable[[int, int, bool], None]) -> None:
        """Set a callback for note events.

        The callback receives (pitch, velocity, is_note_on).
        """
        self._on_note = callback


def transcribe(
    duration: float = 10.0,
    port_name: str | None = None,
    instrument: str = "piano",
    quantize_grid: float = 0.25,
    tempo: float = 120.0,
    on_note: Callable[[int, int, bool], None] | None = None,
    poll_interval: float = 0.01,
) -> "Score":  # noqa: F821
    """Record MIDI input and return a Score.

    This is a blocking function that records for the specified duration.

    Args:
        duration: Recording duration in seconds.
        port_name: MIDI input port name. If None, uses the first available port.
        instrument: Instrument name for the part.
        quantize_grid: Grid size in beats for quantization (0.25 = 16th notes).
        tempo: Tempo in BPM for duration calculations.
        on_note: Optional callback for note events (pitch, velocity, is_note_on).
        poll_interval: How often to poll for MIDI messages (seconds).

    Returns:
        A Score containing the recorded notes.

    Example:
        >>> from aldakit.midi.transcriber import transcribe
        >>> print("Recording for 10 seconds...")
        >>> score = transcribe(duration=10)
        >>> score.play()
    """

    session = TranscribeSession(
        port_name=port_name,
        quantize_grid=quantize_grid,
        default_tempo=tempo,
    )

    if on_note:
        session.on_note(on_note)

    session.start(port_name)

    # Record for the specified duration
    start = time.time()
    while time.time() - start < duration:
        session.poll()
        time.sleep(poll_interval)

    seq = session.stop()

    # Build a Score from the recorded notes

    elements = [Part(instruments=(instrument,)), Tempo(bpm=tempo)]
    elements.extend(seq.elements)

    return Score.from_elements(*elements)


def list_input_ports() -> list[str]:
    """List available MIDI input ports.

    Returns:
        List of port names.
    """
    observer = Observer()
    return [p.display_name for p in observer.get_input_ports()]
