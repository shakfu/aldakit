"""libremidi-based MIDI backend for realtime playback."""

from pathlib import Path
import threading
import time

from .base import MidiBackend
from ..types import MidiSequence


class LibremidiBackend(MidiBackend):
    """MIDI backend using the libremidi library via nanobind.

    Provides low-latency realtime playback using the native libremidi library.
    """

    def __init__(self, port_name: str | None = None) -> None:
        """Initialize the libremidi backend.

        Args:
            port_name: Name of the MIDI output port to use.
                If None, will use a virtual port or first available port.
        """
        from ... import _libremidi

        self._libremidi = _libremidi
        self._port_name = port_name
        self._midi_out: _libremidi.MidiOut | None = None
        self._observer: _libremidi.Observer | None = None
        self._port_opened = False
        self._playing = False
        self._play_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _ensure_port_open(self) -> None:
        """Ensure the MIDI output port is open."""
        if self._port_opened and self._midi_out is not None:
            return

        self._midi_out = self._libremidi.MidiOut()
        self._observer = self._libremidi.Observer()
        ports = self._observer.get_output_ports()

        if self._port_name is not None:
            # Find port by name
            for port in ports:
                if (
                    port.port_name == self._port_name
                    or port.display_name == self._port_name
                ):
                    self._midi_out.open_port(port)
                    self._port_opened = True
                    return
            raise RuntimeError(
                f"Port '{self._port_name}' not found. "
                f"Available ports: {[p.display_name for p in ports]}"
            )
        elif ports:
            # Use first available port
            self._midi_out.open_port(ports[0])
            self._port_opened = True
        else:
            # Create a virtual port
            self._midi_out.open_virtual_port("AldaPyMIDI")
            self._port_opened = True

    def _send_note_on(self, channel: int, note: int, velocity: int) -> None:
        """Send a note on message."""
        if self._midi_out is None:
            return
        status = 0x90 | (channel & 0x0F)
        self._midi_out.send_message(status, note & 0x7F, velocity & 0x7F)

    def _send_note_off(self, channel: int, note: int) -> None:
        """Send a note off message."""
        if self._midi_out is None:
            return
        status = 0x80 | (channel & 0x0F)
        self._midi_out.send_message(status, note & 0x7F, 0)

    def _send_program_change(self, channel: int, program: int) -> None:
        """Send a program change message."""
        if self._midi_out is None:
            return
        status = 0xC0 | (channel & 0x0F)
        self._midi_out.send_message(status, program & 0x7F)

    def _send_control_change(self, channel: int, control: int, value: int) -> None:
        """Send a control change message."""
        if self._midi_out is None:
            return
        status = 0xB0 | (channel & 0x0F)
        self._midi_out.send_message(status, control & 0x7F, value & 0x7F)

    def _send_all_notes_off(self) -> None:
        """Send all notes off on all channels."""
        for channel in range(16):
            self._send_control_change(channel, 123, 0)  # All Notes Off

    def play(self, sequence: MidiSequence) -> None:
        """Play a MIDI sequence in realtime.

        Args:
            sequence: The MIDI sequence to play.
        """
        self.stop()
        self._ensure_port_open()

        self._stop_event.clear()
        self._playing = True

        def play_thread():
            try:
                # Build a timeline of events
                events: list[tuple[float, str, tuple]] = []

                # Add program changes
                for pc in sequence.program_changes:
                    events.append((pc.time, "program", (pc.channel, pc.program)))

                # Add control changes
                for cc in sequence.control_changes:
                    events.append(
                        (cc.time, "control", (cc.channel, cc.control, cc.value))
                    )

                # Add note on/off events
                for note in sequence.notes:
                    events.append(
                        (
                            note.start_time,
                            "note_on",
                            (note.channel, note.pitch, note.velocity),
                        )
                    )
                    events.append(
                        (
                            note.start_time + note.duration,
                            "note_off",
                            (note.channel, note.pitch),
                        )
                    )

                # Sort by time, with note_off before note_on at same time
                events.sort(key=lambda e: (e[0], e[1] != "note_off"))

                # Play events
                start_time = time.perf_counter()
                for event_time, event_type, args in events:
                    if self._stop_event.is_set():
                        self._send_all_notes_off()
                        break

                    # Wait until event time
                    target_time = start_time + event_time
                    current_time = time.perf_counter()
                    if target_time > current_time:
                        # Use a loop with short sleeps to allow for stop events
                        while time.perf_counter() < target_time:
                            if self._stop_event.is_set():
                                self._send_all_notes_off()
                                return
                            remaining = target_time - time.perf_counter()
                            if remaining > 0.01:
                                time.sleep(0.01)
                            elif remaining > 0:
                                time.sleep(remaining)

                    # Send the event
                    if event_type == "note_on":
                        self._send_note_on(*args)
                    elif event_type == "note_off":
                        self._send_note_off(*args)
                    elif event_type == "program":
                        self._send_program_change(*args)
                    elif event_type == "control":
                        self._send_control_change(*args)

            finally:
                self._playing = False

        self._play_thread = threading.Thread(target=play_thread, daemon=True)
        self._play_thread.start()

    def stop(self) -> None:
        """Stop any currently playing sequence."""
        if self._playing:
            self._stop_event.set()
            if self._play_thread and self._play_thread.is_alive():
                self._play_thread.join(timeout=1.0)
            if self._port_opened:
                self._send_all_notes_off()
            self._playing = False

    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        """Save a MIDI sequence to a Standard MIDI File.

        Args:
            sequence: The MIDI sequence to save.
            path: The output file path.
        """
        from ..smf import write_midi_file

        write_midi_file(sequence, path)

    def is_playing(self) -> bool:
        """Check if a sequence is currently playing."""
        return self._playing

    def close(self) -> None:
        """Close the MIDI output port."""
        self.stop()
        if self._midi_out is not None and self._port_opened:
            self._midi_out.close_port()
            self._port_opened = False

    def __enter__(self) -> "LibremidiBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_output_ports(self) -> list[str]:
        """List available MIDI output ports.

        Returns:
            List of available MIDI output port names.
        """
        if self._observer is None:
            self._observer = self._libremidi.Observer()
        return [p.display_name for p in self._observer.get_output_ports()]
