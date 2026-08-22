"""libremidi-based MIDI backend for realtime playback."""

import threading
from pathlib import Path

from ...constants import (
    DEFAULT_VIRTUAL_PORT_NAME,
    MIDI_CC_ALL_NOTES_OFF,
    MIDI_CHANNEL_MASK,
    MIDI_DATA_MASK,
    MIDI_MAX_CHANNELS,
    MIDI_STATUS_CONTROL_CHANGE,
    MIDI_STATUS_NOTE_OFF,
    MIDI_STATUS_NOTE_ON,
    MIDI_STATUS_PROGRAM_CHANGE,
    POLL_INTERVAL_DEFAULT,
)
from ..types import MidiSequence
from .async_playback import AsyncPlaybackManager
from .base import MidiBackend


class LibremidiBackend(MidiBackend):
    """MIDI backend using the libremidi library via nanobind.

    Provides low-latency realtime playback using the native libremidi library.
    Supports concurrent playback mode where multiple sequences can play
    simultaneously (up to MAX_PLAYBACK_SLOTS concurrent slots).

    Example:
        >>> backend = LibremidiBackend()
        >>> backend.concurrent_mode = True  # Enable concurrent playback (default)
        >>> backend.play(sequence1)  # Starts playing
        >>> backend.play(sequence2)  # Layers on top of sequence1
        >>> backend.stop()  # Stop all playback
    """

    def __init__(
        self,
        port_name: str | None = None,
        concurrent: bool = True,
        virtual_port_name: str = DEFAULT_VIRTUAL_PORT_NAME,
    ) -> None:
        """Initialize the libremidi backend.

        Args:
            port_name: Name of the MIDI output port to use.
                If None, will use a virtual port or first available port.
            concurrent: If True, enable concurrent playback mode (default).
                In concurrent mode, multiple sequences can play simultaneously.
            virtual_port_name: Name for the virtual MIDI port when no physical
                ports are available. Defaults to DEFAULT_VIRTUAL_PORT_NAME.
        """
        from ... import _libremidi  # type: ignore[attr-defined]

        self._libremidi = _libremidi
        self._port_name = port_name
        self._virtual_port_name = virtual_port_name
        self._midi_out: _libremidi.MidiOut | None = None
        self._observer: _libremidi.Observer | None = None
        self._port_opened = False
        self._async_manager: AsyncPlaybackManager | None = None
        self._concurrent_mode = concurrent
        # Lock for thread-safe MIDI output
        self._midi_lock = threading.Lock()

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
            self._midi_out.open_virtual_port(self._virtual_port_name)
            self._port_opened = True

    @property
    def concurrent_mode(self) -> bool:
        """Whether concurrent playback is enabled."""
        return self._concurrent_mode

    @concurrent_mode.setter
    def concurrent_mode(self, value: bool) -> None:
        """Set concurrent playback mode."""
        self._concurrent_mode = value
        if self._async_manager:
            self._async_manager.concurrent_mode = value

    @property
    def active_slots(self) -> int:
        """Number of currently active playback slots."""
        if self._async_manager:
            return self._async_manager.active_count
        return 0

    def _send_note_on(self, channel: int, note: int, velocity: int) -> None:
        """Send a note on message (thread-safe)."""
        if self._midi_out is None:
            return
        with self._midi_lock:
            status = MIDI_STATUS_NOTE_ON | (channel & MIDI_CHANNEL_MASK)
            self._midi_out.send_message(
                status, note & MIDI_DATA_MASK, velocity & MIDI_DATA_MASK
            )

    def _send_note_off(self, channel: int, note: int) -> None:
        """Send a note off message (thread-safe)."""
        if self._midi_out is None:
            return
        with self._midi_lock:
            status = MIDI_STATUS_NOTE_OFF | (channel & MIDI_CHANNEL_MASK)
            self._midi_out.send_message(status, note & MIDI_DATA_MASK, 0)

    def _send_program_change(self, channel: int, program: int) -> None:
        """Send a program change message (thread-safe)."""
        if self._midi_out is None:
            return
        with self._midi_lock:
            status = MIDI_STATUS_PROGRAM_CHANGE | (channel & MIDI_CHANNEL_MASK)
            self._midi_out.send_message(status, program & MIDI_DATA_MASK)

    def _send_control_change(self, channel: int, control: int, value: int) -> None:
        """Send a control change message (thread-safe)."""
        if self._midi_out is None:
            return
        with self._midi_lock:
            status = MIDI_STATUS_CONTROL_CHANGE | (channel & MIDI_CHANNEL_MASK)
            self._midi_out.send_message(
                status, control & MIDI_DATA_MASK, value & MIDI_DATA_MASK
            )

    def _send_all_notes_off(self) -> None:
        """Send all notes off on all channels (thread-safe)."""
        for channel in range(MIDI_MAX_CHANNELS):
            self._send_control_change(channel, MIDI_CC_ALL_NOTES_OFF, 0)

    def _ensure_async_manager(self) -> None:
        """Ensure the async playback manager is initialized."""
        if self._async_manager is None:
            self._async_manager = AsyncPlaybackManager(
                send_note_on=self._send_note_on,
                send_note_off=self._send_note_off,
                send_program_change=self._send_program_change,
                send_control_change=self._send_control_change,
                send_all_notes_off=self._send_all_notes_off,
            )
            self._async_manager.concurrent_mode = self._concurrent_mode

    def play(self, sequence: MidiSequence) -> int | None:
        """Play a MIDI sequence in realtime.

        In concurrent mode (default), the sequence starts playing immediately
        alongside any currently playing sequences (up to MAX_PLAYBACK_SLOTS).

        In sequential mode, waits for all current playback to complete first.

        Args:
            sequence: The MIDI sequence to play.

        Returns:
            The slot ID if playback started, or None if all slots are busy.
        """
        self._ensure_port_open()
        self._ensure_async_manager()

        if self._async_manager:
            return self._async_manager.play(sequence)
        return None

    def stop(self) -> None:
        """Stop all currently playing sequences."""
        if self._async_manager:
            self._async_manager.stop()
        if self._port_opened:
            self._send_all_notes_off()

    def save(self, sequence: MidiSequence, path: Path | str) -> None:
        """Save a MIDI sequence to a Standard MIDI File.

        Args:
            sequence: The MIDI sequence to save.
            path: The output file path.
        """
        from ..smf import write_midi_file

        write_midi_file(sequence, path)

    def is_playing(self) -> bool:
        """Check if any sequence is currently playing."""
        if self._async_manager:
            return self._async_manager.is_playing()
        return False

    def wait(self, poll_interval: float = POLL_INTERVAL_DEFAULT) -> None:
        """Block until all playback completes.

        Args:
            poll_interval: Seconds between status checks.
        """
        if self._async_manager:
            self._async_manager.wait(poll_interval)

    def close(self) -> None:
        """Close the MIDI output port and shutdown playback."""
        if self._async_manager:
            self._async_manager.shutdown()
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
