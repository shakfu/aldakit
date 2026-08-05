"""Tests for async/concurrent playback system."""

import time

import pytest

from aldakit.constants import MAX_PLAYBACK_SLOTS
from aldakit.midi.backends.async_playback import (
    AsyncPlaybackManager,
    PlaybackEvent,
    PlaybackSlot,
)
from aldakit.midi.types import MidiNote, MidiSequence


class TestPlaybackSlot:
    """Test PlaybackSlot data class."""

    def test_default_values(self):
        slot = PlaybackSlot(slot_id=0)
        assert slot.slot_id == 0
        assert slot.active is False
        assert slot.events == []
        assert slot.event_index == 0
        assert slot.stop_requested is False
        assert slot.thread is None

    def test_slot_id(self):
        for i in range(MAX_PLAYBACK_SLOTS):
            slot = PlaybackSlot(slot_id=i)
            assert slot.slot_id == i


class TestPlaybackEvent:
    """Test PlaybackEvent data class."""

    def test_note_on_event(self):
        evt = PlaybackEvent(time=1.0, event_type="note_on", args=(0, 60, 100))
        assert evt.time == 1.0
        assert evt.event_type == "note_on"
        assert evt.args == (0, 60, 100)

    def test_note_off_event(self):
        evt = PlaybackEvent(time=2.0, event_type="note_off", args=(0, 60))
        assert evt.time == 2.0
        assert evt.event_type == "note_off"
        assert evt.args == (0, 60)


class TestAsyncPlaybackManager:
    """Test AsyncPlaybackManager."""

    @pytest.fixture
    def events_received(self):
        """Track events received by mock functions."""
        return {"note_on": [], "note_off": [], "program": [], "control": []}

    @pytest.fixture
    def manager(self, events_received):
        """Create a manager with mock send functions."""

        def send_note_on(ch, note, vel):
            events_received["note_on"].append((ch, note, vel))

        def send_note_off(ch, note):
            events_received["note_off"].append((ch, note))

        def send_program(ch, prog):
            events_received["program"].append((ch, prog))

        def send_control(ch, ctrl, val):
            events_received["control"].append((ch, ctrl, val))

        mgr = AsyncPlaybackManager(
            send_note_on=send_note_on,
            send_note_off=send_note_off,
            send_program_change=send_program,
            send_control_change=send_control,
        )
        yield mgr
        mgr.shutdown()

    def test_initial_state(self, manager):
        """Manager starts with no active slots."""
        assert manager.active_count == 0
        assert manager.is_playing() is False
        assert manager.concurrent_mode is True

    def test_concurrent_mode_toggle(self, manager):
        """Concurrent mode can be toggled."""
        assert manager.concurrent_mode is True
        manager.concurrent_mode = False
        assert manager.concurrent_mode is False
        manager.concurrent_mode = True
        assert manager.concurrent_mode is True

    def test_play_empty_sequence(self, manager):
        """Playing empty sequence returns None."""
        seq = MidiSequence()
        result = manager.play(seq)
        assert result is None
        assert manager.is_playing() is False

    def test_play_returns_slot_id(self, manager):
        """Play returns a slot ID."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                )
            ]
        )
        slot_id = manager.play(seq)
        assert slot_id is not None
        assert 0 <= slot_id < MAX_PLAYBACK_SLOTS

    def test_play_increments_active_count(self, manager):
        """Playing increments active count."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                )
            ]
        )
        assert manager.active_count == 0
        manager.play(seq)
        # Give thread time to start
        time.sleep(0.05)
        assert manager.active_count >= 1
        assert manager.is_playing() is True

    def test_stop_all_slots(self, manager):
        """Stop stops all playing slots."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=5.0, channel=0
                )
            ]
        )
        manager.play(seq)
        time.sleep(0.05)
        assert manager.is_playing() is True

        manager.stop()
        time.sleep(0.1)
        assert manager.is_playing() is False
        assert manager.active_count == 0

    def test_wait_blocks_until_complete(self, manager):
        """Wait blocks until playback completes."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                )
            ]
        )
        start = time.perf_counter()
        manager.play(seq)
        manager.wait()
        elapsed = time.perf_counter() - start

        assert manager.is_playing() is False
        assert elapsed >= 0.1  # At least note duration

    def test_concurrent_playback_multiple_slots(self, manager):
        """Multiple sequences can play concurrently."""
        seq1 = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.5, channel=0
                )
            ]
        )
        seq2 = MidiSequence(
            notes=[
                MidiNote(
                    pitch=64, velocity=100, start_time=0.0, duration=0.5, channel=1
                )
            ]
        )

        slot1 = manager.play(seq1)
        slot2 = manager.play(seq2)

        assert slot1 is not None
        assert slot2 is not None
        assert slot1 != slot2

        time.sleep(0.05)
        assert manager.active_count == 2

        manager.stop()

    def test_sequential_mode_waits(self, manager):
        """Sequential mode waits for previous playback."""
        manager.concurrent_mode = False

        seq1 = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                )
            ]
        )
        seq2 = MidiSequence(
            notes=[
                MidiNote(
                    pitch=64, velocity=100, start_time=0.0, duration=0.1, channel=1
                )
            ]
        )

        start = time.perf_counter()
        manager.play(seq1)
        manager.play(seq2)  # Should wait for seq1 to complete
        manager.wait()
        elapsed = time.perf_counter() - start

        # Both should play sequentially
        assert elapsed >= 0.2

    def test_max_slots_limit(self, manager):
        """Cannot exceed MAX_PLAYBACK_SLOTS concurrent playbacks."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=5.0, channel=0
                )
            ]
        )

        # Fill all slots
        slots = []
        for i in range(MAX_PLAYBACK_SLOTS):
            slot_id = manager.play(seq)
            if slot_id is not None:
                slots.append(slot_id)
            time.sleep(0.01)  # Give threads time to start

        # All slots should be used
        assert len(slots) == MAX_PLAYBACK_SLOTS

        # Additional play should return None (all slots busy)
        time.sleep(0.05)
        result = manager.play(seq)
        assert result is None

        manager.stop()

    def test_stop_specific_slot(self, manager):
        """Can stop a specific slot."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=5.0, channel=0
                )
            ]
        )

        slot1 = manager.play(seq)
        manager.play(seq)  # Second slot, not used directly

        time.sleep(0.05)
        assert manager.active_count == 2

        manager.stop_slot(slot1)
        time.sleep(0.1)
        assert manager.active_count == 1

        manager.stop()

    def test_events_sent_correctly(self, manager, events_received):
        """Events are sent via the callbacks."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.05, channel=0
                )
            ]
        )
        manager.play(seq)
        manager.wait()

        # Should have note on and note off
        assert len(events_received["note_on"]) == 1
        assert events_received["note_on"][0] == (0, 60, 100)
        assert len(events_received["note_off"]) == 1
        assert events_received["note_off"][0] == (0, 60)

    def test_get_slot_info(self, manager):
        """Get slot info returns status of all slots."""
        info = manager.get_slot_info()
        assert len(info) == MAX_PLAYBACK_SLOTS
        for slot_info in info:
            assert "slot_id" in slot_info
            assert "active" in slot_info
            assert "event_count" in slot_info
            assert "progress" in slot_info


class TestAsyncPlaybackManagerEvents:
    """Test event building and ordering."""

    @pytest.fixture
    def manager(self):
        mgr = AsyncPlaybackManager(
            send_note_on=lambda *a: None,
            send_note_off=lambda *a: None,
            send_program_change=lambda *a: None,
            send_control_change=lambda *a: None,
        )
        yield mgr
        mgr.shutdown()

    def test_events_sorted_by_time(self, manager):
        """Events are sorted by time."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=64, velocity=100, start_time=0.2, duration=0.1, channel=0
                ),
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                ),
                MidiNote(
                    pitch=62, velocity=100, start_time=0.1, duration=0.1, channel=0
                ),
            ]
        )
        events = manager._build_events(seq)

        # First note_on should be at time 0.0 (pitch 60)
        note_ons = [e for e in events if e.event_type == "note_on"]
        assert note_ons[0].time == 0.0
        assert note_ons[0].args[1] == 60  # pitch

    def test_note_off_before_note_on_at_same_time(self, manager):
        """Note off events come before note on at same time."""
        seq = MidiSequence(
            notes=[
                MidiNote(
                    pitch=60, velocity=100, start_time=0.0, duration=0.1, channel=0
                ),
                MidiNote(
                    pitch=62, velocity=100, start_time=0.1, duration=0.1, channel=0
                ),
            ]
        )
        events = manager._build_events(seq)

        # At time 0.1: note_off for 60, then note_on for 62
        events_at_01 = [e for e in events if e.time == pytest.approx(0.1)]
        assert len(events_at_01) == 2
        assert events_at_01[0].event_type == "note_off"
        assert events_at_01[1].event_type == "note_on"


class TestExampleFilesPlayback:
    """Test playback of example .alda files."""

    @pytest.fixture
    def events_received(self):
        """Track events received by mock functions."""
        return {"note_on": [], "note_off": [], "program": [], "control": []}

    @pytest.fixture
    def manager(self, events_received):
        """Create a manager with mock send functions."""

        def send_note_on(ch, note, vel):
            events_received["note_on"].append((ch, note, vel))

        def send_note_off(ch, note):
            events_received["note_off"].append((ch, note))

        def send_program(ch, prog):
            events_received["program"].append((ch, prog))

        def send_control(ch, ctrl, val):
            events_received["control"].append((ch, ctrl, val))

        mgr = AsyncPlaybackManager(
            send_note_on=send_note_on,
            send_note_off=send_note_off,
            send_program_change=send_program,
            send_control_change=send_control,
        )
        yield mgr
        mgr.shutdown()

    def test_duet_alda_playback(self, manager, events_received):
        """Test playing duet.alda with two instruments (violin and cello)."""
        from pathlib import Path

        from aldakit import parse
        from aldakit.midi import generate_midi

        # Load and parse duet.alda
        duet_path = Path(__file__).parent.parent / "examples" / "duet.alda"
        source = duet_path.read_text(encoding="utf-8")
        ast = parse(source, str(duet_path))
        sequence = generate_midi(ast)

        # Verify sequence has notes from both instruments
        assert len(sequence.notes) > 0

        # Should have notes on two channels (violin=ch0, cello=ch1)
        channels_used = set(note.channel for note in sequence.notes)
        assert len(channels_used) == 2, f"Expected 2 channels, got {channels_used}"

        # Play the sequence
        slot_id = manager.play(sequence)
        assert slot_id is not None

        # Wait for playback to complete
        manager.wait()

        # Verify events were sent for both channels
        note_on_channels = set(ch for ch, _, _ in events_received["note_on"])
        assert len(note_on_channels) == 2, (
            f"Expected notes on 2 channels, got {note_on_channels}"
        )

        # Verify program changes for both instruments
        # Violin = program 40, Cello = program 42
        program_channels = set(ch for ch, _ in events_received["program"])
        assert len(program_channels) == 2, "Expected program changes on 2 channels"

        programs_sent = set(prog for _, prog in events_received["program"])
        assert 40 in programs_sent, "Expected violin program (40)"
        assert 42 in programs_sent, "Expected cello program (42)"

        # Verify all notes have matching note_off events
        assert len(events_received["note_on"]) == len(events_received["note_off"])
