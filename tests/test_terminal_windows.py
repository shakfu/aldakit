"""Tests for the Windows console adapter.

The kernel32 and msvcrt dependencies are injected, so the translation rules
and mode arithmetic are checked on any platform. What cannot be checked here
is the behaviour of a real console.
"""

from __future__ import annotations

import pytest

from aldakit.terminal.keys import KeyDecoder, KeyKind
from aldakit.terminal.windows import (
    ENABLE_ECHO_INPUT,
    ENABLE_LINE_INPUT,
    ENABLE_PROCESSED_INPUT,
    ENABLE_VIRTUAL_TERMINAL_INPUT,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING,
    EXTENDED_KEYS,
    STD_INPUT_HANDLE,
    STD_OUTPUT_HANDLE,
    ConsoleKeySource,
    ConsoleModes,
    ConsoleUnavailable,
)


class FakeKernel32:
    """A kernel32 stand-in recording the modes it is asked to set."""

    def __init__(self, initial=None, reject=()):
        self.modes = dict(initial or {STD_INPUT_HANDLE: 0x1F7, STD_OUTPUT_HANDLE: 0x3})
        self.reject = set(reject)
        self.set_calls: list[tuple[int, int]] = []
        self.readable = True

    def GetStdHandle(self, handle_id):  # noqa: N802 -- mirrors the Win32 name
        return handle_id

    def GetConsoleMode(self, handle, mode_ref):  # noqa: N802
        if not self.readable:
            return 0
        mode_ref._obj.value = self.modes[handle]
        return 1

    def SetConsoleMode(self, handle, mode):  # noqa: N802
        self.set_calls.append((handle, mode))
        if mode in self.reject:
            return 0
        self.modes[handle] = mode
        return 1


class FakeMsvcrt:
    """An msvcrt stand-in yielding scripted characters."""

    def __init__(self, script=""):
        self.script = list(script)

    def kbhit(self):
        return bool(self.script)

    def getwch(self):
        return self.script.pop(0) if self.script else ""


class TestConsoleModes:
    def test_raw_mode_clears_line_echo_and_processed_input(self):
        kernel32 = FakeKernel32()
        modes = ConsoleModes(kernel32)
        modes.apply()
        applied = kernel32.modes[STD_INPUT_HANDLE]
        assert not applied & ENABLE_LINE_INPUT
        assert not applied & ENABLE_ECHO_INPUT
        # Ctrl+C must arrive as a byte, not as a signal.
        assert not applied & ENABLE_PROCESSED_INPUT

    def test_escape_sequences_are_enabled_for_output(self):
        kernel32 = FakeKernel32()
        ConsoleModes(kernel32).apply()
        assert kernel32.modes[STD_OUTPUT_HANDLE] & ENABLE_VIRTUAL_TERMINAL_PROCESSING

    def test_virtual_terminal_input_is_requested(self):
        kernel32 = FakeKernel32()
        modes = ConsoleModes(kernel32)
        modes.apply()
        assert modes.virtual_terminal_input is True
        assert kernel32.modes[STD_INPUT_HANDLE] & ENABLE_VIRTUAL_TERMINAL_INPUT

    def test_an_older_console_falls_back_to_plain_raw_mode(self):
        """A console that rejects VT input still gets raw mode."""
        kernel32 = FakeKernel32()
        rejected = (0x1F7 & ~(
            ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
        )) | ENABLE_VIRTUAL_TERMINAL_INPUT
        kernel32.reject = {rejected}
        modes = ConsoleModes(kernel32)
        modes.apply()
        assert modes.virtual_terminal_input is False
        assert not kernel32.modes[STD_INPUT_HANDLE] & ENABLE_LINE_INPUT

    def test_restore_puts_the_original_modes_back(self):
        kernel32 = FakeKernel32()
        original = dict(kernel32.modes)
        modes = ConsoleModes(kernel32)
        modes.apply()
        assert kernel32.modes != original
        modes.restore()
        assert kernel32.modes == original
        assert modes.active is False

    def test_restore_without_apply_is_harmless(self):
        kernel32 = FakeKernel32()
        ConsoleModes(kernel32).restore()
        assert kernel32.set_calls == []

    def test_apply_is_idempotent(self):
        kernel32 = FakeKernel32()
        modes = ConsoleModes(kernel32)
        modes.apply()
        calls = len(kernel32.set_calls)
        modes.apply()
        assert len(kernel32.set_calls) == calls

    def test_an_unreadable_console_is_reported(self):
        kernel32 = FakeKernel32()
        kernel32.readable = False
        with pytest.raises(ConsoleUnavailable):
            ConsoleModes(kernel32).apply()

    def test_no_kernel32_is_reported(self):
        with pytest.raises(ConsoleUnavailable):
            ConsoleModes(kernel32=None).apply()


class TestConsoleKeySource:
    def read_all(self, source) -> bytes:
        chunks = []
        while True:
            chunk = source.read(1)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def test_typed_characters_become_utf8_bytes(self):
        source = ConsoleKeySource(FakeMsvcrt("ab"))
        assert self.read_all(source) == b"ab"

    def test_non_ascii_is_encoded_not_dropped(self):
        source = ConsoleKeySource(FakeMsvcrt("é"))
        assert self.read_all(source) == "é".encode("utf-8")

    def test_enter_arrives_as_carriage_return(self):
        source = ConsoleKeySource(FakeMsvcrt("\r"))
        assert self.read_all(source) == b"\r"

    @pytest.mark.parametrize(
        ("code", "kind"),
        [
            ("H", KeyKind.ARROW_UP),
            ("P", KeyKind.ARROW_DOWN),
            ("M", KeyKind.ARROW_RIGHT),
            ("K", KeyKind.ARROW_LEFT),
            ("G", KeyKind.HOME),
            ("O", KeyKind.END),
            ("S", KeyKind.DELETE),
            ("I", KeyKind.PAGE_UP),
            ("Q", KeyKind.PAGE_DOWN),
        ],
    )
    def test_extended_keys_decode_to_the_same_events_as_a_terminal(self, code, kind):
        """The adapter normalizes, so the shared decoder needs no Windows case."""
        for lead in ("\x00", "\xe0"):
            source = ConsoleKeySource(FakeMsvcrt(lead + code))
            events = KeyDecoder().feed(self.read_all(source))
            assert [item.kind for item in events] == [kind]

    def test_every_extended_key_maps_to_a_known_sequence(self):
        decoder = KeyDecoder()
        for sequence in EXTENDED_KEYS.values():
            events = decoder.feed(sequence)
            assert events and events[0].kind is not KeyKind.UNKNOWN

    def test_an_unknown_extended_key_is_dropped_not_inserted(self):
        source = ConsoleKeySource(FakeMsvcrt("\x00Z"))
        assert self.read_all(source) == b""

    def test_control_keys_arrive_as_bytes(self):
        source = ConsoleKeySource(FakeMsvcrt("\x03\x04"))
        events = KeyDecoder().feed(self.read_all(source))
        assert [item.kind for item in events] == [KeyKind.CTRL_C, KeyKind.CTRL_D]

    def test_ready_is_false_once_the_script_runs_out(self):
        source = ConsoleKeySource(FakeMsvcrt(""))
        assert source.ready(timeout=0.01) is False

    def test_ready_is_true_while_input_remains(self):
        source = ConsoleKeySource(FakeMsvcrt("a"))
        assert source.ready(timeout=0.01) is True

    def test_a_missing_msvcrt_reports_end_of_input(self):
        """Otherwise a caller would poll a source that can never deliver."""
        source = ConsoleKeySource(msvcrt_module=None)
        assert source.ready(timeout=None) is True
        assert source.read(1) == b""
