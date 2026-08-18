"""Tests for handing a MIDI channel from one part to another over time.

A part only occupies a channel while it is sounding. Alda uses that to play
scores with far more parts than the 15 melodic channels the MIDI spec allows,
which is what ``all-instruments.alda`` (128 instruments) and
``midi-channel-management.alda`` (31 parts) are written to demonstrate.

What has to hold when a channel changes hands:

- the notes themselves do not move or change;
- no two parts sound on one channel at the same time;
- the part taking a channel over gets its own instrument, pan and volume,
  rather than inheriting the previous occupant's;
- a score that fits without reuse is laid out exactly as if this pass did not
  exist, so channel numbers stay predictable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aldakit import parse
from aldakit.analysis import ERROR, inspect_score, lint_score
from aldakit.constants import MIDI_CC_PAN, MIDI_CC_VOLUME, MIDI_DRUM_CHANNEL
from aldakit.midi.channels import (
    CONTROL_DEFAULTS,
    MELODIC_CHANNELS,
    VIRTUAL_CHANNEL_BASE,
    Run,
    _controls_at,
    _program_at,
    _run_at,
    is_virtual,
)
from aldakit.midi.types import MidiControlChange, MidiProgramChange
from aldakit.midi.generator import MidiGenerator

EXAMPLES = Path(__file__).parent.parent / "examples"

# Distinct instruments, so a part's program identifies it in the output.
INSTRUMENTS = [
    "midi-electric-piano-1",
    "violin",
    "viola",
    "cello",
    "flute",
    "oboe",
    "clarinet",
    "bassoon",
    "trumpet",
    "trombone",
    "tuba",
    "harp",
    "banjo",
    "sitar",
    "koto",
    "organ",
    "guitar",
    "midi-square-lead",
]


def generate(source: str):
    """Generate MIDI and return (generator, sequence)."""
    generator = MidiGenerator()
    return generator, generator.generate(parse(source))


def sequential(count: int) -> str:
    """A score where each part plays one whole note after the last has ended.

    At 120 BPM a whole note lasts two seconds and, at the default
    quantization, sounds for 1.8 of them, so no two parts overlap.
    """
    return "\n".join(f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(count))


def program_at(sequence, channel: int, time: float) -> int | None:
    """The program selected on ``channel`` at ``time``."""
    program = None
    for change in sorted(sequence.program_changes, key=lambda p: p.time):
        if change.channel == channel and change.time <= time + 1e-9:
            program = int(change.program)
    return program


def control_at(sequence, channel: int, control: int, time: float) -> int | None:
    """The value of ``control`` on ``channel`` at ``time``."""
    value = None
    for change in sorted(sequence.control_changes, key=lambda c: c.time):
        if (
            change.channel == channel
            and change.control == control
            and change.time <= time + 1e-9
        ):
            value = change.value
    return value


class TestReuseIsAFallback:
    """A score that fits is laid out exactly as it was before reuse existed."""

    def test_fifteen_parts_get_a_channel_each_in_declaration_order(self):
        generator, sequence = generate(sequential(15))
        by_time = sorted(sequence.notes, key=lambda n: n.start_time)
        assert [n.channel for n in by_time] == list(MELODIC_CHANNELS)
        assert generator.channel_assignment.reused is False

    def test_a_short_score_puts_every_program_change_at_the_start(self):
        _, sequence = generate(sequential(15))
        assert {p.time for p in sequence.program_changes} == {0.0}

    def test_one_part_is_still_channel_zero(self):
        _, sequence = generate("piano: c d e")
        assert {n.channel for n in sequence.notes} == {0}


class TestReuseWhenNeeded:
    def test_a_sixteenth_part_takes_the_channel_freed_first(self):
        generator, sequence = generate(sequential(16))
        by_time = sorted(sequence.notes, key=lambda n: n.start_time)
        assert [n.channel for n in by_time] == [*MELODIC_CHANNELS, 0]
        assert generator.channel_assignment.reused is True
        assert generator.channel_assignment.overflowed is False

    def test_reuse_is_not_an_error(self):
        generator, _ = generate(sequential(18))
        assert generator.diagnostics == []

    def test_every_channel_stays_within_the_midi_range(self):
        _, sequence = generate(sequential(18))
        channels = (
            {n.channel for n in sequence.notes}
            | {p.channel for p in sequence.program_changes}
            | {c.channel for c in sequence.control_changes}
        )
        assert all(0 <= channel <= 15 for channel in channels)
        assert not any(is_virtual(channel) for channel in channels)

    def test_no_two_parts_sound_on_one_channel_at_once(self):
        generator, _ = generate(sequential(18))
        assert generator.channel_assignment.conflicts == []

    def test_the_part_taking_a_channel_over_selects_its_own_instrument(self):
        _, sequence = generate(sequential(16))
        sixteenth = max(sequence.notes, key=lambda n: n.start_time)
        # The sixteenth part is "organ"; whatever program it has, the channel
        # it took over must be set to it and not to the part that had it.
        first = min(sequence.notes, key=lambda n: n.start_time)
        assert sixteenth.channel == first.channel
        assert program_at(sequence, sixteenth.channel, sixteenth.start_time) != (
            program_at(sequence, first.channel, first.start_time)
        )

    def test_a_part_keeps_its_channel_across_a_rest(self):
        """Reuse must not shuffle a part between channels for no reason."""
        source = "\n".join(
            [f"{INSTRUMENTS[0]}: c1 r1 c1"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(1, 16)]
        )
        generator, sequence = generate(source)
        assert generator.channel_assignment.channels[VIRTUAL_CHANNEL_BASE] == [0]
        first_part = [n for n in sequence.notes if n.pitch == 60 and n.channel == 0]
        assert len(first_part) == 2


class TestMovingBetweenChannels:
    """A part whose channel is gone when it returns continues on another."""

    # Fifteen parts sound at once and stop, so every channel has been used;
    # one more part then holds the channel freed first while the part that
    # had it is resting, and that part has to come back somewhere else.
    SOURCE = "\n".join(
        [f"{INSTRUMENTS[0]}: c4 r1*2 c4"]
        + [f"{INSTRUMENTS[i]}: c4" for i in range(1, 15)]
        + [f"{INSTRUMENTS[15]}: r2 c1*4"]
    )

    def test_the_part_continues_on_a_free_channel(self):
        generator, _ = generate(self.SOURCE)
        assignment = generator.channel_assignment
        assert assignment.channels[VIRTUAL_CHANNEL_BASE] == [0, 1]
        assert assignment.overflowed is False
        assert assignment.conflicts == []

    def test_the_instrument_is_selected_on_the_new_channel(self):
        _, sequence = generate(self.SOURCE)
        returning = max(
            (n for n in sequence.notes if n.channel == 1), key=lambda n: n.start_time
        )
        first = min(sequence.notes, key=lambda n: n.start_time)
        assert program_at(sequence, 1, returning.start_time) == program_at(
            sequence, first.channel, first.start_time
        )

    def test_inspection_lists_both_channels(self):
        info = inspect_score(self.SOURCE)
        moved = next(p for p in info.parts if p.name == INSTRUMENTS[0])
        assert moved.channels == (0, 1)
        assert moved.channel == 0


class TestStateFollowsThePart:
    """Pan, volume and instrument are restated when a channel changes hands."""

    def _score(self, first_part: str) -> str:
        # Sixteen parts, so channels have to be reused; the first part plays
        # again at the very end, by which time another part has had its
        # channel and left its own settings on it.
        return "\n".join(
            [f"{INSTRUMENTS[0]}: {first_part}"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(1, 16)]
        )

    def test_pan_is_restored_when_a_part_returns_to_its_channel(self):
        _, sequence = generate(self._score("(panning 100) c1 r1*20 c1"))
        first, last = (
            min(sequence.notes, key=lambda n: n.start_time),
            max(sequence.notes, key=lambda n: n.start_time),
        )
        assert first.channel == last.channel == 0
        assert control_at(sequence, 0, MIDI_CC_PAN, first.start_time) == 127
        assert control_at(sequence, 0, MIDI_CC_PAN, last.start_time) == 127

    def test_a_borrowed_channel_does_not_inherit_the_previous_pan(self):
        _, sequence = generate(self._score("(panning 100) c1 r1*20 c1"))
        borrower = [
            n for n in sequence.notes if n.channel == 0 and 29.0 < n.start_time < 31.0
        ]
        assert borrower, "expected another part to borrow channel 0"
        assert (
            control_at(sequence, 0, MIDI_CC_PAN, borrower[0].start_time)
            == (CONTROL_DEFAULTS[MIDI_CC_PAN])
        )

    def test_track_volume_follows_the_part(self):
        _, sequence = generate(self._score("(track-volume 20) c1 r1*20 c1"))
        last = max(sequence.notes, key=lambda n: n.start_time)
        expected = int(20 * 127 / 100)
        assert control_at(sequence, 0, MIDI_CC_VOLUME, last.start_time) == expected

    def test_a_borrowed_channel_does_not_inherit_the_previous_volume(self):
        _, sequence = generate(self._score("(track-volume 20) c1 r1*20 c1"))
        borrower = [
            n for n in sequence.notes if n.channel == 0 and 29.0 < n.start_time < 31.0
        ]
        assert (
            control_at(sequence, 0, MIDI_CC_VOLUME, borrower[0].start_time)
            == (CONTROL_DEFAULTS[MIDI_CC_VOLUME])
        )

    def test_the_instrument_is_reselected_on_return(self):
        _, sequence = generate(self._score("c1 r1*20 c1"))
        notes = sorted(
            (n for n in sequence.notes if n.channel == 0), key=lambda n: n.start_time
        )
        first, last = notes[0], notes[-1]
        assert program_at(sequence, 0, first.start_time) == program_at(
            sequence, 0, last.start_time
        )


class TestReservedChannels:
    def test_percussion_keeps_channel_ten_to_itself(self):
        source = "\n".join(
            ["midi-percussion: o2 c1 r1*20 c1"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(16)]
        )
        _, sequence = generate(source)
        drums = {n.pitch for n in sequence.notes if n.channel == MIDI_DRUM_CHANNEL}
        melodic = {n.pitch for n in sequence.notes if n.channel != MIDI_DRUM_CHANNEL}
        assert drums == {36}
        assert 36 not in melodic

    def test_a_pinned_channel_is_never_handed_to_another_part(self):
        source = "\n".join(
            [f"{INSTRUMENTS[0]}: (midi-channel 3) c1 r1*20 c1"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(1, 17)]
        )
        generator, sequence = generate(source)
        assert generator.channel_assignment.reused is True
        # Channel 3 carries only the pinned part: two notes, at each end.
        on_three = sorted(n.start_time for n in sequence.notes if n.channel == 3)
        assert len(on_three) == 2
        assert generator.channel_assignment.conflicts == []


class TestControlChangesWhileSounding:
    def test_a_pan_change_inside_a_legato_phrase_keeps_its_own_time(self):
        """Full quantization leaves no gap, so the change lands mid-run."""
        source = "\n".join(
            [f"{INSTRUMENTS[0]}: (quant 100) (panning 0) c4 (panning 100) c4 c4 c4"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(1, 16)]
        )
        generator, sequence = generate(source)
        assert generator.channel_assignment.reused is True
        # One unbroken run, so the part never leaves channel 0 mid-phrase.
        assert generator.channel_assignment.channels[VIRTUAL_CHANNEL_BASE] == [0]
        pans = sorted(
            (c.time, c.value)
            for c in sequence.control_changes
            if c.control == MIDI_CC_PAN and c.channel == 0
        )
        assert pans[0] == (0.0, 0)
        assert pans[1] == (0.5, 127)

    def test_a_pan_change_mid_phrase_keeps_its_own_time(self):
        source = "\n".join(
            [f"{INSTRUMENTS[0]}: (panning 0) c4 (panning 100) c4 c4 c4"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(1, 16)]
        )
        generator, sequence = generate(source)
        assert generator.channel_assignment.reused is True
        pans = sorted(
            (c.time, c.channel, c.value)
            for c in sequence.control_changes
            if c.control == MIDI_CC_PAN and c.channel == 0
        )
        # Set at the start, changed half a beat in, and reset to centre when
        # another part takes the channel over.
        assert pans[0] == (0.0, 0, 0)
        assert pans[1] == (0.5, 0, 127)
        assert pans[-1][2] == CONTROL_DEFAULTS[MIDI_CC_PAN]


class TestEveryChannelPinned:
    """Reuse still has to work when (midi-channel) claims all of them."""

    SOURCE = "\n".join(
        [
            f"{INSTRUMENTS[i]}: (midi-channel {channel}) c4"
            for i, channel in enumerate(MELODIC_CHANNELS)
        ]
        + [f"{INSTRUMENTS[15 + i]}: {'r1 ' * (4 + i)}c4" for i in range(3)]
    )

    def test_later_parts_get_channels_the_pinned_parts_have_finished_with(self):
        generator, sequence = generate(self.SOURCE)
        assignment = generator.channel_assignment
        assert assignment.reused is True
        assert assignment.overflowed is False
        assert assignment.conflicts == []
        assert all(0 <= n.channel <= 15 for n in sequence.notes)

    def test_no_diagnostic_is_reported(self):
        generator, _ = generate(self.SOURCE)
        assert generator.diagnostics == []


class TestOverflow:
    def test_parts_that_all_sound_at_once_still_overflow(self):
        source = "\n".join(f"{INSTRUMENTS[i]}: c1" for i in range(16))
        generator, sequence = generate(source)
        assignment = generator.channel_assignment
        assert assignment.overflowed is True
        assert assignment.max_concurrent == 16
        assert [d.code for d in generator.diagnostics] == ["channel-exhaustion"]
        # Even when it cannot fit, nothing lands outside the MIDI range.
        assert all(0 <= n.channel <= 15 for n in sequence.notes)

    def test_overflow_names_the_parts_that_collide(self):
        source = "\n".join(f"{INSTRUMENTS[i]}: c1" for i in range(16))
        findings = lint_score(source)
        codes = sorted({f.code for f in findings if f.severity == ERROR})
        assert codes == ["channel-exhaustion", "shared-channel", "too-many-parts"]
        shared = next(f for f in findings if f.code == "shared-channel")
        assert "play at the same time" in shared.message


class TestSilentParts:
    def test_a_part_that_never_sounds_leaves_no_program_change(self):
        source = "\n".join(
            [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(16)] + ["tuba:"]
        )
        _, sequence = generate(source)
        # Sixteen sounding parts, so sixteen instrument selections; the
        # declared-but-silent part adds none.
        assert len(sequence.program_changes) == 16

    def test_a_silent_part_reports_no_channel(self):
        source = "\n".join(
            [f"{INSTRUMENTS[i]}: {'r1 ' * i}c1" for i in range(16)] + ["midi-tuba:"]
        )
        info = inspect_score(source)
        silent = next(p for p in info.parts if p.name == "midi-tuba")
        assert silent.channel == -1
        assert silent.note_count == 0


class TestInspection:
    def test_note_counts_survive_a_shared_channel(self):
        """Two parts on one channel must not have their notes pooled."""
        source = "\n".join(
            [f"{INSTRUMENTS[0]}: c1 d1"]
            + [f"{INSTRUMENTS[i]}: {'r1 ' * (i + 1)}c1" for i in range(1, 16)]
        )
        info = inspect_score(source)
        first = next(p for p in info.parts if p.name == INSTRUMENTS[0])
        last = next(p for p in info.parts if p.name == INSTRUMENTS[15])
        assert first.channel == last.channel  # the channel was handed on
        assert first.note_count == 2
        assert last.note_count == 1

    def test_reported_channels_are_real_channels(self):
        info = inspect_score(sequential(18))
        assert all(0 <= p.channel <= 15 for p in info.parts)
        assert all(len(p.channels) >= 1 for p in info.parts)


class TestExamples:
    """The two examples written to exercise this."""

    @pytest.mark.parametrize(
        "name", ["all-instruments.alda", "midi-channel-management.alda"]
    )
    def test_example_fits_in_the_available_channels(self, name):
        source = (EXAMPLES / name).read_text(encoding="utf-8")
        generator, sequence = generate(source)
        assignment = generator.channel_assignment
        assert assignment.reused is True
        assert assignment.overflowed is False
        assert assignment.conflicts == []
        assert assignment.max_concurrent <= len(MELODIC_CHANNELS)
        assert all(0 <= n.channel <= 15 for n in sequence.notes)

    @pytest.mark.parametrize(
        "name", ["all-instruments.alda", "midi-channel-management.alda"]
    )
    def test_example_lints_clean(self, name):
        source = (EXAMPLES / name).read_text(encoding="utf-8")
        assert [f for f in lint_score(source, name) if f.severity == ERROR] == []

    def test_all_instruments_plays_every_program(self):
        source = (EXAMPLES / "all-instruments.alda").read_text(encoding="utf-8")
        _, sequence = generate(source)
        assert {int(p.program) for p in sequence.program_changes} == set(range(128))

    def test_no_example_leaks_a_placeholder_channel(self):
        for path in sorted(EXAMPLES.glob("*.alda")):
            _, sequence = generate(path.read_text(encoding="utf-8"))
            channels = (
                {n.channel for n in sequence.notes}
                | {p.channel for p in sequence.program_changes}
                | {c.channel for c in sequence.control_changes}
            )
            assert all(0 <= channel <= 15 for channel in channels), path.name


class TestHelpers:
    """Boundaries of the lookups the rewrite depends on."""

    def test_no_runs_covers_no_time(self):
        assert _run_at([], 0.0) is None

    def test_a_time_before_the_first_run_uses_it(self):
        runs = [Run(owner=16, start=4.0, end=5.0)]
        assert _run_at(runs, 0.0) is runs[0]

    def test_a_time_in_a_gap_belongs_to_the_run_before_it(self):
        runs = [Run(owner=16, start=0.0, end=1.0), Run(owner=16, start=4.0, end=5.0)]
        assert _run_at(runs, 2.0) is runs[0]

    def test_a_part_with_no_program_change_has_no_program(self):
        assert _program_at([], 0.0) is None

    def test_a_program_set_later_is_not_in_effect_yet(self):
        programs = [
            MidiProgramChange(program=1, time=0.0, channel=16),
            MidiProgramChange(program=2, time=5.0, channel=16),
        ]
        assert _program_at(programs, 1.0) == 1
        assert _program_at(programs, 5.0) == 2

    def test_a_program_first_set_after_the_time_asked_for_is_used_anyway(self):
        """A part pinned to a channel selects its instrument on arrival."""
        programs = [MidiProgramChange(program=7, time=5.0, channel=16)]
        assert _program_at(programs, 0.0) == 7

    def test_controls_set_later_are_not_in_effect_yet(self):
        controls = [
            MidiControlChange(control=MIDI_CC_PAN, value=10, time=0.0, channel=16),
            MidiControlChange(control=MIDI_CC_PAN, value=90, time=5.0, channel=16),
            MidiControlChange(control=MIDI_CC_VOLUME, value=50, time=5.0, channel=16),
        ]
        assert _controls_at(controls, 1.0) == {MIDI_CC_PAN: 10}
        assert _controls_at(controls, 5.0) == {MIDI_CC_PAN: 90, MIDI_CC_VOLUME: 50}
