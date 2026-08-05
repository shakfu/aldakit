"""Tests for MIDI channel allocation and percussion routing.

Regression cover for:

- D2: ``midi-percussion`` was not routed to MIDI channel 9, and melodic parts
  were, so four example files played melodic lines as drum hits.
- D3: parts past the 16th all collapsed onto channel 15 and overwrote each
  other's program changes with no warning.
"""

from pathlib import Path

import pytest

from aldakit import parse
from aldakit.constants import MIDI_DRUM_CHANNEL
from aldakit.midi.generator import MELODIC_CHANNELS, MidiGenerator

EXAMPLES = Path(__file__).parent.parent / "examples"


def generate(source: str):
    """Generate MIDI and return (generator, sequence) so diagnostics are visible."""
    generator = MidiGenerator()
    return generator, generator.generate(parse(source))


class TestMelodicChannels:
    def test_drum_channel_excluded_from_melodic_pool(self):
        assert MIDI_DRUM_CHANNEL not in MELODIC_CHANNELS

    def test_fifteen_melodic_channels(self):
        assert len(MELODIC_CHANNELS) == 15

    def test_melodic_parts_never_use_channel_9(self):
        """Declaring more parts than channels must still avoid the drum channel."""
        names = [
            "piano",
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
        _, sequence = generate("\n".join(f"{n}: c" for n in names))
        used = {note.channel for note in sequence.notes}
        assert MIDI_DRUM_CHANNEL not in used

    def test_tenth_part_skips_the_drum_channel(self):
        """The part that would have landed on channel 9 goes to 10 instead."""
        names = [
            "piano",
            "violin",
            "viola",
            "cello",
            "flute",
            "oboe",
            "clarinet",
            "bassoon",
            "trumpet",
            "trombone",
        ]
        _, sequence = generate("\n".join(f"{n}: c" for n in names))
        channels = [
            note.channel for note in sorted(sequence.notes, key=lambda n: n.start_time)
        ]
        assert sorted(set(channels)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

    def test_implicit_part_uses_channel_zero(self):
        _, sequence = generate("c d e")
        assert {n.channel for n in sequence.notes} == {0}


class TestPercussion:
    def test_percussion_uses_drum_channel(self):
        _, sequence = generate("midi-percussion: o2 c8 d e f")
        assert {n.channel for n in sequence.notes} == {MIDI_DRUM_CHANNEL}

    def test_percussion_alias_uses_drum_channel(self):
        _, sequence = generate("percussion: o2 c8 d")
        assert {n.channel for n in sequence.notes} == {MIDI_DRUM_CHANNEL}

    def test_percussion_emits_no_program_change(self):
        """Note numbers select the drum, so a program change is meaningless."""
        _, sequence = generate("midi-percussion: o2 c8 d")
        assert sequence.program_changes == []

    def test_percussion_pitches_are_gm_drum_keys(self):
        """o2 c is MIDI 36 (bass drum 1); o2 d is 38 (acoustic snare)."""
        _, sequence = generate("midi-percussion: o2 c8 d")
        assert [n.pitch for n in sequence.notes] == [36, 38]

    def test_percussion_ignores_transposition(self):
        """Transposing a drum part would change which drums are played."""
        _, plain = generate("midi-percussion: o2 c8 d")
        _, shifted = generate("midi-percussion: (transpose 5) o2 c8 d")
        assert [n.pitch for n in plain.notes] == [n.pitch for n in shifted.notes]

    def test_percussion_ignores_key_signature(self):
        _, plain = generate("midi-percussion: o2 c8 b")
        _, keyed = generate('midi-percussion: (key-sig "b-") o2 c8 b')
        assert [n.pitch for n in plain.notes] == [n.pitch for n in keyed.notes]

    def test_melodic_part_still_transposes(self):
        """The percussion guard must not disable transposition generally."""
        _, plain = generate("piano: c")
        _, shifted = generate("piano: (transpose 5) c")
        assert shifted.notes[0].pitch == plain.notes[0].pitch + 5

    def test_percussion_coexists_with_melodic_parts(self):
        _, sequence = generate("piano: c\nmidi-percussion: o2 c")
        channels = {n.channel for n in sequence.notes}
        assert channels == {0, MIDI_DRUM_CHANNEL}


class TestChannelExhaustion:
    def test_no_diagnostic_within_limit(self):
        names = [f"midi-electric-piano-{i}" for i in (1, 2)] + ["piano"] * 0
        source = "\n".join(f"{n}: c" for n in names)
        generator, _ = generate(source)
        assert generator.diagnostics == []

    def test_diagnostic_when_channels_exhausted(self):
        names = [
            "piano",
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
        ]
        generator, _ = generate("\n".join(f"{n}: c" for n in names))
        messages = [str(d) for d in generator.diagnostics]
        assert any("channels are being reused" in m for m in messages)

    def test_exactly_fifteen_parts_is_silent(self):
        names = [
            "piano",
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
        ]
        generator, sequence = generate("\n".join(f"{n}: c" for n in names))
        assert generator.diagnostics == []
        assert len({n.channel for n in sequence.notes}) == 15


class TestDiagnostics:
    def test_unknown_instrument_reports_position(self):
        generator, _ = generate("bogus-instrument: c")
        assert len(generator.diagnostics) == 1
        message = str(generator.diagnostics[0])
        assert "Unknown instrument 'bogus-instrument'" in message
        assert ":1:1:" in message

    def test_unknown_instrument_falls_back_to_piano(self):
        _, sequence = generate("bogus-instrument: c")
        assert [int(pc.program) for pc in sequence.program_changes] == [0]

    def test_undefined_variable_reported(self):
        generator, _ = generate("piano: c nosuchvariable d")
        assert any("Undefined variable" in str(d) for d in generator.diagnostics)

    def test_undefined_marker_reported(self):
        generator, _ = generate("piano: c @nowhere d")
        assert any("Undefined marker" in str(d) for d in generator.diagnostics)

    def test_valid_score_has_no_diagnostics(self):
        generator, _ = generate("piano: c d e\nviolin: e f g")
        assert generator.diagnostics == []

    def test_score_exposes_diagnostics(self):
        from aldakit import Score

        score = Score("bogus-instrument: c d e")
        assert any("Unknown instrument" in str(d) for d in score.diagnostics)


def _example_files():
    return sorted(EXAMPLES.glob("*.alda"))


class TestExampleFilesRespectDrumChannel:
    """Examples used to route melodic parts to channel 9 and drums elsewhere."""

    @pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
    def test_drum_channel_holds_only_percussion(self, path):
        """Channel 9 is used by percussion parts and by nothing else."""
        generator, sequence = generate(path.read_text(encoding="utf-8"))

        percussion_channels = {
            state.channel
            for state in generator.state.parts.values()
            if state.percussion
        }
        melodic_channels = {
            state.channel
            for state in generator.state.parts.values()
            if not state.percussion
        }

        # Percussion, when present, is always on the drum channel
        assert percussion_channels in ({MIDI_DRUM_CHANNEL}, set())
        # No melodic part ever lands on the drum channel
        assert MIDI_DRUM_CHANNEL not in melodic_channels

        # ... and no stray note ends up there without a percussion part
        if MIDI_DRUM_CHANNEL in {n.channel for n in sequence.notes}:
            assert percussion_channels == {MIDI_DRUM_CHANNEL}

    @pytest.mark.parametrize(
        "filename",
        [
            "rachmaninoff_piano_concerto_2_mvmt_2.alda",
            "orchestra.alda",
            "debussy_quartet.alda",
        ],
    )
    def test_purely_melodic_example_avoids_drum_channel(self, filename):
        """Scores with no percussion part must not touch channel 9 at all."""
        _, sequence = generate((EXAMPLES / filename).read_text(encoding="utf-8"))
        assert MIDI_DRUM_CHANNEL not in {n.channel for n in sequence.notes}

    def test_percussion_example_uses_drum_channel(self):
        path = EXAMPLES / "percussion.alda"
        _, sequence = generate(path.read_text(encoding="utf-8"))
        assert {n.channel for n in sequence.notes} == {MIDI_DRUM_CHANNEL}
