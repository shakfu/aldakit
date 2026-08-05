"""Tests that an imported MIDI file plays back the way it was imported.

Regression cover for D5: ``midi_to_ast`` emitted a ``PartDeclarationNode`` and
an ``EventSequenceNode`` as siblings, but the generator had no branch for a bare
declaration. Declarations were therefore ignored, so every track collapsed onto
channel 0 with program 0 and the parts played one after another instead of
together. ``to_alda()`` looked correct, which is why the existing tests passed.
"""

from __future__ import annotations

import pytest

from aldakit import Score
from aldakit.ast_nodes import EventSequenceNode, PartDeclarationNode, PartNode
from aldakit.constants import MIDI_DRUM_CHANNEL
from aldakit.midi.midi_to_ast import midi_to_ast
from aldakit.midi.smf import write_midi_file
from aldakit.midi.types import (
    MidiNote,
    MidiProgramChange,
    MidiSequence,
    canonical_name,
)


def three_part_sequence() -> MidiSequence:
    """Piano, cello and drums, all sounding together at t=0."""
    return MidiSequence(
        notes=[
            MidiNote(pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0),
            MidiNote(pitch=48, velocity=80, start_time=0.0, duration=0.5, channel=1),
            MidiNote(pitch=36, velocity=100, start_time=0.0, duration=0.25, channel=9),
        ],
        program_changes=[
            MidiProgramChange(program=0, time=0.0, channel=0),
            MidiProgramChange(program=42, time=0.0, channel=1),
        ],
    )


@pytest.fixture
def midi_file(tmp_path):
    path = tmp_path / "three_parts.mid"
    write_midi_file(three_part_sequence(), path)
    return path


class TestImportedAstShape:
    def test_parts_are_wrapped_in_part_nodes(self):
        """A bare PartDeclarationNode is not an executable event."""
        ast = midi_to_ast(three_part_sequence())
        parts = [c for c in ast.children if isinstance(c, PartNode)]
        assert len(parts) == 3

    def test_no_orphan_declarations(self):
        ast = midi_to_ast(three_part_sequence())
        orphans = [c for c in ast.children if isinstance(c, PartDeclarationNode)]
        assert orphans == []

    def test_no_orphan_event_sequences(self):
        """Events must live inside their part, not beside it."""
        ast = midi_to_ast(three_part_sequence())
        orphans = [c for c in ast.children if isinstance(c, EventSequenceNode)]
        assert orphans == []

    def test_each_part_has_events(self):
        ast = midi_to_ast(three_part_sequence())
        for part in (c for c in ast.children if isinstance(c, PartNode)):
            assert part.events.events, f"{part.declaration.names} has no events"

    def test_instrument_names_are_canonical(self):
        ast = midi_to_ast(three_part_sequence())
        names = [
            c.declaration.names[0] for c in ast.children if isinstance(c, PartNode)
        ]
        assert names == [canonical_name(0), canonical_name(42), "midi-percussion"]

    def test_drum_channel_becomes_percussion_part(self):
        ast = midi_to_ast(three_part_sequence())
        names = [
            c.declaration.names[0] for c in ast.children if isinstance(c, PartNode)
        ]
        assert "midi-percussion" in names


class TestImportedPlayback:
    def test_channels_are_preserved(self, midi_file):
        score = Score.from_midi_file(midi_file)
        channels = {n.channel for n in score.midi.notes}
        assert channels == {0, 1, MIDI_DRUM_CHANNEL}

    def test_programs_are_preserved(self, midi_file):
        score = Score.from_midi_file(midi_file)
        programs = {(p.channel, int(p.program)) for p in score.midi.program_changes}
        assert programs == {(0, 0), (1, 42)}

    def test_parts_play_together_not_sequentially(self, midi_file):
        """All three notes started at t=0 and must still do so."""
        score = Score.from_midi_file(midi_file)
        assert {round(n.start_time, 6) for n in score.midi.notes} == {0.0}

    def test_note_count_is_preserved(self, midi_file):
        score = Score.from_midi_file(midi_file)
        assert len(score.midi.notes) == 3

    def test_pitches_are_preserved(self, midi_file):
        score = Score.from_midi_file(midi_file)
        assert sorted(n.pitch for n in score.midi.notes) == [36, 48, 60]

    def test_percussion_is_on_the_drum_channel(self, midi_file):
        score = Score.from_midi_file(midi_file)
        drum_notes = [n for n in score.midi.notes if n.pitch == 36]
        assert drum_notes and all(n.channel == MIDI_DRUM_CHANNEL for n in drum_notes)


class TestImportExportRoundTrip:
    def test_alda_export_reparses_to_the_same_midi(self, midi_file):
        """to_alda() and play() must agree with each other."""
        from aldakit import generate_midi, parse

        score = Score.from_midi_file(midi_file)
        reparsed = generate_midi(parse(score.to_alda()))

        def fingerprint(seq):
            return sorted(
                (round(n.start_time, 6), n.pitch, n.channel) for n in seq.notes
            )

        assert fingerprint(reparsed) == fingerprint(score.midi)

    def test_save_and_reimport_preserves_structure(self, midi_file, tmp_path):
        score = Score.from_midi_file(midi_file)
        out = tmp_path / "resaved.mid"
        score.save(out)

        reimported = Score.from_midi_file(out)
        assert {n.channel for n in reimported.midi.notes} == {
            n.channel for n in score.midi.notes
        }
        assert sorted(n.pitch for n in reimported.midi.notes) == sorted(
            n.pitch for n in score.midi.notes
        )

    def test_single_channel_file_still_works(self, tmp_path):
        path = tmp_path / "one.mid"
        write_midi_file(
            MidiSequence(
                notes=[
                    MidiNote(
                        pitch=60, velocity=80, start_time=0.0, duration=0.5, channel=0
                    ),
                    MidiNote(
                        pitch=62, velocity=80, start_time=0.5, duration=0.5, channel=0
                    ),
                ],
                program_changes=[MidiProgramChange(program=73, time=0.0, channel=0)],
            ),
            path,
        )
        score = Score.from_midi_file(path)
        assert [n.pitch for n in score.midi.notes] == [60, 62]
        assert [int(p.program) for p in score.midi.program_changes] == [73]
