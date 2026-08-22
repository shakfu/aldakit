"""Tests for the AST-to-Alda serializer.

Regression cover for D7: the previous ``_ast_to_alda`` helper raised TypeError
on cram expressions and variable definitions, and silently dropped repeats,
voices, markers, barlines, ties and ms/s durations.

The core property is that a round trip preserves musical meaning:
``parse(write(parse(src)))`` must generate the same MIDI as ``parse(src)``.
"""

import inspect
from pathlib import Path

import pytest

from aldakit import ast_nodes, generate_midi, parse
from aldakit.ast_nodes import ASTNode, NoteNode, RootNode
from aldakit.serialize import AldaWriter, write_alda

EXAMPLES = Path(__file__).parent.parent / "examples"
SHARED_SUITE = Path(__file__).parent / "shared_suite"


def midi_signature(sequence):
    """A comparable summary of everything a sequence sounds like."""
    return (
        sorted(
            (
                round(n.start_time, 6),
                n.pitch,
                n.channel,
                round(n.duration, 6),
                n.velocity,
            )
            for n in sequence.notes
        ),
        sorted(
            (round(p.time, 6), p.channel, int(p.program))
            for p in sequence.program_changes
        ),
        sorted(
            (round(c.time, 6), c.channel, c.control, c.value)
            for c in sequence.control_changes
        ),
        sorted((round(t.time, 6), round(t.bpm, 6)) for t in sequence.tempo_changes),
    )


def assert_round_trips(source: str, name: str = "<test>"):
    """Parsing the serialized form must produce identical MIDI."""
    original_ast = parse(source, name)
    written = write_alda(original_ast)
    reparsed_ast = parse(written, f"{name}:roundtrip")

    expected = midi_signature(generate_midi(original_ast))
    actual = midi_signature(generate_midi(reparsed_ast))
    assert actual == expected, f"round trip changed the music.\nWritten:\n{written}"
    return written


class TestWriterCoverage:
    """Every AST node the parser can produce must be renderable."""

    def test_all_ast_nodes_have_a_visit_method(self):
        node_classes = [
            obj
            for _, obj in inspect.getmembers(ast_nodes, inspect.isclass)
            if issubclass(obj, ASTNode)
            and obj is not ASTNode
            and not inspect.isabstract(obj)
            # Abstract intermediates carry no data of their own
            and obj.__name__ not in {"DurationComponentNode", "LispNode"}
        ]
        assert node_classes, "no AST node classes discovered"

        missing = [
            cls.__name__
            for cls in node_classes
            if not hasattr(AldaWriter, f"visit_{cls.__name__}")
        ]
        assert missing == [], f"AldaWriter is missing visit methods for: {missing}"

    def test_unknown_node_raises_rather_than_dropping(self):
        class MysteryNode(ASTNode):
            position = None

            def accept(self, visitor):
                return visitor.visit(self)

            def _repr_helper(self, indent):
                return "MysteryNode()"

        with pytest.raises(TypeError, match="cannot serialize MysteryNode"):
            write_alda(MysteryNode())

    def test_visit_returning_non_string_is_caught(self):
        class BadWriter(AldaWriter):
            def visit_NoteNode(self, node):
                return 42

        with pytest.raises(TypeError, match="expected str"):
            BadWriter().write(NoteNode(letter="c"))


class TestConstructsThatUsedToBreak:
    """Each of these either crashed or was silently dropped before."""

    def test_cram_does_not_raise(self):
        written = assert_round_trips("piano: {c d e}2")
        assert "{" in written and "}" in written

    def test_variable_definition_does_not_raise(self):
        written = assert_round_trips("m = c d\npiano: m")
        assert "m = c d" in written

    def test_repeat_is_preserved(self):
        written = assert_round_trips("piano: [c d e]*4")
        assert "*4" in written

    def test_voices_are_preserved(self):
        written = assert_round_trips("piano: V1: c d V2: e f V0:")
        assert "V1:" in written and "V2:" in written and "V0:" in written

    def test_markers_are_preserved(self):
        written = assert_round_trips("piano: %a c d @a e")
        assert "%a" in written and "@a" in written

    def test_barline_is_preserved(self):
        written = assert_round_trips("piano: c | d")
        assert "|" in written

    def test_slur_is_preserved(self):
        written = assert_round_trips("piano: c~d e")
        assert "~" in written

    def test_millisecond_duration_is_preserved(self):
        written = assert_round_trips("piano: c500ms d")
        assert "500ms" in written

    def test_second_duration_is_preserved(self):
        written = assert_round_trips("piano: c2s d")
        assert "2s" in written

    def test_tied_duration_is_preserved(self):
        written = assert_round_trips("piano: c1~1")
        assert "1~1" in written

    def test_alternate_endings_are_preserved(self):
        written = assert_round_trips("piano: [c'1 d'2]*2")
        assert "'1" in written and "'2" in written


class TestFormatting:
    @pytest.mark.parametrize(
        "source,expected_fragment",
        [
            ("piano: c d e", "piano:"),
            ('violin/viola "strings": c', 'violin/viola "strings":'),
            ("piano: c4.", "c4."),
            ("piano: c4..", "c4.."),
            ("piano: o5 c", "o5"),
            ("piano: > c", ">"),
            ("piano: < c", "<"),
            ("piano: r4", "r4"),
            ("piano: c+ d- e_", "c+"),
            ("piano: (tempo 120) c", "(tempo 120)"),
            ("piano: (key-signature '(g minor)) c", "'(g minor)"),
            ('piano: (key-sig "f+ c+") c', '"f+ c+"'),
        ],
    )
    def test_fragment_present(self, source, expected_fragment):
        written = assert_round_trips(source)
        assert expected_fragment in written

    def test_whole_number_durations_have_no_decimal_point(self):
        written = write_alda(parse("piano: c4"))
        assert "c4" in written
        assert "4.0" not in written

    def test_chord_octave_changes_attach_to_their_note(self):
        written = assert_round_trips("piano: c/>e/o6 g")
        # Not "c/>/e" - the octave belongs to the note that follows it
        assert "/>e" in written or "/> e" in written

    def test_empty_score_is_empty(self):
        assert write_alda(RootNode(children=[])) == ""


class TestRoundTripCorpus:
    """The full example corpus must survive a round trip unchanged."""

    @pytest.mark.parametrize(
        "path", sorted(EXAMPLES.glob("*.alda")), ids=lambda p: p.name
    )
    def test_example_round_trips(self, path):
        assert_round_trips(path.read_text(encoding="utf-8"), path.name)

    @pytest.mark.parametrize(
        "path", sorted(SHARED_SUITE.glob("*.alda")), ids=lambda p: p.name
    )
    def test_shared_suite_round_trips(self, path):
        assert_round_trips(path.read_text(encoding="utf-8"), path.name)

    @pytest.mark.parametrize(
        "path", sorted(EXAMPLES.glob("*.alda")), ids=lambda p: p.name
    )
    def test_round_trip_is_idempotent(self, path):
        """Writing twice produces the same text as writing once."""
        once = write_alda(parse(path.read_text(encoding="utf-8"), path.name))
        twice = write_alda(parse(once, path.name))
        assert twice == once


class TestScoreIntegration:
    def test_score_save_alda_uses_the_writer(self, tmp_path):
        from aldakit import Score

        out = tmp_path / "out.alda"
        Score("piano: {c d e}2").save(out)
        assert out.read_text(encoding="utf-8") == "piano: {c d e}2"

    def test_deprecated_helper_still_works(self):
        from aldakit.score import _ast_to_alda

        assert _ast_to_alda(parse("piano: c")) == write_alda(parse("piano: c"))


class TestTiesAcrossBarlines:
    """A tie that crossed a barline must survive being written back out."""

    @pytest.mark.parametrize(
        "source",
        [
            "piano: a-8~|2.",
            "piano: c4~|1",
            "piano: c4 | ~2",
            "piano: c1~|1~|1",
            "piano: d4.~4~|\n\n  |~4.~8",
        ],
    )
    def test_round_trips(self, source):
        assert_round_trips(source)

    def test_the_barline_is_written_back(self):
        written = write_alda(parse("piano: c4 | ~2"))
        assert "~|" in written, (
            f"the barline the tie crossed was dropped: {written!r}"
        )
