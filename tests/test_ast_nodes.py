"""Tests for AST node definitions."""

from aldakit.ast_nodes import (
    ASTVisitor,
    RootNode,
    PartNode,
    PartDeclarationNode,
    EventSequenceNode,
    NoteNode,
    RestNode,
    ChordNode,
    DurationNode,
    NoteLengthNode,
    NoteLengthMsNode,
    NoteLengthSecondsNode,
    BarlineNode,
    OctaveSetNode,
    OctaveUpNode,
    OctaveDownNode,
    LispListNode,
    LispSymbolNode,
    LispNumberNode,
    LispStringNode,
    LispQuotedNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    MarkerNode,
    AtMarkerNode,
    VoiceNode,
    VoiceGroupNode,
    CramNode,
    RepeatNode,
    RepetitionRange,
    OnRepetitionsNode,
    BracketedSequenceNode,
)
from aldakit.tokens import SourcePosition


# =============================================================================
# ASTVisitor Tests
# =============================================================================


class TestASTVisitor:
    """Tests for ASTVisitor base class."""

    def test_visitor_dispatches_to_method(self):
        """Visitor dispatches to visit_NodeType method."""
        visited = []

        class TestVisitor(ASTVisitor):
            def visit_NoteNode(self, node):
                visited.append(("note", node.letter))
                return "visited_note"

            def visit_RestNode(self, node):
                visited.append(("rest",))
                return "visited_rest"

        visitor = TestVisitor()
        note = NoteNode(letter="c")
        rest = RestNode()

        result1 = visitor.visit(note)
        result2 = visitor.visit(rest)

        assert result1 == "visited_note"
        assert result2 == "visited_rest"
        assert visited == [("note", "c"), ("rest",)]

    def test_visitor_generic_visit_fallback(self):
        """Visitor falls back to generic_visit for unknown nodes."""
        visited = []

        class TestVisitor(ASTVisitor):
            def generic_visit(self, node):
                visited.append(type(node).__name__)
                return None

        visitor = TestVisitor()
        barline = BarlineNode()

        result = visitor.visit(barline)
        assert result is None
        assert visited == ["BarlineNode"]

    def test_visitor_generic_visit_default(self):
        """Default generic_visit returns None."""
        visitor = ASTVisitor()
        barline = BarlineNode()
        result = visitor.visit(barline)
        assert result is None


# =============================================================================
# RootNode Tests
# =============================================================================


class TestRootNode:
    """Tests for RootNode."""

    def test_repr_helper_empty(self):
        """Empty root node _repr_helper."""
        root = RootNode()
        repr_str = root._repr_helper(0)
        assert "RootNode" in repr_str

    def test_repr_helper_with_children(self):
        """Root node with children _repr_helper."""
        note = NoteNode(letter="c")
        rest = RestNode()
        root = RootNode(children=[note, rest])
        repr_str = root._repr_helper(0)
        assert "RootNode" in repr_str
        assert "NoteNode" in repr_str
        assert "RestNode" in repr_str

    def test_accept(self):
        """RootNode accepts visitor."""
        root = RootNode()

        class TestVisitor(ASTVisitor):
            def visit_RootNode(self, node):
                return "root_visited"

        visitor = TestVisitor()
        result = root.accept(visitor)
        assert result == "root_visited"


# =============================================================================
# PartNode Tests
# =============================================================================


class TestPartNode:
    """Tests for PartNode."""

    def test_repr_helper(self):
        """PartNode _repr_helper includes declaration and events."""
        decl = PartDeclarationNode(names=["piano"])
        events = EventSequenceNode(events=[NoteNode(letter="c")])
        part = PartNode(declaration=decl, events=events)
        repr_str = part._repr_helper(0)
        assert "PartNode" in repr_str
        assert "PartDeclarationNode" in repr_str
        assert "EventSequenceNode" in repr_str

    def test_accept(self):
        """PartNode accepts visitor."""
        decl = PartDeclarationNode(names=["piano"])
        events = EventSequenceNode()
        part = PartNode(declaration=decl, events=events)

        class TestVisitor(ASTVisitor):
            def visit_PartNode(self, node):
                return "part_visited"

        visitor = TestVisitor()
        result = part.accept(visitor)
        assert result == "part_visited"


# =============================================================================
# PartDeclarationNode Tests
# =============================================================================


class TestPartDeclarationNode:
    """Tests for PartDeclarationNode."""

    def test_repr_helper_without_alias(self):
        """PartDeclarationNode _repr_helper without alias."""
        decl = PartDeclarationNode(names=["piano", "violin"])
        repr_str = decl._repr_helper(0)
        assert "PartDeclarationNode" in repr_str
        assert "piano" in repr_str
        assert "violin" in repr_str

    def test_repr_helper_with_alias(self):
        """PartDeclarationNode _repr_helper with alias."""
        decl = PartDeclarationNode(names=["piano"], alias="p")
        repr_str = decl._repr_helper(0)
        assert "PartDeclarationNode" in repr_str
        assert "alias" in repr_str
        assert '"p"' in repr_str

    def test_accept(self):
        """PartDeclarationNode accepts visitor."""
        decl = PartDeclarationNode(names=["piano"])

        class TestVisitor(ASTVisitor):
            def visit_PartDeclarationNode(self, node):
                return "decl_visited"

        visitor = TestVisitor()
        result = decl.accept(visitor)
        assert result == "decl_visited"


# =============================================================================
# EventSequenceNode Tests
# =============================================================================


class TestEventSequenceNode:
    """Tests for EventSequenceNode."""

    def test_repr_helper_empty(self):
        """Empty EventSequenceNode _repr_helper."""
        seq = EventSequenceNode()
        repr_str = seq._repr_helper(0)
        assert repr_str == "EventSequenceNode()"

    def test_repr_helper_with_events(self):
        """EventSequenceNode with events _repr_helper."""
        seq = EventSequenceNode(events=[NoteNode(letter="c"), RestNode()])
        repr_str = seq._repr_helper(0)
        assert "EventSequenceNode" in repr_str
        assert "NoteNode" in repr_str
        assert "RestNode" in repr_str

    def test_accept(self):
        """EventSequenceNode accepts visitor."""
        seq = EventSequenceNode()

        class TestVisitor(ASTVisitor):
            def visit_EventSequenceNode(self, node):
                return "seq_visited"

        visitor = TestVisitor()
        result = seq.accept(visitor)
        assert result == "seq_visited"


# =============================================================================
# NoteNode Tests
# =============================================================================


class TestNoteNode:
    """Tests for NoteNode."""

    def test_repr_helper_basic(self):
        """Basic NoteNode _repr_helper."""
        note = NoteNode(letter="c")
        repr_str = note._repr_helper(0)
        assert "NoteNode" in repr_str
        assert "'c'" in repr_str

    def test_repr_helper_with_accidentals(self):
        """NoteNode with accidentals _repr_helper."""
        note = NoteNode(letter="c", accidentals=["+", "+"])
        repr_str = note._repr_helper(0)
        assert "accidentals" in repr_str
        assert "+" in repr_str

    def test_repr_helper_with_duration(self):
        """NoteNode with duration _repr_helper."""
        note = NoteNode(
            letter="c",
            duration=DurationNode(components=[NoteLengthNode(denominator=4)]),
        )
        repr_str = note._repr_helper(0)
        assert "duration" in repr_str
        assert "DurationNode" in repr_str

    def test_repr_helper_slurred(self):
        """NoteNode slurred _repr_helper."""
        note = NoteNode(letter="c", slurred=True)
        repr_str = note._repr_helper(0)
        assert "slurred=True" in repr_str

    def test_accept(self):
        """NoteNode accepts visitor."""
        note = NoteNode(letter="c")

        class TestVisitor(ASTVisitor):
            def visit_NoteNode(self, node):
                return "note_visited"

        visitor = TestVisitor()
        result = note.accept(visitor)
        assert result == "note_visited"


# =============================================================================
# RestNode Tests
# =============================================================================


class TestRestNode:
    """Tests for RestNode."""

    def test_repr_helper_basic(self):
        """Basic RestNode _repr_helper."""
        rest = RestNode()
        repr_str = rest._repr_helper(0)
        assert repr_str == "RestNode()"

    def test_repr_helper_with_duration(self):
        """RestNode with duration _repr_helper."""
        rest = RestNode(
            duration=DurationNode(components=[NoteLengthNode(denominator=4)])
        )
        repr_str = rest._repr_helper(0)
        assert "duration" in repr_str

    def test_accept(self):
        """RestNode accepts visitor."""
        rest = RestNode()

        class TestVisitor(ASTVisitor):
            def visit_RestNode(self, node):
                return "rest_visited"

        visitor = TestVisitor()
        result = rest.accept(visitor)
        assert result == "rest_visited"


# =============================================================================
# ChordNode Tests
# =============================================================================


class TestChordNode:
    """Tests for ChordNode."""

    def test_repr_helper(self):
        """ChordNode _repr_helper includes notes."""
        chord = ChordNode(
            notes=[NoteNode(letter="c"), NoteNode(letter="e"), NoteNode(letter="g")]
        )
        repr_str = chord._repr_helper(0)
        assert "ChordNode" in repr_str
        assert "NoteNode" in repr_str

    def test_accept(self):
        """ChordNode accepts visitor."""
        chord = ChordNode(notes=[NoteNode(letter="c")])

        class TestVisitor(ASTVisitor):
            def visit_ChordNode(self, node):
                return "chord_visited"

        visitor = TestVisitor()
        result = chord.accept(visitor)
        assert result == "chord_visited"


# =============================================================================
# Duration Node Tests
# =============================================================================


class TestDurationNode:
    """Tests for DurationNode."""

    def test_repr_helper_single_component(self):
        """DurationNode single component _repr_helper."""
        dur = DurationNode(components=[NoteLengthNode(denominator=4)])
        repr_str = dur._repr_helper(0)
        assert "DurationNode" in repr_str
        assert "NoteLengthNode(4)" in repr_str

    def test_repr_helper_multiple_components(self):
        """DurationNode multiple components _repr_helper."""
        dur = DurationNode(
            components=[
                NoteLengthNode(denominator=4),
                NoteLengthNode(denominator=8),
            ]
        )
        repr_str = dur._repr_helper(0)
        assert "DurationNode" in repr_str
        assert "NoteLengthNode(4)" in repr_str
        assert "NoteLengthNode(8)" in repr_str

    def test_accept(self):
        """DurationNode accepts visitor."""
        dur = DurationNode()

        class TestVisitor(ASTVisitor):
            def visit_DurationNode(self, node):
                return "dur_visited"

        visitor = TestVisitor()
        result = dur.accept(visitor)
        assert result == "dur_visited"


class TestNoteLengthNode:
    """Tests for NoteLengthNode."""

    def test_repr_helper_basic(self):
        """Basic NoteLengthNode _repr_helper."""
        length = NoteLengthNode(denominator=4)
        repr_str = length._repr_helper(0)
        assert repr_str == "NoteLengthNode(4)"

    def test_repr_helper_with_dots(self):
        """NoteLengthNode with dots _repr_helper."""
        length = NoteLengthNode(denominator=4, dots=2)
        repr_str = length._repr_helper(0)
        assert "NoteLengthNode(4, dots=2)" in repr_str

    def test_accept(self):
        """NoteLengthNode accepts visitor."""
        length = NoteLengthNode(denominator=4)

        class TestVisitor(ASTVisitor):
            def visit_NoteLengthNode(self, node):
                return "length_visited"

        visitor = TestVisitor()
        result = length.accept(visitor)
        assert result == "length_visited"


class TestNoteLengthMsNode:
    """Tests for NoteLengthMsNode."""

    def test_repr_helper(self):
        """NoteLengthMsNode _repr_helper."""
        length = NoteLengthMsNode(ms=500.0)
        repr_str = length._repr_helper(0)
        assert "NoteLengthMsNode(500.0ms)" in repr_str

    def test_accept(self):
        """NoteLengthMsNode accepts visitor."""
        length = NoteLengthMsNode(ms=500.0)

        class TestVisitor(ASTVisitor):
            def visit_NoteLengthMsNode(self, node):
                return "ms_visited"

        visitor = TestVisitor()
        result = length.accept(visitor)
        assert result == "ms_visited"


class TestNoteLengthSecondsNode:
    """Tests for NoteLengthSecondsNode."""

    def test_repr_helper(self):
        """NoteLengthSecondsNode _repr_helper."""
        length = NoteLengthSecondsNode(seconds=1.5)
        repr_str = length._repr_helper(0)
        assert "NoteLengthSecondsNode(1.5s)" in repr_str

    def test_accept(self):
        """NoteLengthSecondsNode accepts visitor."""
        length = NoteLengthSecondsNode(seconds=1.5)

        class TestVisitor(ASTVisitor):
            def visit_NoteLengthSecondsNode(self, node):
                return "seconds_visited"

        visitor = TestVisitor()
        result = length.accept(visitor)
        assert result == "seconds_visited"


# =============================================================================
# Octave Node Tests
# =============================================================================


class TestOctaveSetNode:
    """Tests for OctaveSetNode."""

    def test_repr_helper(self):
        """OctaveSetNode _repr_helper."""
        node = OctaveSetNode(octave=5)
        repr_str = node._repr_helper(0)
        assert "OctaveSetNode(5)" in repr_str

    def test_accept(self):
        """OctaveSetNode accepts visitor."""
        node = OctaveSetNode(octave=5)

        class TestVisitor(ASTVisitor):
            def visit_OctaveSetNode(self, node):
                return "octave_set_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "octave_set_visited"


class TestOctaveUpNode:
    """Tests for OctaveUpNode."""

    def test_repr_helper(self):
        """OctaveUpNode _repr_helper."""
        node = OctaveUpNode()
        repr_str = node._repr_helper(0)
        assert repr_str == "OctaveUpNode()"

    def test_accept(self):
        """OctaveUpNode accepts visitor."""
        node = OctaveUpNode()

        class TestVisitor(ASTVisitor):
            def visit_OctaveUpNode(self, node):
                return "octave_up_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "octave_up_visited"


class TestOctaveDownNode:
    """Tests for OctaveDownNode."""

    def test_repr_helper(self):
        """OctaveDownNode _repr_helper."""
        node = OctaveDownNode()
        repr_str = node._repr_helper(0)
        assert repr_str == "OctaveDownNode()"

    def test_accept(self):
        """OctaveDownNode accepts visitor."""
        node = OctaveDownNode()

        class TestVisitor(ASTVisitor):
            def visit_OctaveDownNode(self, node):
                return "octave_down_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "octave_down_visited"


class TestBarlineNode:
    """Tests for BarlineNode."""

    def test_repr_helper(self):
        """BarlineNode _repr_helper."""
        node = BarlineNode()
        repr_str = node._repr_helper(0)
        assert repr_str == "BarlineNode()"

    def test_accept(self):
        """BarlineNode accepts visitor."""
        node = BarlineNode()

        class TestVisitor(ASTVisitor):
            def visit_BarlineNode(self, node):
                return "barline_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "barline_visited"


# =============================================================================
# Lisp Node Tests
# =============================================================================


class TestLispListNode:
    """Tests for LispListNode."""

    def test_repr_helper_empty(self):
        """Empty LispListNode _repr_helper."""
        node = LispListNode()
        repr_str = node._repr_helper(0)
        assert repr_str == "LispListNode()"

    def test_repr_helper_with_elements(self):
        """LispListNode with elements _repr_helper."""
        node = LispListNode(
            elements=[
                LispSymbolNode(name="tempo"),
                LispNumberNode(value=120),
            ]
        )
        repr_str = node._repr_helper(0)
        assert "LispListNode" in repr_str
        assert "tempo" in repr_str
        assert "120" in repr_str

    def test_accept(self):
        """LispListNode accepts visitor."""
        node = LispListNode()

        class TestVisitor(ASTVisitor):
            def visit_LispListNode(self, node):
                return "lisp_list_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "lisp_list_visited"


class TestLispSymbolNode:
    """Tests for LispSymbolNode."""

    def test_repr_helper(self):
        """LispSymbolNode _repr_helper."""
        node = LispSymbolNode(name="tempo")
        repr_str = node._repr_helper(0)
        assert repr_str == "tempo"

    def test_accept(self):
        """LispSymbolNode accepts visitor."""
        node = LispSymbolNode(name="tempo")

        class TestVisitor(ASTVisitor):
            def visit_LispSymbolNode(self, node):
                return "symbol_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "symbol_visited"


class TestLispNumberNode:
    """Tests for LispNumberNode."""

    def test_repr_helper_integer(self):
        """LispNumberNode integer _repr_helper."""
        node = LispNumberNode(value=120)
        repr_str = node._repr_helper(0)
        assert repr_str == "120"

    def test_repr_helper_float(self):
        """LispNumberNode float _repr_helper."""
        node = LispNumberNode(value=120.5)
        repr_str = node._repr_helper(0)
        assert repr_str == "120.5"

    def test_accept(self):
        """LispNumberNode accepts visitor."""
        node = LispNumberNode(value=120)

        class TestVisitor(ASTVisitor):
            def visit_LispNumberNode(self, node):
                return "number_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "number_visited"


class TestLispStringNode:
    """Tests for LispStringNode."""

    def test_repr_helper(self):
        """LispStringNode _repr_helper."""
        node = LispStringNode(value="hello")
        repr_str = node._repr_helper(0)
        assert '"hello"' in repr_str

    def test_accept(self):
        """LispStringNode accepts visitor."""
        node = LispStringNode(value="hello")

        class TestVisitor(ASTVisitor):
            def visit_LispStringNode(self, node):
                return "string_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "string_visited"


class TestLispQuotedNode:
    """Tests for LispQuotedNode."""

    def test_repr_helper_symbol(self):
        """LispQuotedNode with symbol _repr_helper."""
        node = LispQuotedNode(value=LispSymbolNode(name="up"))
        repr_str = node._repr_helper(0)
        assert "'up" in repr_str

    def test_repr_helper_list(self):
        """LispQuotedNode with list _repr_helper."""
        node = LispQuotedNode(
            value=LispListNode(
                elements=[
                    LispSymbolNode(name="g"),
                    LispSymbolNode(name="minor"),
                ]
            )
        )
        repr_str = node._repr_helper(0)
        assert "'" in repr_str
        assert "LispListNode" in repr_str

    def test_accept(self):
        """LispQuotedNode accepts visitor."""
        node = LispQuotedNode(value=LispSymbolNode(name="up"))

        class TestVisitor(ASTVisitor):
            def visit_LispQuotedNode(self, node):
                return "quoted_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "quoted_visited"


# =============================================================================
# Variable Node Tests
# =============================================================================


class TestVariableDefinitionNode:
    """Tests for VariableDefinitionNode."""

    def test_repr_helper(self):
        """VariableDefinitionNode _repr_helper."""
        node = VariableDefinitionNode(
            name="motif",
            events=EventSequenceNode(events=[NoteNode(letter="c")]),
        )
        repr_str = node._repr_helper(0)
        assert "VariableDefinitionNode" in repr_str
        assert "'motif'" in repr_str
        assert "EventSequenceNode" in repr_str

    def test_accept(self):
        """VariableDefinitionNode accepts visitor."""
        node = VariableDefinitionNode(
            name="motif",
            events=EventSequenceNode(),
        )

        class TestVisitor(ASTVisitor):
            def visit_VariableDefinitionNode(self, node):
                return "var_def_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "var_def_visited"


class TestVariableReferenceNode:
    """Tests for VariableReferenceNode."""

    def test_repr_helper(self):
        """VariableReferenceNode _repr_helper."""
        node = VariableReferenceNode(name="motif")
        repr_str = node._repr_helper(0)
        assert "VariableReferenceNode" in repr_str
        assert "'motif'" in repr_str

    def test_accept(self):
        """VariableReferenceNode accepts visitor."""
        node = VariableReferenceNode(name="motif")

        class TestVisitor(ASTVisitor):
            def visit_VariableReferenceNode(self, node):
                return "var_ref_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "var_ref_visited"


# =============================================================================
# Marker Node Tests
# =============================================================================


class TestMarkerNode:
    """Tests for MarkerNode."""

    def test_repr_helper(self):
        """MarkerNode _repr_helper."""
        node = MarkerNode(name="verse")
        repr_str = node._repr_helper(0)
        assert "MarkerNode" in repr_str
        assert "'verse'" in repr_str

    def test_accept(self):
        """MarkerNode accepts visitor."""
        node = MarkerNode(name="verse")

        class TestVisitor(ASTVisitor):
            def visit_MarkerNode(self, node):
                return "marker_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "marker_visited"


class TestAtMarkerNode:
    """Tests for AtMarkerNode."""

    def test_repr_helper(self):
        """AtMarkerNode _repr_helper."""
        node = AtMarkerNode(name="verse")
        repr_str = node._repr_helper(0)
        assert "AtMarkerNode" in repr_str
        assert "'verse'" in repr_str

    def test_accept(self):
        """AtMarkerNode accepts visitor."""
        node = AtMarkerNode(name="verse")

        class TestVisitor(ASTVisitor):
            def visit_AtMarkerNode(self, node):
                return "at_marker_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "at_marker_visited"


# =============================================================================
# Voice Node Tests
# =============================================================================


class TestVoiceNode:
    """Tests for VoiceNode."""

    def test_repr_helper(self):
        """VoiceNode _repr_helper."""
        node = VoiceNode(
            number=1,
            events=EventSequenceNode(events=[NoteNode(letter="c")]),
        )
        repr_str = node._repr_helper(0)
        assert "VoiceNode" in repr_str
        assert "number=1" in repr_str
        assert "EventSequenceNode" in repr_str

    def test_accept(self):
        """VoiceNode accepts visitor."""
        node = VoiceNode(number=1, events=EventSequenceNode())

        class TestVisitor(ASTVisitor):
            def visit_VoiceNode(self, node):
                return "voice_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "voice_visited"


class TestVoiceGroupNode:
    """Tests for VoiceGroupNode."""

    def test_repr_helper(self):
        """VoiceGroupNode _repr_helper."""
        node = VoiceGroupNode(
            voices=[
                VoiceNode(number=1, events=EventSequenceNode()),
                VoiceNode(number=2, events=EventSequenceNode()),
            ]
        )
        repr_str = node._repr_helper(0)
        assert "VoiceGroupNode" in repr_str
        assert "VoiceNode" in repr_str

    def test_accept(self):
        """VoiceGroupNode accepts visitor."""
        node = VoiceGroupNode()

        class TestVisitor(ASTVisitor):
            def visit_VoiceGroupNode(self, node):
                return "voice_group_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "voice_group_visited"


# =============================================================================
# Cram Node Tests
# =============================================================================


class TestCramNode:
    """Tests for CramNode."""

    def test_repr_helper_without_duration(self):
        """CramNode without duration _repr_helper."""
        node = CramNode(
            events=EventSequenceNode(
                events=[NoteNode(letter="c"), NoteNode(letter="d")]
            )
        )
        repr_str = node._repr_helper(0)
        assert "CramNode" in repr_str
        assert "EventSequenceNode" in repr_str

    def test_repr_helper_with_duration(self):
        """CramNode with duration _repr_helper."""
        node = CramNode(
            events=EventSequenceNode(events=[NoteNode(letter="c")]),
            duration=DurationNode(components=[NoteLengthNode(denominator=4)]),
        )
        repr_str = node._repr_helper(0)
        assert "CramNode" in repr_str
        assert "duration" in repr_str

    def test_accept(self):
        """CramNode accepts visitor."""
        node = CramNode(events=EventSequenceNode())

        class TestVisitor(ASTVisitor):
            def visit_CramNode(self, node):
                return "cram_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "cram_visited"


# =============================================================================
# Repeat Node Tests
# =============================================================================


class TestRepeatNode:
    """Tests for RepeatNode."""

    def test_repr_helper(self):
        """RepeatNode _repr_helper."""
        node = RepeatNode(
            event=NoteNode(letter="c"),
            times=4,
        )
        repr_str = node._repr_helper(0)
        assert "RepeatNode" in repr_str
        assert "times=4" in repr_str
        assert "NoteNode" in repr_str

    def test_accept(self):
        """RepeatNode accepts visitor."""
        node = RepeatNode(event=NoteNode(letter="c"), times=4)

        class TestVisitor(ASTVisitor):
            def visit_RepeatNode(self, node):
                return "repeat_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "repeat_visited"


class TestRepetitionRange:
    """Tests for RepetitionRange."""

    def test_repr_single(self):
        """RepetitionRange single value repr."""
        rng = RepetitionRange(first=1)
        assert repr(rng) == "1"

    def test_repr_range(self):
        """RepetitionRange range repr."""
        rng = RepetitionRange(first=1, last=3)
        assert repr(rng) == "1-3"


class TestOnRepetitionsNode:
    """Tests for OnRepetitionsNode."""

    def test_repr_helper(self):
        """OnRepetitionsNode _repr_helper."""
        node = OnRepetitionsNode(
            event=NoteNode(letter="c"),
            ranges=[RepetitionRange(first=1, last=3), RepetitionRange(first=5)],
        )
        repr_str = node._repr_helper(0)
        assert "OnRepetitionsNode" in repr_str
        assert "1-3" in repr_str
        assert "5" in repr_str
        assert "NoteNode" in repr_str

    def test_accept(self):
        """OnRepetitionsNode accepts visitor."""
        node = OnRepetitionsNode(event=NoteNode(letter="c"))

        class TestVisitor(ASTVisitor):
            def visit_OnRepetitionsNode(self, node):
                return "on_rep_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "on_rep_visited"


# =============================================================================
# BracketedSequenceNode Tests
# =============================================================================


class TestBracketedSequenceNode:
    """Tests for BracketedSequenceNode."""

    def test_repr_helper(self):
        """BracketedSequenceNode _repr_helper."""
        node = BracketedSequenceNode(
            events=EventSequenceNode(
                events=[NoteNode(letter="c"), NoteNode(letter="d")]
            )
        )
        repr_str = node._repr_helper(0)
        assert "BracketedSequenceNode" in repr_str
        assert "EventSequenceNode" in repr_str

    def test_accept(self):
        """BracketedSequenceNode accepts visitor."""
        node = BracketedSequenceNode(events=EventSequenceNode())

        class TestVisitor(ASTVisitor):
            def visit_BracketedSequenceNode(self, node):
                return "bracketed_visited"

        visitor = TestVisitor()
        result = node.accept(visitor)
        assert result == "bracketed_visited"


# =============================================================================
# Position Tests
# =============================================================================


class TestNodePositions:
    """Tests for source positions on nodes."""

    def test_note_with_position(self):
        """NoteNode can have position."""
        pos = SourcePosition(line=1, column=5)
        note = NoteNode(letter="c", position=pos)
        assert note.position == pos
        assert note.position.line == 1
        assert note.position.column == 5

    def test_root_with_position(self):
        """RootNode can have position."""
        pos = SourcePosition(line=1, column=1)
        root = RootNode(position=pos)
        assert root.position == pos
