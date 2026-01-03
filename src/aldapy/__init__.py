"""aldapy: a pythonic alda music programming language implementation."""

from .tokens import Token, TokenType, SourcePosition
from .scanner import Scanner
from .parser import Parser, parse
from .ast_nodes import (
    ASTNode,
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
    # Phase 2 nodes
    VariableDefinitionNode,
    VariableReferenceNode,
    MarkerNode,
    AtMarkerNode,
    VoiceNode,
    VoiceGroupNode,
    CramNode,
    RepeatNode,
    OnRepetitionsNode,
    RepetitionRange,
    BracketedSequenceNode,
)
from .errors import AldaParseError, AldaScanError, AldaSyntaxError
from .midi import (
    MidiSequence,
    MidiNote,
    MidiGenerator,
    generate_midi,
    MidiBackend,
    LibremidiBackend,
)


__version__ = "0.1.2"


__all__ = [
    # Convenience function
    "parse",
    # Core classes
    "Token",
    "TokenType",
    "SourcePosition",
    "Scanner",
    "Parser",
    # AST nodes - Core
    "ASTNode",
    "ASTVisitor",
    "RootNode",
    "PartNode",
    "PartDeclarationNode",
    "EventSequenceNode",
    "NoteNode",
    "RestNode",
    "ChordNode",
    "DurationNode",
    "NoteLengthNode",
    "NoteLengthMsNode",
    "NoteLengthSecondsNode",
    "BarlineNode",
    "OctaveSetNode",
    "OctaveUpNode",
    "OctaveDownNode",
    "LispListNode",
    "LispSymbolNode",
    "LispNumberNode",
    "LispStringNode",
    # AST nodes - Phase 2
    "VariableDefinitionNode",
    "VariableReferenceNode",
    "MarkerNode",
    "AtMarkerNode",
    "VoiceNode",
    "VoiceGroupNode",
    "CramNode",
    "RepeatNode",
    "OnRepetitionsNode",
    "RepetitionRange",
    "BracketedSequenceNode",
    # Errors
    "AldaParseError",
    "AldaScanError",
    "AldaSyntaxError",
    # MIDI
    "MidiSequence",
    "MidiNote",
    "MidiGenerator",
    "generate_midi",
    "MidiBackend",
    "LibremidiBackend",
]
