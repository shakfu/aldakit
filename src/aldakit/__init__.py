"""aldakit: a pythonic alda music programming language implementation."""

from .api import list_ports, play, play_file, save, save_file
from .ast_nodes import (
    ASTNode,
    ASTVisitor,
    AtMarkerNode,  # Phase 2 nodes
    BarlineNode,
    BracketedSequenceNode,
    ChordNode,
    CramNode,
    DurationNode,
    EventSequenceNode,
    LispListNode,
    LispNumberNode,
    LispStringNode,
    LispSymbolNode,
    MarkerNode,
    NoteLengthMsNode,
    NoteLengthNode,
    NoteLengthSecondsNode,
    NoteNode,
    OctaveDownNode,
    OctaveSetNode,
    OctaveUpNode,
    OnRepetitionsNode,
    PartDeclarationNode,
    PartNode,
    RepeatNode,
    RepetitionRange,
    RestNode,
    RootNode,
    VariableDefinitionNode,
    VariableReferenceNode,
    VoiceGroupNode,
    VoiceNode,
)
from .errors import AldaParseError, AldaScanError, AldaSyntaxError
from .midi import (
    LibremidiBackend,
    MidiBackend,
    MidiGenerator,
    MidiNote,
    MidiSequence,
    generate_midi,
)
from .midi.transcriber import list_input_ports, transcribe
from .parser import Parser, parse
from .scanner import Scanner
from .score import Score
from .tokens import SourcePosition, Token, TokenType

__version__ = "0.1.4"


__all__ = [
    # High-level API
    "Score",
    "play",
    "play_file",
    "save",
    "save_file",
    "list_ports",
    "transcribe",
    "list_input_ports",
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
