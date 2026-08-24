"""Standard-library terminal frontend for the aldakit REPL.

The package is intentionally independent of the existing prompt-toolkit
frontend. Its editor, decoder, completion, history, and renderer can all be
tested with in-memory values and fake terminal streams.
"""

from .color import Span, tokenize_alda
from .completion import Completion, ReplCompleter
from .editor import EditAction, EditActionKind, EditorState, LineEditor
from .history import History
from .keys import KeyDecoder, KeyEvent, KeyKind, decode_bytes
from .render import Renderer
from .app import run_editor, run_line_mode

__all__ = [
    "Completion",
    "EditAction",
    "EditActionKind",
    "EditorState",
    "History",
    "KeyDecoder",
    "KeyEvent",
    "KeyKind",
    "LineEditor",
    "Renderer",
    "ReplCompleter",
    "Span",
    "decode_bytes",
    "run_editor",
    "run_line_mode",
    "tokenize_alda",
]
