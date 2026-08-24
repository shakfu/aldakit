# Standard-Library Terminal Frontend Design

## Summary

Replace the vendored prompt-toolkit frontend with a small terminal editor built
only from the Python standard library. Preserve the current `ReplSession`,
`ReplContext`, command handlers, parser, generator, and backend interfaces. The
new frontend should be an input/rendering layer around those existing pieces,
not a second implementation of score behavior.

The target is practical prompt-toolkit parity for the workflows aldakit
actually supports:

- Single-line and multiline Alda entry
- History navigation and persistent history
- Tab completion for commands, paths, instruments, and attributes
- Syntax-colored Alda source where the terminal supports color
- Ctrl+C playback cancellation and input cancellation
- Ctrl+D exit
- Responsive status/output while playback is active
- The existing `:load`, `:play`, `:save`, `:ls`, `:cd`, `:ports`, `:status`, and
  mode commands

The design deliberately does not reproduce prompt-toolkit's general-purpose
layout engine, mouse support, vi mode, SSH input, or full terminal abstraction.

## Constraints

- Runtime dependencies remain empty in `pyproject.toml`.
- Use only the Python standard library and aldakit modules.
- Work on Python 3.10 through 3.14.
- Support Linux, macOS, and Windows.
- Do not modify process-global `sys.path`.
- All editing behavior must be testable without a real terminal.
- A non-interactive stdin must fail clearly or use a simple line-oriented mode;
  it must never hang waiting for terminal escape sequences.
- Terminal cleanup must run on normal exit, exceptions, and Ctrl+C.

## Proposed Modules

Create a package such as `aldakit/terminal/` with these components:

### `terminal/platform.py`

Provide a narrow platform adapter:

```python
class TerminalInput(Protocol):
    def read_event(self) -> "KeyEvent": ...
    def write(self, text: str) -> None: ...
    def flush(self) -> None: ...

class TerminalMode(Protocol):
    def __enter__(self) -> "TerminalMode": ...
    def __exit__(self, *exc_info: object) -> None: ...
```

On POSIX, use `termios`, `tty`, and `select` to enter cbreak/raw input mode,
read bytes, and restore the original settings. On Windows, use `msvcrt` for
key reads and `colorama`-like behavior implemented locally only where needed;
do not add colorama as a dependency. ANSI output is supported on modern
Windows terminals, with color disabled when the console does not advertise
ANSI support.

The adapter should expose capabilities rather than making the editor infer
them:

```python
@dataclass(frozen=True)
class TerminalCapabilities:
    colors: bool
    unicode: bool
    cursor_movement: bool
    raw_input: bool
    width: int
```

Terminal width comes from `shutil.get_terminal_size()`, with an 80-column
fallback. A resize is detected before each redraw rather than requiring a
background signal handler.

### `terminal/keys.py`

Normalize platform-specific bytes into a small event vocabulary:

```python
class KeyKind(Enum):
    CHARACTER = auto()
    ENTER = auto()
    BACKSPACE = auto()
    DELETE = auto()
    TAB = auto()
    ESCAPE = auto()
    CTRL_C = auto()
    CTRL_D = auto()
    ARROW_LEFT = auto()
    ARROW_RIGHT = auto()
    ARROW_UP = auto()
    ARROW_DOWN = auto()
    HOME = auto()
    END = auto()
    PAGE_UP = auto()
    PAGE_DOWN = auto()
    UNKNOWN = auto()
```

Parse common ANSI sequences such as `ESC [ A`, `ESC [ 1 ~`, and `ESC [ 4 ~`.
The parser must have a bounded escape-sequence timeout. An isolated Escape key
must return promptly instead of blocking indefinitely waiting for more bytes.
Windows extended-key pairs are translated to the same events.

Unknown sequences should be ignored safely. Literal printable input should be
preserved, including non-ASCII text when the terminal encoding supports it.

### `terminal/editor.py`

Implement a model-only line editor with no direct terminal calls:

```python
@dataclass
class EditorState:
    text: str = ""
    cursor: int = 0
    preferred_column: int | None = None
    multiline: bool = False
    completion: CompletionState | None = None

class LineEditor:
    def handle(self, event: KeyEvent) -> EditAction: ...
    def render_lines(self, width: int) -> list["RenderedLine"]: ...
```

Use a Python string plus cursor index initially. This is sufficient for Alda's
expected input sizes and keeps the implementation auditable. Treat cursor
positions as Python character indexes in the model, and calculate terminal
cell widths only while rendering. The first version should document that
complex combining-character editing is best-effort rather than attempting to
reimplement a full Unicode grapheme library.

Required editing behavior:

- Insert printable characters at the cursor
- Backspace and Delete
- Left/Right movement
- Home/End for the current logical line
- Up/Down movement between visual/logical lines
- Ctrl+A/Ctrl+E as Home/End aliases
- Ctrl+W to delete the previous word
- Ctrl+U to clear before the cursor
- Ctrl+K to clear after the cursor
- Enter to submit a single-line entry
- Alt+Enter and Ctrl+J to insert a newline
- Ctrl+C to cancel the current input, or stop playback if input is empty
- Ctrl+D to exit on an empty input, otherwise delete at the cursor

`EditAction` should distinguish `CONTINUE`, `SUBMIT(text)`, `CANCEL`,
`STOP_PLAYBACK`, and `EXIT`. This makes key behavior directly unit-testable.

### `terminal/history.py`

Provide persistent history without importing `readline` as a required runtime
component. Store one submitted entry per record in the existing
`~/.alda_history` path, using a simple escaped format or length-prefixed JSON
records so multiline entries round-trip without ambiguity.

Recommended behavior:

- Load history lazily when the REPL starts.
- Ignore a missing, unreadable, or malformed history file with a warning only
  when verbose mode is enabled.
- Keep a bounded in-memory history, for example 1,000 entries.
- Deduplicate only adjacent identical entries.
- Write atomically through a temporary file and `os.replace`.
- Restrict history file permissions where the platform supports it.
- Never record commands or source that the user cancelled.

History navigation should use a separate draft buffer. Pressing Up from a fresh
buffer enters the newest history entry; editing that entry preserves the draft
until Down returns to it. Page Up/Page Down should move by a larger bounded
step, not exit the history range.

### `terminal/completion.py`

Reuse the semantics of `AldaCompleter`, but make completion independent of
prompt-toolkit's `Document` and `Completion` classes:

```python
@dataclass(frozen=True)
class Completion:
    replacement: str
    start: int
    end: int
    display: str | None = None

class Completer(Protocol):
    def complete(self, text: str, cursor: int) -> list[Completion]: ...
```

Completion contexts should remain:

- At the beginning of an Alda line: instrument names after at least three
  characters
- Inside an open attribute expression: known attribute names
- After `:load`, `:play`, `:save`, or `:cd`: paths
- At the beginning of a command: command names and aliases

Path completion should use `Path.expanduser`, preserve the user's typed prefix,
offer directories with a trailing separator, and limit `:load`, `:play`, and
`:save` files to relevant extensions as the current REPL does. Completion must
not read arbitrary file contents.

Tab behavior:

- No candidates: leave the buffer unchanged.
- One candidate: replace the active token and append a space where appropriate.
- Multiple candidates: insert their longest common prefix; a second Tab prints
  a compact candidate list below the prompt and redraws the editor.
- Escape closes the candidate list without changing the buffer.
- Up/Down navigate candidates only while a candidate list is visible.

### `terminal/color.py`

Move the useful part of `AldaLexer` into a dependency-free tokenizer that emits
spans rather than prompt-toolkit fragments:

```python
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    style: str
```

Recognize the current classes: note, rest, octave, duration, instrument,
attribute, barline, and comment. The tokenizer is for display only; parsing
continues to use the real scanner/parser, avoiding two sources of syntax
semantics.

Map styles to ANSI SGR sequences when `capabilities.colors` is true. Use a
plain-text renderer otherwise. Disable color when `NO_COLOR` is set, when
stdout is not a TTY, or when the terminal is known not to support it.

### `terminal/render.py`

Render the prompt and editor using cursor-relative ANSI operations:

- Clear only the lines previously occupied by the editor.
- Move the cursor to the calculated row and column.
- Redraw the prompt, wrapped/multiline buffer, continuation prompts, and any
  completion list.
- Keep all writes in one flush per render to reduce flicker.
- Never assume the cursor is at column zero after external output; explicitly
  start a new line before status messages and redraw afterwards.

Use a full editor redraw initially. Optimize to partial line updates only if
profiling shows a real need. This keeps cursor accounting manageable and is
adequate for short music-programming entries.

## Input and submission model

Retain the current distinction between source history and the loaded buffer:

- Typed or pasted Alda is submitted to `play_source()` and recorded.
- `:load` loads and records a file but does not play it.
- `:play` replays the loaded buffer without duplicating the session entry.
- `:save` writes the complete session source.

The frontend should call the existing command dispatch function for lines whose
first non-whitespace character is `:`. It should call the existing play
callback for all other submitted text. Parsing and playback must happen outside
the editor model so errors and backend failures can be printed as normal output.

Multiline submission should be explicit. Enter submits the current buffer;
Alt+Enter/Ctrl+J inserts a newline. A continuation prompt is rendered for
subsequent lines. Do not infer submission from incomplete Alda syntax because
the parser cannot reliably distinguish an intentional incomplete edit from a
mistake.

Pasted multiline text is a special usability case. When a terminal sends a
burst containing newlines, insert the entire burst into the buffer and leave
the cursor at the end rather than submitting each line. A future bracketed
paste mode can improve this, but it is not required for the initial version.

## Playback and asynchronous output

The editor loop must remain responsive while playback runs. The simplest safe
model is cooperative polling:

1. Before blocking for a key, poll backend status with a short timeout.
2. If playback state changes, print a one-line status update and redraw.
3. Ctrl+C first stops active playback when the buffer is empty; otherwise it
   cancels the current edit.
4. On exit, stop playback and close the backend before restoring terminal mode.

Do not write from a backend thread directly to the terminal. If background
events are later needed, enqueue them and let the main loop render them. This
avoids interleaved ANSI output and makes tests deterministic.

## TTY modes and fallback behavior

`run_repl()` should select the frontend as follows:

1. If stdin and stdout are TTYs, use the interactive raw/cbreak editor.
2. If either stream is not a TTY, use a line-oriented fallback that calls
   `input()` or reads `sys.stdin.readline()` and does not emit ANSI controls.
3. If raw-mode setup fails, print a warning and use the line-oriented fallback.

The line-oriented mode retains commands, history where possible, multiline
input through an explicit continuation convention, and playback behavior, but
does not promise cursor editing or completion. This makes pipes and CI safe.

## Test Strategy

Tests must focus on deterministic model behavior and use a fake terminal.

### Unit tests

- Decode printable keys, control keys, ANSI arrows, Home/End, Delete, and
  Windows extended-key sequences.
- Handle incomplete and unknown escape sequences without hanging.
- Insert, delete, move, wrap, and multiline cursor behavior.
- History navigation, draft restoration, bounds, deduplication, and malformed
  files.
- Completion replacement ranges, common-prefix behavior, path filtering, and
  command contexts.
- Syntax spans for representative Alda constructs and plain fallback output.
- ANSI renderer output for single-line, wrapped, multiline, and completion
  states.
- Terminal mode restoration when the loop exits normally or by exception.

### Integration tests

- Submit Alda code and verify the existing play callback receives it once.
- Execute every existing REPL command through the new frontend dispatcher.
- Verify Ctrl+C stops playback and does not terminate the process when playback
  is active.
- Verify Ctrl+C cancels non-empty input without recording it.
- Verify Ctrl+D exits only on an empty buffer.
- Verify a multiline entry is submitted as one source string.
- Run the frontend against fake TTY streams with colors enabled and disabled.
- Run the line-oriented fallback with piped input and verify no ANSI escapes are
  emitted.

### Platform checks

Run a small smoke suite on Linux, macOS, and Windows for:

- Terminal mode enter/restore
- Arrow and control-key decoding
- Unicode and terminal width handling
- Ctrl+C and Ctrl+D behavior
- History file replacement

The bulk of the suite should remain platform-independent; only byte decoding
and terminal mode adapters require platform runners.

## Migration Plan

1. Extract the current pure behavior from `repl.py` into reusable frontend
   interfaces where necessary; keep command handlers unchanged.
2. Implement and test `KeyEvent`, the key decoder, and the model-only editor.
3. Implement history and completion adapters, reusing current command names,
   path rules, and instrument tables.
4. Implement ANSI/plain renderers and POSIX/Windows terminal adapters.
5. Add `run_terminal_repl()` and select it behind an internal feature flag.
6. Run the existing REPL command tests against the new dispatcher and add the
   fake-terminal integration tests.
7. Compare the old and new frontends manually for the documented workflows.
8. Make the stdlib frontend the default after the cross-platform smoke suite is
   green.
9. Remove prompt-toolkit imports, `src/aldakit/ext/`, and related exclusions
   only after a release has validated the migration.

## Acceptance Criteria

The replacement is ready when:

- `pyproject.toml` still declares no runtime dependencies.
- `import aldakit` and non-interactive CLI commands never import terminal UI
  modules.
- All existing command behavior and session semantics remain unchanged.
- The new editor passes the model and fake-terminal test suite on all supported
  platforms.
- Interactive use supports history, completion, multiline input, syntax color,
  Ctrl+C, Ctrl+D, arrows, Home/End, and Delete on supported terminals.
- Piped/non-TTY use is deterministic, plain-text, and never hangs.
- Terminal settings are restored after every exit path.
- The wheel contains no prompt-toolkit or wcwidth files.

## Risks and mitigations

- **Terminal complexity:** Keep the platform layer narrow and use a fake
  terminal; do not build a general terminal framework.
- **Unicode cursor errors:** Treat display width as a rendering concern and
  document best-effort behavior for combining characters in the first release.
- **Output corruption:** Centralize all terminal writes and redraw after every
  external message.
- **Windows divergence:** Normalize key events at the adapter boundary and run
  platform smoke tests instead of duplicating editor logic.
- **Feature regression:** Keep the current command tests and add behavior-level
  acceptance tests before deleting the old frontend.
- **Scope growth:** Defer mouse support, vi mode, bracketed paste, inline
  diagnostics, and arbitrary layout support until the basic editor is stable.

## Decision

Implement the stdlib frontend as a separately testable editor and terminal
adapter, while preserving the existing REPL command/session layer. This offers
most of the usability users rely on without carrying prompt-toolkit's broad
dependency tree. A small, explicit fallback is preferable to attempting to
recreate every prompt-toolkit feature.

---

# Implementation Review

Review of `src/aldakit/terminal/` against this design, as a candidate
replacement for the vendored prompt-toolkit in `src/aldakit/ext/`.

Every defect below was reproduced by execution, driving `run_editor` against a
real pty, rather than by inspection alone.

## Verdict

The design is sound and worth keeping. The implementation is an early
prototype -- roughly 40% of the design -- and the interactive path does not
currently work on POSIX. It is not yet a substitute for prompt-toolkit.

The 769 lines in `src/aldakit/terminal/` are well-factored, lint-clean, and
`ty`-clean, but "clean" here mostly reflects that the hard parts have not been
attempted. Nothing imports the package: `run_repl` in `src/aldakit/repl.py`
still constructs a `PromptSession`, so migration step 5 has not happened and
none of this code is exercised outside its own eight unit tests.

## Design assessment

What the design gets right:

- **The model/adapter split.** A pure `LineEditor` returning an `EditAction`
  enum is the correct testability boundary, and the implementation honors it
  faithfully. This is the design's best decision.
- **Explicit multiline submission** rather than inferring completeness from the
  parser. The alternative would give two sources of syntax semantics.
- **A display-only tokenizer** kept separate from the real scanner.
- **Realistic scope exclusions:** no mouse, no vi mode, no layout engine.

Three gaps in the design itself:

1. **`termios` flags are never discussed.** The design says to "use `termios`,
   `tty`, and `select` to enter cbreak/raw input mode" and treats mode-setting
   as a one-liner. It is not, and this omission is the direct cause of the
   worst implementation defect below. The design should name the flags that
   must change: `ICRNL`, `ISIG`, `IXON`, `OPOST`.
2. **Signals are absent entirely.** No mention of `SIGINT`,
   `SIGTSTP`/`SIGCONT` (suspend and resume with the terminal in cbreak mode),
   or `SIGWINCH`. The Ctrl+C model under "Playback and asynchronous output"
   implicitly assumes Ctrl+C arrives as byte `0x03`, which is only true once
   `ISIG` is cleared.
3. **One factual error.** The completion section says path completion should
   limit `:load`, `:play`, and `:save` to relevant extensions "as the current
   REPL does." The current REPL does no such thing -- `AldaCompleter` delegates
   to an unfiltered `PathCompleter`, and the comment in `repl.py` claiming
   otherwise is stale. The new filtering behavior is an improvement, but it
   should be recorded as a change, not as parity.

## Blocking defects

### 1. Enter never submits on POSIX (`platform.py:50`)

`tty.setcbreak()` clears only `ECHO|ICANON` in `LFLAG`; it does not touch
`IFLAG`. `ICRNL` therefore stays on, so Enter (`CR`, `0x0D`) is translated by
the tty driver into `LF` (`0x0A`), which `_CONTROL_KEYS` maps to `CTRL_J` --
insert newline.

Driving `b"piano: c\r\x04"` through `run_editor` left the buffer as
`"piano: c\n"`, `submitted` empty, and the loop still running. There is no way
to submit input or to leave the editor.

```text
ICRNL still on after setcbreak: True
IXON  still on after setcbreak: True
ISIG  still on after setcbreak: True
```

`IXON` also means Ctrl+S silently freezes the terminal.

### 2. Ctrl+C is a signal, not a byte (`platform.py:50`, `app.py:44`)

`ISIG` stays on, so Ctrl+C raises `KeyboardInterrupt` in the main thread and
byte `0x03` is never delivered. The whole `CTRL_C` -> `CANCEL`/`STOP_PLAYBACK`
path in `editor.py` is unreachable on POSIX, and `run_editor` has no
`except KeyboardInterrupt`: the loop dies and `history.save()` is skipped.
Ctrl+Z likewise suspends the process with the terminal left in cbreak mode.

### 3. Non-ASCII input is destroyed (`keys.py:111`, `keys.py:146`)

`TerminalReader.read_event` reads one byte at a time, and `KeyDecoder.feed`
slices one byte at a time for non-escape input. Each byte of a multi-byte
UTF-8 sequence is then decoded alone with `errors="strict"` and fails:

```text
feed 'ce\u0301' one byte at a time -> [('CHARACTER','c'), ('UNKNOWN',''), ('UNKNOWN','')]
```

Separately, `decode_bytes` decodes `data[index:]` -- the entire remainder -- so
one bad byte anywhere discards a valid character before it: `b"a\xff"` yields
two `UNKNOWN` events and loses the `a`. This contradicts the requirement that
literal printable input be preserved, including non-ASCII text.

### 4. Windows is unimplemented

Searching the package for `msvcrt`, `select`, `signal`, `SetConsoleMode`, or
`VIRTUAL_TERMINAL` returns nothing. `TerminalMode.__enter__` returns
immediately on `nt` without enabling VT processing, and `TerminalReader` then
reads a still-canonical stdin. The Windows requirements in this design are
absent, and the acceptance criterion "passes on all supported platforms"
cannot be met.

### 5. An empty `History` is falsy (`history.py:62`, `app.py:39`, `app.py:103`)

`History.__len__` returning `0` makes a fresh `History` false, so
`history = history or History("~/.alda_history")` silently replaces an
explicitly passed history object with the user's real one.

This is not theoretical. `tests/test_terminal_frontend.py` passes
`history=History(tmp_path / "history")` and the test writes to the developer's
real `~/.alda_history` anyway. It only passes because `$HOME` is normally
writable; under a read-only home it fails outright:

```text
FAILED tests/test_terminal_frontend.py::test_line_mode_submits_without_ansi
OSError: [Errno 30] Read-only file system: '/home/sa/..alda_history.pmax3_dy'
```

Fix: test `if history is None:`, and add an explicit `__bool__`.

## Design requirements not implemented

| Requirement | Status |
| --- | --- |
| Bounded escape timeout; lone Escape returns promptly | **Missing.** No `select` anywhere; `KeyDecoder.flush()` exists but is never called. Feeding `b"\x1b"` alone left `run_editor` blocked indefinitely. |
| Alt+Enter inserts a newline | **Inverted.** `decode_bytes(b"\x1b\r")` returns `ENTER`, which submits; incrementally it decodes as `ESCAPE, ENTER`, which also submits. |
| Wrapped rendering; `render_lines(width)` | **Missing.** `Renderer` never reads `capabilities.width`, and the cursor is positioned with `"\x1b[C" * column`, which desynchronizes as soon as a line wraps. |
| Tab: longest common prefix, second-Tab candidate list, Escape to close, Up/Down to navigate | **Missing.** `app.py` applies a completion only when there is exactly one candidate; otherwise Tab does nothing. |
| Page Up/Down history stepping | **Missing.** |
| Cooperative playback polling and status redraw | **Missing.** `run_editor` blocks in `read_event()`; `stop_playback` is reachable only through the dead Ctrl+C path. |
| Warn and fall back if raw-mode setup fails | **Missing.** `termios.error` from `TerminalMode.__enter__` propagates. |
| Start a new line before external output | **Missing.** `submit()` is called with the cursor mid-line, so command output appends to the prompt line. |
| Paste bursts insert rather than submit per line | **Missing.** Every `CR` in a burst submits. |
| Line mode keeps commands and a multiline convention | **Partial.** `run_line_mode` writes no prompt and has no exit path, so `:quit` keeps reading. |
| Select the frontend on stdin *and* stdout | **Partial.** Only stdin is checked. |

Two persistence issues are worth separate mention. History is written only at
loop exit, a regression from prompt-toolkit's `FileHistory`, which appends per
entry. And the new JSON-lines format cannot read the existing file, so every
user silently starts with an empty history:

```text
loading an existing ~/.alda_history -> []
```

## Correctness bugs

- **The continuation prompt is doubled.** `render.py` prefixes `REPL_PROMPT` to
  every line, then prepends `REPL_CONTINUATION_PROMPT` to lines 2 and beyond,
  producing `'aldakit> piano: c\n  ... aldakit> d e'`. The column arithmetic
  uses only the continuation prompt's length, so the cursor lands nine columns
  off on every continuation line.
- **Attribute completion is effectively dead.** `completion.py` returns
  unconditionally once the word is at least three characters, so the attribute
  branch below it is never reached: `complete("(tem")` returns `[]`. The
  original `AldaCompleter` yielded both instruments and attributes.
- **When attribute completion does fire, the range is off by one.**
  `complete("(t")` returns `Completion("(tempo ", 1, 2)`; applying it yields
  `"((tempo "`. `start` should be `open_paren`, not `cursor - len(prefix)`.
- **Arrow-Up is hijacked by history in multiline buffers.** `app.py` routes
  `ARROW_UP` to history whenever `history.entries` is non-empty, so
  `_move_vertical` is unreachable and a multiline entry cannot be navigated.
- **`~` is not preserved in path completion**, contrary to the design:
  `complete(":cd ~/mu")` returns `/tmp/probe_home/mus/`. The same line
  hardcodes `"/"` as the separator.
- **SS3 sequences are not decoded.** `decode_bytes(b"\x1bOA")` returns
  `ESCAPE, 'O', 'A'`, inserting a literal `OA`. xterm sends `ESC O H` and
  `ESC O F` for Home and End.
- **The CSI scanner over-consumes.** It keeps eating `A`-`D`, `H`, and `F`, so
  `b"\x1b[BA"` yields a single `UNKNOWN` and swallows the `A`.

## Structural issues

- **`TerminalCapabilities` is declared twice**, in `platform.py` and
  `render.py`, with different fields, forcing a conversion in `app.py`. The
  design's `unicode` and `raw_input` fields are dropped from both.
- **`COMMAND_NAMES` and `PATH_COMMANDS` are duplicated** between `repl.py` and
  `completion.py` and will drift. They belong in `constants.py`; importing them
  from `repl.py` is impossible without dragging in prompt-toolkit, which is
  exactly the coupling the migration has to break first.
- `EditorState.history_index` is never written -- `app.py` keeps its own local
  -- and the design's `preferred_column` is absent, so vertical movement does
  not sticky-track the column.
- `app.py` import ordering and the unannotated `run_line_mode` signature are
  inconsistent with the rest of the package.

## Test suite

Eight tests, 104 lines. They pass, but they assert only what already works and
never touch the parts that do not.

- **No test constructs `run_editor`.** Every integration test listed under
  "Test Strategy" is missing. A single fake-TTY test driving `b"piano: c\r"`
  would have caught blocking defect 1 immediately.
- `test_incremental_decoder_keeps_escape_pending` asserts
  `feed(b"\x1b") == []`, encoding the hang as correct behavior rather than
  testing the required timeout.
- `test_color_spans_and_plain_renderer` renders only single-line input, so it
  misses the continuation-prompt bug.
- There are no non-ASCII, wrapping, terminal-restoration, or malformed-history
  tests, all of which the unit-test list requires.

## Recommendations

Keep the design; treat the current code as a spike. Before it can replace
`src/aldakit/ext/` (4.3 MB, 153 files):

1. **Fix the platform layer first.** Use `termios` directly rather than
   `tty.setcbreak`: clear `ICRNL|IXON` from `IFLAG` and `ISIG` from `LFLAG`,
   keep `OPOST` on, and read through `select.select(..., timeout)` so that the
   escape timeout and playback polling both fall out naturally. This one change
   unblocks defects 1 and 2 and the escape-timeout requirement.
2. **Buffer bytes, decode text.** Read available bytes with
   `os.read(fd, 1024)` and feed an `incrementaldecoder`. This fixes defect 3
   and yields paste bursts for free.
3. **Add a fake-TTY harness and the integration tests** before writing any more
   features. The current suite gives false confidence.
4. **Move `COMMAND_NAMES` and `PATH_COMMANDS` into `constants.py`** and wire
   `run_terminal_repl()` behind the feature flag from migration step 5, so both
   frontends run against the same command table.
5. Then close the wrapping, completion-UI, and Windows gaps.

Given that signals, `SIGWINCH`, wrapping, the completion UI, and the entire
Windows adapter are still ahead, this is closer to the start of the work than
to the end. `REVIEW.md` proposed a two-tier REPL -- prompt-toolkit when
installed, stdlib fallback otherwise. That remains the lower-risk sequencing:
ship the stdlib editor as the fallback first, let it mature against real use,
and only then make it the default and delete `ext/`.

Two decisions are needed now, because they affect users rather than code:
whether `~/.alda_history` should be migrated or renamed, and whether history
should be appended per entry rather than only at exit.
