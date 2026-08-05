# TODO

## Short-term

- [x] **Visitor-based AST to Alda**: `isinstance()`-based dispatch in `midi/generator.py:_process_node()` replaced with `ASTVisitor`. The AST -> Alda conversion moved out of `score.py` into `aldakit/serialize.py` as `AldaWriter`, a complete visitor covering every node type.
- [x] **Error Messages**: Expand fix suggestions across all error paths. Added `hint` field to `AldaParseError` and contextual hints to all scanner/parser/CLI error messages.
- [x] **Documentation**: Set up MkDocs with Material theme. Removed stale internal docs, fixed broken links, added `docs`, `docs-serve`, `docs-deploy` Makefile targets.
- [x] **General MIDI coverage**: All 128 programs reachable under their canonical Alda names, generated from the language docs by `scripts/gen_instruments.py`.
- [x] **Percussion routing**: `midi-percussion` on MIDI channel 10; melodic parts kept off it.
- [x] **Generation diagnostics**: `MidiGenerator.diagnostics` reports unknown instruments, undefined variables and markers, and channel exhaustion.
- [x] **Golden MIDI fixtures**: `tests/golden/examples.json` pins the output of every example so a change in how a score sounds is a reviewable diff.

## Medium-term

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.
- [ ] **Performance Profiling**: Profile MIDI generation for large scores; add benchmark scripts.
- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Long-term

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.
- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).
- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.
