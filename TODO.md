# TODO

## Short-term

- [x] **Visitor-based AST to Alda**: Replace `isinstance()`-based dispatch in `score.py:node_to_str()` and `midi/generator.py:_process_node()` with proper visitor pattern using `ASTVisitor` from `ast_nodes.py`.
- [x] **Error Messages**: Expand fix suggestions across all error paths. Added `hint` field to `AldaParseError` and contextual hints to all scanner/parser/CLI error messages.
- [x] **Documentation**: Set up MkDocs with Material theme. Removed stale internal docs, fixed broken links, added `docs`, `docs-serve`, `docs-deploy` Makefile targets.

## Medium-term

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.
- [ ] **Performance Profiling**: Profile MIDI generation for large scores; add benchmark scripts.
- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Long-term

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.
- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).
- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.
