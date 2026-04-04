# TODO

## Short-term

- [x] **Visitor-based AST to Alda**: Replace `isinstance()`-based dispatch in `score.py:node_to_str()` and `midi/generator.py:_process_node()` with proper visitor pattern using `ASTVisitor` from `ast_nodes.py`.
- [ ] **Error Messages**: Expand fix suggestions across all error paths. Some CLI errors already include hints (e.g. `aldakit ports`), but parser/scanner errors lack guidance.
- [ ] **Documentation**: Add API documentation (Sphinx or MkDocs) on top of the existing markdown docs in `docs/`.

## Medium-term

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.
- [ ] **Performance Profiling**: Profile MIDI generation for large scores; add benchmark scripts.
- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Long-term

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.
- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).
- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.
