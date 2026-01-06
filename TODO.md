# TODO

## Future Features

### CLI & UX Enhancements

Improve ergonomics for live workflows:
- [x] Added regression tests that execute key CLI paths (stdin streaming, version flag) with mocked backends.
- [ ] Provide `--monitor` and `--metronome` helpers when transcribing to keep performers on grid.

### Conditional Full Bindings

The bundled `_libremidi` extension defaults to the minimal feature set. Add conditional build logic to detect optional dependencies and enable the richer polling/observer APIs when available:

- Check for `boost` and `readerwriterqueue` availability in CMake
- On macOS: `brew install boost readerwriterqueue`
- Define `LIBREMIDI_FULL_BINDINGS` preprocessor macro when deps found
- Use `#ifdef` in `_libremidi.cpp` to conditionally compile full vs minimal bindings

This keeps zero-dependency wheels lean, yet unlocks responsive MIDI I/O for contributors who install the optional toolchain.

## Priorities

### Short-term

1. **Documentation**: Consider adding API documentation (Sphinx/MkDocs) beyond the README
2. **Examples**: Add more example files demonstrating advanced features
3. **Error Messages**: Some error messages could include suggestions for fixes

### Medium-term

1. **Visitor-based AST to Alda**: Replace hasattr-based conversion with proper visitor pattern
2. **Plugin Architecture**: Consider exposing hooks for custom generators/transformers
3. **Performance Profiling**: Profile MIDI generation for large scores

### Long-term

1. **MIDI 2.0**: libremidi supports MIDI 2.0; consider exposing these features
2. **Audio Output**: Direct audio rendering without external synthesizer: an idea would be to embed fluidsynth itself.
3. **IDE Integration**: Language server protocol (LSP) for editor support
