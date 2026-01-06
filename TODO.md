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
