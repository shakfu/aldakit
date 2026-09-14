# TODO

## Critical

## High

- [ ] **Note-level channel reuse**: a channel is freed when the part on it stops sounding, which is enough for every bundled example. Alda decides this per note, so a score where more than 15 parts overlap in span but not in individual notes still reports `too-many-parts`.

## Medium

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.

- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Low

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.

- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).

- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.

