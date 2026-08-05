# TODO

## Short-term

- [x] **Visitor-based AST to Alda**: `isinstance()`-based dispatch in `midi/generator.py:_process_node()` replaced with `ASTVisitor`. The AST -> Alda conversion moved out of `score.py` into `aldakit/serialize.py` as `AldaWriter`, a complete visitor covering every node type.
- [x] **Error Messages**: Expand fix suggestions across all error paths. Added `hint` field to `AldaParseError` and contextual hints to all scanner/parser/CLI error messages.
- [x] **Documentation**: Set up MkDocs with Material theme. Removed stale internal docs, fixed broken links, added `docs`, `docs-serve`, `docs-deploy` Makefile targets.
- [x] **General MIDI coverage**: All 128 programs reachable under their canonical Alda names, generated from the language docs by `scripts/gen_instruments.py`.
- [x] **Percussion routing**: `midi-percussion` on MIDI channel 10; melodic parts kept off it.
- [x] **Generation diagnostics**: `MidiGenerator.diagnostics` reports unknown instruments, undefined variables and markers, and channel exhaustion.
- [x] **Golden MIDI fixtures**: `tests/golden/examples.json` pins the output of every example so a change in how a score sounds is a reviewable diff.

## Short-term (continued)

- [x] **Shared music theory**: Pitch names, scale and mode intervals and key signatures moved to `aldakit/theory.py`, shared by the MIDI generator and the compose module. `constants.py` is wired into the code that used to hardcode its values, and `tests/test_constants.py` fails if a constant goes unused.
- [x] **Attribute handler registry**: Each Alda attribute is a `@handles` method on `MidiGenerator` instead of a branch in an if/elif chain. Unknown attributes now record a diagnostic.
- [x] **Score source strategy**: `Score` holds one `ScoreContent` (source, elements or imported) instead of a mode tag plus fields that only apply in one mode.

- [x] **Unimplemented attributes**: `set-duration`, `set-note-length`, `set-duration-ms`, `track-volume` (`track-vol`) and `midi-channel` are implemented, so `poly.alda`, `multi-poly.alda`, `track-volume.alda` and `midi-channel-management-2.alda` play as written. Attributes the generator still does not know are reported as diagnostics rather than ignored.
- [x] **`aldakit soundfont`**: `list`, `install`, `verify` and `path` reach the SoundFont manager, and audio playback with no SoundFont offers to download one.
- [x] **`aldakit info` and `aldakit lint`**: score inspection and a linter over the generator's diagnostics channel, in `aldakit/analysis.py`. `lint --strict` exits non-zero, so it works as a build step.

- [ ] **Channel reuse over time**: aldakit assigns a channel per part for the life of the score; Alda reuses a channel once a part stops sounding, which is why `all-instruments.alda` and `midi-channel-management.alda` report `too-many-parts` under `aldakit lint`.

## Medium-term

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.
- [ ] **Performance Profiling**: Profile MIDI generation for large scores; add benchmark scripts.
- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Long-term

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.
- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).
- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.
