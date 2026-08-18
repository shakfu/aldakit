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

- [x] **Channel reuse over time**: parts are given placeholder channels while the AST is walked, and `aldakit/midi/channels.py` turns those into real channels once the score's shape is known, handing a channel on when the part holding it stops sounding and restating the new part's instrument, pan and volume. `all-instruments.alda` and `midi-channel-management.alda` lint clean. Reuse is a fallback: 15 or fewer melodic parts still get one channel each, in declaration order.

- [x] **Per-part chord and cram timing**: a chord advanced every active part by the duration of whichever part was processed last, and a cram took its length from the first part, so parts in a group that had diverged in tempo or default note length drifted apart. Each part now advances by its own length; `multi-poly.alda` lines up as the 2:1 polyrhythm it is written as.

- [ ] **Note-level channel reuse**: a channel is freed when the part on it stops sounding, which is enough for every bundled example. Alda decides this per note, so a score where more than 15 parts overlap in span but not in individual notes still reports `too-many-parts`.

## Medium-term

- [x] **Offline audio rendering** (F2 in `REVIEW.md`): `aldakit render` writes a WAV file through the same synthesis loop playback uses, with no audio device and faster than real time.

- [x] **Golden audio fixtures**: `tests/golden/audio.json` pins what every example sounds like, rendered with a checksum-pinned SoundFont, and CI compares it on Linux and macOS. Regenerate with `make golden-audio`. The MIDI fixtures cannot see anything that happens after generation; these can.

- [ ] **`--monitor` and `--metronome` CLI helpers**: Provide real-time grid tracking aids for live transcription workflows.
- [x] **Performance Profiling**: measured rather than done. A synthetic 32,000-note score scans in 53 ms, parses in 183 ms and generates MIDI in 43 ms; `all-instruments.alda` is 14 ms end to end. There is no generation problem to profile, and the cost that does dominate is parsing, not MIDI generation. Reopen with a score that is actually slow.
- [ ] **Conditional Full Bindings**: Detect `boost` and `readerwriterqueue` in CMake and define `LIBREMIDI_FULL_BINDINGS` to conditionally compile richer polling/observer APIs in `_libremidi.cpp`. Keeps zero-dependency wheels lean while unlocking responsive MIDI I/O for contributors.

## Long-term

- [ ] **Plugin Architecture**: Expose hooks for custom generators/transformers.
- [ ] **MIDI 2.0**: Expose libremidi's MIDI 2.0 / UMP features (currently only MIDI 1.0 is bound).
- [ ] **IDE Integration**: Language server protocol (LSP) for editor support.
