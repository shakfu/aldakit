# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8]

### Added

- **Key signature support** (`key-sig`, `key-signature`) - Full implementation in MIDI generator
  - String format: `(key-sig "f+ c+ g+")` for explicit accidentals
  - Quoted list format: `(key-sig '(g major))`, `(key-sig '(d minor))`
  - Modal key signatures: `(key-sig '(d dorian))`, `(key-sig '(e phrygian))`
  - Explicit accidentals: `(key-sig '(b (flat) e (flat)))`
  - Natural sign (`_`) overrides key signature on individual notes
  - Per-part and global (`key-sig!`) variants

- **Transposition support** (`transpose`) - Full implementation in MIDI generator
  - Semitone-based transposition: `(transpose 5)` for up a fourth, `(transpose -7)` for down a fifth
  - Useful for transposing instruments: `clarinet: (transpose -2)` for Bb clarinet
  - Resets with `(transpose 0)`
  - Per-part and global (`transpose!`) variants
  - Correctly clamps to valid MIDI range (0-127)

- **Quoted list syntax** in S-expressions (`'(...)`) - Lisp-style quoted lists now fully parsed

- **Concurrent playback mode** for LibremidiBackend
  - Up to 8 simultaneous playback slots
  - `concurrent_mode` property (default True) to enable/disable layered playback
  - `active_slots` property to check number of active playbacks
  - `wait()` method to block until all playback completes
  - `stop()` stops all slots, `stop_slot(id)` stops a specific one
  - Thread-safe MIDI message sending
  - Inspired by alda-midi's libuv-based async system

- **REPL concurrent playback integration**
  - Concurrent mode enabled by default - inputs layer on each other
  - `:concurrent` command to enable concurrent mode
  - `:sequential` command to enable sequential mode (wait for each input)
  - `:status` command to show playback mode and active slots
  - `--sequential` CLI flag to start REPL in sequential mode

- **CLI audio backend option**
  - `-sf` / `--soundfont FILE` to use TinySoundFont audio backend
  - Available for both REPL (`aldakit repl -sf ...`) and play (`aldakit play -sf ... file.alda`)
  - Clear error message when no MIDI ports available, directing user to use `-sf`

- **CLI reorganization**
  - New `eval` subcommand: `aldakit eval "piano: c d e"`
  - Top-level now only has `--version` and `--help`
  - All options moved to their respective subcommands
  - `aldakit` with no args opens the REPL

- **Configuration file support** (`~/.aldakit/config.ini`)
  - INI-format config files using Python's built-in `configparser` (zero dependencies)
  - User config: `~/.aldakit/config.ini`
  - Project-local config: `./aldakit.ini` (overrides user config)
  - Supported options: `soundfont`, `backend`, `port`, `tempo`, `verbose`
  - Environment variable `ALDAKIT_SOUNDFONT` overrides config files
  - CLI arguments always take highest priority
  - Smart fallback: MIDI is preferred by default; audio is used only if explicitly requested (`-sf` or `backend=audio`) or as fallback when no MIDI ports are available

### Fixed

- **Windows build failure in `_tsf.cpp`** - Added `#define NOMINMAX` to prevent Windows SDK `min`/`max` macro conflicts with `std::min`/`std::max`
- **Type checking errors in `tsf_backend.py` and `soundfont.py`**
  - Fixed `callable` type hints to use `Callable[[int, int], None]` from `collections.abc`
  - Added type ignore comments for native module imports
  - Fixed dictionary value type inference with explicit `str()` casts
- **Flaky `test_current_time_advances` test** - Replaced fixed sleep with polling loop to handle audio thread startup latency

## [0.1.7]

### Added

#### Direct Audio Playback via TinySoundFont

Self-contained audio synthesis without requiring an external MIDI synthesizer:

- **New `_tsf` native module** - TinySoundFont + miniaudio integration via nanobind
  - Header-only libraries for minimal binary size (~50KB)
  - Cross-platform audio output (CoreAudio, ALSA, WASAPI)
  - Real-time SoundFont synthesis at 44.1kHz stereo

- **New `TsfBackend` class** (`aldakit.midi.backends.TsfBackend`)
  - Drop-in replacement for `LibremidiBackend` when no external synth is available
  - Automatic SoundFont discovery in common locations
  - Environment variable support (`ALDAKIT_SOUNDFONT`)
  - `play()`, `stop()`, `wait()`, `is_playing()`, `current_time()` methods
  - Preset inspection: `preset_count`, `preset_name(index)`
  - Adjustable gain via `set_gain()`

- **`Score.play()` backend parameter**
  - `score.play(backend="audio")` - direct audio output via TinySoundFont
  - `score.play(backend="midi")` - MIDI output via libremidi (default, unchanged)
  - `score.play(backend="audio", soundfont="/path/to/sf2")` - explicit SoundFont

- **SoundFont utilities** (`aldakit.midi.soundfont`)
  - `find_soundfont()` - search common locations for GM SoundFont
  - `list_soundfonts()` - list all discovered SoundFont files
  - `download_soundfont(name)` - download from public archives
  - `setup_soundfont()` - interactive setup with progress display
  - Catalog includes TimGM6mb (5.8MB), FluidR3_GM (141MB), GeneralUser_GS (31MB)

- **29 new tests** for TsfBackend (`tests/test_tsf_backend.py`)

### Changed

- Development status changed from Beta to Alpha in `pyproject.toml`
- **CLI now opens REPL by default** when run without arguments (`aldakit` is now equivalent to `aldakit repl`)

## [0.1.6]

### Fixed

- **Vendored prompt_toolkit imports in REPL now use sys.path approach correctly**
  - `repl.py` now imports `ext` first to initialize sys.path, then uses absolute imports for prompt_toolkit.
  - Previously, relative imports (`.ext.prompt_toolkit`) bypassed the sys.path setup needed for vendored packages.

- **Test assertions now use explicit `assert` statements**
  - Converted mock assertion methods (e.g., `mock.assert_called_once()`) to explicit `assert` statements in test_api.py and test_midi_import.py.
  - Fixes pytest-review warnings about tests with no assertions.

- **Compose API part declarations now generate correct AST structure**
  - `Score.from_elements(part("violin"), note("c"))` now wraps declarations and events in `PartNode`, so instruments are properly honored during MIDI generation.
  - Previously, bare `PartDeclarationNode` objects were ignored by the MIDI generator, causing all parts to play as piano (program 0).

- **`to_alda()` now handles `PartNode` and `ChordNode` correctly**
  - `PartNode` is now explicitly rendered with its declaration and events, preserving part structure in round-trips.
  - `ChordNode` no longer crashes with `AttributeError` (it has no `duration` attribute; individual notes carry their durations).

- **Variable definitions no longer emit sound immediately**
  - `theme = c d e` now only stores the events; sound is emitted only when the variable is referenced (`theme`).
  - Previously, definitions would double-play content (once at definition, once at each reference).

- **MIDI file writer now handles tempo changes correctly**
  - Introduced `TempoMap` class that properly integrates tick positions across tempo segments.
  - Channel tracks now use the same tempo map as the tempo track, ensuring notes align with tempo changes.
  - Previously, tempo changes mid-score caused timing drift because timestamps were converted using a single constant tempo.

## [0.1.5]

### Added

- **Transcription expressiveness**
  - Real-time transcribe now supports swing, triplet, and quintuplet feels with true ties and dotted durations.
  - Tuplet groups (including block chords) collapse into `{ ... }` cram expressions instead of long tie chains.
  - Chord transcription preserves ties/tuplets so block chords inherit the same rhythmic structure as monophonic lines.
  - Per-recording metadata (feel, swing ratio, tuplet division) is surfaced via `Seq.metadata` for downstream tweaking.
  - CLI `transcribe` command exposes `--feel` and `--swing-ratio` flags; transcription tuples populate the metadata.
  - MIDI import emits in-part tempo changes whenever `MidiTempoChange` events appear after the initial tempo.
- **CLI improvements**
  - `aldakit ports` now lists both output and input ports by default, with `-o/--outputs` or `-i/--inputs` filtering as needed.
  - `--port` accepts both port names and numeric indices (e.g., `--port 0` uses the first port from `aldakit ports`).
  - Single-port auto-selection: when only one MIDI port is available, it is automatically selected without requiring `--port`.

### Changed

- `Score.save()` writes `.mid` files directly via the SMF writer, dropping the libremidi dependency for exports.
- `Seq` now carries an optional metadata dictionary that concatenation preserves; `seq(..., metadata=...)` helper added.
- `Chord` objects support dotted durations so tuplets and other rhythmic transforms can express chords without losing dots.
- CLI stdin mode and version flag use `aldakit.__version__` and context-managed backends to avoid leaked ports.
- Transcription timing constants extracted as named module-level constants for clarity and tuning.

### Fixed

- `beat_to_duration` now handles pytest `approx` inputs and the expanded duration catalog adds triplet/quintuplet denominators.
- Libremidi-dependent tests are skipped automatically when the native extension is unavailable.
- CLI validates `--swing-ratio` is in range (0, 1) before transcription.

## [0.1.4]

### Added

#### High-Level Python API

- New `Score` class for working with Alda music (`from aldakit import Score`)
  - `Score(source)` and `Score.from_file(path)` constructors
  - `Score.from_elements(*elements)` for programmatic composition
  - `Score.from_parts(*parts)` convenience constructor
  - `play(port=None, wait=True)` method for MIDI playback
  - `save(path)` method for MIDI/Alda file export
  - `to_alda()` method for Alda source code export
  - Lazy `ast` and `midi` properties (computed and cached on first access)
  - `duration` property for total score length in seconds
  - Builder methods: `add()`, `with_part()`, `with_tempo()`, `with_volume()`
- Module-level convenience functions for one-liner usage:
  - `aldakit.play(source)` - parse and play Alda code
  - `aldakit.play_file(path)` - parse and play an Alda file
  - `aldakit.save(source, path)` - parse and save as MIDI
  - `aldakit.save_file(source_path, output_path)` - convert Alda file to MIDI
  - `aldakit.list_ports()` - list available MIDI output ports

#### Compose Module (`aldakit.compose`)

Programmatic music composition with domain objects that generate AST directly (no text parsing):

- **Core elements:**
  - `note(pitch, *, duration, octave, accidental, dots, ms, seconds, slurred)` - create notes
  - `rest(*, duration, dots, ms, seconds)` - create rests
  - `chord(*notes_or_pitches, duration)` - create chords
  - `seq(*elements)` - create sequences
  - `Seq.from_alda(source)` - parse Alda into a sequence
- **Part declarations:**
  - `part(*instruments, alias)` - instrument declarations
- **Attributes:**
  - `tempo(bpm, global_)` - tempo setting
  - `volume(level)` / `vol(level)` - volume control
  - `octave(n)` - set octave
  - `octave_up()` / `octave_down()` - relative octave changes
  - `quant(level)` - quantization/legato
  - `panning(level)` - stereo panning
  - Dynamic markings: `pp()`, `p()`, `mp()`, `mf()`, `f()`, `ff()`
- **Transformations:**
  - `Note.sharpen()` / `Note.flatten()` - accidental changes
  - `Note.transpose(semitones)` - pitch transposition
  - `Note.with_duration(n)` / `Note.with_octave(n)` - property changes
  - `note * n` / `seq * n` - repeat syntax
- **Advanced elements:**
  - `cram(*elements, duration)` - tuplet/cram expressions
  - `voice(number, *elements)` - voice for polyphonic writing
  - `voice_group(*voices)` - group of parallel voices
  - `var(name, *elements)` - variable definition
  - `var_ref(name)` - variable reference
  - `marker(name)` - marker (synchronization point)
  - `at_marker(name)` - jump to marker
- **Output methods:**
  - `to_ast()` - generate AST node directly
  - `to_alda()` - generate Alda source code

#### Scales Module (`aldakit.compose.scales`)

Music theory utilities for scales and modes:

- **Scale generation:**
  - `scale(root, scale_type)` - get pitch names for a scale
  - `scale_notes(root, scale_type, octave, duration, ascending)` - generate Seq of notes
  - `scale_degree(root, scale_type, degree, octave)` - get specific scale degree
  - `mode(root, mode_name)` - alias for scale (ionian, dorian, etc.)
- **Key relationships:**
  - `relative_minor(major_root)` - get relative minor (C -> A)
  - `relative_major(minor_root)` - get relative major (A -> C)
  - `parallel_minor(major_root)` - same root, minor mode
  - `parallel_major(minor_root)` - same root, major mode
- **Utilities:**
  - `transpose_scale(pitches, semitones)` - transpose pitch list
  - `interval_name(semitones)` - get interval name (0 -> "unison", 7 -> "perfect fifth")
  - `list_scales()` - list available scale types
  - `SCALE_INTERVALS` - dictionary of scale interval patterns
- **Scale types:** major, minor, harmonic-minor, melodic-minor, pentatonic, blues, chromatic, whole-tone, diminished, dorian, phrygian, lydian, mixolydian, locrian, japanese, arabic, hungarian-minor, spanish, bebop-dominant, bebop-major

#### Chords Module (`aldakit.compose.chords`)

Chord voicing utilities for building common chord types:

- **Core builder:**
  - `build_chord(root, chord_type, octave, duration, inversion)` - build any chord type
- **Triad constructors:**
  - `major(root)`, `minor(root)`, `dim(root)`, `aug(root)`, `sus2(root)`, `sus4(root)`
- **Seventh chord constructors:**
  - `maj7(root)`, `min7(root)`, `dom7(root)`, `dim7(root)`, `half_dim7(root)`, `min_maj7(root)`, `aug7(root)`
- **Extended chord constructors:**
  - `maj6(root)`, `min6(root)`, `dom9(root)`, `maj9(root)`, `min9(root)`, `add9(root)`, `power(root)`
- **Chord utilities:**
  - `arpeggiate(chord, pattern, duration)` - convert chord to arpeggio note sequence
  - `invert_chord(chord, inversion)` - apply inversion to existing chord
  - `voicing(chord, octaves)` - apply custom octave voicing (spread, close, etc.)
  - `list_chord_types()` - list available chord types
  - `CHORD_INTERVALS` - dictionary of chord interval patterns

#### Transform Module (`aldakit.compose.transform`)

AST-level transformers for musical sequences (can be exported back to Alda):

- **Pitch transformers:**
  - `transpose(seq, semitones)` - transpose all notes by semitones
  - `invert(seq, axis)` - invert intervals around an axis pitch
  - `reverse(seq)` - retrograde (reverse order)
  - `shuffle(seq, seed)` - random permutation
  - `retrograde_inversion(seq)` - combined reverse + invert
- **Structural transformers:**
  - `augment(seq, factor)` - lengthen durations (e.g., 8th -> quarter)
  - `diminish(seq, factor)` - shorten durations (e.g., quarter -> 8th)
  - `fragment(seq, length)` - take first N elements
  - `loop(seq, times)` - repeat sequence (explicit duplication)
  - `interleave(*seqs)` - alternate elements from multiple sequences
  - `rotate(seq, positions)` - rotate elements left/right
  - `take_every(seq, n, offset)` - sample every Nth element
  - `split(seq, size)` - split into chunks
  - `concat(*seqs)` - concatenate sequences
- **Helpers:**
  - `pipe(seq, *transforms)` - chain multiple transformations
  - `identity(seq)` - return sequence unchanged (placeholder)

#### MIDI Transform Module (`aldakit.midi.transform`)

MIDI-level transformers for post-MIDI-generation processing (operates on absolute timing, cannot be reversed to Alda):

- **Timing transformers:**
  - `quantize(seq, grid, strength)` - snap note start times to a grid
  - `humanize(seq, timing, velocity, duration, seed)` - add random variations
  - `swing(seq, grid, amount)` - apply swing feel to offbeat notes
  - `stretch(seq, factor)` - scale all timings by a factor
  - `shift(seq, seconds)` - shift all notes forward/backward in time
- **Velocity transformers:**
  - `accent(seq, pattern, amount, base_velocity)` - apply accent pattern
  - `crescendo(seq, start_vel, end_vel, start_time, end_time)` - gradual velocity increase
  - `diminuendo(seq, start_vel, end_vel, start_time, end_time)` - gradual velocity decrease
  - `normalize(seq, target)` - scale velocities to target maximum
  - `velocity_curve(seq, func)` - apply custom velocity transformation
  - `compress(seq, threshold, ratio)` - compress dynamic range
- **Filtering:**
  - `filter_notes(seq, predicate)` - keep notes matching a condition
  - `trim(seq, start, end)` - extract a time range
- **Combining:**
  - `merge(*seqs)` - combine multiple sequences into one
  - `concatenate(*seqs, gap)` - append sequences end-to-end

#### Generate Module (`aldakit.compose.generate`)

Generative functions for algorithmic music composition:

- **Random selection:**
  - `random_note(scale, duration, octave, seed)` - random note from a scale
  - `random_choice(options, seed)` - random selection from options
  - `weighted_choice(weighted_options, seed)` - probability-weighted selection
- **Random walks:**
  - `random_walk(start, steps, intervals, duration, ...)` - pitch random walk
  - `drunk_walk(start, steps, max_step, bias, ...)` - biased toward smaller intervals
- **Rhythmic generators:**
  - `euclidean(hits, steps, pitch, rotate)` - Euclidean rhythm patterns (tresillo, cinquillo, etc.)
  - `probability_seq(notes, length, probability)` - probabilistic note/rest sequence
  - `rest_probability(seq, probability)` - add random rests to existing sequence
- **Markov chains:**
  - `markov_chain(transitions)` - create chain from transition probabilities
  - `learn_markov(sequence, order)` - learn transitions from existing melody
  - `MarkovChain.generate(start, length)` - generate new melodies
- **L-Systems:**
  - `lsystem(axiom, rules, iterations, note_map)` - Lindenmayer system patterns
- **Cellular automata:**
  - `cellular_automaton(rule, width, steps, pitch_on)` - Wolfram elementary automata (rules 0-255)
- **Shift registers:**
  - `shift_register(length, taps, bits, scale, mode)` - Linear Feedback Shift Register patterns
  - `turing_machine(length, bits, scale, probability)` - Music Thing Modular-style evolving loops

#### MIDI Import (`aldakit.midi.smf_reader`, `aldakit.midi.midi_to_ast`)

Import MIDI files and convert them to Alda for editing, manipulation, and playback:

- **Score methods:**
  - `Score.from_midi_file(path, quantize_grid)` - import a MIDI file as a Score
  - `Score.from_file(path)` - now auto-detects .mid/.midi files and imports them
  - `score.to_alda()` - export imported MIDI as Alda source code
  - `score.save(path)` - re-save imported MIDI or convert to Alda
- **MIDI file reader:**
  - Full Standard MIDI File (SMF) format 0 and 1 support
  - Tempo map parsing with proper timing conversion
  - Note on/off pairing with duration calculation
  - Program change detection for instrument assignment
  - Variable-length quantity parsing
- **MIDI to AST conversion:**
  - MIDI pitch to note name mapping (with sharps for black keys)
  - Timing quantization to standard note values (whole, half, quarter, etc.)
  - Configurable quantization grid (default: 16th notes)
  - Chord detection for simultaneous notes
  - Rest insertion for gaps between notes
  - Multi-channel support (each channel becomes a separate part)
  - General MIDI program to instrument name mapping
- **Round-trip workflow:**
  - Import MIDI -> Edit as Alda -> Export back to MIDI
  - Import MIDI -> Apply transformations -> Play or save

#### Real-Time MIDI Transcription (`aldakit.midi.transcriber`)

Record MIDI input from a keyboard or controller and convert to Alda:

- **Module-level functions:**
  - `transcribe(duration, port_name, instrument, ...)` - record MIDI for a duration and return a Score
  - `list_input_ports()` - list available MIDI input ports
- **TranscribeSession class:**
  - `start(port_name)` - start recording from a MIDI port
  - `stop()` - stop recording and return a Seq
  - `poll()` - poll for incoming messages (call periodically)
  - `on_note(callback)` - set callback for note events (pitch, velocity, is_on)
- **Features:**
  - Thread-safe message queue (callbacks from C++ thread)
  - Automatic note duration detection
  - Configurable quantization grid
  - Gap detection and rest insertion
  - Pending note handling (notes still held when recording stops)

#### CLI Transcription Commands

- **New subcommands:**
  - `aldakit input-ports` - list available MIDI input ports
  - `aldakit transcribe` - record MIDI input and output Alda code
- **Transcribe options:**
  - `-d, --duration SECONDS` - recording duration (default: 10)
  - `-i, --instrument NAME` - instrument name (default: piano)
  - `-t, --tempo BPM` - tempo for quantization (default: 120)
  - `-q, --quantize GRID` - quantize grid in beats (default: 0.25 = 16th notes)
  - `-o, --output FILE` - save to file (.alda or .mid)
  - `--port NAME` - MIDI input port name
  - `--play` - play back the recording after transcription
  - `-v, --verbose` - show notes as they are played
  - `--alda-notes` - show notes in Alda notation (with -v)

## [0.1.3]

### Changed

- **Project renamed from `pyalda` to `aldakit`** for PyPI availability
- Package import changed from `from pyalda import ...` to `from aldakit import ...`
- Virtual MIDI port renamed from "PyAldaMIDI" to "AldaKitMIDI"
- Environment variables renamed from `PYALDA_SF2_DIR`/`PYALDA_SF2_DEFAULT` to `ALDAKIT_SF2_DIR`/`ALDAKIT_SF2_DEFAULT`
- FluidSynth helper script converted from shell to Python (`scripts/fluidsynth-gm.py`)
  - Cross-platform support (macOS, Linux, Windows)
  - Added `--list`, `--gain`, `--audio-driver`, `--midi-driver` options
  - Configuration via environment variables instead of hardcoded paths
- README improvements for PyPI presentation
  - Added platform badges (PyPI, Python version, platforms, license)
  - Fixed relative URLs to use absolute GitHub URLs
  - Clarified zero-dependency claim
  - Added Python version requirement

## [0.1.1]

### Added

#### Interactive REPL

- New `aldakit repl` subcommand for interactive music composition
- Syntax highlighting with custom color scheme (notes, durations, instruments, attributes)
- Auto-completion for instrument names (triggers on 3+ characters)
- Persistent command history across sessions
- Multi-line input support (Alt+Enter)
- REPL commands: `:help`, `:quit`, `:ports`, `:instruments`, `:tempo`, `:stop`
- Ctrl+C to stop playback without exiting

#### CLI Subcommands

- `aldakit repl` - Interactive REPL with line editing and history
- `aldakit ports` - List available MIDI output ports
- `aldakit play` - Explicit play command (also default behavior)

#### libremidi Backend

- Replaced mido and python-rtmidi with libremidi via nanobind
- Low-latency realtime MIDI playback
- Virtual MIDI port support ("AldaKitMIDI") for DAW integration
- Cross-platform support (macOS CoreMIDI, Linux ALSA, Windows)
- Explicit platform API selection (CoreMIDI on macOS, ALSA on Linux, WinMM on Windows)
- Support for both hardware and virtual/software MIDI ports (FluidSynth, IAC Driver, etc.)

#### Pure Python MIDI File Writer

- New `smf.py` module for Standard MIDI File output
- No external dependencies for MIDI file generation
- Support for tempo changes, program changes, control changes

#### Scripts and Documentation

- `scripts/fluidsynth-gm.sh` helper script for FluidSynth General MIDI setup
- Architecture diagram (`docs/architecture.d2`)
- Design document for programmatic API extension (`docs/extending-aldakit.md`)

### Changed

- Project renamed from `alda` to `aldakit`
- CLI uses subcommands instead of flags for major modes
- Virtual port name changed to "AldaKitMIDI"
- REPL prompt changed to `aldakit>`

### Removed

- mido dependency
- python-rtmidi dependency
- MidoBackend and RtMidiBackend classes

## [0.1.0]

### Added

#### Parser

- Hand-written recursive descent parser for the Alda music language
- Scanner (lexer) with context-sensitive tokenization for S-expressions
- 32 token types supporting all core Alda syntax
- 25+ AST node types with visitor pattern for tree traversal
- Source position tracking for error reporting

#### Core Syntax Support

- Notes (a-g) with accidentals (sharp `+`, flat `-`, natural `_`)
- Rests (`r`)
- Durations: numeric (`4`, `8`, `16`), dotted (`4.`), milliseconds (`500ms`), seconds (`2s`)
- Ties (`c1~1`) and slurs (`c~d`)
- Chords (`c/e/g`)
- Octave controls: set (`o4`), up (`>`), down (`<`)
- Parts with instrument declarations (`piano:`, `violin "v1":`)
- Multi-instrument parts (`violin/viola/cello "strings":`)
- S-expressions for attributes: `(tempo 120)`, `(vol 80)`, `(quant 90)`
- Dynamic markings: `(pp)`, `(p)`, `(mp)`, `(mf)`, `(f)`, `(ff)`, etc.
- Barlines (`|`) and comments (`# comment`)

#### Extended Syntax Support

- Variables: definition (`riff = c d e`) and reference (`riff`)
- Markers (`%verse`) and jumps (`@verse`)
- Voice groups (`V1:`, `V2:`, `V0:`)
- Cram expressions (`{c d e}2` for triplets)
- Bracketed sequences with repeats (`[c d e]*4`)
- On-repetitions (`c'1-3,5`)

#### MIDI Generation

- AST to MIDI conversion with `generate_midi()` function
- Support for 128 General MIDI instruments
- Tempo, volume, panning, and quantization handling
- Proper timing calculations for all duration types

#### MIDI Backends

- `MidoBackend`: File output (.mid) and playback via mido library
- `RtMidiBackend`: Low-latency realtime playback via python-rtmidi

#### Command-Line Interface

- `aldakit` command (or `python -m aldakit`)
- Play Alda files: `aldakit song.alda`
- Inline evaluation: `aldakit -e "piano: c d e"`
- MIDI export: `aldakit song.alda -o output.mid`
- Parse-only mode: `aldakit song.alda --parse-only`
- Backend selection: `--backend mido` or `--backend rtmidi`
- Port listing: `--list-ports`

#### Examples

- 13 example files demonstrating various features
- Simple melodies, chord progressions, dynamics
- Multi-instrument arrangements (duet, orchestra, jazz)
- Bach Prelude in C Major (simplified)

#### Testing

- 142 tests covering scanner, parser, and MIDI generation
- pytest-based test suite with `make test`
