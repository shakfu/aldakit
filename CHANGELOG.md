# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### High-Level Python API

- New `Score` class for working with Alda music (`from aldakit import Score`)
  - `Score(source)` and `Score.from_file(path)` constructors
  - `play(port=None, wait=True)` method for MIDI playback
  - `save(path)` method for MIDI file export
  - Lazy `ast` and `midi` properties (computed and cached on first access)
  - `duration` property for total score length in seconds
- Module-level convenience functions for one-liner usage:
  - `aldakit.play(source)` - parse and play Alda code
  - `aldakit.play_file(path)` - parse and play an Alda file
  - `aldakit.save(source, path)` - parse and save as MIDI
  - `aldakit.save_file(source_path, output_path)` - convert Alda file to MIDI
  - `aldakit.list_ports()` - list available MIDI output ports

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
