# aldakit

[![PyPI version](https://badge.fury.io/py/aldakit.svg)](https://pypi.org/project/aldakit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency Python parser and MIDI generator for the [Alda](https://alda.io) music programming language[^1].

[^1]: Includes a rich repl and native MIDI support via bundled [prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) and [libremidi](https://github.com/jcelerier/libremidi) respectively.

## Installation

Requires Python 3.10+

```sh
pip install aldakit
```

Or with [uv](https://github.com/astral-sh/uv):

```sh
uv add aldakit
```

## Quick Start

### Command Line

```sh
# Evaluate inline code
aldakit -e "piano: c d e f g"

# Interactive REPL
aldakit repl

# Play an Alda file (examples available in the repository)
aldakit examples/twinkle.alda

# Export to MIDI file
aldakit examples/bach-prelude.alda -o bach.mid
```

### Python API

```python
import aldakit

# Play directly
aldakit.play("piano: c d e f g")

# Save to MIDI file
aldakit.save("piano: c d e f g", "output.mid")

# Play from file
aldakit.play_file("song.alda")

# List available MIDI ports
print(aldakit.list_ports())
```

For more control, use the `Score` class:

```python
from aldakit import Score

score = Score("""
piano:
  (tempo 120)
  o4 c4 d e f | g a b > c
""")

# Play with options
score.play(port="FluidSynth", wait=False)

# Save to file
score.save("output.mid")

# Access internals
print(f"Duration: {score.duration}s")
print(score.ast)   # Parsed AST
print(score.midi)  # MIDI sequence
```

### Programmatic Composition

Build music programmatically using the compose module:

```python
from aldakit import Score
from aldakit.compose import part, note, rest, chord, seq, tempo, volume

# Create a score from compose elements
score = Score.from_elements(
    part("piano"),
    tempo(120),
    note("c", duration=4),
    note("d"),
    note("e"),
    chord("c", "e", "g", duration=2),
)
score.play()

# Builder pattern with method chaining
score = (
    Score.from_elements(part("violin"))
    .with_tempo(90)
    .add(note("g", duration=8), note("a"), note("b"))
)

# Note transformations
c = note("c", duration=4)
c_sharp = c.sharpen()           # C#
c_up_octave = c.transpose(12)   # Up one octave

# Repeat syntax
pattern = seq(note("c"), note("d"), note("e"))
repeated = pattern * 4  # Repeat 4 times

# Export to Alda source
print(score.to_alda())  # "violin: (tempo 90) g8 a b"
```

Available compose elements:
- **Notes**: `note("c", duration=4, octave=5, accidental="+", dots=1)`
- **Rests**: `rest(duration=4)`, `rest(ms=500)`
- **Chords**: `chord("c", "e", "g")`, `chord(note("c"), note("e", accidental="+"))`
- **Sequences**: `seq(note("c"), note("d"))`, `Seq.from_alda("c d e")`
- **Parts**: `part("piano")`, `part("violin", alias="v1")`
- **Attributes**: `tempo(120)`, `volume(80)`, `octave(5)`, `panning(50)`
- **Dynamics**: `pp()`, `p()`, `mp()`, `mf()`, `f()`, `ff()`

### Transformers

Transform sequences with pitch and structural operations:

```python
from aldakit.compose import (
    note, seq,
    transpose, invert, reverse, shuffle,
    augment, diminish, fragment, loop, interleave,
    pipe,
)

# Create a motif
motif = seq(note("c", duration=8), note("d", duration=8), note("e", duration=8))

# Pitch transformers
up_fourth = transpose(motif, 5)      # Transpose up 5 semitones
inverted = invert(motif)             # Invert intervals around first note
backwards = reverse(motif)           # Retrograde

# Structural transformers
longer = augment(motif, 2)           # Double durations (8th -> quarter)
shorter = diminish(motif, 2)         # Halve durations (8th -> 16th)
first_two = fragment(motif, 2)       # Take first 2 elements
repeated = loop(motif, 4)            # Repeat 4 times (explicit)

# Chain transformations with pipe
result = pipe(
    motif,
    lambda s: transpose(s, 5),
    reverse,
    lambda s: augment(s, 2),
)

# All transforms preserve to_alda() export
print(result.to_alda())
```

### MIDI Transformers

For post-MIDI-generation processing, use MIDI-level transformers that operate on absolute timing:

```python
from aldakit import Score
from aldakit.midi.transform import (
    quantize, humanize, swing, stretch,
    accent, crescendo, normalize,
    filter_notes, trim, merge,
)

# Get MIDI sequence from a score
score = Score("piano: c d e f g a b > c")
midi_seq = score.midi

# Timing transformers
quantized = quantize(midi_seq, grid=0.25, strength=0.8)  # Snap to quarter-note grid
humanized = humanize(midi_seq, timing=0.02, velocity=10)  # Add subtle variations
swung = swing(midi_seq, grid=0.5, amount=0.3)            # Apply swing feel

# Velocity transformers
accented = accent(midi_seq, pattern=[1.0, 0.5, 0.5, 0.5])  # 4/4 accent pattern
crescendo_seq = crescendo(midi_seq, start_vel=50, end_vel=100)
normalized = normalize(midi_seq, target=100)

# Filtering and combining
filtered = filter_notes(midi_seq, lambda n: n.pitch >= 60)  # Keep notes >= middle C
trimmed = trim(midi_seq, start=0.0, end=2.0)               # First 2 seconds
merged = merge(midi_seq, another_seq)                       # Combine sequences
```

Note: MIDI transformers operate on absolute timing (seconds) and cannot be converted back to Alda notation.

## CLI Reference

```sh
aldakit [-h] [--version] [-e CODE] [-o FILE] [--port NAME]
       [--stdin] [--parse-only] [--no-wait] [-v]
       {repl,ports,play} [file]
```

### Subcommands

| Command | Description |
| ------- | ----------- |
| `repl` | Interactive REPL with syntax highlighting and auto-completion |
| `ports` | List available MIDI output ports |
| `play` | Play an Alda file or code (default behavior) |

### Options

| Option | Description |
| ------ | ----------- |
| `file` | Alda file to play (use `-` for stdin) |
| `-e, --eval CODE` | Evaluate Alda code directly |
| `-o, --output FILE` | Save to MIDI file instead of playing |
| `--port NAME` | MIDI output port name |
| `--stdin` | Read from stdin (blank line to play) |
| `--parse-only` | Print AST without playing |
| `--no-wait` | Don't wait for playback to finish |
| `-v, --verbose` | Verbose output |

### Examples

```bash
# Interactive REPL with syntax highlighting
aldakit repl

# List available MIDI ports
aldakit ports

# Play with verbose output
aldakit -v examples/jazz.alda

# Read from stdin
echo "piano: c d e f g" | aldakit -

# Parse and show AST
aldakit --parse-only -e "piano: c/e/g"

# Export to MIDI file
aldakit examples/twinkle.alda -o twinkle.mid
```

## Interactive REPL

The REPL provides an interactive environment for composing and playing Alda code:

```bash
aldakit repl
```

Features:

- Syntax highlighting
- Auto-completion for instruments (3+ characters)
- Command history (persistent across sessions)
- Multi-line paste (use platform-specific paste: ctrl-v, shift-ctrl-v, cmd-v, etc.)
- Multi-line input (Alt+Enter)
- MIDI playback control (Ctrl+C to stop)

REPL Commands:

- `:help` - Show help
- `:quit` - Exit REPL
- `:ports` - List MIDI ports
- `:instruments` - List available instruments
- `:tempo [BPM]` - Show/set default tempo
- `:stop` - Stop playback

## Alda Syntax Reference

### Notes and Rests

```alda
piano:
  c d e f g a b   # Notes
  r               # Rest
  c4 d8 e16       # With duration (4=quarter, 8=eighth, etc.)
  c4. d4..        # Dotted notes
  c500ms d2s      # Milliseconds and seconds
```

### Accidentals

```alda
c+    # Sharp
c-    # Flat
c_    # Natural
c++   # Double sharp
```

### Octaves

```alda
o4 c    # Set octave to 4
> c     # Octave up
< c     # Octave down
```

### Chords

```alda
c/e/g           # C major chord
c1/e/g          # Whole note chord
c/e/g/>c        # With octave change
```

### Ties and Slurs

```alda
c1~1            # Tied notes (duration adds)
c4~d~e~f        # Slurred notes (legato)
```

### Parts

```alda
piano: c d e

violin "v1": c d e    # With alias

violin/viola/cello "strings":   # Multi-instrument
  c d e
```

### Attributes

```alda
(tempo 120)     # Set tempo (BPM)
(tempo! 120)    # Global tempo

(vol 80)        # Volume (0-100)
(volume 80)

(quant 90)      # Quantization/legato (0-100)

(panning 50)    # Pan (0=left, 100=right)

# Dynamic markings
(pp) (p) (mp) (mf) (f) (ff)
```

### Variables

```alda
riff = c8 d e f g4

piano:
  riff riff > riff
```

### Repeats

```alda
c*4             # Repeat note 4 times
[c d e]*4       # Repeat sequence
[c d e f]*8     # 8 times
```

### Cram (Tuplets)

```alda
{c d e}4        # Triplet in quarter note
{c d e f g}2    # Quintuplet in half note
{c {d e} f}4    # Nested cram
```

### Voices

```alda
piano:
  V1: c4 d e f
  V2: e4 f g a
  V0:           # End voices
```

### Markers

```alda
piano:
  c d e f
  %chorus
  g a b > c

violin:
  @chorus       # Jump to chorus marker
  e f g a
```

## Supported Instruments

All 128 General MIDI instruments are supported. Common examples:

- `piano`, `acoustic-grand-piano`
- `violin`, `viola`, `cello`, `contrabass`
- `flute`, `oboe`, `clarinet`, `bassoon`
- `trumpet`, `trombone`, `french-horn`, `tuba`
- `acoustic-guitar`, `electric-guitar-clean`, `electric-bass`
- `choir`, `strings`, `brass-section`

See [midi/types.py](https://github.com/shakfu/aldakit/blob/main/src/aldakit/midi/types.py) for the complete mapping.

## MIDI Backend

aldakit uses [libremidi](https://github.com/jcelerier/libremidi) via [nanobind](https://github.com/wjakob/nanobind) for cross-platform MIDI I/O:

- Low-latency realtime playback
- Virtual MIDI port support (AldaPyMIDI), makes it easy to just send to your DAW.
- Pure Python MIDI file writing (no external dependencies)
- Cross-platform: macOS (CoreMIDI), Linux (ALSA), Windows (WinMM)
- Supports hardware and software/virtual MIDI ports (FluidSynth, IAC Driver, etc.)

```python
import aldakit

# List available ports
print(aldakit.list_ports())

# Play to virtual port (visible in DAWs like Ableton Live)
aldakit.play("piano: c d e f g")

# Play to a specific port
aldakit.play("piano: c d e f g", port="FluidSynth")

# Save to MIDI file
aldakit.save("piano: c d e f g", "output.mid")
```

## MIDI Playback Setup

### Virtual Port (Recommended)

When no hardware MIDI ports are available, aldakit creates a virtual port named "AldaPyMIDI". This port is visible to DAWs and other MIDI software:

1. Start the REPL: `aldakit repl`
2. In your DAW (Ableton Live, Logic Pro, etc.), look for "AldaPyMIDI" in MIDI input settings
3. Play code in the REPL - notes will be sent to your DAW

### Software Synthesizer (FluidSynth)

For high-quality General MIDI playback without hardware, use [FluidSynth](https://www.fluidsynth.org/):

```sh
# Install FluidSynth (macOS)
brew install fluidsynth

# Install FluidSynth (Debian/Ubuntu)
sudo apt install fluidsynth 

# Download a SoundFont (e.g., FluidR3_GM.sf2)
# eg. sudo apt install fluid-soundfont-gm
# Place in ~/Music/sf2/

# Start FluidSynth with CoreMIDI (macOS)
fluidsynth -a coreaudio -m coremidi ~/Music/sf2/FluidR3_GM.sf2

# In another terminal, start aldakit
aldakit repl
# aldakit> piano: c d e f g
```

A helper script is available in the [repository](https://github.com/shakfu/aldakit/tree/main/scripts):

```sh
# Set the SoundFont directory (add to your shell profile)
export ALDAPY_SF2_DIR=~/Music/sf2

# Run with default SoundFont (FluidR3_GM.sf2)
python scripts/fluidsynth-gm.py

# Or specify a SoundFont directly
python scripts/fluidsynth-gm.py /path/to/soundfont.sf2

# List available SoundFonts
python scripts/fluidsynth-gm.py --list
```

### Hardware MIDI

Connect a USB MIDI interface or synthesizer, then:

```sh
# List available ports
aldakit ports

# Play to a specific port
aldakit --port "My MIDI Device" examples/twinkle.alda
```

### MIDI File Export

If you don't have MIDI playback set up, export to a file:

```bash
# Save to MIDI file
aldakit examples/twinkle.alda -o twinkle.mid

# Open with default app
open twinkle.mid
```

## Development

### Setup

```sh
git clone https://github.com/shakfu/aldakit.git
cd aldakit
make  # Build the libremidi extension
```

### Run Tests

```sh
make test
# or
uv run pytest tests/ -v
```

### Architecture

![aldakit architecture](https://raw.githubusercontent.com/shakfu/aldakit/main/docs/assets/architecture.svg)

## License

MIT

## See Also

- [Alda](https://alda.io) - The original Alda language and reference implementation
- [Alda Cheat Sheet](https://alda.io/cheat-sheet/) - Syntax reference
- [Extending aldakit](https://github.com/shakfu/aldakit/blob/main/docs/extending-aldakit.md) - Design document for programmatic API
- [libremidi](https://github.com/celtera/libremidi) - A modern C++ MIDI 1 / MIDI 2 real-time & file I/O library. Supports Windows, macOS, Linux and WebMIDI.
- [nanobind](https://github.com/wjakob/nanobind) - a tiny and efficient C++/Python bindings