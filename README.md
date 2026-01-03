# pyalda

[![PyPI version](https://badge.fury.io/py/pyalda.svg)](https://pypi.org/project/pyalda/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A zero-dependency Python parser and MIDI generator for the [Alda](https://alda.io) music programming language[^1].

[^1]: Includes a rich repl and native MIDI support via bundled [prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) and [libremidi](https://github.com/jcelerier/libremidi) respectively.

## Installation

Requires Python 3.10+

```sh
pip install pyalda
```

Or with [uv](https://github.com/astral-sh/uv):

```sh
uv add pyalda
```

## Quick Start

### Command Line

```sh
# Evaluate inline code
pyalda -e "piano: c d e f g"

# Interactive REPL
pyalda repl

# Play an Alda file (examples available in the repository)
pyalda examples/twinkle.alda

# Export to MIDI file
pyalda examples/bach-prelude.alda -o bach.mid
```

### Python API

```python
from pyalda import parse, generate_midi, LibremidiBackend

# Parse Alda source code
ast = parse("""
piano:
  (tempo 120)
  o4 c4 d e f | g a b > c
""")

# Generate MIDI sequence
sequence = generate_midi(ast)

# Save to file
backend = LibremidiBackend()
backend.save(sequence, "output.mid")

# Or play directly
with LibremidiBackend() as backend:
    backend.play(sequence)
```

## CLI Reference

```sh
pyalda [-h] [--version] [-e CODE] [-o FILE] [--port NAME]
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
pyalda repl

# List available MIDI ports
pyalda ports

# Play with verbose output
pyalda -v examples/jazz.alda

# Read from stdin
echo "piano: c d e f g" | pyalda -

# Parse and show AST
pyalda --parse-only -e "piano: c/e/g"

# Export to MIDI file
pyalda examples/twinkle.alda -o twinkle.mid
```

## Interactive REPL

The REPL provides an interactive environment for composing and playing Alda code:

```bash
pyalda repl
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

See [midi/types.py](https://github.com/shakfu/pyalda/blob/main/src/pyalda/midi/types.py) for the complete mapping.

## MIDI Backend

pyalda uses [libremidi](https://github.com/jcelerier/libremidi) via [nanobind](https://github.com/wjakob/nanobind) for cross-platform MIDI I/O:

- Low-latency realtime playback
- Virtual MIDI port support (PyAldaMIDI), makes it easy to just send to your DAW.
- Pure Python MIDI file writing (no external dependencies)
- Cross-platform: macOS (CoreMIDI), Linux (ALSA), Windows (WinMM)
- Supports hardware and software/virtual MIDI ports (FluidSynth, IAC Driver, etc.)

```python
from pyalda import LibremidiBackend

backend = LibremidiBackend()

# List available ports
print(backend.list_output_ports())

# Play to virtual port (visible in DAWs like Ableton Live)
backend.play(sequence)

# Save to MIDI file
backend.save(sequence, "output.mid")
```

## MIDI Playback Setup

### Virtual Port (Recommended)

When no hardware MIDI ports are available, pyalda creates a virtual port named "PyAldaMIDI". This port is visible to DAWs and other MIDI software:

1. Start the REPL: `pyalda repl`
2. In your DAW (Ableton Live, Logic Pro, etc.), look for "PyAldaMIDI" in MIDI input settings
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

# In another terminal, start pyalda
pyalda repl
# pyalda> piano: c d e f g
```

A helper script is available in the [repository](https://github.com/shakfu/pyalda/tree/main/scripts):

```sh
# Set the SoundFont directory (add to your shell profile)
export PYALDA_SF2_DIR=~/Music/sf2

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
pyalda ports

# Play to a specific port
pyalda --port "My MIDI Device" examples/twinkle.alda
```

### MIDI File Export

If you don't have MIDI playback set up, export to a file:

```bash
# Save to MIDI file
pyalda examples/twinkle.alda -o twinkle.mid

# Open with default app
open twinkle.mid
```

## Development

### Setup

```sh
git clone https://github.com/shakfu/pyalda.git
cd pyalda
make  # Build the libremidi extension
```

### Run Tests

```sh
make test
# or
uv run pytest tests/ -v
```

### Architecture

![pyalda architecture](https://raw.githubusercontent.com/shakfu/pyalda/main/docs/assets/architecture.svg)

## License

MIT

## See Also

- [Alda](https://alda.io) - The original Alda language and reference implementation
- [Alda Cheat Sheet](https://alda.io/cheat-sheet/) - Syntax reference
- [Extending pyalda](https://github.com/shakfu/pyalda/blob/main/docs/extending-pyalda.md) - Design document for programmatic API
- [libremidi](https://github.com/celtera/libremidi) - A modern C++ MIDI 1 / MIDI 2 real-time & file I/O library. Supports Windows, macOS, Linux and WebMIDI.
- [nanobind](https://github.com/wjakob/nanobind) - a tiny and efficient C++/Python bindings