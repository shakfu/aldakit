# aldakit

A zero-dependency Python parser and MIDI generator for the [Alda](https://alda.io) music programming language.

## Features

- **Alda Parser** — Full parser for the Alda music language with AST generation
- **MIDI Playback** — Low-latency playback via libremidi (CoreMIDI, ALSA, WinMM)
- **Audio Playback** — Built-in synthesis via TinySoundFont (no external synth required)
- **MIDI Export** — Save compositions as Standard MIDI Files
- **MIDI Import** — Load MIDI files and convert to Alda notation
- **Real-time Transcription** — Record from MIDI keyboards and convert to Alda
- **Programmatic Composition** — Build music with Python using the compose module
- **Music Theory** — Scale, chord, and interval utilities
- **Transformers** — Transpose, invert, augment, diminish, and more
- **Generative Music** — Markov chains, L-systems, cellular automata, Euclidean rhythms
- **Interactive REPL** — Syntax highlighting, auto-completion, and live playback
- **CLI Tools** — Play, transcribe, and convert from the command line

## Quick Start

### Installation

Requires Python 3.10+

```sh
pip install aldakit
```

### Command Line

```sh
# Interactive REPL (default when no args)
aldakit

# Evaluate inline code
aldakit eval "piano: c d e f g"

# Play an Alda file
aldakit play examples/twinkle.alda

# Export to MIDI file
aldakit play examples/bach-prelude.alda -o bach.mid

# Use built-in audio instead of MIDI
aldakit play -a examples/twinkle.alda
```

### Python API

```python
import aldakit

# Play directly
aldakit.play("piano: c d e f g")

# Save to MIDI file
aldakit.save("piano: c d e f g", "output.mid")

# Use the Score class for more control
from aldakit import Score

score = Score("piano: (tempo 120) o4 c4 d e f | g a b > c")
score.play()
score.save("output.mid")
print(f"Duration: {score.duration}s")
```

## Architecture

![aldakit architecture](assets/architecture.svg)

The AST is the central hub — all inputs flow into it, all outputs derive from it:

| Input | Operation | Output | Operation |
|-------|-----------|--------|-----------|
| Alda Source | `parse()` | Alda Source | `export()` |
| MIDI File | `import()` | MIDI File | `save()` |
| Python API | `to_ast()` | MIDI Playback | `play()` |
| MIDI Input | `transcribe()` | Audio Output | `play(backend="audio")` |
