# aldakit

[![PyPI version](https://badge.fury.io/py/aldakit.svg)](https://pypi.org/project/aldakit/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python parser and MIDI generator for the [Alda](https://alda.io) music programming language, with no install-time dependencies[^1].

[^1]: Includes a rich REPL, native MIDI, and built-in audio via bundled [prompt-toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit), [libremidi](https://github.com/jcelerier/libremidi), and [TinySoundFont](https://github.com/schellingb/TinySoundFont) respectively.

## Features

- **Alda Parser** - Full parser for the Alda music language with AST generation
- **MIDI Playback** - Low-latency playback via libremidi (CoreMIDI, ALSA, WinMM)
- **Audio Playback** - Built-in synthesis via TinySoundFont (no external synth required)
- **MIDI Export** - Save compositions as Standard MIDI Files
- **Audio Export** - Render a score to a WAV file, many times faster than real time
- **MIDI Import** - Load MIDI files and convert to Alda notation
- **Alda Export** - Serialize any AST back to Alda source, round-trip safe
- **Real-time Transcription** - Record from MIDI keyboards and convert to Alda
- **Programmatic Composition** - Build music with Python using the compose module
- **Music Theory** - Scale, chord, and interval utilities
- **Transformers** - Transpose, invert, augment, diminish, and more
- **Generative Music** - Markov chains, L-systems, cellular automata, Euclidean rhythms
- **Interactive REPL** - Syntax highlighting, auto-completion, and live playback
- **Score Checking** - `aldakit lint` and `aldakit info` report what will sound wrong before you play it
- **SoundFont Management** - Find, download and verify SoundFonts from the CLI
- **CLI Tools** - Play, transcribe, and convert from the command line

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
# Interactive REPL (default when no args)
aldakit

# Evaluate inline code
aldakit eval "piano: c d e f g"

# Play an Alda file
aldakit play examples/twinkle.alda

# Export to MIDI file
aldakit play examples/bach-prelude.alda -o bach.mid

# Use built-in audio (TinySoundFont) instead of MIDI
aldakit play -sf ~/Music/sf2/FluidR3_GM.sf2 examples/twinkle.alda
aldakit repl -sf ~/Music/sf2/FluidR3_GM.sf2

# Use audio backend with pre-configured soundfont (from config or env)
aldakit play -a examples/twinkle.alda
aldakit repl -a

# Create virtual MIDI port with custom name
aldakit repl -vp MyMIDI

# Get a SoundFont for the built-in audio backend
aldakit soundfont install

# Inspect or check a score without playing it
aldakit info examples/twinkle.alda
aldakit lint examples/twinkle.alda
```

### Python API

```python
import aldakit

# Play directly
aldakit.play("piano: c d e f g")

# Save to MIDI file
aldakit.save("piano: c d e f g", "output.mid")

# Render to audio, faster than real time and without an audio device
aldakit.render("piano: c d e f g", "output.wav")

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

# Play, blocking until finished
score.play(port="FluidSynth")

# Or play in the background and keep control of the playback
handle = score.play(port="FluidSynth", wait=False)
handle.is_playing()
handle.stop()

# Save to file
score.save("output.mid")

# Render to a WAV file
score.render("output.wav")

# Access internals
print(f"Duration: {score.duration}s")
print(score.ast)   # Parsed AST
print(score.midi)  # MIDI sequence
```

### Concurrent Playback

Layer multiple sequences for polyphonic REPL-style playback:

```python
from aldakit.midi.backends import LibremidiBackend

# Create backend with concurrent mode (default)
backend = LibremidiBackend(concurrent=True)

# Play multiple sequences - they layer on top of each other
backend.play(score1.midi)  # Starts immediately
backend.play(score2.midi)  # Layers on top of score1
backend.play(score3.midi)  # Up to 8 concurrent slots

# Check status
print(f"Active slots: {backend.active_slots}")
print(f"Playing: {backend.is_playing()}")

# Wait for all playback to complete
backend.wait()

# Or stop all playback immediately
backend.stop()

# Sequential mode - each play waits for previous to finish
backend.concurrent_mode = False
backend.play(score1.midi)  # Plays first
backend.play(score2.midi)  # Waits, then plays second
```

### MIDI Import

Import existing MIDI files and work with them as Alda:

```python
from aldakit import Score

# Import a MIDI file
score = Score.from_midi_file("recording.mid")

# Or use from_file (auto-detects .mid/.midi)
score = Score.from_file("song.mid")

# View as Alda source
print(score.to_alda())
# piano:
# o4 c4 d e f | g a b > c

# Play the imported MIDI
score.play()

# Export to Alda file
score.save("song.alda")

# Re-export to MIDI
score.save("output.mid")

# Import with custom quantization grid
# Default is 0.25 (16th notes), use 0.5 for 8th notes
score = Score.from_midi_file("recording.mid", quantize_grid=0.5)
```

Features:
- Multi-track MIDI files (each channel becomes a separate part, and stays on
  its own channel when played or re-exported)
- Channel 10 is imported as a `midi-percussion` part
- Tempo detection and preservation
- General MIDI instrument mapping across all 128 programs
- Chord detection for simultaneous notes
- Configurable timing quantization

### Real-Time MIDI Transcription

Record MIDI input from a keyboard or controller:

```python
import aldakit

# List available MIDI input ports
print(aldakit.list_input_ports())

# Record for 10 seconds from the first available port
score = aldakit.transcribe(duration=10)

# Play back what was recorded
score.play()

# Export to Alda source
print(score.to_alda())

# Record with options
score = aldakit.transcribe(
    duration=30,
    port_name="My MIDI Keyboard",
    instrument="piano",
    tempo=120,
    quantize_grid=0.25,  # Quantize to 16th notes
)
```

For more control, use `TranscribeSession`:

```python
from aldakit.midi.transcriber import TranscribeSession

session = TranscribeSession(quantize_grid=0.25, default_tempo=120)

# Set a callback for note events (optional)
session.on_note(lambda pitch, vel, on: print(f"Note: {pitch}, vel={vel}, on={on}"))

# Start recording
session.start()

# Poll periodically (in a loop or timer)
import time
for _ in range(100):
    session.poll()
    time.sleep(0.1)

# Stop and get the recorded notes
seq = session.stop()
print(seq.to_alda())
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
- **Advanced**: `cram()`, `voice()`, `voice_group()`, `var()`, `var_ref()`, `marker()`, `at_marker()`

Accidentals use Alda's characters: `"+"` (sharp), `"-"` (flat), `"_"` (natural),
repeated for double accidentals. Anything else raises `ValueError`.

The `octave` a note declares is preserved through both `to_alda()` and MIDI
generation. Octave is stateful in Alda, so a note only emits an octave change
where the octave actually changes:

```python
from aldakit import Score
from aldakit.compose import part, note

score = Score.from_elements(part("piano"), note("c", octave=5), note("d"))
print(score.to_alda())  # 'piano: o5 c d'
print([n.pitch for n in score.midi.notes])  # [72, 74]
```

### Instrument Names

All 128 General MIDI programs are available under their Alda names. Canonical
names carry the `midi-` prefix, and most have shorter aliases:

```alda
midi-acoustic-grand-piano:  # canonical
piano:                      # alias for the same program
midi-square-lead:           # synth lead
midi-percussion:            # drum kit on MIDI channel 10
```

See [docs/alda-language/list-of-instruments.md](docs/alda-language/list-of-instruments.md)
for the full list. An unrecognised instrument name falls back to acoustic grand
piano and reports a warning rather than failing silently:

```python
score = Score("bogus-instrument: c d e")
print([str(d) for d in score.diagnostics])
# ["<input>:1:1: Unknown instrument 'bogus-instrument'; falling back to acoustic grand piano."]
```

`midi-percussion` (alias `percussion`) is placed on MIDI channel 10, where note
numbers select drum sounds; key signatures and transposition do not apply to it.
Melodic parts never use that channel.

That leaves 15 channels for pitched parts, but a score is not limited to 15 of
them: a part only holds a channel while it is sounding, so a part that has
finished hands its channel to one that is about to start, and the instrument,
pan and volume are set again for the part taking over. This is how
`examples/all-instruments.alda` plays all 128 General MIDI instruments. Scores
that fit without reuse keep one channel per part, in declaration order. A
diagnostic is reported only when more than 15 pitched parts sound at the same
moment, which no amount of reuse can accommodate.

### Inspecting and Checking a Score

The problems the generator finds are available as values, not just as CLI
output, so a build step or an editor plugin can use them:

```python
from aldakit import inspect_score, lint_score

info = inspect_score("piano: c d e\ncello: c2")
print(info.note_count, info.duration)
for part in info.parts:
    print(part.name, part.instrument, part.channel, part.note_count)

for finding in lint_score("piano: nosuchvar"):
    print(finding.severity, finding.code, finding.message)
    # error undefined-variable Undefined variable 'nosuchvar'.
```

`lint_score()` reports unknown instruments and attributes, undefined variables
and markers, notes clamped into the MIDI range, unused and redefined variables,
and parts that collide on a channel. Each finding carries a `code`, a
`severity` and the source position. The `aldakit info` and `aldakit lint`
commands are thin wrappers over these two functions.

### Scales and Chords

Build melodies and harmonies using music theory helpers:

```python
from aldakit import Score
from aldakit.compose import part, tempo
from aldakit.compose import (
    # Scale functions
    scale, scale_notes, scale_degree, mode,
    relative_minor, relative_major,
    # Chord builders
    major, minor, dim, aug, maj7, min7, dom7,
    arpeggiate, invert_chord, voicing,
)

# Get scale pitches
c_major = scale("c", "major")       # ['c', 'd', 'e', 'f', 'g', 'a', 'b']
a_blues = scale("a", "blues")       # ['a', 'c', 'd', 'd+', 'e', 'g']

# Generate scale as playable notes
melody = scale_notes("c", "pentatonic", duration=8)

# Key relationships
rel_min = relative_minor("c")  # 'a' (C major -> A minor)
rel_maj = relative_major("a")  # 'c' (A minor -> C major)

# Build chords
c_maj = major("c")                    # C E G
a_min7 = min7("a")                    # A C E G
g_dom7 = dom7("g", inversion=1)       # B D F G (first inversion)

# Arpeggiate a chord
arp = arpeggiate(maj7("c"), pattern=[0, 1, 2, 3, 2, 1], duration=16)

# Custom voicing (spread chord across octaves)
spread = voicing(major("c"), [3, 4, 5])  # C3 E4 G5

# Create a I-IV-V-I progression
pitches = scale("c", "major")
progression = [
    major(pitches[0], duration=2),  # C major (I)
    major(pitches[3], duration=2),  # F major (IV)
    major(pitches[4], duration=2),  # G major (V)
    major(pitches[0], duration=1),  # C major (I)
]

score = Score.from_elements(
    part("piano"),
    tempo(100),
    *progression,
)
score.play()
```

Available scales: major, minor, harmonic-minor, melodic-minor, pentatonic, blues, chromatic, whole-tone, dorian, phrygian, lydian, mixolydian, locrian, japanese, arabic, hungarian-minor, spanish, bebop-dominant, bebop-major

Available chords: major, minor, dim, aug, sus2, sus4, maj7, min7, dom7, dim7, half_dim7, min_maj7, aug7, maj6, min6, dom9, maj9, min9, add9, power

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
crescendo_seq = crescendo(midi_seq, start_velocity=50, end_velocity=100)
normalized = normalize(midi_seq, target=100)

# Filtering and combining
filtered = filter_notes(midi_seq, lambda n: n.pitch >= 60)  # Keep notes >= middle C
trimmed = trim(midi_seq, start=0.0, end=2.0)               # First 2 seconds
merged = merge(midi_seq, another_seq)                       # Combine sequences
```

Note: MIDI transformers operate on absolute timing (seconds) and cannot be converted back to Alda notation.

### Generative Functions

Create algorithmic compositions with generative functions:

```python
from aldakit import Score
from aldakit.compose import part, tempo
from aldakit.compose.generate import (
    random_walk, euclidean, markov_chain, lsystem, cellular_automaton,
    shift_register, turing_machine,
)

# Random walk melody
melody = random_walk("c", steps=16, intervals=[-2, -1, 1, 2], duration=8, seed=42)

# Euclidean rhythms (e.g., Cuban tresillo: 3 hits over 8 steps)
rhythm = euclidean(hits=3, steps=8, pitch="c", duration=16)

# Markov chain
chain = markov_chain({
    "c": {"d": 0.5, "e": 0.3, "g": 0.2},
    "d": {"e": 0.6, "c": 0.4},
    "e": {"c": 0.5, "g": 0.5},
    "g": {"c": 1.0},
})
markov_melody = chain.generate(start="c", length=16, duration=8, seed=42)

# L-System (Fibonacci pattern)
from aldakit.compose import note, rest
fib = lsystem(
    axiom="A",
    rules={"A": "AB", "B": "A"},
    iterations=5,
    note_map={"A": note("c", duration=8), "B": note("e", duration=8)},
)

# Cellular automaton (Rule 110)
automaton = cellular_automaton(rule=110, width=8, steps=4, pitch_on="c", duration=16)

# Shift register (LFSR) - classic analog sequencer
lfsr = shift_register(16, bits=4, scale=["c", "e", "g", "b"], duration=16)

# Turing Machine - evolving loop (probability=0 for locked, higher for chaos)
turing = turing_machine(32, bits=8, probability=0.1, seed=42)

# Combine into a score
score = Score.from_elements(
    part("piano"),
    tempo(120),
    *melody.elements,
)
score.play()
```

## CLI Reference

```sh
aldakit [--version] [-h] {repl,play,eval,info,lint,ports,soundfont,transcribe} ...
```

### Subcommands

| Command | Description |
| ------- | ----------- |
| (none) | Opens the interactive REPL (default when no args) |
| `repl` | Interactive REPL with syntax highlighting and auto-completion |
| `play` | Play an Alda file |
| `eval` | Evaluate Alda code directly |
| `info` | Summarise a score: parts, instruments, channels, duration |
| `render` | Render a score to a WAV file, faster than real time |
| `lint` | Report problems in a score without playing it |
| `ports` | List available MIDI ports (both input and output) |
| `soundfont` | Find, download and verify SoundFonts for the audio backend |
| `transcribe` | Record MIDI input and output Alda code |

### Global Options

| Option | Description |
| ------ | ----------- |
| `--version` | Show version number and exit |
| `-h, --help` | Show help message |

### `play` Subcommand

```sh
aldakit play [-v] [-e CODE] [-o FILE] [--port NAME|INDEX] [-sf FILE] [-a] [-vp NAME] [--stdin] [--parse-only] [--no-wait] FILE
```

| Option | Description |
| ------ | ----------- |
| `FILE` | Alda file to play (use `-` for stdin) |
| `-e, --eval CODE` | Play Alda code given on the command line instead of a file |
| `-v, --verbose` | Verbose output |
| `-o, --output FILE` | Save to MIDI file instead of playing |
| `--port NAME\|INDEX` | MIDI port by name or index (see `aldakit ports`) |
| `-sf, --soundfont FILE` | Use TinySoundFont audio backend with specified SoundFont |
| `-a, --audio` | Use audio backend with pre-configured soundfont |
| `-vp, --virtual-port NAME` | Custom virtual MIDI port name (default: AldakitMIDI) |
| `--stdin` | Read from stdin (blank line to play) |
| `--parse-only` | Print AST without playing |
| `--no-wait` | Return without waiting for playback to finish (playback stops when the command exits) |

### `eval` Subcommand

```sh
aldakit eval [-v] [-o FILE] [-p NAME|INDEX] [-sf FILE] [-a] [-vp NAME] [--parse-only] [--no-wait] CODE
```

| Option | Description |
| ------ | ----------- |
| `CODE` | Alda code to evaluate |
| `-v, --verbose` | Verbose output |
| `--parse-only` | Print AST without playing |
| `--no-wait` | Return without waiting for playback to finish |
| `-o, --output FILE` | Save to MIDI file instead of playing |
| `-p, --port NAME\|INDEX` | MIDI port by name or index |
| `-sf, --soundfont FILE` | Use TinySoundFont audio backend |
| `-a, --audio` | Use audio backend with pre-configured soundfont |
| `-vp, --virtual-port NAME` | Custom virtual MIDI port name (default: AldakitMIDI) |

### `repl` Subcommand

```sh
aldakit repl [-v] [--port NAME|INDEX] [-sf FILE] [-a] [-vp NAME] [--sequential] [FILE]
```

| Option | Description |
| ------ | ----------- |
| `FILE` | Alda file to load on startup (use `:play` to hear it) |
| `-v, --verbose` | Verbose output |
| `--port NAME\|INDEX` | MIDI port by name or index |
| `-sf, --soundfont FILE` | Use TinySoundFont audio backend |
| `-a, --audio` | Use audio backend with pre-configured soundfont |
| `-vp, --virtual-port NAME` | Custom virtual MIDI port name (default: AldakitMIDI) |
| `--sequential` | Start in sequential mode (wait for each input) |

### `soundfont` Subcommand

The audio backend needs a General MIDI SoundFont. This finds, fetches and
checks them; downloads land in `~/.aldakit/soundfonts/` and are verified
against a SHA256 checksum.

```sh
aldakit soundfont list              # installed files and the download catalog
aldakit soundfont install           # fetch the default (TimGM6mb, 5.8 MB)
aldakit soundfont install FluidR3_GM --force
aldakit soundfont install --all
aldakit soundfont verify            # re-check the downloaded files
aldakit soundfont path              # print the one playback would use
```

If you ask for audio playback and no SoundFont can be found, aldakit offers to
download one, so the usual first run is a single prompt rather than an error.
Non-interactive runs (scripts, CI) get the error instead of a prompt.

### `info` Subcommand

```sh
aldakit info song.alda
aldakit info -e "piano: c d e"
```

Prints the parts, their instruments, MIDI programs, channels and note counts,
along with the tempo, duration, variables and markers:

```
song.alda
  parts:    2
  notes:    5
  duration: 0:02.0 (2.00s)
  tempo:    90 bpm

  part   instrument                 prog  chan   notes
  ---------------------------------------------------
  piano  midi-acoustic-grand-piano     0     0       3
  cello  midi-cello                   42     1       2
```

### `render` Subcommand

```sh
aldakit render song.alda                  # writes song.wav
aldakit render song.alda -o out.wav
aldakit render -e "piano: c d e" -o scale.wav
aldakit render song.alda --gain 0.5 --tail 2
```

Synthesizes the score with a SoundFont and writes a 16-bit stereo WAV, with no
audio device involved and without waiting for the score to play: a two and a
half minute score renders in about twelve seconds. The synthesis is the same
code path playback uses, so the file and the speakers agree.

| Option | Description |
| ------ | ----------- |
| `-o, --output FILE` | Output file (default: the input file with a `.wav` suffix) |
| `-sf, --soundfont FILE` | SoundFont to synthesize with (default: the one playback uses) |
| `-g, --gain GAIN` | Volume factor, 0.0 to 2.0, where 1.0 is unity |
| `--tail SECONDS` | Audio rendered after the last note, so release tails are not cut off |

A mix loud enough to clip is reported along with a gain that will not:

```
Warning: the mix peaked at 2.15 of full scale and was clipped. Try --gain 0.46.
```

### `lint` Subcommand

```sh
aldakit lint song.alda
aldakit lint -e "piano: c" --strict
```

Reports what will make a score sound wrong without playing it: unknown
instruments, undefined variables and markers, unknown attributes, notes clamped
into the MIDI range, unused variables, and parts that collide on a channel.

| Option | Description |
| ------ | ----------- |
| `FILE` | Alda file to check (use `-` for stdin) |
| `-e, --eval CODE` | Check Alda code given on the command line |
| `-q, --quiet` | Print nothing; report through the exit status |
| `--strict` | Exit non-zero on warnings as well as errors |

Exit status is 0 when clean, 1 when an error was found (or any finding under
`--strict`), and 2 when the score does not parse -- so `aldakit lint --strict`
works as a build step.

### `transcribe` Subcommand

```sh
aldakit transcribe [-d SEC] [-i INST] [-t BPM] [-q GRID] [-o FILE] [--port NAME] [--play] [-v] [--alda-notes] [--feel FEEL] [--swing-ratio RATIO]
```

| Option | Description |
| ------ | ----------- |
| `-d, --duration SEC` | Recording duration in seconds (default: 10) |
| `-i, --instrument NAME` | Instrument name (default: piano) |
| `-t, --tempo BPM` | Tempo for quantization (default: 120) |
| `-q, --quantize GRID` | Quantize grid in beats (default: 0.25 = 16th notes) |
| `-o, --output FILE` | Save to file (.alda or .mid) |
| `--port NAME` | MIDI input port name |
| `--play` | Play back the recording after transcription |
| `-v, --verbose` | Show notes as they are played |
| `--alda-notes` | Show notes in Alda notation (with -v) |
| `--feel FEEL` | Rhythm feel: straight, swing, triplet, quintuplet |
| `--swing-ratio RATIO` | Swing ratio between 0 and 1 (default: 0.67) |

### Examples

```bash
# Interactive REPL (default when no args)
aldakit
aldakit repl

# Evaluate inline code
aldakit eval "piano: c d e f g"

# Play a file
aldakit play examples/jazz.alda
aldakit play -v examples/jazz.alda  # verbose

# Play to a specific port (by index or name)
aldakit play --port 0 examples/twinkle.alda
aldakit play --port FluidSynth examples/twinkle.alda

# Use built-in audio (TinySoundFont) instead of MIDI
aldakit play -sf ~/Music/sf2/FluidR3_GM.sf2 examples/twinkle.alda
aldakit repl -sf ~/Music/sf2/FluidR3_GM.sf2

# Read from stdin
echo "piano: c d e f g" | aldakit play -
aldakit play --stdin

# Parse and show AST
aldakit play --parse-only examples/twinkle.alda
aldakit eval --parse-only "piano: c/e/g"

# Export to MIDI file
aldakit play examples/twinkle.alda -o twinkle.mid
aldakit eval "piano: c d e f g" -o output.mid

# List available MIDI ports
aldakit ports
aldakit ports -o  # output ports only
aldakit ports -i  # input ports only

# Record MIDI input for 10 seconds (default)
aldakit transcribe

# Record from a specific input port
aldakit transcribe --port 0
aldakit transcribe --port "My MIDI Keyboard"

# Record for 30 seconds with verbose note display
aldakit transcribe -d 30 -v

# Record with Alda-style note display
aldakit transcribe -d 10 -v --alda-notes

# Record and save to file
aldakit transcribe -o recording.alda
aldakit transcribe -o recording.mid

# Record and play back
aldakit transcribe --play

# Record with custom settings (swing feel, triplet quantization)
aldakit transcribe -d 20 -t 90 -i guitar --feel triplet --play
```

## Configuration File

aldakit supports INI-format configuration files to set default values for common options. Configuration is loaded from these locations (in priority order):

1. `./aldakit.ini` - Project-local config (current working directory)
2. `~/.aldakit/config.ini` - User config (home directory)
3. `ALDAKIT_SOUNDFONT` environment variable (for soundfont only)

CLI arguments always override config file settings.

### Example Configuration

Create `~/.aldakit/config.ini`:

```ini
[aldakit]
# Default SoundFont for audio backend
soundfont = ~/Music/sf2/FluidR3_GM.sf2

# Default backend: "midi" or "audio"
backend = midi

# Default MIDI output port (name or index)
port = FluidSynth

# Default tempo for REPL (BPM)
tempo = 120

# Enable verbose output by default
verbose = false
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `soundfont` | path | none | SoundFont path for audio backend |
| `backend` | string | `midi` | `midi` = external synths/DAWs/virtual port; `audio` = built-in TinySoundFont |
| `port` | string | none | Default MIDI output port name |
| `tempo` | integer | `120` | Default tempo for REPL (BPM) |
| `verbose` | boolean | `false` | Enable verbose output |

**Backend values:**
- `midi` (default): Uses libremidi for MIDI output. Sends to external synthesizers (FluidSynth, hardware), DAWs, or creates a virtual port ("AldakitMIDI") for routing.
- `audio`: Uses built-in TinySoundFont for direct audio output. Requires a `soundfont` to be configured. No external MIDI setup needed.

### Backend Selection Priority

1. CLI `-sf /path/to/soundfont.sf2` forces the audio backend with that SoundFont
2. CLI `-a` / `--audio` forces the audio backend, using a configured or
   discovered SoundFont
3. Config `backend = audio` uses the audio backend
4. If MIDI output ports are available, use MIDI (default)
5. If no MIDI ports are available and a SoundFont can be found, use audio
6. Otherwise create a virtual MIDI port ("AldakitMIDI") and warn that nothing
   will be heard until a synth or DAW connects to it

A SoundFont counts as available if it is named by `-sf`, by `soundfont` in the
config file, or by `ALDAKIT_SOUNDFONT`, or if one is discovered in a standard
location such as `~/.aldakit/soundfonts/` or `~/Music/sf2/`.

If `aldakit play` produces no sound, run `aldakit ports` to see whether any MIDI
destination exists. With no ports and no SoundFont, MIDI is being sent to a
virtual port that nothing is listening to.

### Project-Local Configuration

Create `aldakit.ini` in your project directory to override user settings:

```ini
[aldakit]
# Use audio backend with project-specific SoundFont
backend = audio
soundfont = ./sounds/project-soundfont.sf2
tempo = 140
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

| Command | Description |
| ------- | ----------- |
| `:load FILE` | Load an Alda file (does not play) |
| `:play [FILE]` | Play the loaded score, or load and play `FILE` |
| `:save FILE` | Save the session to `.alda` or `.mid` |
| `:ls [DIR]` | List Alda files and directories |
| `:cd [DIR]` | Change directory |
| `:pwd` | Show the current directory |
| `:clear` | Forget the session so far |
| `:ports` | List MIDI ports |
| `:instruments` | List available instruments |
| `:tempo [BPM]` | Show/set the default tempo |
| `:stop` | Stop playback |
| `:status` | Show playback status |
| `:concurrent` / `:sequential` | Switch playback mode |
| `:help` | Show help |
| `:quit` | Exit the REPL |

Open a file directly from the command line. It is loaded, not played, so the
REPL is ready immediately:

```sh
$ aldakit repl examples/twinkle.alda
Loaded examples/twinkle.alda (1 part, 42 notes, 28.7s)
Type :play to hear it.

aldakit> :play
Playing examples/twinkle.alda...
```

Working with files inside the REPL:

```
aldakit> :ls
  examples/
  sketch.alda
aldakit> :load sketch.alda
Loaded sketch.alda (2 parts, 64 notes, 12.4s)
Type :play to hear it.
aldakit> :play
Playing sketch.alda...
aldakit> piano: c d e f g
aldakit> :save take-2.alda
Saved /home/you/music/take-2.alda
```

`:load` never plays: a file can be inspected, saved or edited before it is
heard, and opening a long score does not tie up the prompt. `:play` with no
argument replays whatever is loaded; `:play FILE` is shorthand for loading and
playing in one step. Typed input still plays as soon as you press Enter.

`:save` writes everything accepted during the session -- typed, pasted or
loaded -- as a single score. A `.mid` or `.midi` extension exports MIDI
instead. Note that this is the session's *source*, not a recording of what you
heard: in concurrent mode inputs are layered as they play, whereas the saved
file is one score read top to bottom.

A file opened with `:load` (or given on the command line) keeps its own tempo.
The REPL only prepends its default tempo to input you type that does not set
one.

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

# Key signatures
(key-sig '(g major))     # G major (F#)
(key-sig '(d minor))     # D minor (Bb)
(key-sig "f+ c+")        # Explicit accidentals

# Transposition
(transpose 5)   # Up 5 semitones
(transpose -2)  # Down 2 semitones (Bb instrument)
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
- Virtual MIDI port support (AldakitMIDI), makes it easy to just send to your DAW.
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

## Audio Backend (Built-in)

For self-contained audio playback without external synthesizers, aldakit includes a built-in audio backend powered by [TinySoundFont](https://github.com/schellingb/TinySoundFont) and [miniaudio](https://github.com/mackron/miniaudio):

- Direct audio output (no FluidSynth or DAW required)
- Cross-platform: macOS (CoreAudio), Linux (ALSA/PulseAudio), Windows (WASAPI)
- Requires a SoundFont file (.sf2) for instrument sounds
- Header-only libraries for minimal binary size

### Basic Usage

```python
from aldakit import Score

# Play with built-in audio (requires SoundFont)
score = Score("piano: c d e f g")
score.play(backend="audio")

# Specify SoundFont explicitly
score.play(backend="audio", soundfont="/path/to/FluidR3_GM.sf2")
```

### SoundFont Setup

The audio backend requires a General MIDI SoundFont file. aldakit searches these locations automatically:

- `$ALDAKIT_SOUNDFONT` environment variable
- `~/Music/sf2/`
- `~/.aldakit/soundfonts/`
- `/usr/share/soundfonts/` (Linux)

**Option 1: Download manually**

Download a SoundFont and place it in a folder such as `~/Music/sf2/`. These are
mirrored on aldakit's [soundfonts-v1](https://github.com/shakfu/aldakit/releases/tag/soundfonts-v1)
release, which is where `aldakit soundfont install` fetches them from:

- [FluidR3_GM.sf2](https://github.com/shakfu/aldakit/releases/download/soundfonts-v1/FluidR3_GM.sf2) (148 MB, high quality, MIT)
- [GeneralUser-GS.sf2](https://github.com/shakfu/aldakit/releases/download/soundfonts-v1/GeneralUser-GS.sf2) (32 MB, balanced, author's own licence)
- [TimGM6mb.sf2](https://github.com/shakfu/aldakit/releases/download/soundfonts-v1/TimGM6mb.sf2) (6 MB, compact, GPL-2)

They are third-party works, not part of aldakit;
[SOUNDFONT-LICENSES.txt](https://github.com/shakfu/aldakit/releases/download/soundfonts-v1/SOUNDFONT-LICENSES.txt)
on the same release records each licence in full.

Suggest using a `sha256sum` (macOs or Linux) or similar to verify file integrity after download:

```sh
% sha256sum FluidR3_GM.sf2
74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0  FluidR3_GM.sf2

% sha256sum GeneralUser-GS.sf2
c278464b823daf9c52106c0957f752817da0e52964817ff682fe3a8d2f8446ce  GeneralUser-GS.sf2

% sha256sum TimGM6mb.sf2
82475b91a76de15cb28a104707d3247ba932e228bada3f47bba63c6b31aaf7a1  TimGM6mb.sf2
```

On Windows (PowerShell): `Get-FileHash -Algorithm SHA256`

**Option 2: Auto-download**

```python
from aldakit.midi.soundfont import setup_soundfont, setup_all_soundfonts

# Downloads TimGM6mb.sf2 (~6 MB) to ~/.aldakit/soundfonts/
setup_soundfont()

# Or download all available SoundFonts from the catalog
setup_all_soundfonts()
```

**Option 3: Using SoundFontManager**

For more control, use the `SoundFontManager` class:

```python
from aldakit.midi.soundfont import SoundFontManager

manager = SoundFontManager()

# Find existing SoundFont
sf = manager.find()

# List all found SoundFonts
for path in manager.list():
    print(path)

# Download a specific SoundFont (with SHA256 verification)
path = manager.download("FluidR3_GM")

# Download all SoundFonts from catalog
paths = manager.setup_all()

# Verify checksums of downloaded files
results = manager.verify_checksums()
for name, valid in results.items():
    print(f"{name}: {'OK' if valid else 'FAILED'}")

# List available downloads
for name, info in manager.list_available_downloads().items():
    print(f"{name}: {info['size_mb']} MB - {info['description']}")
```

**Option 4: Environment variable**

```bash
export ALDAKIT_SOUNDFONT=/path/to/your/soundfont.sf2
```

### Using TsfBackend Directly

```python
from aldakit import Score
from aldakit.midi.backends import TsfBackend

# Create backend with specific SoundFont
with TsfBackend(soundfont="~/Music/sf2/FluidR3_GM.sf2") as backend:
    score = Score("piano: c/e/g")
    backend.play(score.midi)
    backend.wait()  # Block until playback completes

# Inspect SoundFont presets
backend = TsfBackend()
print(f"Presets: {backend.preset_count}")
for i in range(min(10, backend.preset_count)):
    print(f"  {i}: {backend.preset_name(i)}")
```

### Audio vs MIDI Backend

| Feature | Audio (`backend="audio"`) | MIDI (`backend="midi"`) |
|---------|---------------------------|-------------------------|
| External synth required | No | Yes (FluidSynth, DAW, hardware) |
| Setup complexity | Just needs SoundFont | Requires MIDI routing |
| Sound quality | Depends on SoundFont | Depends on synth |
| DAW integration | No | Yes (virtual port) |
| Latency | Very low | Very low |
| Effects (reverb, etc.) | No | Depends on synth |

**Recommendation:** Use `backend="audio"` for quick playback and standalone use. Use `backend="midi"` (default) for DAW integration, hardware synths, or when you need effects.

## MIDI Playback Setup

### Virtual Port (Recommended)

When no hardware MIDI ports are available, aldakit creates a virtual port named "AldakitMIDI". This port is visible to DAWs and other MIDI software:

1. Start the REPL: `aldakit repl`
2. In your DAW (Ableton Live, Logic Pro, etc.), look for "AldakitMIDI" in MIDI input settings
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
aldakit play --port "My MIDI Device" examples/twinkle.alda
```

### MIDI File Export

If you don't have MIDI playback set up, export to a file:

```bash
# Save to MIDI file
aldakit play examples/twinkle.alda -o twinkle.mid

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

### Golden Fixtures

Two sets of fixtures pin what the examples produce, so that an unintended
change shows up as a reviewable diff rather than as music that quietly sounds
different:

- `tests/golden/examples.json` pins the notes, channels, programs, timings and
  velocities of every example. Regenerate with `make golden`.
- `tests/golden/audio.json` pins what they sound like: every example is
  rendered with a checksum-pinned SoundFont and its loudness per channel,
  peak and length are compared. This catches what MIDI cannot -- an instrument
  that never sounds because its program change went to the wrong channel, or a
  pan that never reached the synthesizer. Regenerate with `make golden-audio`.

The audio fixtures need the SoundFont they are pinned to, which is not in the
repository, so they skip if it is absent:

```sh
make soundfont     # downloads TimGM6mb (6 MB, checksum verified)
make test-audio    # fails rather than skips if it is missing
```

CI runs `test-audio` on Linux and macOS.

### Architecture

![aldakit architecture](https://raw.githubusercontent.com/shakfu/aldakit/main/docs/assets/architecture.svg)

## License

MIT

## See Also

- [Alda](https://alda.io) - The original Alda language and reference implementation
- [Alda Cheat Sheet](https://alda.io/cheat-sheet/) - Syntax reference
- [Extending aldakit](https://github.com/shakfu/aldakit/blob/main/docs/api-design.md) - Design document for programmatic API
- [libremidi](https://github.com/celtera/libremidi) - A modern C++ MIDI 1 / MIDI 2 real-time & file I/O library. Supports Windows, macOS, Linux and WebMIDI.
- [TinySoundFont](https://github.com/schellingb/TinySoundFont) - SoundFont2 synthesizer library in a single C/C++ header
- [miniaudio](https://github.com/mackron/miniaudio) - Single-header audio playback and capture library
- [nanobind](https://github.com/wjakob/nanobind) - a tiny and efficient C++/Python bindings