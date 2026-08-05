# Extending aldakit: A Programmatic API

This document describes the design of aldakit's programmatic composition API,
which extends the library beyond parsing the Alda language. The key insight is
that **the AST is the central hub** - all inputs flow into it, all outputs
derive from it.

Everything described here is implemented. The "Implementation Phases" section
is kept as a record of how the API was built up, not as a roadmap. For a
task-oriented guide to the same API, see the "Programmatic Composition" section
of the [README](https://github.com/shakfu/aldakit#programmatic-composition).

## Architecture Overview

![aldakit architecture](assets/architecture.svg)

**AST as the central hub** - symmetric operations:

| Input | Operation | Output | Operation |
|-------|-----------|--------|-----------|
| Alda Source | `parse()` | Alda Source | `export()` |
| MIDI File | `import()` | MIDI File | `save()` |
| Python API | `to_ast()` | MIDI Playback | `play()` |
| MIDI Input | `transcribe()` | | |

The programmatic API creates domain objects that can:

1. Generate AST nodes directly via `to_ast()` (for MIDI output)
2. Serialize to Alda source code via `to_alda()` (for debugging, export, or interop)

## Design Principles

### 1. AST as Central Hub, Alda as Text Format

The AST is the canonical internal representation. The Alda language serves as the human-readable text format for import/export:

```python
from aldakit import Score
from aldakit.compose import note, part, tempo

# Build programmatically
score = Score.from_elements(
    part("piano"),
    tempo(120),
    note("c", duration=4),
    note("d"),
    note("e")
)

# Export to Alda source
print(score.to_alda())
# Output: piano: (tempo 120) c4 d e

# Or play directly (bypasses text, goes straight to AST -> MIDI)
score.play()
```

### 2. Composable Domain Objects

Every musical element is a first-class Python object that can be composed, transformed, and introspected:

```python
# Notes are objects - all parameters are explicit keywords
n = note("c", duration=4, accidental="+")   # "+" sharp, "-" flat, "_" natural
n.transpose(2)  # Returns new note: d#4

# Chords are collections of notes
c_major = chord(note("c"), note("e"), note("g"))
c_minor = chord(note("c"), note("e", accidental="-"), note("g"))

# Sequences are transformed by functions, which return new sequences
from aldakit.compose import invert, reverse, transpose

melody = seq(note("c"), note("d"), note("e"), note("f"))
reverse(melody)        # f e d c (retrograde)
invert(melody)         # Invert intervals around the first note
transpose(melody, 5)   # Up a perfect fourth
```

## API Design

### Core Domain Objects

```python
from aldakit import Score
from aldakit.compose import (
    Part, Voice,
    note, rest, chord, seq,
    tempo, volume, octave,
)
```

#### Notes

All note parameters use explicit keyword arguments to avoid ambiguity:

```python
# Basic note - pitch is the only positional argument
note("c")                              # c (quarter note, default octave 4)
note("c", duration=4)                  # c4 (quarter note, explicit)
note("c", duration=8, accidental="+")  # c+8 (eighth note, sharp)
note("c", duration=4, dots=1)          # c4. (dotted quarter)
note("c", ms=500)                      # c500ms (milliseconds)
note("c", seconds=2)                   # c2s (seconds)
note("c", octave=5)                    # o5 c (set octave)

# Note attributes
n = note("c", duration=4)
n.pitch                      # "c"
n.duration                   # 4
n.octave                     # 4 (default)
n.accidental                 # None
n.midi_pitch                 # 60 (computed from pitch + octave + accidental)

# Transformations (return new notes, immutable)
n.sharpen()                  # c+4
n.flatten()                  # c-4
n.transpose(semitones=2)     # d4 (up 2 semitones)
n.transpose(semitones=-12)   # c3 (down an octave)
n.with_duration(8)           # c8 (new note with different duration)
n.with_octave(5)             # c4 in octave 5
```

#### Rests

```python
rest()                       # r (quarter rest)
rest(duration=2)             # r2 (half rest)
rest(ms=1000)                # r1000ms (one second rest)
```

#### Chords

```python
# Building chords
chord(note("c"), note("e"), note("g"))           # c/e/g
chord("c", "e", "g")                              # Shorthand
chord("c", "e", "g", duration=1)                  # c1/e/g

# Named chords (convenience constructors)
from aldakit.compose.chords import major, minor, dim, aug, dom7

major("c")                   # c/e/g
minor("a")                   # a/c/e
dom7("g")                    # g/b/d/f
```

#### Sequences

```python
# Explicit sequence
melody = seq(note("c"), note("d"), note("e"), note("f"))

# From string (parsed as Alda)
melody = Seq.from_alda("c d e f g")

# Repeat
melody * 4                   # [c d e f]*4

# Concatenate
intro + verse + chorus       # Sequences combine
```

#### Parts and Instruments

```python
# A part declares instruments; the events that follow it belong to it
from aldakit import Score

Score.from_elements(part("piano"), note("c"), note("d"), note("e"))

# With alias
part("violin", alias="v1")

# Multi-instrument
part("violin", "viola", "cello", alias="strings")
```

#### Attributes

```python
tempo(120)                   # (tempo 120)
tempo(120, global_=True)     # (tempo! 120)
volume(80)                   # (vol 80)
quant(90)                    # (quant 90)
panning(50)                  # (panning 50)
octave(5)                    # o5
octave_up()                  # >
octave_down()                # <

# Dynamics
from aldakit.compose import pp, p, mp, mf, f, ff
```

### The Unified Score Class

The `Score` class serves as the unified entry point for all inputs. It provides multiple
class methods for construction and instance methods for manipulation and output.

**Note on Parts vs Voices:**
- **Parts** = instruments (piano, violin) - independent channels with different timbres
- **Voices** = parallel lines *within* a single part (e.g., V1/V2 for counterpoint)

```python
from aldakit import Score

class Score:
    """Unified score class for parsing, building, and playing music."""

    # === Construction: one per kind of input ===

    def __init__(self, source: str, filename: str = "<input>") -> None:
        """Create from Alda source code."""

    @classmethod
    def from_source(cls, source: str, filename: str = "<input>") -> "Score":
        """Same as the constructor, spelled to match the others."""

    @classmethod
    def from_file(cls, path: str | Path) -> "Score":
        """Create from a .alda or .mid/.midi file, chosen by extension."""

    @classmethod
    def from_midi_file(cls, path: str | Path, *, quantize_grid: float = 0.25) -> "Score":
        """Import a MIDI file: one part per channel, instruments preserved."""

    @classmethod
    def from_elements(cls, *elements: ComposeElement) -> "Score":
        """Create from compose domain objects (notes, parts, attributes)."""

    @classmethod
    def from_parts(cls, *parts: Part) -> "Score":
        """Create from Part objects alone."""

    # === Builder methods (return self, so they chain) ===

    def add(self, *elements: ComposeElement) -> "Score":
        """Append elements. Only valid for a score built from elements."""

    def with_part(self, *instruments: str, alias: str | None = None) -> "Score": ...
    def with_tempo(self, bpm: int | float, global_: bool = False) -> "Score": ...
    def with_volume(self, level: int | float) -> "Score": ...

    # === Properties (lazy, cached, invalidated by add()) ===

    @cached_property
    def ast(self) -> RootNode:
        """The AST, however this score was built."""

    @cached_property
    def midi(self) -> MidiSequence:
        """The generated MIDI sequence."""

    @property
    def diagnostics(self) -> list[Diagnostic]:
        """Non-fatal problems found while generating MIDI."""

    @property
    def elements(self) -> list[ComposeElement]:
        """The compose elements this score was built from (a copy)."""

    @property
    def source(self) -> str: ...
    @property
    def duration(self) -> float: ...

    # === Output ===

    def to_alda(self) -> str:
        """Export as Alda source."""

    def play(self, ..., wait: bool = True) -> PlaybackHandle | None:
        """Play the score. With wait=False, returns a handle that owns the
        backend, so playback continues until you stop it."""

    def save(self, path: str | Path) -> None:
        """Save as .alda or .mid, chosen by extension."""
```

Where a score's music comes from -- source text, compose elements, or an
imported AST -- is a `ScoreContent` in `aldakit/score_content.py`. Each of the
three answers the same three questions (build an AST, export Alda source,
describe itself), so `Score` holds one of them rather than branching on how it
was constructed.

## Use Cases

### 1. Algorithmic Composition

```python
import random
from aldakit import Score
from aldakit.compose import note, chord, seq, part, tempo

def random_melody(length=8, scale=["c", "d", "e", "f", "g", "a", "b"]):
    """Generate a random melody from a scale."""
    return seq(*[note(random.choice(scale), duration=8) for _ in range(length)])

def arpeggiate(chord_notes, pattern=[0, 1, 2, 1]):
    """Arpeggiate a chord with a pattern."""
    return seq(*[note(chord_notes[i % len(chord_notes)], duration=16) for i in pattern])

score = Score.from_elements(
    part("piano"),
    tempo(120),
    random_melody(16),
    arpeggiate(["c", "e", "g"]) * 4
)
score.play()
```

### 2. Data Sonification

```python
from aldakit import Score
from aldakit.compose import note, part, tempo

def weather_to_music(temperatures: list[float]) -> Score:
    """Convert temperature data to music."""
    min_t, max_t = min(temperatures), max(temperatures)

    def temp_to_note(t):
        # Map to pentatonic scale (C D E G A)
        scale = ["c", "d", "e", "g", "a"]
        idx = int((t - min_t) / (max_t - min_t) * (len(scale) - 1))
        return note(scale[idx], duration=8)

    notes = [temp_to_note(t) for t in temperatures]
    return Score.from_elements(
        part("vibraphone"),
        tempo(140),
        *notes
    )

# Sonify a week of temperatures
temps = [45, 52, 48, 61, 58, 55, 50]
weather_to_music(temps).play()
```

### 3. Music Theory Operations

```python
from aldakit import Score
from aldakit.compose import seq, rest, part
from aldakit.compose import Seq
from aldakit.compose.transform import transpose, invert, reverse

# Define a motif
motif = Seq.from_alda("c8 d e- g")

# Transform it
motif_up = transpose(motif, semitones=5)   # Up a fourth
motif_inv = invert(motif)                   # Invert intervals
motif_ret = reverse(motif)                  # Retrograde

# Build a fugue-like structure
score = Score.from_elements(
    part("piano"),
    motif,
    rest(duration=2),
    motif_up,
    rest(duration=2),
    motif_inv
)
score.play()
```

### 4. Live Coding / REPL Workflow

```python
>>> from aldakit import Score
>>> from aldakit.compose import note, chord, part, tempo

# Start with empty score, build incrementally
>>> s = Score.from_elements(part("piano"), tempo(120))
>>> s.add(note("c", duration=4), note("e"), note("g"))
>>> s.play()
# Hear: C E G

>>> s.add(chord("c", "e", "g", duration=1))
>>> s.play()
# Hear: C E G, then C major chord (whole note)

>>> print(s.to_alda())
piano:
(tempo 120)
c4 e g
c1/e/g
```

### 5. Interoperability with Alda

```python
from aldakit import Score
from aldakit.compose import Seq
from aldakit.compose.transform import transpose

# Load an Alda file
score = Score.from_file("song.alda")

# Transform a sequence, then rebuild a score from it
motif = Seq.from_alda(score.to_alda())
transposed = transpose(motif, 2)

new_score = Score.from_elements(transposed)
new_score.save("song_transposed.alda")
new_score.save("song_transposed.mid")
```

## Implementation Strategy

### Phase 1: Core Domain Objects with Direct AST Generation

All domain objects implement `to_ast()` from the start - no text round-trip:

1. Implement `note()`, `rest()`, `chord()`, `seq()` with `to_ast()` methods
2. Implement `Part`, `tempo()`, `volume()`, and other attributes with `to_ast()`
3. Extend `Score` class with `from_elements()`, `from_parts()`, builder methods
4. Add `to_alda()` methods for debugging/export (generates text from AST, not vice versa)

```python
# Domain objects create AST nodes directly
class Note:
    def to_ast(self) -> NoteNode:
        return NoteNode(
            letter=self.pitch,
            accidentals=self._accidentals_list(),
            duration=self._duration_node(),
            slurred=self.slurred,
            position=None,
        )

    def to_alda(self) -> str:
        # For debugging/export only - derived from object state, not used for AST
        return f"{self.pitch}{self._accidental_str()}{self._duration_str()}"
```

### Phase 2: AST-Level Transformers

1. Pitch transformers: `transpose()`, `invert()`, `reverse()`, `shuffle()`
2. Structural transformers: `augment()`, `diminish()`, `fragment()`, `loop()`
3. Located in `aldakit.compose.transform`

### Phase 3: MIDI-Level Transformers

1. Timing transformers: `quantize()`, `humanize()`, `swing()`, `stretch()`
2. Velocity transformers: `accent()`, `crescendo()`, `diminuendo()`
3. Located in `aldakit.midi.transform`

### Phase 4: Generative Functions

1. Random selection: `random_note()`, `random_choice()`, `weighted_choice()`
2. Random walks: `random_walk()`, `drunk_walk()`
3. Rhythmic generators: `euclidean()`, `probability_seq()`
4. Pattern-based: `markov_chain()`, `lsystem()`, `cellular_automaton()`
5. Circuits: `shift_register()`, `turing_machine()`

### Phase 5: Advanced Features

1. Variables and references
2. Markers and jumps
3. Voices (parallel lines within a part)
4. Cram expressions (tuplets)
5. Scale and mode helpers
6. Chord voicing utilities

### Phase 6: MIDI Import

1. MIDI file import to AST
2. Real-time MIDI transcription

## Module Structure

```text
src/aldakit/
  score.py              # Unified Score class
  score_content.py      # Where a Score's music comes from: source, elements, import
  serialize.py          # AST -> Alda source (AldaWriter)
  theory.py             # Pitch names, scale and mode intervals, key signatures
  analysis.py           # Score inspection and linting (aldakit info / lint)
  constants.py          # Defaults and MIDI protocol values
  compose/
    __init__.py         # Public API: note, rest, chord, seq, part, tempo, etc.
    core.py             # note(), rest(), chord(), seq() domain objects
    duration.py         # Shared duration handling for notes and rests
    part.py             # Part, Voice
    attributes.py       # tempo(), volume(), octave(), dynamics
    chords.py           # Chord constructors: major(), minor(), dom7(), etc.
    scales.py           # Scale and mode helpers over aldakit.theory
    transform.py        # AST-level transformers: transpose, invert, reverse
    generate.py         # Generative functions: euclidean, markov, etc.
  midi/
    generator.py        # AST -> MidiSequence (an ASTVisitor)
    transform.py        # MIDI-level transformers: humanize, swing, quantize
```

Facts that more than one layer needs -- pitch spellings, scale intervals, key
signatures -- live in `theory.py`, and tunable defaults and MIDI protocol
values live in `constants.py`. Neither imports the rest of the package, so
either can be read without following the pipeline.

## Transformers

Transformers are functions that take a sequence and return a modified version. They are
organized into two categories based on what level they operate at:

### AST-Level vs MIDI-Level Transformers

| Level | Operates On | Examples | Reversible to Alda? |
|-------|-------------|----------|---------------------|
| **AST-Level** | Symbolic notation (notes, durations) | transpose, invert, reverse, augment | Yes |
| **MIDI-Level** | Absolute timing (seconds, ticks) | humanize, swing, quantize | No (timing is baked in) |

**AST-Level Transformers** modify the symbolic representation:
- Work with note names, intervals, and relative durations
- Output can be exported back to Alda source
- Located in `aldakit.compose.transform`

**MIDI-Level Transformers** modify the generated MIDI:
- Work with absolute time (seconds), MIDI pitch numbers, velocities
- Cannot be reversed to Alda (information is lost)
- Located in `aldakit.midi.transform`

```python
from aldakit.compose import Seq
from aldakit.compose.transform import transpose, reverse  # AST-level
from aldakit.midi.transform import humanize, swing        # MIDI-level

# AST-level: can export to Alda
melody = Seq.from_alda("c d e f g")
transposed = transpose(melody, semitones=5)
print(transposed.to_alda())  # "f g a a+ > c"

# MIDI-level: works on MidiSequence
midi_seq = score.midi
humanized = humanize(midi_seq, timing=0.1, velocity=0.05)
# Cannot convert back to Alda - timing is now in absolute seconds
```

### Pitch Transformers

```python
from aldakit.compose import Seq
from aldakit.compose.transform import transpose, invert, reverse, shuffle

melody = Seq.from_alda("c d e f g")

transpose(melody, 5)        # Up a perfect fourth
transpose(melody, -12)      # Down an octave
invert(melody)              # Invert intervals around first note
reverse(melody)             # Retrograde: g f e d c
shuffle(melody)             # Random permutation of notes
```

### Timing Transformers

```python
from aldakit import Score
from aldakit.midi.transform import quantize, humanize, swing, stretch

# These operate on a MidiSequence (absolute seconds), not on a compose Seq
midi_seq = Score("piano: c d e f g").midi

# Quantize to a grid, given in seconds
quantize(midi_seq, grid=0.25)              # Snap to the 0.25s grid
quantize(midi_seq, grid=0.25, strength=0.5)  # Halfway to the grid

# Humanize (add subtle timing and velocity variation)
humanize(midi_seq, timing=0.02)                 # +/- 20ms
humanize(midi_seq, timing=0.02, velocity=10)    # Also vary velocity

# Swing (delay offbeat notes)
swing(midi_seq, amount=0.3)   # 30% swing feel
swing(midi_seq, amount=0.5)   # Heavy shuffle

# Time stretch
stretch(midi_seq, 2.0)        # Double duration (half speed)
stretch(midi_seq, 0.5)        # Half duration (double speed)
```

### Velocity Transformers

```python
from aldakit import Score
from aldakit.midi.transform import accent, crescendo, diminuendo, normalize

midi_seq = Score("piano: c d e f g").midi

accent(midi_seq, pattern=[1.0, 0.6, 0.6, 0.6])          # Accent every 4th note
crescendo(midi_seq, start_velocity=40, end_velocity=100)  # Increase velocity
diminuendo(midi_seq, start_velocity=100, end_velocity=40) # Decrease velocity
normalize(midi_seq, target=80)                            # Even out velocities
```

### Structural Transformers

```python
from aldakit.compose.transform import (
    augment, diminish, fragment, loop, interleave
)

augment(melody, 2)          # Double all durations
diminish(melody, 2)         # Halve all durations
fragment(melody, 4)         # Take first 4 notes
loop(melody, 4)             # Repeat 4 times
interleave(melody1, melody2)  # Alternate notes from each
```

### Chaining Transformers

```python
from aldakit.compose import note, seq
from aldakit.compose.transform import augment, pipe, reverse, transpose

# pipe() chains AST-level transformers, which all take and return a Seq.
# MIDI-level transformers (humanize, swing) work on a MidiSequence instead,
# so they belong after generation rather than in this chain.
melody = seq(note("c"), note("d"), note("e"))

result = pipe(
    melody,
    lambda m: transpose(m, 5),
    reverse,
    lambda m: augment(m, 2),
)

# Or use functional composition
transformed = augment(reverse(transpose(melody, 5)), 2)
```

## Generative Functions

Generative functions create musical material algorithmically, useful for composition, experimentation, and live coding.

### Random Selection

```python
from aldakit.compose.generate import random_note, random_choice, weighted_choice

# Random note from scale
random_note(scale=["c", "d", "e", "g", "a"])  # Pentatonic

# Random choice from options
random_choice([
    chord("c", "e", "g"),
    chord("f", "a", "c"),
    chord("g", "b", "d"),
])

# Weighted random (probability distribution)
weighted_choice([
    (note("c"), 0.4),   # 40% chance
    (note("e"), 0.3),   # 30% chance
    (note("g"), 0.3),   # 30% chance
])
```

### Random Walk

```python
from aldakit.compose.generate import random_walk, drunk_walk

# Random walk: each step is random interval from previous
random_walk(
    start="c",
    steps=16,
    intervals=[-2, -1, 1, 2],  # Allowed intervals (semitones)
    duration=8
)

# Drunk walk: biased toward smaller intervals
drunk_walk(
    start="c",
    steps=16,
    max_step=3,         # Maximum interval size
    duration=8
)
```

### Probability-Based Generation

```python
from aldakit.compose.generate import probability_seq, rest_probability

# Each note has probability of appearing
probability_seq(
    notes=["c", "d", "e", "f", "g"],
    length=16,
    probability=0.7,    # 70% chance each step has a note
    duration=16
)

# Add random rests to existing sequence
rest_probability(melody, probability=0.2)  # 20% of notes become rests
```

### Euclidean Rhythms

```python
from aldakit.compose.generate import euclidean

# Euclidean rhythm: distribute k hits over n steps
euclidean(hits=3, steps=8, pitch="c")   # [x . . x . . x .]
euclidean(hits=5, steps=8, pitch="c")   # [x . x x . x x .]
euclidean(hits=7, steps=12, pitch="c")  # West African bell pattern

# With rotation
euclidean(hits=3, steps=8, pitch="c", rotate=1)  # Rotate pattern
```

### Markov Chains

```python
from aldakit.compose.generate import markov_chain, learn_markov

# Define transition probabilities manually
chain = markov_chain({
    "c": {"d": 0.5, "e": 0.3, "g": 0.2},
    "d": {"e": 0.6, "c": 0.4},
    "e": {"f": 0.5, "g": 0.3, "c": 0.2},
    "f": {"g": 0.7, "e": 0.3},
    "g": {"c": 0.6, "e": 0.4},
})
melody = chain.generate(start="c", length=16)

# Learn from existing melody
learned = learn_markov(existing_melody, order=1)
new_melody = learned.generate(length=32)

# Higher-order Markov (considers more context)
learned2 = learn_markov(existing_melody, order=2)
```

### L-Systems (Lindenmayer Systems)

```python
from aldakit.compose.generate import lsystem

# Define L-system rules
rules = {
    "A": "AB",
    "B": "A",
}

# Map symbols to notes
note_map = {
    "A": note("c", duration=8),
    "B": note("e", duration=8),
}

# Generate and expand
melody = lsystem(
    axiom="A",
    rules=rules,
    iterations=5,
    note_map=note_map
)
```

### Cellular Automata

```python
from aldakit.compose.generate import cellular_automaton

# Rule 30, 90, 110, etc.
melody = cellular_automaton(
    rule=110,
    width=8,
    steps=16,
    pitch_on="c",   # cells that are "off" become rests
    duration=16,
)
```

### Combining Generators

```python
from aldakit import Score
from aldakit.compose import part
from aldakit.compose.generate import euclidean, random_walk, markov_chain

# Layer different generative techniques
chain = markov_chain({"c": {"e": 0.5, "g": 0.5}, "e": {"c": 1.0}, "g": {"c": 1.0}})

score = Score.from_elements(
    part("midi-percussion"),
    euclidean(hits=5, steps=16, pitch="c"),  # Kick pattern

    part("acoustic-bass"),
    random_walk("c", steps=16, duration=8),

    part("piano"),
    chain.generate(start="c", length=16, duration=8),
)

score.play()
```

## Relationship to Existing Code

The compose API extends the existing parser/generator with programmatic construction:

| Direction | Operation | Status |
|-----------|-----------|--------|
| Alda -> AST | `parse()` | Implemented |
| AST -> MIDI Playback | `Score.play()` | Implemented |
| AST -> MIDI File | `Score.save()` | Implemented |
| Alda File -> Score | `Score.from_file()` | Implemented |
| Source String -> Score | `Score.from_source()` | Implemented |
| Python Objects -> AST | `to_ast()` | Implemented (compose module) |
| Python Objects -> Alda | `to_alda()` | Implemented (compose module) |
| MIDI File -> Score | `Score.from_file()` | Implemented |
| MIDI Input -> Score | `transcribe()` | Implemented |
| AST -> Alda | `write_alda()` | Implemented (serialize module) |

The parser remains essential for:

- The `Score.from_source()` constructor
- The `Seq.from_alda()` convenience method for parsing snippets
- Interop with other Alda tools

## MIDI Import

The unified Score architecture supports MIDI import through the same interface:

### MIDI File Import

```python
from aldakit import Score

# Import a MIDI file to Score
score = Score.from_file("recording.mid")

# Export to Alda for human-readable notation
print(score.to_alda())
# Output: piano: c4 d e f | g2 r2

# Or manipulate and re-export
from aldakit.compose import Seq
from aldakit.compose.transform import transpose

transposed = Score.from_elements(transpose(Seq.from_alda(score.to_alda()), 5))
transposed.save("transposed.mid")
transposed.save("transposed.alda")
```

### Real-time MIDI Transcription

```python
from aldakit.midi.transcriber import transcribe

# Transcribe live MIDI input for 10 seconds
score = transcribe(duration=10)

# Export as Alda notation
print(score.to_alda())
```

These features enable:

- Converting existing MIDI files to Alda notation
- Real-time transcription from MIDI keyboards
- Round-trip workflows: play -> transcribe -> edit -> play

## Conclusion

By treating the **AST as the central hub** and the **Score as the unified interface**, aldakit provides a complete platform where:

1. **Multiple inputs** (Alda text, Python API, MIDI files) all flow through `Score.from_*()` constructors
2. **Multiple outputs** (playback, MIDI files, Alda text) all derive from `Score` methods
3. **Transformations** operate at appropriate levels (AST for symbolic, MIDI for timing)
4. **Symmetric operations** enable round-trip transformations

```python
from aldakit import Score

# All roads lead to Score
score = Score.from_source("piano: c d e")       # From Alda text
score = Score.from_file("song.alda")            # From Alda file
score = Score.from_file("song.mid")             # From MIDI file
score = Score.from_elements(part, tempo, notes) # From Python objects

# All outputs derive from Score
score.play()                 # Playback
score.save("out.mid")        # MIDI file
score.save("out.alda")       # Alda text
print(score.to_alda())       # Alda string
```

This positions aldakit as a complete platform for music programming in Python, whether you prefer text-based notation, programmatic construction, or MIDI-based workflows.
