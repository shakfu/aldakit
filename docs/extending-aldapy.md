# Extending aldapy: A Programmatic API

This document explores extending aldapy beyond parsing the Alda language to provide a programmatic Python API for music composition. The key insight is that **the AST is the central hub** - all inputs flow into it, all outputs derive from it.

## Architecture Overview

![aldapy architecture](assets/architecture.svg)

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
score = Score()
score.add(part("piano"), note("c", 4), note("d"), note("e"))

# Export to Alda source
print(score.to_alda())
# Output: piano: c4 d e

# Or play directly (bypasses text, goes straight to AST -> MIDI)
score.play()
```

### 2. Composable Domain Objects

Every musical element is a first-class Python object that can be composed, transformed, and introspected:

```python
# Notes are objects
n = note("c", duration=4, accidental="sharp")
n.transpose(2)  # Returns new note: d#4

# Chords are collections of notes
c_major = chord(note("c"), note("e"), note("g"))
c_minor = c_major.flatten(2)  # Flatten the third

# Sequences can be manipulated
melody = seq(note("c"), note("d"), note("e"), note("f"))
melody.reverse()      # f e d c
melody.retrograde()   # Same as reverse for pitches
melody.invert()       # Invert intervals
```

## API Design

### Core Domain Objects

```python
from aldapy.compose import (
    Score, Part, Voice,
    note, rest, chord, seq,
    tempo, volume, octave,
)
```

#### Notes

```python
# Basic note
note("c")                    # c (quarter note, default octave)
note("c", 4)                 # c4 (quarter note)
note("c", 8, "sharp")        # c+8 (eighth note, sharp)
note("c", dots=1)            # c. (dotted)
note("c", ms=500)            # c500ms (milliseconds)

# Note attributes
n = note("c", 4)
n.pitch                      # "c"
n.duration                   # 4
n.midi_pitch                 # 60 (at default octave 4)

# Transformations (return new notes)
n.sharpen()                  # c+4
n.flatten()                  # c-4
n.transpose(2)               # d4 (up 2 semitones)
n.transpose(-12)             # c4 (down an octave)
```

#### Rests

```python
rest()                       # r (quarter rest)
rest(2)                      # r2 (half rest)
rest(ms=1000)                # r1s (one second rest)
```

#### Chords

```python
# Building chords
chord(note("c"), note("e"), note("g"))           # c/e/g
chord("c", "e", "g")                              # Shorthand
chord("c", "e", "g", duration=1)                  # c1/e/g

# Named chords (convenience constructors)
from aldapy.compose.chords import major, minor, dim, aug, dom7

major("c")                   # c/e/g
minor("a")                   # a/c/e
dom7("g")                    # g/b/d/f
```

#### Sequences

```python
# Explicit sequence
melody = seq(note("c"), note("d"), note("e"), note("f"))

# From string (parsed as Alda)
melody = seq.from_alda("c d e f g")

# Repeat
melody * 4                   # [c d e f]*4

# Concatenate
intro + verse + chorus       # Sequences combine
```

#### Parts and Instruments

```python
# Simple part
Part("piano").add(note("c"), note("d"), note("e"))

# With alias
Part("violin", alias="v1")

# Multi-instrument
Part("violin", "viola", "cello", alias="strings")
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
from aldapy.compose import pp, p, mp, mf, f, ff
```

### The Score Class

```python
class Score:
    def __init__(self):
        self.elements = []
        self.variables = {}

    def add(self, *elements) -> "Score":
        """Add elements to the score."""
        self.elements.extend(elements)
        return self

    def part(self, *instruments, alias=None) -> "Score":
        """Add a part declaration."""
        return self.add(Part(*instruments, alias=alias))

    def notes(self, alda_string: str) -> "Score":
        """Parse and add notes from Alda syntax."""
        # Convenience for quick note entry
        return self.add(seq.from_alda(alda_string))

    def tempo(self, bpm: int, global_=False) -> "Score":
        return self.add(tempo(bpm, global_=global_))

    def var(self, name: str, *elements) -> "Score":
        """Define a variable."""
        self.variables[name] = seq(*elements)
        return self

    def use(self, name: str) -> "Score":
        """Reference a variable."""
        return self.add(VarRef(name))

    # Output methods

    def to_alda(self) -> str:
        """Export as Alda source code."""
        return "\n".join(e.to_alda() for e in self.elements)

    def to_ast(self) -> "AldaAST":
        """Convert to AST (for MIDI generation)."""
        # Either parse to_alda() or build AST directly
        ...

    def to_midi(self) -> "MidiSequence":
        """Generate MIDI sequence."""
        from aldapy import generate_midi
        return generate_midi(self.to_ast())

    def play(self, backend=None):
        """Play the score."""
        from aldapy import LibremidiBackend
        backend = backend or LibremidiBackend()
        backend.play(self.to_midi())

    def save(self, path: str):
        """Save to file (.alda or .mid based on extension)."""
        if path.endswith(".alda"):
            Path(path).write_text(self.to_alda())
        elif path.endswith(".mid"):
            from aldapy import LibremidiBackend
            LibremidiBackend().save(self.to_midi(), path)
```

## Use Cases

### 1. Algorithmic Composition

```python
import random
from aldapy.compose import Score, Part, note, chord, seq, tempo

def random_melody(length=8, scale=["c", "d", "e", "f", "g", "a", "b"]):
    """Generate a random melody from a scale."""
    return seq(*[note(random.choice(scale), 8) for _ in range(length)])

def arpeggiate(chord_notes, pattern=[0, 1, 2, 1]):
    """Arpeggiate a chord with a pattern."""
    return seq(*[note(chord_notes[i % len(chord_notes)], 16) for i in pattern])

score = Score()
score.add(Part("piano"))
score.add(tempo(120))
score.add(random_melody(16))
score.add(arpeggiate(["c", "e", "g"]) * 4)
score.play()
```

### 2. Data Sonification

```python
from aldapy.compose import Score, Part, note, tempo

def weather_to_music(temperatures: list[float]):
    """Convert temperature data to music."""
    min_t, max_t = min(temperatures), max(temperatures)

    def temp_to_note(t):
        # Map to pentatonic scale (C D E G A)
        scale = ["c", "d", "e", "g", "a"]
        idx = int((t - min_t) / (max_t - min_t) * (len(scale) - 1))
        return note(scale[idx], 8)

    score = Score()
    score.add(Part("vibraphone"))
    score.add(tempo(140))
    for t in temperatures:
        score.add(temp_to_note(t))
    return score

# Sonify a week of temperatures
temps = [45, 52, 48, 61, 58, 55, 50]
weather_to_music(temps).play()
```

### 3. Music Theory Operations

```python
from aldapy.compose import Score, Part, seq, rest
from aldapy.compose.transform import transpose, invert, reverse

# Define a motif
motif = seq.from_alda("c8 d e- g")

# Transform it
motif_up = transpose(motif, 5)    # Up a fourth
motif_inv = invert(motif)         # Invert intervals
motif_ret = reverse(motif)        # Retrograde

# Build a fugue-like structure
score = Score()
score.add(Part("piano"))
score.add(motif)
score.add(rest(2))
score.add(motif_up)
score.add(rest(2))
score.add(motif_inv)
score.play()
```

### 4. Live Coding / REPL Workflow

```python
>>> from aldapy.compose import Score, Part, note, chord, tempo
>>> s = Score()
>>> s.add(Part("piano"), tempo(120))
>>> s.add(note("c", 4), note("e"), note("g"))
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
# Load an Alda file, modify it, save back
from aldapy import parse
from aldapy.compose import Score

# Parse existing Alda file to AST
with open("song.alda") as f:
    ast = parse(f.read())

# Wrap in Score for manipulation (future feature)
score = Score.from_ast(ast)
score.transpose(2)  # Transpose entire score up 2 semitones
score.save("song_transposed.alda")
```

## Implementation Strategy

### Phase 1: Core Domain Objects

1. Implement `note()`, `rest()`, `chord()`, `seq()` with `to_alda()` methods
2. Implement `Part`, `tempo()`, `volume()`, and other attributes
3. Implement `Score` class with `add()`, `to_alda()`, `play()`

### Phase 2: AST Integration

1. Add `to_ast()` methods that create AST nodes directly
2. Bypass text parsing for better performance
3. Ensure round-trip: `Score -> AST -> Alda text -> AST` produces equivalent results

### Phase 3: Transformers

1. Pitch transformers: `transpose()`, `invert()`, `reverse()`, `shuffle()`
2. Timing transformers: `quantize()`, `humanize()`, `swing()`, `stretch()`
3. Velocity transformers: `accent()`, `crescendo()`, `diminuendo()`
4. Structural transformers: `augment()`, `diminish()`, `fragment()`, `loop()`

### Phase 4: Generative Functions

1. Random selection: `random_note()`, `random_choice()`, `weighted_choice()`
2. Random walks: `random_walk()`, `drunk_walk()`
3. Rhythmic generators: `euclidean()`, `probability_seq()`
4. Pattern-based: `markov_chain()`, `lsystem()`, `cellular_automaton()`

### Phase 5: Advanced Features

1. Variables and references
2. Markers and jumps
3. Voices
4. Cram expressions (tuplets)
5. Scale and mode helpers
6. Chord voicing utilities

## Module Structure

```text
src/aldapy/
  compose/
    __init__.py       # Public API exports
    core.py           # note, rest, chord, seq
    score.py          # Score class
    part.py           # Part, Voice
    attributes.py     # tempo, volume, octave, etc.
    chords.py         # Chord constructors (major, minor, etc.)
    theory.py         # transpose, invert, retrograde
    transform.py      # MIDI transformers
    generate.py       # Generative functions
```

## Transformers

Transformers are functions that take a sequence (or MIDI data) and return a modified version. They operate on timing, pitch, velocity, and structure.

### Pitch Transformers

```python
from aldapy.compose.transform import transpose, invert, reverse, shuffle

melody = seq.from_alda("c d e f g")

transpose(melody, 5)        # Up a perfect fourth
transpose(melody, -12)      # Down an octave
invert(melody)              # Invert intervals around first note
reverse(melody)             # Retrograde: g f e d c
shuffle(melody)             # Random permutation of notes
```

### Timing Transformers

```python
from aldapy.compose.transform import quantize, humanize, swing, stretch

# Quantize to grid (snap to nearest division)
quantize(melody, 16)        # Quantize to 16th notes
quantize(melody, 8)         # Quantize to 8th notes

# Humanize (add subtle timing variations)
humanize(melody, amount=0.1)  # 10% timing deviation
humanize(melody, amount=0.2, velocity=0.15)  # Also vary velocity

# Swing (delay offbeat notes)
swing(melody, amount=0.3)   # 30% swing feel
swing(melody, amount=0.5)   # Heavy shuffle

# Time stretch
stretch(melody, 2.0)        # Double duration (half speed)
stretch(melody, 0.5)        # Half duration (double speed)
```

### Velocity Transformers

```python
from aldapy.compose.transform import accent, crescendo, diminuendo, normalize

accent(melody, pattern=[1, 0, 0, 0])  # Accent every 4th note
crescendo(melody, start=40, end=100)  # Gradually increase velocity
diminuendo(melody, start=100, end=40) # Gradually decrease velocity
normalize(melody, target=80)          # Normalize all velocities
```

### Structural Transformers

```python
from aldapy.compose.transform import (
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
from aldapy.compose.transform import pipe

# Apply multiple transformations
result = pipe(
    melody,
    lambda m: transpose(m, 5),
    lambda m: humanize(m, 0.1),
    lambda m: swing(m, 0.2),
)

# Or use functional composition
transformed = swing(humanize(transpose(melody, 5), 0.1), 0.2)
```

## Generative Functions

Generative functions create musical material algorithmically, useful for composition, experimentation, and live coding.

### Random Selection

```python
from aldapy.compose.generate import random_note, random_choice, weighted_choice

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
from aldapy.compose.generate import random_walk, drunk_walk

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
from aldapy.compose.generate import probability_seq, rest_probability

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
from aldapy.compose.generate import euclidean

# Euclidean rhythm: distribute k hits over n steps
euclidean(hits=3, steps=8, note="c")   # [x . . x . . x .]
euclidean(hits=5, steps=8, note="c")   # [x . x x . x x .]
euclidean(hits=7, steps=12, note="c")  # West African bell pattern

# With rotation
euclidean(hits=3, steps=8, note="c", rotate=1)  # Rotate pattern
```

### Markov Chains

```python
from aldapy.compose.generate import markov_chain, learn_markov

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
from aldapy.compose.generate import lsystem

# Define L-system rules
rules = {
    "A": "AB",
    "B": "A",
}

# Map symbols to notes
note_map = {
    "A": note("c", 8),
    "B": note("e", 8),
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
from aldapy.compose.generate import cellular_automaton

# Rule 30, 90, 110, etc.
melody = cellular_automaton(
    rule=110,
    width=8,
    steps=16,
    note_on="c",
    note_off=rest()
)
```

### Combining Generators

```python
from aldapy.compose import Score
from aldapy.compose.generate import euclidean, random_walk, markov_chain

# Layer different generative techniques
score = Score()
score.add(part("drums"))
score.add(euclidean(hits=5, steps=16, note="c"))  # Kick pattern

score.add(part("bass"))
score.add(random_walk(start="c", steps=16, octave=2))

score.add(part("piano"))
chain = markov_chain(...)
score.add(chain.generate(length=16))

score.play()
```

## Relationship to Existing Code

The compose API complements the existing parser/generator:

| Direction | Operation | Status |
|-----------|-----------|--------|
| Alda -> AST | `parse()` | Implemented |
| AST -> MIDI Playback | `play()` | Implemented |
| AST -> MIDI File | `save()` | Implemented |
| AST -> Alda | `export()` | Planned |
| Python API -> AST | `to_ast()` | Planned (compose module) |
| Python API -> Alda | `to_alda()` | Planned (compose module) |
| MIDI File -> AST | `import()` | Future |
| MIDI Input -> AST | `transcribe()` | Future |

The parser remains essential for:

- Loading `.alda` files
- The `notes()` convenience method (parses Alda snippets)
- Interop with other Alda tools

## Future: MIDI Import

The architecture supports two forms of MIDI import:

### MIDI File Import

```python
from aldapy import import_midi

# Import a MIDI file to AST
ast = import_midi("recording.mid")

# Export to Alda for human-readable notation
alda_source = export(ast)
print(alda_source)
# Output: piano: c4 d e f | g2 r2

# Or manipulate and re-export
ast_transposed = transpose(ast, 5)
save(ast_transposed, "transposed.mid")
```

### Real-time MIDI Transcription

```python
from aldapy import MidiInput

# Transcribe live MIDI input to Alda
with MidiInput() as midi_in:
    for ast_fragment in midi_in.transcribe():
        print(export(ast_fragment))
        # Prints Alda notation as you play
```

These features would enable:

- Converting existing MIDI files to Alda notation
- Real-time transcription from MIDI keyboards
- Round-trip workflows: play -> transcribe -> edit -> play

## Conclusion

By treating the **AST as the central hub**, aldapy provides a unified platform where:

1. **Multiple inputs** (Alda text, Python API, MIDI) all flow into AST
2. **Multiple outputs** (playback, MIDI files, Alda text) all derive from AST
3. **Symmetric operations** enable round-trip transformations

This positions aldapy as a complete platform for music programming in Python, whether you prefer text-based notation, programmatic construction, or MIDI-based workflows.
