# Alda Language Reference (aldakit)

This document provides a comprehensive reference for the Alda music programming language as implemented in aldakit.

For detailed Alda language documentation, see the [Alda Language](alda-language/alda-overview.md) section.

---

## Quick Reference

| Feature | Syntax | Example |
|---------|--------|---------|
| Note | `[a-g][accidentals][duration]` | `c4`, `f#+8`, `b-2.` |
| Rest | `r[duration]` | `r4`, `r2.` |
| Octave set | `o[0-9]` | `o4`, `o5` |
| Octave up/down | `>` / `<` | `> c`, `< d` |
| Chord | `note/note/...` | `c/e/g` |
| Part | `instrument:` | `piano:` |
| Tempo | `(tempo N)` | `(tempo 120)` |
| Volume | `(vol N)` | `(vol 80)` |
| Variable | `name = events` | `theme = c d e` |
| Marker | `%name` / `@name` | `%verse`, `@verse` |
| Voice | `V[1-8]:` / `V0:` | `V1: c d e` |
| Repeat | `*N` | `c*4`, `[c d]*2` |
| Cram | `{events}duration` | `{c d e}2` |

---

## Notes

Notes are the fundamental building blocks of Alda scores.

### Basic Syntax

```alda
c d e f g a b      # Natural notes (C to B)
```

### Accidentals

| Symbol | Meaning |
|--------|---------|
| `+` or `#` | Sharp (+1 semitone) |
| `-` or `b` | Flat (-1 semitone) |
| `_` | Natural (explicit) |

```alda
c+                 # C sharp
c#                 # C sharp (alternate)
b-                 # B flat
bb                 # B flat (alternate)
c++                # C double-sharp
d--                # D double-flat
f_                 # F natural (explicit)
```

### Durations

| Value | Name | Duration (at 120 BPM) |
|-------|------|----------------------|
| 1 | Whole note | 2.0 seconds |
| 2 | Half note | 1.0 second |
| 4 | Quarter note | 0.5 seconds |
| 8 | Eighth note | 0.25 seconds |
| 16 | Sixteenth note | 0.125 seconds |
| 32 | Thirty-second | 0.0625 seconds |

```alda
c1                 # Whole note
c2                 # Half note
c4                 # Quarter note (default)
c8                 # Eighth note
c16                # Sixteenth note
```

### Dotted Notes

Dots extend duration by 50% of the previous value:

```alda
c4.                # Dotted quarter (1.5 beats)
c4..               # Double-dotted (1.75 beats)
c2.                # Dotted half (3 beats)
```

### Duration in Time Units

```alda
c500ms             # 500 milliseconds
c2s                # 2 seconds
c1.5s              # 1.5 seconds
```

### Tied Notes

Connect notes of the same pitch with `~`:

```alda
c4~4               # Quarter tied to quarter (2 beats)
c2~8               # Half tied to eighth
c1~2~4             # Multiple ties
```

A tie may cross a barline, which is how a note held across a bar is written.
The tie can go before the bar, after it, or on both sides; all three mean the
same thing, and all three mean the same as writing the tie with no bar at all:

```alda
c4~|2              # Tie before the barline
c4 | ~2            # Tie after it
c4~|~2             # On both sides
c4~2               # The same note, no barline
```

### Slurred Notes

Connect different notes with `~` for legato:

```alda
c~d~e              # Slurred phrase (no gap between notes)
```

---

## Rests

Rests indicate silence:

```alda
r                  # Rest (uses current default duration)
r4                 # Quarter rest
r2                 # Half rest
r1                 # Whole rest
r4.                # Dotted quarter rest
```

---

## Octaves

### Setting Octave

The default octave is 4 (middle C = C4 = MIDI note 60).

```alda
o4 c               # C4 (MIDI 60) - middle C
o5 c               # C5 (MIDI 72)
o3 c               # C3 (MIDI 48)
o0 c               # C0 (MIDI 12) - lowest
o8 c               # C8 (MIDI 108) - highest
```

### Octave Shifts

```alda
> c                # Up one octave then play C
< c                # Down one octave then play C
>> c               # Up two octaves
<< c               # Down two octaves
```

### MIDI Note Numbers

| Octave | C | D | E | F | G | A | B |
|--------|---|---|---|---|---|---|---|
| 0 | 12 | 14 | 16 | 17 | 19 | 21 | 23 |
| 4 | 60 | 62 | 64 | 65 | 67 | 69 | 71 |
| 8 | 108 | 110 | 112 | 113 | 115 | 117 | 119 |

---

## Chords

Simultaneous notes are separated by `/`:

```alda
c/e/g              # C major triad
c/e-/g             # C minor triad
c/e/g/b            # C major 7th
c4/e/g             # Quarter note chord
c/>e/g             # With octave change (C4, E5, G5)
```

---

## Parts (Instruments)

### Part Declaration

```alda
piano:
  c d e f g

violin:
  c d e f g
```

### Part Groups

Apply events to multiple instruments:

```alda
piano/violin:
  c d e f g        # Both play the same notes
```

### Part Aliases

Name specific instances:

```alda
violin "v1":
  c d e f

violin "v2":
  e f g a
```

### Supported Instruments

All 128 General MIDI instruments are supported. Common examples:

**Keyboards**: piano, electric-piano-1, harpsichord, organ
**Strings**: violin, viola, cello, contrabass
**Brass**: trumpet, trombone, french-horn, tuba
**Woodwinds**: flute, oboe, clarinet, bassoon
**Guitars**: acoustic-guitar-nylon, electric-guitar-clean
**Percussion**: vibraphone, marimba, xylophone, timpani

See [alda-language/list-of-instruments.md](alda-language/list-of-instruments.md) for the complete list.

---

## Attributes (S-expressions)

Attributes modify playback using Lisp-style S-expression syntax.

### Tempo

```alda
(tempo 120)        # Set tempo to 120 BPM
(tempo 60)         # 60 BPM (slower)
(tempo 180)        # 180 BPM (faster)
(tempo! 100)       # Global tempo (affects all parts)
```

### Volume

```alda
(volume 100)       # Full volume (0-100 scale)
(volume 50)        # Half volume
(vol 80)           # Shorthand
```

### Dynamics

```alda
(ppp)              # Pianississimo (very soft)
(pp)               # Pianissimo
(p)                # Piano
(mp)               # Mezzo-piano
(mf)               # Mezzo-forte (default)
(f)                # Forte
(ff)               # Fortissimo
(fff)              # Fortississimo (very loud)
```

Extended dynamics: `pppppp`, `ppppp`, `pppp` and `ffff`, `fffff`, `ffffff`

### Quantization

Controls note articulation (percentage of duration actually sounded):

```alda
(quant 100)        # Legato (full duration)
(quant 90)         # Default (slightly detached)
(quant 50)         # Staccato (half duration)
(quantization 80)  # Full name
(quant! 95)        # Global form: applies to every part
```

Any attribute can be written with a trailing `!` to apply it to the whole
score rather than the current part, including parts declared later.

### Panning

```alda
(panning 0)        # Hard left
(panning 50)       # Center
(panning 100)      # Hard right
(pan 25)           # Shorthand
```

### Track volume

The instrument's overall level (MIDI channel volume), as opposed to `volume`,
which sets the velocity of individual notes:

```alda
(track-volume 80)  # 0-100 scale
(track-vol 100)    # Shorthand
```

### Default note length

The length a note gets when it does not spell one out:

```alda
(set-duration 2)      # Two beats, i.e. a half note
(set-duration 2.5)    # A dotted half note
(set-note-length 1)   # A whole note, given as a note value
(set-duration-ms 500) # Half a second, converted at the part's tempo
```

### MIDI channel

Parts are assigned a channel automatically. To choose one:

```alda
piano: (midi-channel 2) c d e
```

Channel 9 is reserved for percussion, so a melodic part asking for it is
reported by `aldakit lint` and left where it was.

### Octave

```alda
(octave 5)         # Set octave to 5
```

### Transposition

Transpose all subsequent notes by a number of semitones:

```alda
(transpose 5)      # Transpose up a perfect fourth
(transpose -7)     # Transpose down a perfect fifth
(transpose 12)     # Transpose up one octave
(transpose 0)      # Reset to no transposition
(transposition 3)  # Full name
```

Transposing instruments (e.g., clarinet in Bb, horn in F):

```alda
clarinet: (transpose -2) c d e f    # Sounds Bb C D Eb
french-horn: (transpose -7) c d e f # Sounds F G A Bb
```

---

## Variables

Store and reuse musical phrases:

### Definition

```alda
theme = c d e f g
motif = c8 d e f g a b > c
chord_prog = c/e/g d/f/a e/g/b
```

**Important**: Variable definitions do not produce sound. They only store the events.

### Reference

```alda
theme              # Plays the stored events
theme theme        # Plays twice
```

### Example

```alda
verse = c4 d e f | g2 g | a4 a a a | g1
chorus = c8 c c c d d d d | e4 e f f | g1

piano:
  verse
  chorus
  verse
```

---

## Markers

Mark positions in time for synchronization.

### Marker Definition

```alda
piano:
  c d e f %verse g a b c
```

The marker `%verse` records the time position after `f`.

### Marker Reference

```alda
piano:
  c d e %verse f g a b

violin:
  @verse c d e f    # Starts at the %verse position
```

---

## Voices

Enable polyphonic writing within a single part.

### Basic Voices

```alda
piano:
  V1: c1 d e f     # Voice 1 (melody)
  V2: e1 g a b     # Voice 2 (plays simultaneously)
  V0:              # Merge voices, continue from latest
```

### Voice Numbering

Voices are numbered 1-8. Each voice has independent timing.

### Voice Merge (V0:)

`V0:` merges all voices and continues from the latest position:

```alda
piano:
  V1: c4 d e f     # 4 quarter notes (2 beats)
  V2: c2 d         # 2 half notes (2 beats)
  V0: c1           # Continues after both finish
```

**Note**: aldakit requires explicit `V0:` to close voice groups in bracketed sequences.

---

## Repeats

### Simple Repeat

```alda
c*4                # Repeat note 4 times
[c d e]*3          # Repeat sequence 3 times
```

### On-Repetitions (Alternate Endings)

Play different content on specific repetitions:

```alda
[c d e'1 f'2]*2
# Iteration 1: c d e
# Iteration 2: c d f

[a b c'1,2 d'3]*3
# Iterations 1,2: a b c
# Iteration 3: a b d

[c'1-3 d'4-6]*6
# Iterations 1-3: c
# Iterations 4-6: d
```

---

## Cram Expressions (Tuplets)

Fit multiple notes into a specified duration:

```alda
{c d e}2           # Triplet: 3 notes in a half note
{c d e f g}4       # Quintuplet: 5 notes in a quarter
{c d}4             # Duplet: 2 notes in a quarter
```

---

## Barlines

Barlines are purely visual and have no effect on timing:

```alda
c4 d e f | g a b c | d1
```

Because they carry no timing, a barline does not interrupt a tie: see
[Tied Notes](#tied-notes).

---

## Comments

```alda
# This is a line comment

c d e    # Inline comment
```

---

## Implementation Status

### Fully Implemented in aldakit

- Notes with all accidentals and durations
- Rests
- Octave setting and shifts
- Chords
- Tied and slurred notes
- Dotted notes (single and double)
- Duration in milliseconds and seconds
- All 128 GM instruments
- Part declarations and groups
- Part aliases
- Tempo (local and global)
- Volume
- All dynamic markings (pppppp to ffffff)
- Quantization
- Panning (`panning`, `pan`)
- Track volume (`track-volume`, `track-vol`)
- Default note length (`set-duration`, `set-note-length`, `set-duration-ms`)
- Explicit MIDI channel assignment (`midi-channel`)
- Channel reuse: channels are handed on as parts stop sounding, so a score can
  have more parts than there are channels
- Offline audio rendering to WAV (`aldakit render`)
- Octave attribute
- The global (`!`) form of every attribute
- Variables (definition and reference)
- Markers and marker jumps
- Voices (V1-V8, V0)
- Repeats with on-repetitions
- Cram expressions
- Bracketed sequences
- Barlines
- Comments

### Recently Added

- Transposition (`transpose`, `transposition`) - fully supported with semitone offsets
- Key signature (`key-sig`, `key-signature`) - fully supported with string and quoted list formats
- Quoted list syntax in S-expressions (`'(g minor)`) - now fully supported

### Known Limitations

- An attribute aldakit does not recognise is reported by `aldakit lint` and
  ignored, rather than stopping generation.

- **Channel reuse is decided per part, not per note.** A channel is freed when
  the part on it stops sounding, which is enough for both
  `examples/all-instruments.alda` and `examples/midi-channel-management.alda`.
  More than 15 pitched parts sounding at the same instant still cannot fit, and
  `aldakit lint` reports which parts collide.

---

## Differences from alda-midi (C implementation)

| Feature | aldakit | alda-midi |
|---------|---------|-----------|
| Markers | Implemented | Parsed only |
| On-repetitions | Implemented | Parsed only |
| Slurs | Implemented | Parsed only |
| Key signature | Implemented | Parsed only |
| Quoted S-expr lists | Supported | Supported |
| Voice termination | Requires V0: in brackets | Auto-closes |
| MIDI file export | Yes | No |
| MIDI file import | Yes | No |
| Real-time transcription | Yes | No |
| Concurrent playback | Yes (8 slots) | Yes |
| Built-in audio | Yes (TinySoundFont) | No |

