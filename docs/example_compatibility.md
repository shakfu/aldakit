# Example File Compatibility Report

This document tracks how the bundled `.alda` examples behave in
[aldakit](https://github.com/shakfu/aldakit).

## What "compatible" means here

Earlier revisions of this document reported 40/40 compatible without stating a
criterion. That number meant "parses without raising", which is a much weaker
claim than it appeared: several files parsed cleanly while producing music that
was plainly wrong (every instrument resolving to piano, drums playing as piano,
melodic parts landing on the drum channel).

Compatibility is now measured against three checks, all enforced by the test
suite rather than by hand:

| Check | Enforced by |
| --- | --- |
| Parses and generates MIDI without error | `tests/test_golden_midi.py` |
| Generates the exact expected notes, channels, programs, timings and velocities | `tests/golden/examples.json` |
| Survives a `parse -> write -> parse` round trip with identical MIDI | `tests/test_serialize.py` |

Two further invariants are checked across every example:

- No melodic part is ever assigned MIDI channel 9, and `midi-percussion` always
  is (`tests/test_channel_allocation.py`).
- No example references an instrument name that falls back to piano
  (`tests/test_instruments.py`).

## Status

All 40 examples pass all three checks and both invariants.

Generation is otherwise diagnostic-free, with one expected exception described
below.

## Expected diagnostics

Two examples deliberately declare more parts than General MIDI has channels:

| Example | Melodic parts | Diagnostics |
| --- | --- | --- |
| all-instruments.alda | 129 | 114 channel-reuse warnings |
| midi-channel-management.alda | 31 | 16 channel-reuse warnings |

General MIDI provides 16 channels, one of which (channel 10) is reserved for
percussion, leaving 15 for pitched instruments. A score with more pitched parts
than that must reuse channels, and reused channels share a program, so some
parts sound with the wrong instrument. aldakit reports this rather than failing
silently:

```
Warning: More than 15 melodic parts declared; MIDI channels are being reused
and instrument assignments will collide.
```

These files exist to exercise the whole General MIDI sound set, so the warning
is correct and expected. Every other example generates without diagnostics.

## Language feature coverage

All Alda language constructs used by the examples are implemented:

- Notes, rests, accidentals, octaves, chords, barlines
- Durations: note lengths, dots, ties, milliseconds, seconds
- Attributes: tempo, volume, quantization, panning, dynamics, key signature,
  transposition, octave, and their global (`!`) variants
- Variables, markers, voices, cram expressions, repeats, alternate endings
- Instrument groups, aliases and the group-member dot accessor
  (`strings.cello:`)
- All 128 General MIDI programs under their canonical Alda names and aliases,
  plus `midi-percussion`

## Fixes applied

1. **midi-channel-management.alda**: added `V0:` to close a voice group inside a
   bracketed variable definition. aldakit requires explicit voice group
   termination.

2. **Quoted list syntax**: added support for Lisp-style quoted lists `'(...)` in
   S-expressions, enabling files using `(key-signature '(g minor))`.

3. **Instrument names**: the instrument table is now generated from
   `docs/alda-language/list-of-instruments.md` and covers all 128 GM programs
   under their canonical `midi-` prefixed names and aliases. Previously
   `all-instruments.alda` resolved 128 of its 129 instruments to acoustic grand
   piano.

4. **Percussion routing**: `midi-percussion` is placed on MIDI channel 10 with
   no program change, and melodic parts are kept off that channel. Previously
   `percussion.alda` played as piano and four other examples sent melodic lines
   to the drum channel.

5. **Dot accessor**: `strings.cello:` now resolves to the cello of the aliased
   group. Previously it created a separate cello instance on a new channel
   starting at time zero, so `dot_accessor.alda` produced the wrong music.

## Regenerating the golden fixtures

After an intentional change to the generator:

```sh
make golden      # or: python scripts/gen_golden_midi.py
```

Review the resulting diff. It shows exactly which notes changed, so an
unintended change to how a score sounds is visible rather than silent.
