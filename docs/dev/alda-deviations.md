# Differences from Alda

aldakit is correct when it produces the MIDI that Alda produces. The oracle is
`alda export` from Alda 2.4.7, committed under `tests/alda_reference/2.4.7/`.
`tests/test_alda_reference.py` fails on any difference not listed here, and
`scripts/alda_diff.py` (`make alda-diff`) reports differences in detail. Alda
source links are to the `release-2.4.7` tag.

Every score in `examples/` and `tests/shared_suite/` matches. Bugs found by
the first comparison are recorded in `CHANGELOG.md`.

## Deliberate deviations

### D1. Channel numbers

Alda picks a channel per note when the note is generated
([`client/model/midi.go:122-187`](https://github.com/alda-lang/alda/blob/release-2.4.7/client/model/midi.go#L122-L187)).
After a voice group ends, the part moves to a fresh channel, because the
merged voice becomes a new `origin`
([`voice.go:160`](https://github.com/alda-lang/alda/blob/release-2.4.7/client/model/voice.go#L160)).
aldakit assigns a channel when a part is declared and keeps a part's voices on
it (`src/aldakit/midi/channels.py`).

The move is audible in one case: a note from inside a voice group still
sounding when the part plays the same pitch after the group, which needs
`quant` above 100. On one channel the new note-on cuts the old note off; on
Alda's new channel both sound. aldakit moves the part to a new channel in that
case only, so the result matches Alda. A unison inside a voice group shares one
channel in Alda too.

Rationale: a General MIDI synth does not hear a channel number. It hears the
program and controller state a note plays with. Copying Alda would spend a
channel on every voice group, so a score with many voice groups would run out
of the 15 melodic channels sooner. The upstream channel hop looks unintended
(inference: nothing in `voice.go` mentions channels).

The comparison therefore checks, for every note, the program and controllers
7, 10 and 11 in effect on its channel when it starts, not the channel number
or when those were sent.

Limit: this does not hold for a synth configured per channel, such as a
multitimbral module with fixed patches. Pin channels there with
`(midi-channel N)`.

### D2. Errors do not stop generation

Alda refuses a score with an out-of-range attribute value, an unknown
instrument or attribute, or an ambiguous instrument reference. aldakit reports
each as a diagnostic, skips the offending value, and keeps generating.
`MidiGenerator(strict=True)`, `Score(..., strict=True)` and `--strict` on
`play`, `eval` and `render` raise `AldaGenerationError` on the first
diagnostic instead.

Rationale: the REPL and editor integrations need a result for a score that is
mid-edit, and `aldakit lint` reports every problem rather than the first.
Strict mode gives Alda's behaviour where a build should fail.

## Alda against its own documentation

Where they disagree, aldakit follows the code.

- `quant`: documented as 0-100; the code accepts any non-negative value.
- Cram: the docs say a cram's first note is a quarter note; the code uses the
  part's current duration.
- Dynamics: the docs give rounded volumes; the code uses exact fractions,
  which differ by one velocity step for `ppppp`, `pp`, `p` and `mp`.

## Limits of the reference

- `alda export` drops leading silence: `piano: r1 c` exports `c` at 0 s. A
  corpus score that starts with silence would fail the comparison; none does.
- Alda writes 128 ticks per beat and rounds events to whole milliseconds. The
  comparison tolerates one tick on starts and two on durations.
- `alda export` sometimes hangs on the first export after starting its player.
  The script times out after 120 s; rerun to retry.
