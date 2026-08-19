# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0]

Scores can now be rendered to audio files, and a part no longer holds a MIDI channel for the length of a score: it gives the channel up once it has stopped sounding, so a score can have more parts than MIDI has channels. That is what `examples/all-instruments.alda` (128 instruments) and `examples/midi-channel-management.alda` (31 parts) were written to demonstrate, and both were reported as broken by `aldakit lint` until now. Underneath it, a timing bug that desynchronised parts in a group. **Three of the 40 examples generate different MIDI**: two only in which channel each part lands on, and `multi-poly.alda` in how it sounds. No score lost a note. The diff is in `tests/golden/examples.json`.

### Added

- **MIDI channels are handed on as parts stop sounding.** A part occupies a channel only while it is sounding, so a channel a part has finished with goes to the next part that needs one. The part taking it over selects its own instrument and restates its pan and volume, and any controller it does not set itself is returned to the General MIDI default rather than inherited from the part before it -- without which an organ borrowing a piano's channel would also borrow its panning. Reuse is a fallback rather than the default: a score with 15 or fewer melodic parts still gets one channel per part in declaration order, and its MIDI is unchanged. Within reuse a part prefers the channel it last used, so a part with rests in it keeps one channel instead of moving between whichever are free.

- **`aldakit/midi/channels.py`** decides this, after the AST has been walked and the shape of the score is known. `MidiGenerator.channel_assignment` exposes what it decided: which real channels each part used, which parts overlap on a channel, and how many parts sound at once.

- **`aldakit info` reports every channel a part uses** (`0,1` for a part that moved), or `-` for a part that never sounds and so needs none.

- **`aldakit render FILE` writes a score to a WAV file**, synthesized with a SoundFont, with no audio device involved and without waiting for the score to play: `all-instruments.alda` is two and a half minutes long and renders in twelve seconds. `-o` names the output (the input file with a `.wav` suffix by default), `-sf` the SoundFont, `-g` the gain and `--tail` how much is rendered after the last note so that release tails are not cut off. A mix loud enough to clip is reported along with a gain that will not clip. `aldakit.render()`, `aldakit.render_file()` and `Score.render()` are the same thing from Python.

- **Golden audio fixtures.** `tests/golden/audio.json` pins what every example *sounds* like -- loudness per channel over quarter-second windows, the peak, and the length -- next to the golden MIDI fixtures that pin what the generator decided. It closes the gap that every defect in this project's history has fallen through: MIDI that is perfectly well formed and sounds wrong. Deleting the control changes on the way to the synthesizer, for instance, leaves all 203 golden MIDI assertions passing and fails four audio fixtures. Regenerate with `make golden-audio`; CI runs the comparison on Linux and macOS.

- **`aldakit/midi/render.py`** holds it, on top of a new `render_pcm16()` in the TinySoundFont binding. Offline rendering and real-time playback now run the same synthesis loop -- the audio callback is a caller of it rather than the owner of it -- so a rendered file and the sound coming out of the speakers cannot disagree.

### Fixed

- **A chord desynchronised parts that were at different tempi.** In a multi-instrument part (`violin/viola:`) a chord advanced every active part by the duration of whichever part happened to be processed last. Two parts at 60 and 240 BPM came out of a shared quarter-note chord with the slower one three quarters of a second early, and everything after it stayed that far out. Each part now advances by its own longest note in the chord.

- **A cram took its length from the first part in the group.** `{c d e}` fills the part's current default duration, which parts in a group need not agree on. `multi-poly.alda` is the case this was hiding: the piano crams three notes into a whole note and the harp three into a half note, a 2:1 polyrhythm, but the harp was given the piano's whole note and played on for two seconds after the piano had finished, half as long again as the score. Both parts now end together, twelve harp notes against six.

- **`all-instruments.alda` and `midi-channel-management.alda` play as written.** All 128 General MIDI programs are heard in the first, where previously the parts past the fifteenth overwrote each other's instrument on a shared channel. Both lint clean.

- **`aldakit lint` no longer reports a score as broken for declaring more parts than there are channels.** `too-many-parts` is reported when more melodic parts sound at the same moment than there are channels to put them on, which is the number that has to fit, and `shared-channel` when two parts actually overlap in time on one channel rather than merely being assigned to it.

- **The audio backend's gain was in the wrong unit.** `gain` is documented as a volume factor from 0.0 to 2.0 where 1.0 is unity, but it was passed to TinySoundFont's `tsf_set_output`, whose argument is decibels. Every gain in the documented range was therefore a small boost: the default of 1.0 was +1 dB rather than unity, 0.5 was +0.5 dB rather than half, and 0.0 was 0 dB -- full volume, not silence. It now goes through `tsf_set_volume`, so halving the gain halves the level and zero is silent. Audio playback at the default is about 12% quieter than it was, which is what unity means.

### Changed - internal

- **The generator hands out placeholder channels** (`VIRTUAL_CHANNEL_BASE` and up, allocated without limit) while it walks the AST, because whether reuse is needed cannot be known until the last part has been declared. `PartState.channel` holds a real channel again once `generate()` returns, or -1 for a part that never sounds; `PartState.allocated_channel` keeps the placeholder so the linter can attribute a shared channel back to the parts sharing it.

- **Finding severities are an enum**, `aldakit.Severity`, rather than three module-level strings. The comment above those strings said they were "ordered by how much they should worry the reader", but the order was not in them: it was repeated in a dict inside `lint_score`, and again in the test that checked the ordering. `Severity.rank` reads it back from the order the members are declared in, so it is stated once. The members subclass `str` and keep the old names as aliases, so `finding.severity == "error"` and `from aldakit.analysis import ERROR` both still work. `Severity.__str__` is defined explicitly because `Enum.__format__` changed in 3.11, and without it `aldakit lint` would print "error" on 3.10 and "Severity.ERROR" from 3.11 on.

- **`MELODIC_CHANNELS` moved to `aldakit/midi/channels.py`** and is re-exported from `midi/generator.py`, so existing imports are unaffected.

### Changed - build and CI
- **The SoundFonts are mirrored on aldakit's own `soundfonts-v1` release.** The site all three were downloaded from has put itself behind a bot challenge that answers every automated request with 403, which broke `aldakit soundfont install` completely -- for every SoundFont, for every user, no matter how often it retried. The catalog now points at byte-identical copies attached to a release of this repository, verified against the same checksums as before, so nothing that depends on the exact bytes had to change. `SOUNDFONT-LICENSES.txt` on that release records each licence in full: FluidR3_GM is MIT, TimGM6mb is GPL-2, and GeneralUser GS carries its author's own permissive licence. They remain third-party works, not part of aldakit.

- **A refused download explains itself.** A host behind a bot challenge answers every retry with 403, so reporting the status code alone leaves the reader with nothing to do. The error now says that retrying will not help, and names both the URL a browser can still fetch and the path to save it to. An ordinary failure such as a 404 is still reported plainly.

- **A `Golden audio` job** renders every example against the pinned fixtures on Linux and macOS. It caches the SoundFont, keyed on the download catalog, so a run does not depend on a third-party site being up, retries the download if the cache misses and the site is unreachable, and sets `ALDAKIT_REQUIRE_AUDIO_FIXTURES`, which turns "no SoundFont, nothing to compare" from a skip into a failure -- a green tick for a comparison that never ran is worse than no comparison.

- **`make golden-audio`, `make soundfont` and `make test-audio`.** `make generated` is unchanged and still needs no download.

- **The test matrix now covers the ends of the supported range on every platform.** It pinned 3.10 and 3.13 across the three operating systems and filled in 3.11, 3.12 and 3.14 on Linux, which left 3.14 -- the newest version, and one the wheels are built for -- untested on macOS and Windows. The base matrix is now 3.10 and 3.14 everywhere, with 3.11 to 3.13 filled in on Linux: the same nine jobs, covering the two versions most likely to break on a platform-specific problem. Verified locally on macOS: the extensions build against 3.14.7 and the suite passes.

### Changed - documentation

- The README's channel section described the 15-channel limit as a hard one; it now describes what reuse does and when a diagnostic is still reported.

- `docs/reference.md` drops "channel reuse" from its known limitations and lists it as supported, with the remaining limitation stated precisely: reuse is decided per part rather than per note. Offline rendering joins the supported list.

- The README documents the `render` subcommand and its options, `aldakit.render()` / `Score.render()` in the quick start, and gains an "Audio Export" line in the feature list and a `render` row in the subcommand table.

- The README's Development section gains a "Golden Fixtures" section: what each of the two fixture sets pins, how to regenerate them, and how to get the SoundFont the audio ones need.

- The module map in `docs/api-design.md` covers `midi/channels.py` and `midi/render.py`.

### Tests

- 2115 tests, 92% coverage (from 1989 tests). `aldakit/midi/channels.py` and `aldakit/midi/render.py` are both at 100%.

- `tests/test_channel_reuse.py` (42 tests) covers reuse as a fallback, a part keeping its channel across a rest, a part continuing on another channel when its own is gone, instrument and controller state following the part across a handover, percussion and `(midi-channel)` pins being left alone, overflow when more than 15 parts sound at once, and both examples. Removing the controller reset, the channel stickiness, the pinned-channel reservation or the handover program change each fails the test written for it.

- `tests/test_multipart_timing.py` (9 tests) covers chords and crams across parts that differ in tempo and in default note length, including that a cram restores the default duration it borrowed. Six of the nine fail against the previous implementation.

- `tests/test_analysis.py::TestExamples` asserted that two examples were expected to have errors; it now asserts that none does.

- `tests/test_golden_audio.py` (48 tests) compares rendered audio against the fixtures. A failure reports where the audio diverged, by how much, and how many windows are out, because one window out is a note that changed while nearly all of them out by the same factor is a change in level or a tolerance too tight for the platform. It also asserts invariants on the fixtures themselves: that none is silent, that none clips at the fixture gain (a clamped mix hides a change in level -- one example did clip, which is why the fixtures render at 0.15 rather than 0.4), that both fixture sets cover the same examples, and that the panned example really does differ between channels, without which a pan regression could not fail.

- `tests/test_render.py` (25 tests) renders audio and asserts on it: length against `MidiSequence.duration()` plus the tail, silence where the score rests, sound where it does not, gain as a linear factor, determinism between two renders of one score, and the `render` subcommand's arguments and clipping warning. It also renders `all-instruments.alda` and fails if the audio falls silent anywhere, which is the only test in the suite that hears whether a channel handover actually worked. All of it skips when no SoundFont is installed.

## [0.2.1]

Three new subcommands -- `soundfont`, `info` and `lint` -- and a structural cleanup underneath them. The cleanup made several silent bugs visible, so **22 of the 40 examples now generate different MIDI**: 14 only in which channel each part lands on, and eight in how they sound. No score lost a note. The diff is in `tests/golden/examples.json`.

### Added

- **`aldakit soundfont`** reaches the SoundFont manager that has been in the package all along but had no CLI. `list` shows what is installed and what can be fetched, `install [NAME]` downloads from a catalog of three (SHA256 verified, `--all` and `--force` supported), `verify` re-checks the downloads, and `path` prints the file playback would use. When audio playback is asked for and no SoundFont can be found, aldakit now offers to download one instead of only reporting the error; non-interactive runs still get the error.

- **`aldakit info FILE`** summarises a score without playing it: parts with their instruments, MIDI programs, channels and note counts, plus tempo, duration, key signatures, transpositions, variables and markers.

- **`aldakit lint FILE`** reports what will make a score sound wrong: unknown instruments and attributes, undefined variables and markers, notes clamped into the MIDI range, unused or redefined variables, scores with no notes, and parts that collide on a MIDI channel. Exit status is 0 clean, 1 on an error (or any finding under `--strict`), 2 when the score does not parse, so it works as a build step. `--quiet` reports through the status alone.

- **`aldakit.analysis`** is the module behind both: `inspect_score()` and `lint_score()` return values, not printed text, so an editor plugin or a CI check can use them directly.

- **Four Alda attributes that parsed but did nothing** are implemented: `set-duration` (beats), `set-note-length` (a note value), `set-duration-ms` (converted at the part's tempo), `track-volume` / `track-vol` (MIDI channel volume, as opposed to note velocity), and `midi-channel` (pinning a part to a channel, refused for channel 9 in a melodic part). This is what `poly.alda`, `multi-poly.alda`, `track-volume.alda` and `midi-channel-management-2.alda` were waiting on; all four now play as written, and no bundled example produces an unknown-attribute warning.

- **Notes pushed outside MIDI's 0-127 by an octave or a transposition** are reported rather than silently clamped.

- **Diagnostics carry a code** (`unknown-instrument`, `undefined-variable`, `note-out-of-range`, ...) so tools can group and filter them without matching on prose. An S-expression the generator does not recognise now records one instead of being ignored in silence.

- **`Score.elements`**, the public form of the compose elements a score was built from. Returns a copy; use `add()` to extend a score.

### Fixed

- **A global attribute above the first part declaration is no longer dropped.** Only `(tempo! N)` was inherited by parts declared later; `(quant! N)`, `(vol! N)`, `(key-sig! ...)` and `(transpose! N)` applied to nothing at all when written at the top of a score, which is where scores put them. `across_the_sea.alda` and `bach_cello_suite_no_1.alda` ignored their `(quant! ...)`, and `debussy_quartet.alda` and `rachmaninoff_piano_concerto_2_mvmt_2.alda` played in C major instead of the key they declare.

- **A leading global attribute no longer consumes a MIDI channel.** Writing one above the first part declaration created an implicit part that took channel 0, so the score's first instrument started on channel 1 and one of the 15 melodic channels was lost.

- **`(pan N)` and `(quant! N)` now work.** `pan` is Alda's documented abbreviation for `panning` and `quant!` its global form, but neither was in the generator's dispatch chain, so both were silently ignored. `(transposition N)` -- the canonical spelling of `transpose` -- was missing for the same reason. `tests/shared_suite/20_panning.expected` changed accordingly: the three `(pan N)` calls in that score now emit pan control changes.

- **A dotted note or rest built through the compose API no longer loses the dot in one of its two output paths.** `note("c", dots=1).to_ast()` produced a dotted quarter while `to_alda()` produced a bare `c`, and `rest(dots=1)` dropped the dot in both. Both now produce `c4.` / `r4.`.

### Changed - internal

- **`aldakit/theory.py`** holds the pitch names, scale and mode intervals, and key signatures that the MIDI generator and the compose module previously encoded separately. `compose.scales` re-exports its tables under their old names, and the key-signature parsing that lived on `MidiGenerator` is now pure functions there.

- **`constants.py` is now wired in.** Eighteen constants were defined and never imported while the values they named sat hardcoded elsewhere -- including a dynamics table where eleven of the fourteen velocities disagreed with the generator's live copy. The generator's values (derived from Alda's volume scale, `velocity = volume * 127 / 100`) won; `constants.py` now holds them and the generator reads them, and `PartState`'s defaults, the SMF writer's status bytes, the tempo arithmetic and the SoundFont environment variable all read from the module too. `SOUNDFONT_NAMES`, `MODE_INTERVALS`, the accidental characters and `MIDI_MAX_PROGRAM` are gone from it, having moved to the code that owns them or nowhere. `tests/test_constants.py` fails if a constant goes unused again.

- **Attribute handling is a registry.** Each Alda attribute is a `@handles` decorated method on `MidiGenerator` rather than a branch in a 120-line if/elif chain, so adding one is a method plus a decorator line.

- **`Score` no longer carries a mode tag.** The three ways a score can be built are three small classes in `aldakit/score_content.py`, each answering how to build an AST, how to export Alda source, and how to print. This removes the three `cls.__new__(cls)` constructors that set five attributes by hand.

- **`ASTNode.accept()` is defined once** on the base class instead of thirty identical copies on the node types.

- **Duration handling shared by `Note` and `Rest`** moved to `compose/duration.py`; it was two copies of the same method.

- `tests/demo.py`, which pytest never collected, moved to `scripts/demo_parser.py`, and the unused `Parser._peek_next` is gone.

### Changed - documentation

- The README's `play` option table was missing `-e, --eval`, and its `eval` table gave the port flag as `--port` where the parser accepts `-p, --port`.

- "Zero-dependency" now reads "no install-time dependencies", with the three bundled projects named in the same breath, in both the README and the docs home page.

- The module map in `docs/api-design.md` covers `theory.py`, `constants.py`, `score_content.py` and `compose/duration.py`, and says what belongs in the two shared-fact modules.

- The README documents the `soundfont`, `info` and `lint` subcommands and the `inspect_score()` / `lint_score()` functions behind them, and its feature list and quick start cover both. `docs/index.md` matches.

- `docs/reference.md` documents the attributes implemented in this release along with `pan`, `transposition` and the global `!` form, and gains a "Known Limitations" section recording that a part holds its MIDI channel for the whole score where Alda frees it once the part stops sounding.

- The `Score` listing in `docs/api-design.md` described pre-refactor internals (`cls.__new__`, `_elements`, `hasattr(self, "_source")`) and claimed MIDI import was unimplemented. It is now an interface-level listing, which is what a design document should carry, plus a pointer to `score_content.py`.

### Tests

- 1989 tests, 92% coverage (from 1835 and 89%).

- `tests/test_constants.py` checks that every constant has a caller, that the pitch and scale tables are the same objects everywhere, that mode intervals and scale intervals agree, and covers the key-signature calculations moved to `theory.py`.

- `tests/test_midi.py` gains `TestGlobalAttributes`, `TestAttributeRegistry`, `TestDurationAttributes`, `TestTrackVolume` and `TestMidiChannel`; `tests/test_compose.py` gains `TestDurationSpelling`, which asserts that every way of writing a duration survives both `to_ast()` and `to_alda()` identically.

- `tests/test_analysis.py`, `tests/test_cli_info_lint.py` and `tests/test_cli_soundfont.py` cover the new commands. The SoundFont tests never touch the network: the catalog is replaced with an entry whose checksum matches what the stubbed downloader writes, so verification still runs for real.

## [0.2.0]

This release fixes fourteen defects that made scores sound wrong rather than fail. **Existing scores may sound different** -- see "Changed" for what moved.

It also includes everything listed under 0.1.11 below, which was developed in parallel and merged in: visitor-based AST dispatch, error-message hints, the MkDocs site, and the opt-in publish/prerelease workflow.

### Fixed - correctness

These all produced well-formed output that simply sounded wrong, which is why they survived a passing test suite.

- **All 128 General MIDI programs are now reachable under their Alda names.** The instrument table omitted Alda's canonical `midi-` prefixed names and 47 GM programs had no name at all, so `examples/all-instruments.alda` resolved 128 of its 129 instruments to acoustic grand piano. The table is now generated from `docs/alda-language/list-of-instruments.md` by `scripts/gen_instruments.py` (`make instruments`) and covers 318 names across all 128 programs. Names that earlier releases accepted but that are not Alda-canonical are retained in `LEGACY_INSTRUMENT_ALIASES`.

- **`midi-percussion` is routed to MIDI channel 10, and melodic parts are kept off it.** Percussion previously became program 0 on whatever channel it was handed, so `percussion.alda` played as piano; conversely the tenth declared part landed on the drum channel, so `all-instruments.alda`, `midi-channel-management.alda`, `nicechord-alda-demo.alda` and `rachmaninoff_piano_concerto_2_mvmt_2.alda` played melodic lines as drum hits. Percussion parts emit no program change, and key signatures and transposition no longer shift their note numbers.

- **The compose API preserves the octave a note declares.** `Note.octave` was stored and used by `Note.midi_pitch` but discarded by both `to_ast()` and `to_alda()`, so `note("c", octave=5)` generated middle C. This silently broke `voicing()`, `build_chord(octave=...)`, `arpeggiate()`, `Note.transpose()` across octave boundaries, and the register of every `transcribe()` result. Compose elements now thread an `OctaveContext` through conversion, emitting an octave change only where the octave actually changes.

- **Imported MIDI files play back the way they were imported.** `midi_to_ast` emitted a `PartDeclarationNode` and an `EventSequenceNode` as siblings, but the generator had no branch for a bare declaration, so declarations were ignored: every track collapsed onto channel 0 with program 0 and the parts played one after another instead of together. Imports now emit `PartNode`, channel 10 is imported as a `midi-percussion` part, and the generator also handles a bare declaration defensively.

- **`Score.play(wait=False)` no longer stops playback immediately.** The backend was created inside a `with` block, so the block exited right after `play()` and closed it, shutting down the playback threads and broadcasting all-notes-off. Non-blocking playback now returns a `PlaybackHandle` that owns the backend; see Added below.

- **The AST-to-Alda exporter is complete.** The previous helper raised `TypeError` on cram expressions and variable definitions, and silently dropped repeats, voices, markers, barlines, ties and millisecond/second durations. It is replaced by `aldakit.serialize.AldaWriter`, a visitor covering every AST node, verified by round-trip tests over every example and shared-suite file.

- **The group-member dot accessor works.** `strings.cello:` created a new cello instance on its own channel starting at time zero instead of addressing the cello of the aliased group, so `examples/dot_accessor.alda` produced the wrong music. The scanner now treats a dot between identifier characters as part of a name, and the generator resolves `alias.instrument` against recorded group membership.

- **`aldakit eval` accepts `--parse-only` and `--no-wait`.** Both were documented in the README; neither was registered on the subparser, so the documented invocation exited with a usage error. `play` and `eval` now share one argument definition.

- **`Note` validates accidentals.** `note("c", accidental="sharp")` produced the literal string `csharp`; anything other than `+`, `-`, `_` and repetitions now raises `ValueError`.

- **The audio backend applies control changes.** `TsfBackend` scheduled programs and notes but ignored `sequence.control_changes`, so `(panning ...)` was a silent no-op in audio mode. The `_tsf` extension gained `schedule_control()`, and it now also selects the General MIDI drum bank for channel 10 so percussion parts sound as drums rather than as a melodic instrument.

- **`SoundFontManager.list()` renamed to `list_installed()`.** Binding the name `list` inside the class body shadowed the builtin and invalidated the `list[Path]` annotations on three methods. `list()` remains as a deprecated alias.

- **Removed a production workaround for a test double.** `beats_to_duration()` unwrapped a `.expected` attribute so that two tests could pass `pytest.approx(...)` as an *input*. The tests now pass plain floats.

- **Playback no longer falls silent when no MIDI ports exist.** On a machine with no MIDI output ports and no `ALDAKIT_SOUNDFONT` or config entry, `aldakit play file.alda` opened a virtual MIDI port and played into it; with nothing connected, that is silence, reported as success. The CLI decided whether a SoundFont was available by consulting only explicit configuration, never `find_soundfont()`. Consequently `-a` also failed with "No soundfont configured" while SoundFonts sat in `~/.aldakit/soundfonts/`. Backend selection is now one function, `cli.resolve_backend()`, shared by `play`, `eval` and `repl` (it previously existed as three near-copies that disagreed), and it uses SoundFont discovery. When audio genuinely is not possible, the virtual-port fallback now warns that nothing will be heard.

- **Alda files with non-ASCII characters no longer fail on Windows.** `cli.read_source()` called `read_text()` with no encoding, so Python used the locale codepage (cp1252 on Windows) and any non-ASCII character in a score raised `UnicodeDecodeError`. Every other read in the codebase already said UTF-8. Found by the first Windows CI run; `tests/test_docs.py` now fails if any text IO in `src/` omits its encoding.

- **`TsfBackend` expands `~` and `$VARS` in SoundFont paths.** The config file expanded them but the backend did not, so the documented `TsfBackend(soundfont="~/Music/sf2/FluidR3_GM.sf2")` raised `FileNotFoundError`.

### Documentation

A pre-release audit executed every code block in `README.md` and `docs/`, which found claims no test covered:

- `README.md` showed `aldakit FILE` and `aldakit --port X FILE` without the `play` subcommand, a form removed when subcommands were introduced.

- `README.md` documented `crescendo(start_vel=, end_vel=)`; the parameters are `start_velocity` / `end_velocity`.

- `docs/extending-aldakit.md` imported eight MIDI-level transformers from `aldakit.compose.transform` instead of `aldakit.midi.transform`, imported `Score` from `aldakit.compose`, called `seq.from_alda()` on the factory function rather than the `Seq` class, and used wrong keywords for `euclidean()`, `cellular_automaton()`, `rest()` and `note()`. It also described MIDI import as future work and mutation methods that do not exist. It is now accurate and framed as a description of the shipped design rather than a proposal.

- `docs/architecture.d2` labelled edges `export()` and `import()`; the functions are `write_alda()` and `Score.from_midi_file()`. Assets regenerated.

`tests/test_docs.py` now guards against recurrence: every documented `aldakit` import must resolve, every ```alda block must parse and generate, every documented `aldakit ...` command line must be accepted by the real parser, and the concrete expectations in `docs/test_specification.md` must match generated output. Prose still needs a human; these cover what a machine can check.

### Added

- **The REPL can reach the filesystem.** It previously had no way to open the files `aldakit play` exists to read, nor to save an improvisation; the only input was typing or pasting.

  - `:load FILE` reads a file into the session without playing it, reporting its parts, note count and duration. A missing `.alda` suffix is tried, so `:load twinkle` finds `twinkle.alda`. Loading a long score therefore does not tie up the prompt, and the file can be saved or inspected first.

  - `:play` plays the loaded score; `:play FILE` loads and plays in one step. Loaded scores keep their own tempo: the REPL's default is only prepended to typed input that sets none.

  - `:save FILE` writes the session as one score, or exports MIDI for `.mid` and `.midi`.

  - `:ls [DIR]`, `:cd [DIR]`, `:pwd` for navigation, and `:clear` to discard the session.

  - Tab completion now covers command names and, for `:load`, `:save` and `:cd`, filesystem paths.

  - `aldakit repl FILE` loads a file on startup, ready for `:play`.

  Command handling moved out of the prompt loop into `handle_command()`. `PromptSession` requires a TTY, so anything inside the loop cannot be tested; the commands now have direct test coverage.

- **`aldakit.serialize`** - `AldaWriter` and `write_alda()` render any AST back to Alda source. Round-tripping preserves musical meaning: `parse(write(parse(src)))` generates identical MIDI.

- **`PlaybackHandle`** - returned by `Score.play(wait=False)` and by `aldakit.play(..., wait=False)`. Exposes `is_playing()`, `wait()`, `stop()` and `close()`, and works as a context manager. `Score.stop()` stops background playback started from that score.

- **Generation diagnostics** - `MidiGenerator.diagnostics` and `Score.diagnostics` report problems that change what is heard without stopping generation: unknown instrument names, undefined variable and marker references, unresolved group members, and MIDI channel exhaustion. The CLI prints them as warnings.

- **Instrument lookup helpers** - `lookup_instrument()`, `is_percussion()`, `normalize_instrument_name()`, `canonical_name()` and `PROGRAM_NAMES` in `aldakit.midi.types`.

- **`_tsf.TsfPlayer.schedule_control(channel, control, value, time)`** for scheduled MIDI control changes in the audio backend.

### Changed

- `organ` now resolves to church organ (program 19), matching Alda, rather than drawbar organ (program 16). Use `midi-drawbar-organ` or the retained legacy alias `drawbar-organ` for the previous behaviour.

- `MidiGenerator` skips MIDI channel 9 when allocating channels for pitched instruments, leaving 15 melodic channels.

### Testing and CI

- **CI now runs the test suite.** The workflow previously only triggered on `workflow_dispatch` and had no test job at all. It now runs on push and pull request across Linux, macOS and Windows on Python 3.10-3.14, running ruff, ty and pytest, and gates wheel building and publishing on the result.

- **Golden MIDI fixtures** (`tests/golden/examples.json`, regenerate with `make golden`) pin the exact notes, channels, programs, timings and velocities of all 40 examples, so a change to how a score sounds shows up as a reviewable diff instead of passing silently.

- **New test modules**: `test_instruments.py`, `test_channel_allocation.py`, `test_serialize.py`, `test_compose_octave.py`, `test_golden_midi.py`, `test_midi_import_playback.py`, `test_playback_handle.py`, `test_cli_arguments.py`, `test_tsf_control_changes.py`, `test_dot_accessor.py`. The suite grew from 1032 to 1678 tests; each fix was verified to fail when reverted.

- `ruff` and `ty` are clean across `src/`, `tests/` and `scripts/`.

## [0.1.11]

### Changed

- **Visitor-based AST dispatch** - Replaced `isinstance()`-based dispatch in `score.py` (`node_to_str()`) and `midi/generator.py` (`_process_node()`) with proper visitor pattern using `ASTVisitor` from `ast_nodes.py`

  - `score.py`: New `_AldaStringVisitor` class handles AST-to-Alda string conversion via `visit_*` methods, replacing nested closures with `isinstance` chains and `hasattr` fallbacks

  - `midi/generator.py`: `MidiGenerator` now extends `ASTVisitor`; the monolithic 20-branch `_process_node()` dispatch is replaced by individual `visit_*Node` methods with dynamic dispatch via `ASTVisitor.visit()`

- **Error messages with fix suggestions** - Added a `hint` field to `AldaParseError` and contextual hints across all scanner, parser, and CLI error paths

  - Scanner: hints for unexpected characters, unmatched parens, unterminated strings, empty markers/repeats

  - Parser: hints for missing colons, unclosed brackets/braces/parens, quoted expressions, variable syntax, durations

  - CLI: actionable suggestions for missing input, missing files, no notes generated, unavailable backends

### Added

- **MkDocs documentation site** - Added `mkdocs.yml` with Material theme, `docs/index.md` landing page, and organized navigation covering home, quick reference, Alda language guide (15 pages), and API design

  - New Makefile targets: `docs` (build), `docs-serve` (local preview), `docs-deploy` (GitHub Pages)

  - Added `mkdocs` and `mkdocs-material` as dev dependencies

- **GitHub Actions prerelease workflow** - Added opt-in `prerelease` checkbox to the `Build and Publish` workflow dispatch. When enabled, reads version from `pyproject.toml`, creates a git tag, and uploads all wheels and sdist to a GitHub prerelease titled `aldakit v{version}`. PyPI publishing is also now opt-in via a separate `publish` checkbox (both default to off).

### Removed

- **Unused C parser** (`thirdparty/alda-parser`) - Removed incomplete, unbundled C parser. The Python recursive descent parser is the sole parser and has full Alda language coverage.

- **Stale internal docs** - Removed `test_specification.md`, `example_compatibility.md`, `offset.md` (garbled), `implementing-an-alda-library.md`, `writing-music-programmatically.md`, `instance-and-group-assignment.md` (all original Alda project docs not relevant to aldakit users)

### Fixed

- **Path expansion tests now work cross-platform** - Fixed brittle `test_expands_tilde` test that failed on Windows due to path separator differences (`/` vs `\`). Tests now use `Path` objects for comparison instead of string matching.

- **Added platform-specific path tests** - Unix-style path tests skip on Windows; new Windows-specific tests added for full coverage.

- **Path expansion tests now skippable** - Set `SKIP_PATH_EXPANSION_TESTS=1` environment variable to skip all path expansion tests if needed.

## [0.1.10]

### Added

- **`SoundFontManager` class** (`aldakit.midi.soundfont`) - Centralized SoundFont management

  - `find()` - Search common locations for a General MIDI SoundFont

  - `list()` - List all SoundFont files found in common locations

  - `download(name)` - Download a SoundFont from the catalog with SHA256 verification

  - `ensure(name)` - Find existing or download if not found

  - `setup(name)` - Interactive setup with progress display

  - `setup_all(force)` - Download all SoundFonts from the catalog

  - `verify_checksums()` - Verify SHA256 checksums for all downloaded SoundFonts

  - `get_search_paths()` - Get the list of paths searched for SoundFont files

  - `list_available_downloads()` - List SoundFonts available for download

  - Configurable `soundfont_dir` and `catalog` via constructor

  - All module-level functions preserved for backwards compatibility

- **New module-level functions** (`aldakit.midi.soundfont`)

  - `setup_all_soundfonts(force=False)` - Download all (3) SoundFonts from the catalog

  - `verify_soundfont_checksums()` - Verify SHA256 checksums for all downloaded SoundFonts

### Changed

- **SoundFont management refactored** - All SoundFont discovery, downloading, and verification logic now consolidated in `soundfont.py` module (previously split between `soundfont.py` and `tsf_backend.py`)

- **SoundFont search order updated** - `~/.aldakit/soundfonts/` is now searched first (after `ALDAKIT_SOUNDFONT` env var), before user music folders and system locations

## [0.1.9]

### Added

- **`--audio` / `-a` flag** for `play`, `eval`, and `repl` subcommands

  - Explicitly selects the built-in TinySoundFont audio backend

  - Uses pre-configured soundfont from config file (`~/.aldakit/config.ini`) or `ALDAKIT_SOUNDFONT` environment variable

  - Avoids need to specify `-sf /path/to/soundfont.sf2` each time

  - Example: `aldakit play -a examples/twinkle.alda`

- **`--virtual-port` / `-vp` flag** for `play`, `eval`, and `repl` subcommands

  - Allows customizing the virtual MIDI port name (default: "AldakitMIDI")

  - Example: `aldakit repl -vp MyMIDI`

- **Centralized constants** (`src/aldakit/constants.py`)

  - All default values, magic numbers, and configuration constants now in one place

  - Includes: `DEFAULT_VIRTUAL_PORT_NAME`, `DEFAULT_TEMPO`, `DEFAULT_BACKEND`, `MAX_PLAYBACK_SLOTS`, MIDI protocol constants, timing intervals

### Fixed

- **Virtual MIDI port creation** now works when no physical MIDI ports are available

  - Previously, the CLI would error with "No MIDI output ports available" instead of creating a virtual port

  - Now correctly falls through to let the libremidi backend create the virtual port

## [0.1.8]

### Added

- **Key signature support** (`key-sig`, `key-signature`) - Full implementation in MIDI generator

  - String format: `(key-sig "f+ c+ g+")` for explicit accidentals

  - Quoted list format: `(key-sig '(g major))`, `(key-sig '(d minor))`

  - Modal key signatures: `(key-sig '(d dorian))`, `(key-sig '(e phrygian))`

  - Explicit accidentals: `(key-sig '(b (flat) e (flat)))`

  - Natural sign (`_`) overrides key signature on individual notes

  - Per-part and global (`key-sig!`) variants

- **Transposition support** (`transpose`) - Full implementation in MIDI generator

  - Semitone-based transposition: `(transpose 5)` for up a fourth, `(transpose -7)` for down a fifth

  - Useful for transposing instruments: `clarinet: (transpose -2)` for Bb clarinet

  - Resets with `(transpose 0)`

  - Per-part and global (`transpose!`) variants

  - Correctly clamps to valid MIDI range (0-127)

- **Quoted list syntax** in S-expressions (`'(...)`) - Lisp-style quoted lists now fully parsed

- **Concurrent playback mode** for LibremidiBackend

  - Up to 8 simultaneous playback slots

  - `concurrent_mode` property (default True) to enable/disable layered playback

  - `active_slots` property to check number of active playbacks

  - `wait()` method to block until all playback completes

  - `stop()` stops all slots, `stop_slot(id)` stops a specific one

  - Thread-safe MIDI message sending

  - Inspired by alda-midi's libuv-based async system

- **REPL concurrent playback integration**

  - Concurrent mode enabled by default - inputs layer on each other

  - `:concurrent` command to enable concurrent mode

  - `:sequential` command to enable sequential mode (wait for each input)

  - `:status` command to show playback mode and active slots

  - `--sequential` CLI flag to start REPL in sequential mode

- **CLI audio backend option**

  - `-sf` / `--soundfont FILE` to use TinySoundFont audio backend

  - Available for both REPL (`aldakit repl -sf ...`) and play (`aldakit play -sf ... file.alda`)

  - Clear error message when no MIDI ports available, directing user to use `-sf`

- **CLI reorganization**

  - New `eval` subcommand: `aldakit eval "piano: c d e"`

  - Top-level now only has `--version` and `--help`

  - All options moved to their respective subcommands

  - `aldakit` with no args opens the REPL

- **Configuration file support** (`~/.aldakit/config.ini`)

  - INI-format config files using Python's built-in `configparser` (zero dependencies)

  - User config: `~/.aldakit/config.ini`

  - Project-local config: `./aldakit.ini` (overrides user config)

  - Supported options: `soundfont`, `backend`, `port`, `tempo`, `verbose`

  - Environment variable `ALDAKIT_SOUNDFONT` overrides config files

  - CLI arguments always take highest priority

  - Smart fallback: MIDI is preferred by default; audio is used only if explicitly requested (`-sf` or `backend=audio`) or as fallback when no MIDI ports are available

### Fixed

- **Windows build failure in `_tsf.cpp`** - Added `#define NOMINMAX` to prevent Windows SDK `min`/`max` macro conflicts with `std::min`/`std::max`

- **Type checking errors in `tsf_backend.py` and `soundfont.py`**

  - Fixed `callable` type hints to use `Callable[[int, int], None]` from `collections.abc`

  - Added type ignore comments for native module imports

  - Fixed dictionary value type inference with explicit `str()` casts

- **Flaky `test_current_time_advances` test** - Replaced fixed sleep with polling loop to handle audio thread startup latency

## [0.1.7]

### Added

#### Direct Audio Playback via TinySoundFont

Self-contained audio synthesis without requiring an external MIDI synthesizer:

- **New `_tsf` native module** - TinySoundFont + miniaudio integration via nanobind

  - Header-only libraries for minimal binary size (~50KB)

  - Cross-platform audio output (CoreAudio, ALSA, WASAPI)

  - Real-time SoundFont synthesis at 44.1kHz stereo

- **New `TsfBackend` class** (`aldakit.midi.backends.TsfBackend`)

  - Drop-in replacement for `LibremidiBackend` when no external synth is available

  - Automatic SoundFont discovery in common locations

  - Environment variable support (`ALDAKIT_SOUNDFONT`)

  - `play()`, `stop()`, `wait()`, `is_playing()`, `current_time()` methods

  - Preset inspection: `preset_count`, `preset_name(index)`

  - Adjustable gain via `set_gain()`

- **`Score.play()` backend parameter**

  - `score.play(backend="audio")` - direct audio output via TinySoundFont

  - `score.play(backend="midi")` - MIDI output via libremidi (default, unchanged)

  - `score.play(backend="audio", soundfont="/path/to/sf2")` - explicit SoundFont

- **SoundFont utilities** (`aldakit.midi.soundfont`)

  - `find_soundfont()` - search common locations for GM SoundFont

  - `list_soundfonts()` - list all discovered SoundFont files

  - `download_soundfont(name)` - download from public archives

  - `setup_soundfont()` - interactive setup with progress display

  - Catalog includes TimGM6mb (5.8MB), FluidR3_GM (141MB), GeneralUser_GS (31MB)

- **29 new tests** for TsfBackend (`tests/test_tsf_backend.py`)

### Changed

- Development status changed from Beta to Alpha in `pyproject.toml`

- **CLI now opens REPL by default** when run without arguments (`aldakit` is now equivalent to `aldakit repl`)

## [0.1.6]

### Fixed

- **Vendored prompt_toolkit imports in REPL now use sys.path approach correctly**

  - `repl.py` now imports `ext` first to initialize sys.path, then uses absolute imports for prompt_toolkit.

  - Previously, relative imports (`.ext.prompt_toolkit`) bypassed the sys.path setup needed for vendored packages.

- **Test assertions now use explicit `assert` statements**

  - Converted mock assertion methods (e.g., `mock.assert_called_once()`) to explicit `assert` statements in test_api.py and test_midi_import.py.

  - Fixes pytest-review warnings about tests with no assertions.

- **Compose API part declarations now generate correct AST structure**

  - `Score.from_elements(part("violin"), note("c"))` now wraps declarations and events in `PartNode`, so instruments are properly honored during MIDI generation.

  - Previously, bare `PartDeclarationNode` objects were ignored by the MIDI generator, causing all parts to play as piano (program 0).

- **`to_alda()` now handles `PartNode` and `ChordNode` correctly**

  - `PartNode` is now explicitly rendered with its declaration and events, preserving part structure in round-trips.

  - `ChordNode` no longer crashes with `AttributeError` (it has no `duration` attribute; individual notes carry their durations).

- **Variable definitions no longer emit sound immediately**

  - `theme = c d e` now only stores the events; sound is emitted only when the variable is referenced (`theme`).

  - Previously, definitions would double-play content (once at definition, once at each reference).

- **MIDI file writer now handles tempo changes correctly**

  - Introduced `TempoMap` class that properly integrates tick positions across tempo segments.

  - Channel tracks now use the same tempo map as the tempo track, ensuring notes align with tempo changes.

  - Previously, tempo changes mid-score caused timing drift because timestamps were converted using a single constant tempo.

## [0.1.5]

### Added

- **Transcription expressiveness**

  - Real-time transcribe now supports swing, triplet, and quintuplet feels with true ties and dotted durations.

  - Tuplet groups (including block chords) collapse into `{ ... }` cram expressions instead of long tie chains.

  - Chord transcription preserves ties/tuplets so block chords inherit the same rhythmic structure as monophonic lines.

  - Per-recording metadata (feel, swing ratio, tuplet division) is surfaced via `Seq.metadata` for downstream tweaking.

  - CLI `transcribe` command exposes `--feel` and `--swing-ratio` flags; transcription tuples populate the metadata.

  - MIDI import emits in-part tempo changes whenever `MidiTempoChange` events appear after the initial tempo.

- **CLI improvements**

  - `aldakit ports` now lists both output and input ports by default, with `-o/--outputs` or `-i/--inputs` filtering as needed.

  - `--port` accepts both port names and numeric indices (e.g., `--port 0` uses the first port from `aldakit ports`).

  - Single-port auto-selection: when only one MIDI port is available, it is automatically selected without requiring `--port`.

### Changed

- `Score.save()` writes `.mid` files directly via the SMF writer, dropping the libremidi dependency for exports.

- `Seq` now carries an optional metadata dictionary that concatenation preserves; `seq(..., metadata=...)` helper added.

- `Chord` objects support dotted durations so tuplets and other rhythmic transforms can express chords without losing dots.

- CLI stdin mode and version flag use `aldakit.__version__` and context-managed backends to avoid leaked ports.

- Transcription timing constants extracted as named module-level constants for clarity and tuning.

### Fixed

- `beat_to_duration` now handles pytest `approx` inputs and the expanded duration catalog adds triplet/quintuplet denominators.

- Libremidi-dependent tests are skipped automatically when the native extension is unavailable.

- CLI validates `--swing-ratio` is in range (0, 1) before transcription.

## [0.1.4]

### Added

#### High-Level Python API

- New `Score` class for working with Alda music (`from aldakit import Score`)

  - `Score(source)` and `Score.from_file(path)` constructors

  - `Score.from_elements(*elements)` for programmatic composition

  - `Score.from_parts(*parts)` convenience constructor

  - `play(port=None, wait=True)` method for MIDI playback

  - `save(path)` method for MIDI/Alda file export

  - `to_alda()` method for Alda source code export

  - Lazy `ast` and `midi` properties (computed and cached on first access)

  - `duration` property for total score length in seconds

  - Builder methods: `add()`, `with_part()`, `with_tempo()`, `with_volume()`

- Module-level convenience functions for one-liner usage:

  - `aldakit.play(source)` - parse and play Alda code

  - `aldakit.play_file(path)` - parse and play an Alda file

  - `aldakit.save(source, path)` - parse and save as MIDI

  - `aldakit.save_file(source_path, output_path)` - convert Alda file to MIDI

  - `aldakit.list_ports()` - list available MIDI output ports

#### Compose Module (`aldakit.compose`)

Programmatic music composition with domain objects that generate AST directly (no text parsing):

- **Core elements:**

  - `note(pitch, *, duration, octave, accidental, dots, ms, seconds, slurred)` - create notes

  - `rest(*, duration, dots, ms, seconds)` - create rests

  - `chord(*notes_or_pitches, duration)` - create chords

  - `seq(*elements)` - create sequences

  - `Seq.from_alda(source)` - parse Alda into a sequence

- **Part declarations:**

  - `part(*instruments, alias)` - instrument declarations

- **Attributes:**

  - `tempo(bpm, global_)` - tempo setting

  - `volume(level)` / `vol(level)` - volume control

  - `octave(n)` - set octave

  - `octave_up()` / `octave_down()` - relative octave changes

  - `quant(level)` - quantization/legato

  - `panning(level)` - stereo panning

  - Dynamic markings: `pp()`, `p()`, `mp()`, `mf()`, `f()`, `ff()`

- **Transformations:**

  - `Note.sharpen()` / `Note.flatten()` - accidental changes

  - `Note.transpose(semitones)` - pitch transposition

  - `Note.with_duration(n)` / `Note.with_octave(n)` - property changes

  - `note * n` / `seq * n` - repeat syntax

- **Advanced elements:**

  - `cram(*elements, duration)` - tuplet/cram expressions

  - `voice(number, *elements)` - voice for polyphonic writing

  - `voice_group(*voices)` - group of parallel voices

  - `var(name, *elements)` - variable definition

  - `var_ref(name)` - variable reference

  - `marker(name)` - marker (synchronization point)

  - `at_marker(name)` - jump to marker

- **Output methods:**

  - `to_ast()` - generate AST node directly

  - `to_alda()` - generate Alda source code

#### Scales Module (`aldakit.compose.scales`)

Music theory utilities for scales and modes:

- **Scale generation:**

  - `scale(root, scale_type)` - get pitch names for a scale

  - `scale_notes(root, scale_type, octave, duration, ascending)` - generate Seq of notes

  - `scale_degree(root, scale_type, degree, octave)` - get specific scale degree

  - `mode(root, mode_name)` - alias for scale (ionian, dorian, etc.)

- **Key relationships:**

  - `relative_minor(major_root)` - get relative minor (C -> A)

  - `relative_major(minor_root)` - get relative major (A -> C)

  - `parallel_minor(major_root)` - same root, minor mode

  - `parallel_major(minor_root)` - same root, major mode

- **Utilities:**

  - `transpose_scale(pitches, semitones)` - transpose pitch list

  - `interval_name(semitones)` - get interval name (0 -> "unison", 7 -> "perfect fifth")

  - `list_scales()` - list available scale types

  - `SCALE_INTERVALS` - dictionary of scale interval patterns

- **Scale types:** major, minor, harmonic-minor, melodic-minor, pentatonic, blues, chromatic, whole-tone, diminished, dorian, phrygian, lydian, mixolydian, locrian, japanese, arabic, hungarian-minor, spanish, bebop-dominant, bebop-major

#### Chords Module (`aldakit.compose.chords`)

Chord voicing utilities for building common chord types:

- **Core builder:**

  - `build_chord(root, chord_type, octave, duration, inversion)` - build any chord type

- **Triad constructors:**

  - `major(root)`, `minor(root)`, `dim(root)`, `aug(root)`, `sus2(root)`, `sus4(root)`

- **Seventh chord constructors:**

  - `maj7(root)`, `min7(root)`, `dom7(root)`, `dim7(root)`, `half_dim7(root)`, `min_maj7(root)`, `aug7(root)`

- **Extended chord constructors:**

  - `maj6(root)`, `min6(root)`, `dom9(root)`, `maj9(root)`, `min9(root)`, `add9(root)`, `power(root)`

- **Chord utilities:**

  - `arpeggiate(chord, pattern, duration)` - convert chord to arpeggio note sequence

  - `invert_chord(chord, inversion)` - apply inversion to existing chord

  - `voicing(chord, octaves)` - apply custom octave voicing (spread, close, etc.)

  - `list_chord_types()` - list available chord types

  - `CHORD_INTERVALS` - dictionary of chord interval patterns

#### Transform Module (`aldakit.compose.transform`)

AST-level transformers for musical sequences (can be exported back to Alda):

- **Pitch transformers:**

  - `transpose(seq, semitones)` - transpose all notes by semitones

  - `invert(seq, axis)` - invert intervals around an axis pitch

  - `reverse(seq)` - retrograde (reverse order)

  - `shuffle(seq, seed)` - random permutation

  - `retrograde_inversion(seq)` - combined reverse + invert

- **Structural transformers:**

  - `augment(seq, factor)` - lengthen durations (e.g., 8th -> quarter)

  - `diminish(seq, factor)` - shorten durations (e.g., quarter -> 8th)

  - `fragment(seq, length)` - take first N elements

  - `loop(seq, times)` - repeat sequence (explicit duplication)

  - `interleave(*seqs)` - alternate elements from multiple sequences

  - `rotate(seq, positions)` - rotate elements left/right

  - `take_every(seq, n, offset)` - sample every Nth element

  - `split(seq, size)` - split into chunks

  - `concat(*seqs)` - concatenate sequences

- **Helpers:**

  - `pipe(seq, *transforms)` - chain multiple transformations

  - `identity(seq)` - return sequence unchanged (placeholder)

#### MIDI Transform Module (`aldakit.midi.transform`)

MIDI-level transformers for post-MIDI-generation processing (operates on absolute timing, cannot be reversed to Alda):

- **Timing transformers:**

  - `quantize(seq, grid, strength)` - snap note start times to a grid

  - `humanize(seq, timing, velocity, duration, seed)` - add random variations

  - `swing(seq, grid, amount)` - apply swing feel to offbeat notes

  - `stretch(seq, factor)` - scale all timings by a factor

  - `shift(seq, seconds)` - shift all notes forward/backward in time

- **Velocity transformers:**

  - `accent(seq, pattern, amount, base_velocity)` - apply accent pattern

  - `crescendo(seq, start_vel, end_vel, start_time, end_time)` - gradual velocity increase

  - `diminuendo(seq, start_vel, end_vel, start_time, end_time)` - gradual velocity decrease

  - `normalize(seq, target)` - scale velocities to target maximum

  - `velocity_curve(seq, func)` - apply custom velocity transformation

  - `compress(seq, threshold, ratio)` - compress dynamic range

- **Filtering:**

  - `filter_notes(seq, predicate)` - keep notes matching a condition

  - `trim(seq, start, end)` - extract a time range

- **Combining:**

  - `merge(*seqs)` - combine multiple sequences into one

  - `concatenate(*seqs, gap)` - append sequences end-to-end

#### Generate Module (`aldakit.compose.generate`)

Generative functions for algorithmic music composition:

- **Random selection:**

  - `random_note(scale, duration, octave, seed)` - random note from a scale

  - `random_choice(options, seed)` - random selection from options

  - `weighted_choice(weighted_options, seed)` - probability-weighted selection

- **Random walks:**

  - `random_walk(start, steps, intervals, duration, ...)` - pitch random walk

  - `drunk_walk(start, steps, max_step, bias, ...)` - biased toward smaller intervals

- **Rhythmic generators:**

  - `euclidean(hits, steps, pitch, rotate)` - Euclidean rhythm patterns (tresillo, cinquillo, etc.)

  - `probability_seq(notes, length, probability)` - probabilistic note/rest sequence

  - `rest_probability(seq, probability)` - add random rests to existing sequence

- **Markov chains:**

  - `markov_chain(transitions)` - create chain from transition probabilities

  - `learn_markov(sequence, order)` - learn transitions from existing melody

  - `MarkovChain.generate(start, length)` - generate new melodies

- **L-Systems:**

  - `lsystem(axiom, rules, iterations, note_map)` - Lindenmayer system patterns

- **Cellular automata:**

  - `cellular_automaton(rule, width, steps, pitch_on)` - Wolfram elementary automata (rules 0-255)

- **Shift registers:**

  - `shift_register(length, taps, bits, scale, mode)` - Linear Feedback Shift Register patterns

  - `turing_machine(length, bits, scale, probability)` - Music Thing Modular-style evolving loops

#### MIDI Import (`aldakit.midi.smf_reader`, `aldakit.midi.midi_to_ast`)

Import MIDI files and convert them to Alda for editing, manipulation, and playback:

- **Score methods:**

  - `Score.from_midi_file(path, quantize_grid)` - import a MIDI file as a Score

  - `Score.from_file(path)` - now auto-detects .mid/.midi files and imports them

  - `score.to_alda()` - export imported MIDI as Alda source code

  - `score.save(path)` - re-save imported MIDI or convert to Alda

- **MIDI file reader:**

  - Full Standard MIDI File (SMF) format 0 and 1 support

  - Tempo map parsing with proper timing conversion

  - Note on/off pairing with duration calculation

  - Program change detection for instrument assignment

  - Variable-length quantity parsing

- **MIDI to AST conversion:**

  - MIDI pitch to note name mapping (with sharps for black keys)

  - Timing quantization to standard note values (whole, half, quarter, etc.)

  - Configurable quantization grid (default: 16th notes)

  - Chord detection for simultaneous notes

  - Rest insertion for gaps between notes

  - Multi-channel support (each channel becomes a separate part)

  - General MIDI program to instrument name mapping

- **Round-trip workflow:**

  - Import MIDI -> Edit as Alda -> Export back to MIDI

  - Import MIDI -> Apply transformations -> Play or save

#### Real-Time MIDI Transcription (`aldakit.midi.transcriber`)

Record MIDI input from a keyboard or controller and convert to Alda:

- **Module-level functions:**

  - `transcribe(duration, port_name, instrument, ...)` - record MIDI for a duration and return a Score

  - `list_input_ports()` - list available MIDI input ports

- **TranscribeSession class:**

  - `start(port_name)` - start recording from a MIDI port

  - `stop()` - stop recording and return a Seq

  - `poll()` - poll for incoming messages (call periodically)

  - `on_note(callback)` - set callback for note events (pitch, velocity, is_on)

- **Features:**

  - Thread-safe message queue (callbacks from C++ thread)

  - Automatic note duration detection

  - Configurable quantization grid

  - Gap detection and rest insertion

  - Pending note handling (notes still held when recording stops)

#### CLI Transcription Commands

- **New subcommands:**

  - `aldakit input-ports` - list available MIDI input ports

  - `aldakit transcribe` - record MIDI input and output Alda code

- **Transcribe options:**

  - `-d, --duration SECONDS` - recording duration (default: 10)

  - `-i, --instrument NAME` - instrument name (default: piano)

  - `-t, --tempo BPM` - tempo for quantization (default: 120)

  - `-q, --quantize GRID` - quantize grid in beats (default: 0.25 = 16th notes)

  - `-o, --output FILE` - save to file (.alda or .mid)

  - `--port NAME` - MIDI input port name

  - `--play` - play back the recording after transcription

  - `-v, --verbose` - show notes as they are played

  - `--alda-notes` - show notes in Alda notation (with -v)

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
