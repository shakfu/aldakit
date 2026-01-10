"""Test runner for shared Alda test suite.

Validates aldakit's MIDI output against .expected files.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from aldakit import parse
from aldakit.midi import generate_midi

SUITE_DIR = Path(__file__).parent / "shared_suite"

# Tolerance for floating point comparisons
TIME_TOLERANCE = 0.001  # 1ms tolerance for timing
DURATION_TOLERANCE = 0.001


@dataclass
class ExpectedNote:
    """Expected note from .expected file."""
    pitch: int
    start: float
    duration: float
    velocity: int
    channel: int


@dataclass
class ExpectedProgram:
    """Expected program change from .expected file."""
    program: int
    channel: int
    time: float


@dataclass
class ExpectedCC:
    """Expected control change from .expected file."""
    control: int
    value: int
    channel: int
    time: float


@dataclass
class ExpectedTempo:
    """Expected tempo change from .expected file."""
    bpm: float
    time: float


@dataclass
class ExpectedOutput:
    """Parsed expected output from .expected file."""
    notes: list[ExpectedNote]
    programs: list[ExpectedProgram]
    control_changes: list[ExpectedCC]
    tempos: list[ExpectedTempo]


def parse_expected_file(path: Path) -> ExpectedOutput:
    """Parse a .expected file into structured data."""
    notes = []
    programs = []
    control_changes = []
    tempos = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        record_type = parts[0]

        if record_type == "NOTE":
            notes.append(ExpectedNote(
                pitch=int(parts[1]),
                start=float(parts[2]),
                duration=float(parts[3]),
                velocity=int(parts[4]),
                channel=int(parts[5]),
            ))
        elif record_type == "PROGRAM":
            programs.append(ExpectedProgram(
                program=int(parts[1]),
                channel=int(parts[2]),
                time=float(parts[3]),
            ))
        elif record_type == "CC":
            control_changes.append(ExpectedCC(
                control=int(parts[1]),
                value=int(parts[2]),
                channel=int(parts[3]),
                time=float(parts[4]),
            ))
        elif record_type == "TEMPO":
            tempos.append(ExpectedTempo(
                bpm=float(parts[1]),
                time=float(parts[2]),
            ))

    return ExpectedOutput(notes, programs, control_changes, tempos)


def parse_and_generate(alda_file: Path):
    """Parse an Alda file and generate MIDI sequence."""
    text = alda_file.read_text()
    ast = parse(text)
    return generate_midi(ast)


@pytest.fixture(scope="module")
def all_test_files() -> list[Path]:
    """Get all .alda test files."""
    return sorted(SUITE_DIR.glob("*.alda"))


class TestNotesBasic:
    """Test 01_notes_basic.alda."""

    def test_notes_match_expected(self):
        expected = parse_expected_file(SUITE_DIR / "01_notes_basic.expected")
        seq = parse_and_generate(SUITE_DIR / "01_notes_basic.alda")

        assert len(seq.notes) == len(expected.notes), (
            f"Expected {len(expected.notes)} notes, got {len(seq.notes)}"
        )

        # Sort both by start time then pitch for comparison
        actual_notes = sorted(seq.notes, key=lambda n: (n.start_time, n.pitch))
        expected_notes = sorted(expected.notes, key=lambda n: (n.start, n.pitch))

        for i, (actual, exp) in enumerate(zip(actual_notes, expected_notes)):
            assert actual.pitch == exp.pitch, f"Note {i}: pitch mismatch"
            assert abs(actual.start_time - exp.start) < TIME_TOLERANCE, f"Note {i}: start mismatch"
            assert abs(actual.duration - exp.duration) < DURATION_TOLERANCE, f"Note {i}: duration mismatch"
            assert actual.velocity == exp.velocity, f"Note {i}: velocity mismatch"
            assert actual.channel == exp.channel, f"Note {i}: channel mismatch"


class TestAccidentals:
    """Test 02_notes_accidentals.alda."""

    def test_pitches_match_expected(self):
        expected = parse_expected_file(SUITE_DIR / "02_notes_accidentals.expected")
        seq = parse_and_generate(SUITE_DIR / "02_notes_accidentals.alda")

        actual_pitches = sorted([n.pitch for n in seq.notes])
        expected_pitches = sorted([n.pitch for n in expected.notes])

        assert actual_pitches == expected_pitches, (
            f"Pitch mismatch: expected {expected_pitches}, got {actual_pitches}"
        )


class TestDurations:
    """Test 03_notes_durations.alda."""

    def test_durations_match_expected(self):
        expected = parse_expected_file(SUITE_DIR / "03_notes_durations.expected")
        seq = parse_and_generate(SUITE_DIR / "03_notes_durations.alda")

        actual_durations = sorted([round(n.duration, 4) for n in seq.notes])
        expected_durations = sorted([round(n.duration, 4) for n in expected.notes])

        assert actual_durations == expected_durations


class TestOctaves:
    """Test 04_octaves.alda."""

    def test_octave_pitches_match(self):
        expected = parse_expected_file(SUITE_DIR / "04_octaves.expected")
        seq = parse_and_generate(SUITE_DIR / "04_octaves.alda")

        actual_pitches = sorted([n.pitch for n in seq.notes])
        expected_pitches = sorted([n.pitch for n in expected.notes])

        assert actual_pitches == expected_pitches


class TestRests:
    """Test 05_rests.alda."""

    def test_rests_create_timing_gaps(self):
        expected = parse_expected_file(SUITE_DIR / "05_rests.expected")
        seq = parse_and_generate(SUITE_DIR / "05_rests.alda")

        actual_starts = sorted([round(n.start_time, 4) for n in seq.notes])
        expected_starts = sorted([round(n.start, 4) for n in expected.notes])

        assert actual_starts == expected_starts


class TestChords:
    """Test 06_chords.alda."""

    def test_chord_notes_match(self):
        expected = parse_expected_file(SUITE_DIR / "06_chords.expected")
        seq = parse_and_generate(SUITE_DIR / "06_chords.alda")

        # Check simultaneous notes at time 0 (first chord)
        actual_at_zero = sorted([n.pitch for n in seq.notes if abs(n.start_time) < TIME_TOLERANCE])
        expected_at_zero = sorted([n.pitch for n in expected.notes if abs(n.start) < TIME_TOLERANCE])

        assert actual_at_zero == expected_at_zero, (
            f"First chord mismatch: expected {expected_at_zero}, got {actual_at_zero}"
        )


class TestTies:
    """Test 07_ties.alda."""

    def test_tied_durations_match(self):
        expected = parse_expected_file(SUITE_DIR / "07_ties.expected")
        seq = parse_and_generate(SUITE_DIR / "07_ties.alda")

        # Find the longest duration note (tied note)
        actual_max = max(n.duration for n in seq.notes)
        expected_max = max(n.duration for n in expected.notes)

        assert abs(actual_max - expected_max) < DURATION_TOLERANCE


class TestTempo:
    """Test 08_tempo.alda."""

    def test_tempo_affects_timing(self):
        expected = parse_expected_file(SUITE_DIR / "08_tempo.expected")
        seq = parse_and_generate(SUITE_DIR / "08_tempo.alda")

        actual_durations = sorted(set(round(n.duration, 4) for n in seq.notes))
        expected_durations = sorted(set(round(n.duration, 4) for n in expected.notes))

        assert actual_durations == expected_durations


class TestVolume:
    """Test 09_volume.alda."""

    def test_volume_velocities_match(self):
        expected = parse_expected_file(SUITE_DIR / "09_volume.expected")
        seq = parse_and_generate(SUITE_DIR / "09_volume.alda")

        actual_velocities = sorted([n.velocity for n in seq.notes])
        expected_velocities = sorted([n.velocity for n in expected.notes])

        assert actual_velocities == expected_velocities


class TestDynamics:
    """Test 10_dynamics.alda."""

    def test_dynamics_velocities_match(self):
        expected = parse_expected_file(SUITE_DIR / "10_dynamics.expected")
        seq = parse_and_generate(SUITE_DIR / "10_dynamics.alda")

        actual_velocities = sorted([n.velocity for n in seq.notes])
        expected_velocities = sorted([n.velocity for n in expected.notes])

        assert actual_velocities == expected_velocities


class TestParts:
    """Test 11_parts.alda."""

    def test_instrument_programs_match(self):
        expected = parse_expected_file(SUITE_DIR / "11_parts.expected")
        seq = parse_and_generate(SUITE_DIR / "11_parts.alda")

        actual_programs = sorted([pc.program for pc in seq.program_changes])
        expected_programs = sorted([pc.program for pc in expected.programs])

        assert actual_programs == expected_programs


class TestVariables:
    """Test 12_variables.alda."""

    def test_variable_notes_match(self):
        expected = parse_expected_file(SUITE_DIR / "12_variables.expected")
        seq = parse_and_generate(SUITE_DIR / "12_variables.alda")

        assert len(seq.notes) == len(expected.notes)


class TestMarkers:
    """Test 13_markers.alda."""

    def test_marker_sync_timing(self):
        expected = parse_expected_file(SUITE_DIR / "13_markers.expected")
        seq = parse_and_generate(SUITE_DIR / "13_markers.alda")

        # Verify multiple channels have notes at same time points
        actual_starts = sorted([round(n.start_time, 4) for n in seq.notes])
        expected_starts = sorted([round(n.start, 4) for n in expected.notes])

        assert actual_starts == expected_starts


class TestVoices:
    """Test 14_voices.alda."""

    def test_voices_parallel_timing(self):
        expected = parse_expected_file(SUITE_DIR / "14_voices.expected")
        seq = parse_and_generate(SUITE_DIR / "14_voices.alda")

        # Multiple notes should start at time 0
        actual_at_zero = len([n for n in seq.notes if abs(n.start_time) < TIME_TOLERANCE])
        expected_at_zero = len([n for n in expected.notes if abs(n.start) < TIME_TOLERANCE])

        assert actual_at_zero == expected_at_zero


class TestRepeats:
    """Test 15_repeats.alda."""

    def test_repeat_count_match(self):
        expected = parse_expected_file(SUITE_DIR / "15_repeats.expected")
        seq = parse_and_generate(SUITE_DIR / "15_repeats.alda")

        assert len(seq.notes) == len(expected.notes)


class TestCram:
    """Test 16_cram.alda."""

    def test_cram_timing_match(self):
        expected = parse_expected_file(SUITE_DIR / "16_cram.expected")
        seq = parse_and_generate(SUITE_DIR / "16_cram.alda")

        # Cram notes should have short durations
        actual_short = len([n for n in seq.notes if n.duration < 0.2])
        expected_short = len([n for n in expected.notes if n.duration < 0.2])

        assert actual_short == expected_short


class TestKeySignature:
    """Test 17_key_signature.alda."""

    def test_key_sig_pitches_match(self):
        expected = parse_expected_file(SUITE_DIR / "17_key_signature.expected")
        seq = parse_and_generate(SUITE_DIR / "17_key_signature.alda")

        actual_pitches = sorted([n.pitch for n in seq.notes])
        expected_pitches = sorted([n.pitch for n in expected.notes])

        assert actual_pitches == expected_pitches


class TestTranspose:
    """Test 18_transpose.alda."""

    def test_transpose_pitches_match(self):
        expected = parse_expected_file(SUITE_DIR / "18_transpose.expected")
        seq = parse_and_generate(SUITE_DIR / "18_transpose.alda")

        actual_pitches = sorted([n.pitch for n in seq.notes])
        expected_pitches = sorted([n.pitch for n in expected.notes])

        assert actual_pitches == expected_pitches


class TestQuantization:
    """Test 19_quantization.alda."""

    def test_quant_durations_match(self):
        expected = parse_expected_file(SUITE_DIR / "19_quantization.expected")
        seq = parse_and_generate(SUITE_DIR / "19_quantization.alda")

        actual_durations = sorted([round(n.duration, 4) for n in seq.notes])
        expected_durations = sorted([round(n.duration, 4) for n in expected.notes])

        assert actual_durations == expected_durations


class TestPanning:
    """Test 20_panning.alda."""

    def test_panning_cc_match(self):
        expected = parse_expected_file(SUITE_DIR / "20_panning.expected")
        seq = parse_and_generate(SUITE_DIR / "20_panning.alda")

        # Check CC#10 (pan) values
        actual_pan = sorted([cc.value for cc in seq.control_changes if cc.control == 10])
        expected_pan = sorted([cc.value for cc in expected.control_changes if cc.control == 10])

        assert actual_pan == expected_pan


class TestAllFilesValidation:
    """Validate all test files against their .expected files."""

    def test_all_files_parse(self, all_test_files):
        """Ensure every .alda file parses successfully."""
        errors = []
        for alda_file in all_test_files:
            try:
                parse(alda_file.read_text())
            except Exception as e:
                errors.append((alda_file.name, str(e)))
        assert not errors, f"Parse errors: {errors}"

    def test_all_files_have_expected(self, all_test_files):
        """Ensure every .alda file has a corresponding .expected file."""
        missing = []
        for alda_file in all_test_files:
            expected_file = alda_file.with_suffix(".expected")
            if not expected_file.exists():
                missing.append(alda_file.name)
        assert not missing, f"Missing .expected files for: {missing}"

    def test_all_notes_match(self, all_test_files):
        """Verify note count matches for all files."""
        mismatches = []
        for alda_file in all_test_files:
            expected_file = alda_file.with_suffix(".expected")
            if not expected_file.exists():
                continue

            expected = parse_expected_file(expected_file)
            seq = parse_and_generate(alda_file)

            if len(seq.notes) != len(expected.notes):
                mismatches.append(
                    f"{alda_file.name}: expected {len(expected.notes)}, got {len(seq.notes)}"
                )

        assert not mismatches, f"Note count mismatches:\n" + "\n".join(mismatches)
