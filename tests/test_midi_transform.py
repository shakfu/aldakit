"""Tests for MIDI-level transformers."""

import pytest

from aldakit.midi.types import MidiNote, MidiSequence
from aldakit.midi.transform import (
    # Timing transformers
    quantize,
    humanize,
    swing,
    stretch,
    shift,
    # Velocity transformers
    accent,
    crescendo,
    diminuendo,
    normalize,
    velocity_curve,
    compress,
    # Filtering
    filter_notes,
    trim,
    # Combining
    merge,
    concatenate,
)


def make_sequence(notes: list[tuple[int, int, float, float]]) -> MidiSequence:
    """Helper to create a MidiSequence from note tuples (pitch, velocity, start, duration)."""
    return MidiSequence(
        notes=[
            MidiNote(pitch=p, velocity=v, start_time=s, duration=d)
            for p, v, s, d in notes
        ]
    )


class TestQuantize:
    """Test quantize transformer."""

    def test_quantize_snaps_to_grid(self):
        seq = make_sequence(
            [
                (60, 80, 0.12, 0.5),  # Slightly after 0
                (62, 80, 0.48, 0.5),  # Slightly before 0.5
                (64, 80, 0.76, 0.5),  # Slightly after 0.75
            ]
        )
        quantized = quantize(seq, 0.25)  # Quarter note grid
        assert abs(quantized.notes[0].start_time - 0.0) < 0.01
        assert abs(quantized.notes[1].start_time - 0.5) < 0.01
        assert abs(quantized.notes[2].start_time - 0.75) < 0.01

    def test_quantize_partial_strength(self):
        seq = make_sequence([(60, 80, 0.1, 0.5)])
        # With strength 0.5, should move halfway to grid
        quantized = quantize(seq, 0.25, strength=0.5)
        # 0.1 should move toward 0.0 by 50%: 0.1 - 0.05 = 0.05
        assert abs(quantized.notes[0].start_time - 0.05) < 0.01

    def test_quantize_zero_strength(self):
        seq = make_sequence([(60, 80, 0.1, 0.5)])
        quantized = quantize(seq, 0.25, strength=0.0)
        assert quantized.notes[0].start_time == 0.1

    def test_quantize_preserves_note_properties(self):
        seq = make_sequence([(60, 100, 0.1, 0.5)])
        quantized = quantize(seq, 0.25)
        assert quantized.notes[0].pitch == 60
        assert quantized.notes[0].velocity == 100
        assert quantized.notes[0].duration == 0.5

    def test_quantize_invalid_grid(self):
        seq = make_sequence([(60, 80, 0.1, 0.5)])
        with pytest.raises(ValueError):
            quantize(seq, 0)
        with pytest.raises(ValueError):
            quantize(seq, -0.1)


class TestHumanize:
    """Test humanize transformer."""

    def test_humanize_with_seed_reproducible(self):
        seq = make_sequence([(60, 80, 0.0, 0.5), (62, 80, 0.5, 0.5)])
        h1 = humanize(seq, timing=0.02, velocity=10, seed=42)
        h2 = humanize(seq, timing=0.02, velocity=10, seed=42)
        assert h1.notes[0].start_time == h2.notes[0].start_time
        assert h1.notes[0].velocity == h2.notes[0].velocity

    def test_humanize_timing_within_range(self):
        seq = make_sequence([(60, 80, 1.0, 0.5)])
        humanized = humanize(seq, timing=0.02, seed=123)
        assert 0.98 <= humanized.notes[0].start_time <= 1.02

    def test_humanize_velocity_within_range(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        humanized = humanize(seq, velocity=10, seed=123)
        assert 70 <= humanized.notes[0].velocity <= 90

    def test_humanize_velocity_clamped(self):
        seq = make_sequence([(60, 5, 0.0, 0.5)])  # Low velocity
        humanized = humanize(seq, velocity=20, seed=42)
        assert humanized.notes[0].velocity >= 1

    def test_humanize_duration(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        humanized = humanize(seq, duration=0.05, seed=123)
        assert 0.45 <= humanized.notes[0].duration <= 0.55

    def test_humanize_no_changes_with_zero_amounts(self):
        seq = make_sequence([(60, 80, 0.5, 0.5)])
        humanized = humanize(seq, timing=0.0, velocity=0.0, duration=0.0)
        assert humanized.notes[0].start_time == 0.5
        assert humanized.notes[0].velocity == 80
        assert humanized.notes[0].duration == 0.5


class TestSwing:
    """Test swing transformer."""

    def test_swing_delays_offbeats(self):
        # Notes on grid: 0, 0.5, 1.0, 1.5
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.4),  # On beat
                (62, 80, 0.5, 0.4),  # Offbeat (should be delayed)
                (64, 80, 1.0, 0.4),  # On beat
                (65, 80, 1.5, 0.4),  # Offbeat (should be delayed)
            ]
        )
        swung = swing(seq, amount=0.33, grid=1.0)

        # On-beat notes should not move
        assert swung.notes[0].start_time == 0.0
        assert swung.notes[2].start_time == 1.0

        # Offbeat notes should be delayed
        assert swung.notes[1].start_time > 0.5
        assert swung.notes[3].start_time > 1.5

    def test_swing_amount_affects_delay(self):
        seq = make_sequence([(60, 80, 0.5, 0.4)])  # Offbeat
        light_swing = swing(seq, amount=0.2, grid=1.0)
        heavy_swing = swing(seq, amount=0.5, grid=1.0)

        assert heavy_swing.notes[0].start_time > light_swing.notes[0].start_time

    def test_swing_invalid_grid(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        with pytest.raises(ValueError):
            swing(seq, amount=0.33, grid=0)


class TestStretch:
    """Test stretch transformer."""

    def test_stretch_doubles_time(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
            ]
        )
        stretched = stretch(seq, 2.0)
        assert stretched.notes[0].start_time == 0.0
        assert stretched.notes[0].duration == 1.0
        assert stretched.notes[1].start_time == 2.0
        assert stretched.notes[1].duration == 1.0

    def test_stretch_halves_time(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 1.0),
                (62, 80, 2.0, 1.0),
            ]
        )
        stretched = stretch(seq, 0.5)
        assert stretched.notes[0].duration == 0.5
        assert stretched.notes[1].start_time == 1.0

    def test_stretch_preserves_pitch_velocity(self):
        seq = make_sequence([(60, 100, 0.0, 0.5)])
        stretched = stretch(seq, 2.0)
        assert stretched.notes[0].pitch == 60
        assert stretched.notes[0].velocity == 100

    def test_stretch_invalid_factor(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        with pytest.raises(ValueError):
            stretch(seq, 0)
        with pytest.raises(ValueError):
            stretch(seq, -1.0)


class TestShift:
    """Test shift transformer."""

    def test_shift_positive(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
            ]
        )
        shifted = shift(seq, 1.0)
        assert shifted.notes[0].start_time == 1.0
        assert shifted.notes[1].start_time == 2.0

    def test_shift_negative_removes_notes(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
            ]
        )
        shifted = shift(seq, -0.6)
        # First note starts at -0.6, ends at -0.1, so it's completely before 0
        # Second note starts at 0.4
        assert len(shifted.notes) == 1
        assert abs(shifted.notes[0].start_time - 0.4) < 0.01

    def test_shift_negative_truncates_note(self):
        seq = make_sequence([(60, 80, 0.0, 1.0)])
        shifted = shift(seq, -0.3)
        # Note would start at -0.3 but extends to 0.7, so truncate
        assert shifted.notes[0].start_time == 0.0
        assert abs(shifted.notes[0].duration - 0.7) < 0.01


class TestAccent:
    """Test accent transformer."""

    def test_accent_pattern(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 0.5, 0.5),
                (64, 80, 1.0, 0.5),
                (65, 80, 1.5, 0.5),
            ]
        )
        accented = accent(seq, [1.25, 0.75])
        assert accented.notes[0].velocity == 100  # 80 * 1.25
        assert accented.notes[1].velocity == 60  # 80 * 0.75
        assert accented.notes[2].velocity == 100  # Pattern repeats
        assert accented.notes[3].velocity == 60

    def test_accent_with_base_velocity(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        accented = accent(seq, [1.0], base_velocity=100)
        assert accented.notes[0].velocity == 100

    def test_accent_empty_pattern(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        accented = accent(seq, [])
        assert accented.notes[0].velocity == 80  # Unchanged

    def test_accent_clamped(self):
        seq = make_sequence([(60, 120, 0.0, 0.5)])
        accented = accent(seq, [2.0])  # Would be 240
        assert accented.notes[0].velocity == 127


class TestCrescendo:
    """Test crescendo transformer."""

    def test_crescendo_basic(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
                (64, 80, 2.0, 0.5),
            ]
        )
        cresc = crescendo(seq, start_velocity=40, end_velocity=100)
        assert cresc.notes[0].velocity == 40
        assert cresc.notes[1].velocity == 70  # Midpoint
        assert cresc.notes[2].velocity == 100

    def test_crescendo_time_range(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
                (64, 80, 2.0, 0.5),
                (65, 80, 3.0, 0.5),
            ]
        )
        cresc = crescendo(seq, 40, 100, start_time=1.0, end_time=2.0)
        assert cresc.notes[0].velocity == 40  # Before range
        assert cresc.notes[1].velocity == 40  # Start of range
        assert cresc.notes[2].velocity == 100  # End of range
        assert cresc.notes[3].velocity == 100  # After range

    def test_crescendo_empty_sequence(self):
        seq = MidiSequence()
        cresc = crescendo(seq, 40, 100)
        assert len(cresc.notes) == 0


class TestDiminuendo:
    """Test diminuendo transformer."""

    def test_diminuendo_basic(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
                (64, 80, 2.0, 0.5),
            ]
        )
        dim = diminuendo(seq, start_velocity=100, end_velocity=40)
        assert dim.notes[0].velocity == 100
        assert dim.notes[2].velocity == 40


class TestNormalize:
    """Test normalize transformer."""

    def test_normalize_scales_to_target(self):
        seq = make_sequence(
            [
                (60, 50, 0.0, 0.5),
                (62, 100, 0.5, 0.5),  # Max velocity
                (64, 75, 1.0, 0.5),
            ]
        )
        normalized = normalize(seq, target=80)
        # Scale factor: 80/100 = 0.8
        assert normalized.notes[0].velocity == 40  # 50 * 0.8
        assert normalized.notes[1].velocity == 80  # 100 * 0.8
        assert normalized.notes[2].velocity == 60  # 75 * 0.8

    def test_normalize_empty_sequence(self):
        seq = MidiSequence()
        normalized = normalize(seq, target=100)
        assert len(normalized.notes) == 0


class TestVelocityCurve:
    """Test velocity_curve transformer."""

    def test_velocity_curve_custom(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        curved = velocity_curve(seq, lambda v: v // 2)
        assert curved.notes[0].velocity == 40

    def test_velocity_curve_clamped(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        curved = velocity_curve(seq, lambda v: v * 3)  # Would be 240
        assert curved.notes[0].velocity == 127


class TestCompress:
    """Test compress transformer."""

    def test_compress_above_threshold(self):
        seq = make_sequence(
            [
                (60, 60, 0.0, 0.5),  # Below threshold
                (62, 100, 0.5, 0.5),  # Above threshold
            ]
        )
        compressed = compress(seq, threshold=80, ratio=2.0)
        assert compressed.notes[0].velocity == 60  # Unchanged
        # 100 is 20 above threshold, compressed: 80 + 20/2 = 90
        assert compressed.notes[1].velocity == 90

    def test_compress_invalid_ratio(self):
        seq = make_sequence([(60, 80, 0.0, 0.5)])
        with pytest.raises(ValueError):
            compress(seq, threshold=80, ratio=0)


class TestFilterNotes:
    """Test filter_notes transformer."""

    def test_filter_by_velocity(self):
        seq = make_sequence(
            [
                (60, 50, 0.0, 0.5),
                (62, 100, 0.5, 0.5),
                (64, 30, 1.0, 0.5),
            ]
        )
        filtered = filter_notes(seq, lambda n: n.velocity > 40)
        assert len(filtered.notes) == 2
        assert filtered.notes[0].velocity == 50
        assert filtered.notes[1].velocity == 100

    def test_filter_by_pitch(self):
        seq = make_sequence(
            [
                (48, 80, 0.0, 0.5),  # Low
                (60, 80, 0.5, 0.5),  # Middle
                (72, 80, 1.0, 0.5),  # High
            ]
        )
        # Keep only middle register (C4 to C5)
        filtered = filter_notes(seq, lambda n: 60 <= n.pitch <= 72)
        assert len(filtered.notes) == 2

    def test_filter_by_time(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
                (64, 80, 2.0, 0.5),
            ]
        )
        filtered = filter_notes(seq, lambda n: n.start_time >= 1.0)
        assert len(filtered.notes) == 2


class TestTrim:
    """Test trim transformer."""

    def test_trim_basic(self):
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 1.0, 0.5),
                (64, 80, 2.0, 0.5),
                (65, 80, 3.0, 0.5),
            ]
        )
        trimmed = trim(seq, start=1.0, end=3.0)
        assert len(trimmed.notes) == 2
        # Times should be shifted to start at 0
        assert trimmed.notes[0].start_time == 0.0
        assert trimmed.notes[1].start_time == 1.0

    def test_trim_preserves_note_properties(self):
        seq = make_sequence([(60, 100, 1.0, 0.5)])
        trimmed = trim(seq, start=0.0, end=2.0)
        assert trimmed.notes[0].pitch == 60
        assert trimmed.notes[0].velocity == 100
        assert trimmed.notes[0].duration == 0.5


class TestMerge:
    """Test merge transformer."""

    def test_merge_two_sequences(self):
        seq1 = make_sequence([(60, 80, 0.0, 1.0)])
        seq2 = make_sequence([(64, 80, 0.0, 1.0)])
        merged = merge(seq1, seq2)
        assert len(merged.notes) == 2

    def test_merge_sorts_by_time(self):
        seq1 = make_sequence([(60, 80, 1.0, 0.5)])
        seq2 = make_sequence([(64, 80, 0.0, 0.5)])
        merged = merge(seq1, seq2)
        assert merged.notes[0].start_time == 0.0
        assert merged.notes[1].start_time == 1.0

    def test_merge_empty(self):
        merged = merge()
        assert len(merged.notes) == 0


class TestConcatenate:
    """Test concatenate transformer."""

    def test_concatenate_two_sequences(self):
        seq1 = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 0.5, 0.5),
            ]
        )  # Duration: 1.0
        seq2 = make_sequence(
            [
                (64, 80, 0.0, 0.5),
            ]
        )
        concat = concatenate(seq1, seq2)
        assert len(concat.notes) == 3
        assert concat.notes[0].start_time == 0.0
        assert concat.notes[1].start_time == 0.5
        assert concat.notes[2].start_time == 1.0  # After seq1 ends

    def test_concatenate_with_gap(self):
        seq1 = make_sequence([(60, 80, 0.0, 0.5)])  # Duration: 0.5
        seq2 = make_sequence([(64, 80, 0.0, 0.5)])
        concat = concatenate(seq1, seq2, gap=0.5)
        # seq1 ends at 0.5, gap of 0.5, seq2 starts at 1.0
        assert concat.notes[1].start_time == 1.0

    def test_concatenate_empty(self):
        concat = concatenate()
        assert len(concat.notes) == 0


class TestIntegration:
    """Integration tests for MIDI transformers."""

    def test_humanize_then_quantize(self):
        """Humanize and then re-quantize."""
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 0.5, 0.5),
                (64, 80, 1.0, 0.5),
            ]
        )
        humanized = humanize(seq, timing=0.1, seed=42)
        requantized = quantize(humanized, 0.5)

        # After re-quantizing, notes should be close to original grid
        assert abs(requantized.notes[0].start_time - 0.0) < 0.1
        assert abs(requantized.notes[1].start_time - 0.5) < 0.1
        assert abs(requantized.notes[2].start_time - 1.0) < 0.1

    def test_stretch_and_crescendo(self):
        """Stretch timing and apply crescendo."""
        seq = make_sequence(
            [
                (60, 80, 0.0, 0.5),
                (62, 80, 0.5, 0.5),
            ]
        )
        stretched = stretch(seq, 2.0)
        cresc = crescendo(stretched, 40, 100)

        assert stretched.notes[1].start_time == 1.0
        assert cresc.notes[0].velocity == 40
        assert cresc.notes[1].velocity == 100

    def test_with_score(self):
        """Test using MIDI transforms with a Score."""
        from aldakit import Score
        from aldakit.compose import part, note, tempo

        score = Score.from_elements(
            part("piano"),
            tempo(120),
            note("c", duration=8),
            note("d", duration=8),
            note("e", duration=8),
        )

        midi = score.midi
        humanized = humanize(midi, timing=0.01, velocity=5, seed=42)

        assert len(humanized.notes) == 3
        # Notes should be slightly different from original
        # (can't assert exact values due to randomness, but structure preserved)
