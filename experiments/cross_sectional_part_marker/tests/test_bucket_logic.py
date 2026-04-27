"""
Tests for the score_to_bucket() helper in scoring.py.

Run with:
    pytest experiments/cross_sectional_part_marker/tests/test_bucket_logic.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.cross_sectional_part_marker.src.schemas import BucketDefinition, RubricSpec
from experiments.cross_sectional_part_marker.src.scoring import score_to_bucket

_FIXTURES = Path(__file__).parent / "fixtures"


def _make_bucket_defs_10() -> list[BucketDefinition]:
    """Standard 0–10 bucket definitions for a 10-mark question."""
    return [
        BucketDefinition(bucket="A", label="Excellent",  mark_range=(9, 10), description=""),
        BucketDefinition(bucket="B", label="Good",       mark_range=(7, 8),  description=""),
        BucketDefinition(bucket="C", label="Adequate",   mark_range=(5, 6),  description=""),
        BucketDefinition(bucket="D", label="Poor",       mark_range=(3, 4),  description=""),
        BucketDefinition(bucket="E", label="Very Poor",  mark_range=(1, 2),  description=""),
        BucketDefinition(bucket="F", label="Fail",       mark_range=(0, 0),  description=""),
    ]


def _make_bucket_defs_5() -> list[BucketDefinition]:
    """Standard 0–5 bucket definitions for a 5-mark question."""
    return [
        BucketDefinition(bucket="A", label="Excellent",  mark_range=(5, 5), description=""),
        BucketDefinition(bucket="B", label="Good",       mark_range=(4, 4), description=""),
        BucketDefinition(bucket="C", label="Adequate",   mark_range=(3, 3), description=""),
        BucketDefinition(bucket="D", label="Poor",       mark_range=(2, 2), description=""),
        BucketDefinition(bucket="E", label="Very Poor",  mark_range=(1, 1), description=""),
        BucketDefinition(bucket="F", label="Fail",       mark_range=(0, 0), description=""),
    ]


class TestScoreToBucket:
    def test_top_score_maps_to_A(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(10.0, defs) == "A"

    def test_score_9_maps_to_A(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(9.0, defs) == "A"

    def test_score_8_maps_to_B(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(8.0, defs) == "B"

    def test_score_7_maps_to_B(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(7.0, defs) == "B"

    def test_score_6_maps_to_C(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(6.0, defs) == "C"

    def test_score_5_maps_to_C(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(5.0, defs) == "C"

    def test_score_4_maps_to_D(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(4.0, defs) == "D"

    def test_score_3_maps_to_D(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(3.0, defs) == "D"

    def test_score_2_maps_to_E(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(2.0, defs) == "E"

    def test_score_1_maps_to_E(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(1.0, defs) == "E"

    def test_score_0_maps_to_F(self):
        defs = _make_bucket_defs_10()
        assert score_to_bucket(0.0, defs) == "F"

    def test_boundary_lower_of_A_range(self):
        """Exactly 8.0 should map to A when range is [8, 10]."""
        defs = [
            BucketDefinition(bucket="A", label="A", mark_range=(8, 10), description=""),
            BucketDefinition(bucket="B", label="B", mark_range=(5, 7),  description=""),
            BucketDefinition(bucket="C", label="C", mark_range=(3, 4),  description=""),
            BucketDefinition(bucket="F", label="F", mark_range=(0, 2),  description=""),
        ]
        assert score_to_bucket(8.0, defs) == "A"

    def test_boundary_upper_of_B_range(self):
        """Exactly 7.0 should map to B when range is [5, 7]."""
        defs = [
            BucketDefinition(bucket="A", label="A", mark_range=(8, 10), description=""),
            BucketDefinition(bucket="B", label="B", mark_range=(5, 7),  description=""),
            BucketDefinition(bucket="C", label="C", mark_range=(3, 4),  description=""),
            BucketDefinition(bucket="F", label="F", mark_range=(0, 2),  description=""),
        ]
        assert score_to_bucket(7.0, defs) == "B"

    def test_zero_maps_to_F_5mark(self):
        defs = _make_bucket_defs_5()
        assert score_to_bucket(0.0, defs) == "F"

    def test_empty_bucket_defs_returns_F(self):
        assert score_to_bucket(5.0, []) == "F"

    def test_fractional_score_5mark(self):
        """Score 4.5 should fall in the bucket whose range contains it."""
        defs = [
            BucketDefinition(bucket="A", label="A", mark_range=(4.5, 5.0), description=""),
            BucketDefinition(bucket="B", label="B", mark_range=(3.0, 4.4), description=""),
            BucketDefinition(bucket="F", label="F", mark_range=(0.0, 2.9), description=""),
        ]
        assert score_to_bucket(4.5, defs) == "A"

    def test_score_above_max_returns_highest_bucket(self):
        """
        A score above all defined ranges should return the bucket with the
        lowest lower bound (the fallback). This tests that the function
        does not raise and returns a sensible value.
        """
        defs = _make_bucket_defs_10()
        # 12 > 10, above all ranges
        result = score_to_bucket(12.0, defs)
        assert result in ("A", "F")  # implementation-defined; should not raise

    def test_score_to_bucket_with_fixture_rubric(self):
        """Integration test using the fixture rubric spec."""
        fixture_path = _FIXTURES / "sample_rubric_spec.json"
        with open(fixture_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        spec = RubricSpec.model_validate(raw)

        q1a = spec.get_part_spec("Q1", "a")
        assert q1a is not None
        defs = q1a.bucket_definitions

        assert score_to_bucket(5.0, defs) == "A"
        assert score_to_bucket(4.0, defs) == "B"
        assert score_to_bucket(3.0, defs) == "C"
        assert score_to_bucket(2.0, defs) == "D"
        assert score_to_bucket(1.0, defs) == "E"
        assert score_to_bucket(0.0, defs) == "F"
