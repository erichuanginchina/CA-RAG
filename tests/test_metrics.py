"""Tests for the CAS / PCS / KGR evaluation metrics."""

import unittest

from carg.config import BEGINNER, INTERMEDIATE, EXPERT
from carg.judge import Judge
from carg.metrics import (
    cognitive_alignment_score,
    knowledge_gap_reduction,
    prerequisite_coverage_score,
)


class TestCognitiveAlignmentScore(unittest.TestCase):
    def test_perfect_alignment(self):
        self.assertEqual(cognitive_alignment_score(BEGINNER, BEGINNER), 1.0)
        self.assertEqual(cognitive_alignment_score(EXPERT, EXPERT), 1.0)

    def test_max_mismatch(self):
        self.assertEqual(cognitive_alignment_score(BEGINNER, EXPERT), 0.0)

    def test_partial_mismatch(self):
        self.assertEqual(cognitive_alignment_score(BEGINNER, INTERMEDIATE), 0.5)


class TestKnowledgeGapReduction(unittest.TestCase):
    def test_full_reduction(self):
        self.assertAlmostEqual(knowledge_gap_reduction(0.5, 0.0), 1.0)

    def test_no_reduction(self):
        self.assertAlmostEqual(knowledge_gap_reduction(0.5, 0.5), 0.0)

    def test_zero_pre_gap_excluded(self):
        self.assertIsNone(knowledge_gap_reduction(0.0, 0.0))


class TestPrerequisiteCoverageScore(unittest.TestCase):
    def setUp(self):
        self.judge = Judge()

    def test_full_coverage_and_ordering(self):
        # Response explains prerequisite then target, in order.
        text = ("Blood circulation moves blood around the body. "
                "An ischemic stroke happens when blood flow is blocked by a clot.")
        score = prerequisite_coverage_score(
            text, "ischemic_stroke",
            required_prereqs=["blood_circulation"],
            missing_concepts=["blood_circulation"],
            judge=self.judge,
        )
        self.assertGreater(score, 0.5)

    def test_no_coverage(self):
        text = "An ischemic stroke happens when blood flow is blocked."
        score = prerequisite_coverage_score(
            text, "ischemic_stroke",
            required_prereqs=["blood_circulation"],
            missing_concepts=["blood_circulation"],
            judge=self.judge,
        )
        # Coverage is 0, dependency 0, detection 0 -> average 0.
        self.assertAlmostEqual(score, 0.0)

    def test_no_required_prereqs_is_perfect(self):
        text = "Something about hypertension."
        score = prerequisite_coverage_score(
            text, "hypertension",
            required_prereqs=[],
            missing_concepts=[],
            judge=self.judge,
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
