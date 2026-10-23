"""Tests for the Cognitive Modeling (CM) module."""

import unittest

from carg.cognitive_model import (
    CognitiveModel,
    InteractionSignals,
    KnowledgeVector,
)
from carg.config import BEGINNER, INTERMEDIATE, EXPERT


class TestKnowledgeVector(unittest.TestCase):
    def test_default_values(self):
        kv = KnowledgeVector(default=0.5)
        self.assertEqual(kv["blood_circulation"], 0.5)
        self.assertEqual(len(kv), 11)

    def test_clipping(self):
        kv = KnowledgeVector({"blood_circulation": 1.5, "blood_vessels": -0.3})
        self.assertEqual(kv["blood_circulation"], 1.0)
        self.assertEqual(kv["blood_vessels"], 0.0)

    def test_missing_concepts(self):
        kv = KnowledgeVector({"blood_circulation": 0.3, "blood_vessels": 0.8})
        missing = kv.missing_concepts(threshold=0.5)
        self.assertIn("blood_circulation", missing)
        self.assertNotIn("blood_vessels", missing)
        self.assertAlmostEqual(missing["blood_circulation"], 0.7)


class TestCognitiveModel(unittest.TestCase):
    def setUp(self):
        self.cm = CognitiveModel()

    def test_update_success_raises_mastery(self):
        kv = KnowledgeVector(default=0.5)
        signals = InteractionSignals(concepts=["blood_circulation"], success=True)
        self.cm.update(kv, signals)
        self.assertGreater(kv["blood_circulation"], 0.5)

    def test_update_clarification_lowers_mastery(self):
        kv = KnowledgeVector(default=0.5)
        signals = InteractionSignals(concepts=["blood_circulation"],
                                     clarification_requests=1)
        self.cm.update(kv, signals)
        self.assertLess(kv["blood_circulation"], 0.5)

    def test_update_excessive_dwell_lowers_mastery(self):
        kv = KnowledgeVector(default=0.5)
        signals = InteractionSignals(concepts=["blood_circulation"],
                                     dwell_time=100.0)
        self.cm.update(kv, signals)
        self.assertLess(kv["blood_circulation"], 0.5)

    def test_skip_foundational_raises_mastery(self):
        kv = KnowledgeVector(default=0.5)
        signals = InteractionSignals(concepts=["blood_circulation"],
                                     skip_foundational=True)
        self.cm.update(kv, signals)
        self.assertGreater(kv["blood_circulation"], 0.5)

    def test_mastery_stays_in_unit_interval(self):
        kv = KnowledgeVector(default=0.99)
        signals = InteractionSignals(concepts=["blood_circulation"],
                                     success=True, skip_foundational=True)
        for _ in range(10):
            self.cm.update(kv, signals)
        self.assertLessEqual(kv["blood_circulation"], 1.0)
        self.assertGreaterEqual(kv["blood_circulation"], 0.0)

    def test_infer_level(self):
        self.assertEqual(self.cm.infer_level(KnowledgeVector(default=0.2)),
                         BEGINNER)
        self.assertEqual(self.cm.infer_level(KnowledgeVector(default=0.55)),
                         INTERMEDIATE)
        self.assertEqual(self.cm.infer_level(KnowledgeVector(default=0.85)),
                         EXPERT)


if __name__ == "__main__":
    unittest.main()
