"""Tests for the Knowledge Gap Detection (KGD) module."""

import unittest

from carg.cognitive_model import KnowledgeVector
from carg.gap_detection import (
    ConstraintInstruction,
    KnowledgeGapDetector,
    knowledge_gap_vector,
)


class TestKnowledgeGapVector(unittest.TestCase):
    def test_gap_is_complement(self):
        kv = KnowledgeVector({"blood_circulation": 0.3, "blood_vessels": 0.9})
        kgv = knowledge_gap_vector(kv)
        self.assertAlmostEqual(kgv["blood_circulation"], 0.7)
        self.assertAlmostEqual(kgv["blood_vessels"], 0.1)


class TestKnowledgeGapDetector(unittest.TestCase):
    def setUp(self):
        self.kgd = KnowledgeGapDetector(missing_threshold=0.5)

    def test_weakest_concept(self):
        kv = KnowledgeVector({"blood_circulation": 0.3, "blood_vessels": 0.9})
        self.assertEqual(self.kgd.weakest_concept(kv), "blood_circulation")

    def test_missing_concepts_sorted_by_mastery(self):
        kv = KnowledgeVector({"blood_circulation": 0.2, "blood_vessels": 0.4,
                              "blood_pressure": 0.9})
        missing = self.kgd.missing_concepts(kv)
        self.assertEqual(missing, ["blood_circulation", "blood_vessels"])

    def test_build_instruction_lists_weak_and_missing_prereqs(self):
        kv = KnowledgeVector({"blood_circulation": 0.2, "blood_vessels": 0.3,
                              "blood_clot": 0.9})
        instruction = self.kgd.build_instruction(kv, target_concept="ischemic_stroke")
        self.assertIsInstance(instruction, ConstraintInstruction)
        self.assertIn("blood_circulation", instruction.weak_concepts)
        self.assertIn("blood_vessels", instruction.weak_concepts)
        self.assertTrue(instruction.policy)

    def test_constraint_input_contains_query_and_policy(self):
        kv = KnowledgeVector(default=0.3)
        instruction = self.kgd.build_instruction(kv, target_concept="ischemic_stroke")
        text = instruction.to_text()
        self.assertIn("Generation Policy", text)
        self.assertIn("explain prerequisite concepts first", text)


if __name__ == "__main__":
    unittest.main()
