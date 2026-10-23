"""End-to-end tests for the CA-RAG pipeline and baselines."""

import unittest

from carg.baselines import PromptEnhancedRAG, VanillaRAG
from carg.dataset import build_test_set
from carg.judge import Judge
from carg.pipeline import CARAGPipeline
from carg.vectorstore import InMemoryVectorStore, KnowledgeBase


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase(InMemoryVectorStore())
        cls.pipeline = CARAGPipeline(cls.kb)
        cls.vanilla = VanillaRAG(cls.kb)
        cls.prompt = PromptEnhancedRAG(cls.kb)
        cls.judge = Judge()
        cls.samples = build_test_set()

    def test_answer_returns_nonempty_text(self):
        for sample in self.samples[:3]:
            response = self.pipeline.answer_sample(sample)
            self.assertTrue(response.text.strip())

    def test_beginner_response_is_rated_beginner(self):
        beginner = [s for s in self.samples if s.user_level == 0][0]
        response = self.pipeline.answer_sample(beginner)
        self.assertEqual(self.judge.rate_difficulty(response.text), 0)

    def test_expert_response_is_rated_expert(self):
        expert = [s for s in self.samples if s.user_level == 2][0]
        response = self.pipeline.answer_sample(expert)
        self.assertEqual(self.judge.rate_difficulty(response.text), 2)

    def test_prerequisites_explained_before_core(self):
        beginner = [s for s in self.samples if s.user_level == 0][0]
        response = self.pipeline.answer_sample(beginner)
        for prereq, target in response.ordered_pairs:
            self.assertTrue(self.judge.is_ordered(response.text, prereq, target))

    def test_ablation_flags_change_output(self):
        sample = self.samples[0]
        full = self.pipeline.answer_sample(sample, cm=True, kgd=True, ag=True)
        no_kgd = self.pipeline.answer_sample(sample, cm=True, kgd=False, ag=True)
        self.assertNotEqual(full.text, no_kgd.text)
        self.assertGreaterEqual(len(full.addressed_gaps), len(no_kgd.addressed_gaps))

    def test_ca_rag_outperforms_vanilla(self):
        cas_ca = []
        cas_vanilla = []
        for s in self.samples:
            r_ca = self.pipeline.answer_sample(s)
            r_v = self.vanilla.answer(s)
            cas_ca.append(1 - abs(s.user_level - self.judge.rate_difficulty(r_ca.text)) / 2)
            cas_vanilla.append(1 - abs(s.user_level - self.judge.rate_difficulty(r_v.text)) / 2)
        self.assertGreater(sum(cas_ca), sum(cas_vanilla))


if __name__ == "__main__":
    unittest.main()
