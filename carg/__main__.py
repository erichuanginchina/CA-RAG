"""Command-line entry point: ``python -m carag``.

Runs the two experiments from the paper and prints the resulting tables.
"""

import sys

from .baselines import PromptEnhancedRAG, VanillaRAG
from .dataset import build_test_set
from .experiments import (
    PAPER_EXPERIMENT_I,
    PAPER_EXPERIMENT_II,
    format_table,
    run_experiment_i,
    run_experiment_ii,
)
from .judge import Judge
from .pipeline import CARAGPipeline
from .vectorstore import InMemoryVectorStore, KnowledgeBase


def main() -> int:
    kb = KnowledgeBase(InMemoryVectorStore())
    judge = Judge()
    pipeline = CARAGPipeline(kb)
    vanilla = VanillaRAG(kb)
    prompt = PromptEnhancedRAG(kb)

    samples = build_test_set()
    print(f"Test set: {len(samples)} samples "
          f"({len(samples) // 3} queries x 3 cognitive levels)")
    print()

    exp1 = run_experiment_i(samples, pipeline, vanilla, prompt, judge)
    print("Experiment I -- comparative evaluation (simulated, offline)")
    print(format_table(exp1, ["Vanilla RAG", "Prompt-enhanced RAG", "CA-RAG"]))
    print()
    print("Experiment I -- paper's reported results (Gemma-3-4B)")
    print(format_table(PAPER_EXPERIMENT_I,
                       ["Vanilla RAG", "Prompt-enhanced RAG", "CA-RAG"]))
    print()

    exp2 = run_experiment_ii(samples, pipeline, vanilla, judge)
    order = ["CA-RAG", "-CM", "-KGD", "-AG", "-CM-KGD", "-CM-AG", "-KGD-AG",
             "Vanilla RAG"]
    print("Experiment II -- ablation study (simulated, offline)")
    print(format_table(exp2, order))
    print()
    print("Experiment II -- paper's reported results (Gemma-3-4B)")
    print(format_table(PAPER_EXPERIMENT_II, order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
