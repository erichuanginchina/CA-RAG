"""Baseline RAG methods used in Experiment I.

Two baselines are implemented alongside the CA-RAG pipeline:

* ``VanillaRAG``         -- standard retrieval-augmented generation with no
                            cognitive modeling, gap detection, or adaptation.
* ``PromptEnhancedRAG``  -- Vanilla RAG extended with a *static* user profile
                            baked into the prompt. It does not dynamically
                            update a Knowledge Vector or detect gaps.

Both baselines are deliberately kept simple so that the only thing that
differs from CA-RAG is the cognitive-adaptation machinery, mirroring the
controlled setup described in Section 4.2.1.
"""

from .config import INTERMEDIATE
from .types import Response, Sample
from .vectorstore import KnowledgeBase


class VanillaRAG:
    """Retrieves relevant documents and generates a generic, unadapted answer.

    Because there is no notion of a user level, the deterministic simulation
    represents the answer with the corpus passage at the fixed intermediate
    level for the target concept. A real deployment would instead feed the
    retrieved documents to the LLM verbatim.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def answer(self, sample: Sample) -> Response:
        passage = self.kb.passages_for_concept(sample.target_concept, INTERMEDIATE)
        return Response(
            text=passage.text,
            target_concept=sample.target_concept,
            intended_level=INTERMEDIATE,
            covered_concepts=[sample.target_concept],
        )


class PromptEnhancedRAG:
    """Vanilla RAG with a *static* user profile supplied in the prompt.

    The static profile is a coarse, occasionally-incorrect prior over the
    user's level, modelling the paper's observation that prompt-level
    personalization "provides limited support for dynamically tracking users'
    evolving knowledge states". It has no Knowledge Vector, no gap detection,
    and no prerequisite coverage.
    """

    def __init__(self, kb: KnowledgeBase, assumed_level: int = INTERMEDIATE,
                 correct_fraction: float = 2 / 3):
        self.kb = kb
        self.assumed_level = assumed_level
        self.correct_fraction = correct_fraction

    def answer(self, sample: Sample) -> Response:
        # Deterministic "stale profile" model: the static profile is correct
        # for a fixed fraction of samples and otherwise collapses to the
        # assumed (default) level.
        level = self._static_level(sample)
        passage = self.kb.passages_for_concept(sample.target_concept, level)
        return Response(
            text="Based on the user's profile:\n\n" + passage.text,
            target_concept=sample.target_concept,
            intended_level=level,
            covered_concepts=[sample.target_concept],
        )

    def _static_level(self, sample: Sample) -> int:
        """Return the static profile's estimate of the user's level.

        Uses a deterministic, hash-seed-independent key derived from the
        sample so results are reproducible across runs.
        """
        key = _stable_key(sample.query, sample.target_concept, sample.user_level)
        if key % 3 != 0:  # correct 2/3 of the time
            return sample.user_level
        return self.assumed_level


def _stable_key(*parts) -> int:
    """A stable integer key independent of Python's randomized string hash."""
    value = 0
    for part in parts:
        for ch in str(part):
            value = (value * 131 + ord(ch)) % (2 ** 32)
    return value
