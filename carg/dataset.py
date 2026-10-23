"""Synthetic cognitive interaction dataset.

The paper constructs a synthetic dataset because public QA corpora lack user
cognitive-state annotations. This module reproduces that idea: it builds
``Sample`` objects -- (query, target concept, user level, Knowledge Vector,
required prerequisites, missing concepts) -- spanning the three cognitive
levels across the stroke-education concept vocabulary.

Everything is deterministic (no randomness), so experiments reproduce exactly.
"""

from typing import List

from .cognitive_model import KnowledgeVector
from .config import BEGINNER, INTERMEDIATE, EXPERT, MISSING_THRESHOLD
from .knowledge import CONCEPTS, QUERIES, prerequisites_of
from .types import Sample


def make_kv(level: int, sample_index: int = 0) -> KnowledgeVector:
    """Build a Knowledge Vector consistent with ``level``.

    Mastery scores are spread so that:
      * Beginner      -> mean mastery below 0.4 (most concepts missing)
      * Intermediate  -> mean mastery around 0.5-0.6 (mixed mastery)
      * Expert        -> mean mastery above 0.7 (most concepts mastered)

    The values use a deterministic modular jitter so that ``CognitiveModel``'s
    level inference maps each KV back to its labelled level.
    """
    kv = {}
    for i, concept in enumerate(CONCEPTS):
        if level == BEGINNER:
            mastery = 0.18 + 0.03 * ((i * 3 + sample_index) % 7)
        elif level == INTERMEDIATE:
            mastery = 0.40 + 0.04 * ((i * 5 + sample_index) % 8)
        else:  # EXPERT
            mastery = 0.74 + 0.03 * ((i * 7 + sample_index) % 7)
        kv[concept] = round(mastery, 3)
    return KnowledgeVector(kv)


def build_test_set() -> List[Sample]:
    """Build the evaluation test set: every query x every cognitive level."""
    samples: List[Sample] = []
    index = 0
    for query, target in QUERIES.items():
        for level in (BEGINNER, INTERMEDIATE, EXPERT):
            kv = make_kv(level, sample_index=index)
            required = prerequisites_of(target, closure=True)
            missing = [c for c in CONCEPTS if kv[c] < MISSING_THRESHOLD]
            samples.append(Sample(
                query=query,
                target_concept=target,
                user_level=level,
                kv=kv,
                required_prereqs=required,
                missing_concepts=missing,
            ))
            index += 1
    return samples
