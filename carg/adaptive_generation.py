"""Adaptive Generation (AG).

The AG module produces responses through three progressive layers
(Section 3.3.3):

  1. Prerequisite Explanation -- foundational concepts the user is missing
  2. Core Concept Explanation  -- the target concept at the user's level
  3. Advanced Explanation      -- only if the user shows sufficient readiness

The generated text is assembled from the knowledge-base corpus passages at the
appropriate difficulty level. In a production system each layer would be
produced by prompting an LLM with the Constraint Instruction; the composition
logic here is identical either way, so swapping the mock corpus for a real LLM
only changes how each layer's prose is written, not the framework's behaviour.
"""

from typing import List, Optional

from .config import EXPERT, ADVANCED_READINESS_TAU
from .knowledge import prerequisites_of, explanation
from .types import Response
from .vectorstore import KnowledgeBase


class AdaptiveGenerator:
    """Hierarchical prerequisite-aware response generator (the AG component)."""

    def __init__(self, kb: KnowledgeBase,
                 tau: float = ADVANCED_READINESS_TAU):
        self.kb = kb
        self.tau = tau

    def generate(self, target_concept: str, level: int,
                 missing_prereqs: Optional[List[str]] = None,
                 kv=None) -> Response:
        """Compose a hierarchical response for ``target_concept``.

        ``missing_prereqs`` is the ordered list of prerequisite concepts that
        should be explained first (from KGD). ``kv`` is the current Knowledge
        Vector, used to decide whether to append the advanced stage.
        """
        parts: List[str] = []
        covered: List[str] = []
        ordered_pairs: List[tuple] = []

        # Stage 1: prerequisite explanation -----------------------------------
        for prereq in (missing_prereqs or []):
            text = self.kb.passages_for_concept(prereq, level).text
            if text:
                parts.append(text)
                covered.append(prereq)
                ordered_pairs.append((prereq, target_concept))

        # Stage 2: core concept explanation -----------------------------------
        core = self.kb.passages_for_concept(target_concept, level).text
        if core:
            parts.append(core)
            covered.append(target_concept)

        # Stage 3: advanced explanation ---------------------------------------
        ready = kv is not None and kv[target_concept] >= self.tau
        if ready and level != EXPERT:
            advanced = self.kb.passages_for_concept(target_concept, EXPERT).text
            if advanced and advanced != core:
                parts.append(advanced)

        text = "\n\n".join(parts)
        return Response(
            text=text,
            target_concept=target_concept,
            intended_level=level,
            covered_concepts=covered,
            ordered_pairs=ordered_pairs,
            addressed_gaps=list(missing_prereqs or []),
        )
