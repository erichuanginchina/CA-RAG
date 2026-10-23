"""The evaluator judge.

The paper evaluates with a "unified evaluation pipeline that combines
rule-based checks with LLM-assisted judgment". This module implements that
hybrid judge:

* rule-based concept detection  -- keyword markers from ``carg.knowledge``
* LLM-assisted difficulty rating -- delegated to the LLM backend (a rubric in
  production, a deterministic heuristic in the offline ``MockLLM``)
"""

from typing import List, Optional

from .llm import LLM, MockLLM
from .knowledge import KEYWORDS


class Judge:
    """Rates response difficulty and detects concept coverage."""

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or MockLLM()

    def rate_difficulty(self, response_text: str) -> int:
        return self.llm.rate_difficulty(response_text)

    def concept_explained(self, response_text: str, concept: str) -> bool:
        return self.llm.concept_explained(response_text, concept)

    def first_position(self, response_text: str, concept: str) -> Optional[int]:
        """Earliest character offset at which any keyword of ``concept`` appears.

        Returns ``None`` if the concept is not mentioned. Used to verify
        prerequisite ordering (dependency correctness).
        """
        text = response_text.lower()
        positions = []
        for kw in KEYWORDS.get(concept, [concept]):
            idx = text.find(kw)
            if idx != -1:
                positions.append(idx)
        return min(positions) if positions else None

    def is_ordered(self, response_text: str, before: str, after: str) -> bool:
        """Whether ``before`` is explained before ``after`` in the response."""
        pos_before = self.first_position(response_text, before)
        pos_after = self.first_position(response_text, after)
        if pos_before is None or pos_after is None:
            return False
        return pos_before < pos_after
