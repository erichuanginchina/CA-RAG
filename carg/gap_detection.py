"""Knowledge Gap Detection (KGD).

The KGD module transforms the Knowledge Vector (KV) into a Knowledge Gap
Vector (KGV) via ``KGV = 1 - KV`` (Section 3.3.2). Larger values indicate
weaker mastery. The weakest concept -- the one with the largest gap -- is
identified as the highest-priority learning need, and a human/machine readable
Constraint Instruction is produced to guide retrieval and generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .cognitive_model import KnowledgeVector
from .knowledge import PREREQUISITES, prerequisites_of


def knowledge_gap_vector(kv: KnowledgeVector) -> Dict[str, float]:
    """Return KGV = 1 - KV for every concept."""
    return {c: 1.0 - k for c, k in kv.items()}


@dataclass
class ConstraintInstruction:
    """A User-Adaptive Instruction constraining downstream generation.

    Mirrors the example in Section 3.3.2 of the paper: a set of weak concepts
    plus a generation policy that the Adaptive Generation stage follows.
    """

    weak_concepts: List[str] = field(default_factory=list)
    policy: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        weak = ", ".join(self.weak_concepts) or "(none)"
        policy = "\n".join(f"- {p}" for p in self.policy) or "- (none)"
        return (
            f"User Profile: weak in {weak}\n"
            f"Generation Policy:\n{policy}"
        )


class KnowledgeGapDetector:
    """Identifies weak concepts and builds a Constraint Instruction."""

    def __init__(self, missing_threshold: float = 0.5):
        self.missing_threshold = missing_threshold

    def gap_vector(self, kv: KnowledgeVector) -> Dict[str, float]:
        return knowledge_gap_vector(kv)

    def weakest_concept(self, kv: KnowledgeVector) -> Optional[str]:
        """Return the concept with the largest gap (lowest mastery)."""
        if len(kv) == 0:
            return None
        return min(kv.to_dict(), key=lambda c: kv[c])

    def missing_concepts(self, kv: KnowledgeVector) -> List[str]:
        """Concepts whose mastery is below the missing threshold, weakest first."""
        gaps = knowledge_gap_vector(kv)
        missing = [c for c in kv.keys() if kv[c] < self.missing_threshold]
        missing.sort(key=lambda c: kv[c])
        return missing

    def build_instruction(self, kv: KnowledgeVector,
                          target_concept: Optional[str] = None,
                          level: Optional[int] = None) -> ConstraintInstruction:
        """Build a Constraint Instruction from the current KV.

        The weak-concept list combines the globally weakest concepts with the
        missing prerequisites of ``target_concept``, so that the instruction
        covers both broad gaps and the specific prerequisites the current query
        depends on.
        """
        from .config import LEVEL_NAMES

        missing = self.missing_concepts(kv)

        weak = set(missing)
        if target_concept is not None:
            for prereq in prerequisites_of(target_concept, closure=True):
                if kv[prereq] < self.missing_threshold:
                    weak.add(prereq)

        weak = sorted(weak, key=lambda c: kv[c])

        policy = self._generation_policy(level)
        return ConstraintInstruction(weak_concepts=weak, policy=policy)

    def _generation_policy(self, level: Optional[int]) -> List[str]:
        """Return the generation policy for a given cognitive level.

        These rules mirror the example policy in Section 3.3.2 (avoid advanced
        notation, explain prerequisites first, use analogies, keep sentences
        short) and are extended per level.
        """
        policy = [
            "explain prerequisite concepts first",
            "use analogies and simple examples",
        ]
        if level is None:
            policy.append("adjust complexity to the user's inferred level")
        elif level == 0:  # Beginner
            policy += [
                "avoid technical terminology and mathematical notation",
                "generate short, concise sentences",
            ]
        elif level == 1:  # Intermediate
            policy += [
                "use plain terminology with brief definitions",
                "limit mathematical abstraction",
            ]
        else:  # Expert
            policy += [
                "include precise mechanisms and supporting detail",
                "permit formal notation and quantitative data",
            ]
        return policy


def constraint_input(query: str, instruction: ConstraintInstruction) -> str:
    """Combine the original query with the adaptive instruction (Section 3.3.2).

    The paper defines the final retrieval input as the user query together with
    the constraint instruction ``P_constraint``.
    """
    return f"{query}\n\n[Constraint Instruction]\n{instruction.to_text()}"
