"""Cognitive Modeling (CM).

The CM module maintains a user's dynamic Knowledge Vector (KV) -- a mapping
from domain concepts to mastery scores in [0, 1] -- and updates it online from
interaction signals, exactly as described in Section 3.3.1 of the paper.

Signals (Section 3.3.1):
  * clarification requests  -- may indicate insufficient understanding
  * excessive dwell time    -- may suggest cognitive difficulty
  * skipping foundational   -- may indicate existing knowledge / familiarity

Simplified update rule (Section 3.3.1):
  k_i' = clip(k_i + alpha * P_i - beta * Q_i - gamma * D_i, 0, 1)

where ``P_i`` is a successful-understanding signal, ``Q_i`` a clarification
request, and ``D_i`` an excessive-dwell indicator for concept ``i``.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from .config import ALPHA, BETA, GAMMA, DWELL_THRESHOLD
from .knowledge import CONCEPTS


@dataclass
class InteractionSignals:
    """Interaction signals extracted during a single dialogue turn."""

    #: The concept(s) the current turn is primarily about.
    concepts: Iterable[str] = field(default_factory=list)

    #: Number of clarification requests the user made this turn.
    clarification_requests: int = 0

    #: Dwell time in seconds on the concept explanation.
    dwell_time: float = 0.0

    #: Whether the user skipped the foundational explanation.
    skip_foundational: bool = False

    #: Whether the user showed successful understanding (e.g. answered a
    #: follow-up correctly or proceeded without clarification).
    success: bool = False

    def excessive_dwell(self, threshold: float = DWELL_THRESHOLD) -> bool:
        return self.dwell_time > threshold


class KnowledgeVector:
    """A dynamic knowledge vector mapping concept -> mastery score in [0, 1]."""

    def __init__(self, values: Optional[Dict[str, float]] = None,
                 default: float = 0.5):
        self._kv: Dict[str, float] = {}
        for concept in CONCEPTS:
            if values is not None and concept in values:
                self._kv[concept] = self._clip(values[concept])
            else:
                self._kv[concept] = self._clip(default)

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    # -- container protocol ------------------------------------------------- #
    def __getitem__(self, concept: str) -> float:
        return self._kv[concept]

    def __setitem__(self, concept: str, value: float) -> None:
        self._kv[concept] = self._clip(value)

    def __iter__(self):
        return iter(self._kv)

    def __len__(self) -> int:
        return len(self._kv)

    def items(self):
        return self._kv.items()

    def keys(self):
        return self._kv.keys()

    def to_dict(self) -> Dict[str, float]:
        return dict(self._kv)

    def copy(self) -> "KnowledgeVector":
        return KnowledgeVector(self.to_dict())

    # -- helpers ------------------------------------------------------------ #
    def missing_concepts(self, threshold: float) -> Dict[str, float]:
        """Concepts whose mastery falls below ``threshold`` (their gap)."""
        return {c: 1.0 - k for c, k in self._kv.items() if k < threshold}

    def mean_mastery(self) -> float:
        return sum(self._kv.values()) / len(self._kv)

    def __repr__(self) -> str:
        inner = ", ".join(f"{c}={k:.2f}" for c, k in self._kv.items())
        return f"KnowledgeVector({{{inner}}})"


class CognitiveModel:
    """Estimates and updates a user's Knowledge Vector from interaction signals.

    This is the CM component. It is stateless apart from the KV it is handed,
    so the same instance can model any number of users.
    """

    def __init__(self, alpha: float = ALPHA, beta: float = BETA,
                 gamma: float = GAMMA, dwell_threshold: float = DWELL_THRESHOLD):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dwell_threshold = dwell_threshold

    def update(self, kv: KnowledgeVector,
               signals: InteractionSignals) -> KnowledgeVector:
        """Apply one turn's signals to ``kv`` in place and return it.

        The update follows the paper's simplified rule. Clarification requests
        and excessive dwell lower mastery on the affected concepts; skipping
        foundational content and successful understanding raise it.
        """
        concepts = list(signals.concepts) or list(kv.keys())

        # Successful-understanding signal (P_i): a positive indicator for the
        # concepts the user demonstrated understanding of.
        success = 1.0 if signals.success else 0.0
        # Clarification signal (Q_i): frequency of clarification requests.
        clarifications = float(signals.clarification_requests)
        # Excessive-dwell signal (D_i).
        dwell = 1.0 if signals.excessive_dwell(self.dwell_threshold) else 0.0

        for concept in concepts:
            k = kv[concept]
            delta = 0.0

            delta += self.alpha * success
            delta -= self.beta * clarifications
            delta -= self.gamma * dwell

            if signals.skip_foundational:
                # Skipping foundational content is treated as evidence of
                # existing knowledge (Section 3.3.1).
                delta += self.alpha

            kv[concept] = k + delta

        return kv

    def infer_level(self, kv: KnowledgeVector) -> int:
        """Map a Knowledge Vector to a coarse cognitive level.

        Uses mean mastery to place the user in one of the three levels. This
        is used for evaluation when no explicit level label is provided.
        """
        from .config import BEGINNER, INTERMEDIATE, EXPERT

        m = kv.mean_mastery()
        if m < 0.4:
            return BEGINNER
        if m > 0.7:
            return EXPERT
        return INTERMEDIATE
