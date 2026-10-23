"""Cognition-oriented evaluation metrics (Section 4.3).

Three metrics are implemented, computed at the sample level and then averaged
over the test set:

* CAS -- Cognitive Alignment Score
* PCS -- Prerequisite Coverage Score (three sub-metrics)
* KGR -- Knowledge Gap Reduction

None of these rely on lexical-overlap metrics such as BLEU/ROUGE.
"""

from typing import List, Optional, Set

from .config import MAX_LEVEL_DIFF
from .judge import Judge


def cognitive_alignment_score(user_level: int, response_difficulty: int) -> float:
    """CAS = 1 - |user_level - response_difficulty| / max_level_diff."""
    return 1.0 - abs(user_level - response_difficulty) / MAX_LEVEL_DIFF


def _coverage(covered: Set[str], required: Set[str]) -> float:
    if not required:
        return 1.0
    return len(covered & required) / len(required)


def prerequisite_coverage_score(response_text: str, target_concept: str,
                                required_prereqs: List[str],
                                missing_concepts: List[str],
                                judge: Judge) -> float:
    """PCS, decomposed into three sub-metrics (Section 4.3.2).

    1. Prerequisite Coverage: what fraction of the *required-and-missing*
       prerequisites are explicitly explained.
    2. Dependency Correctness: what fraction of prerequisite->target dependency
       pairs are explained in the correct order.
    3. Missing Prerequisite Detection: what fraction of the user's missing
       prerequisites are actively identified and addressed.

    The sample-level PCS is the arithmetic mean of the three sub-metrics.
    """
    # 1. Prerequisite coverage ------------------------------------------------
    # P_required = task-required prerequisites ∩ user-missing concepts.
    p_required = set(required_prereqs) & set(missing_concepts)
    p_covered = {c for c in p_required if judge.concept_explained(response_text, c)}
    coverage = _coverage(p_covered, p_required)

    # 2. Dependency correctness ----------------------------------------------
    pairs = [(p, target_concept) for p in required_prereqs]
    if pairs:
        correct = sum(1 for a, b in pairs if judge.is_ordered(response_text, a, b))
        dependency = correct / len(pairs)
    else:
        dependency = 1.0

    # 3. Missing prerequisite detection --------------------------------------
    m_missing = set(missing_concepts)
    m_detected = {c for c in m_missing if judge.concept_explained(response_text, c)}
    detection = _coverage(m_detected, m_missing)

    return (coverage + dependency + detection) / 3.0


def knowledge_gap_reduction(gap_pre: float, gap_post: float) -> Optional[float]:
    """KGR = (g_pre - g_post) / g_pre on the target dimension.

    Returns ``None`` when ``g_pre == 0`` (the sample is excluded from
    aggregation, as specified in Section 4.3.3).
    """
    if gap_pre <= 0:
        return None
    return (gap_pre - gap_post) / gap_pre


def mean(values: List[float]) -> float:
    """Arithmetic mean of a (possibly empty) list; 0.0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)
