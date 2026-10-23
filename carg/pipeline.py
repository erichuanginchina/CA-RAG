"""The CA-RAG pipeline: Cognitive Modeling -> Knowledge Gap Detection ->
Adaptive Generation (Section 3.1, Figure 1).

The pipeline orchestrates the three modules and exposes ``answer`` for both the
interactive demo and the evaluation experiments. Each module can be disabled
independently, which is exactly what the Experiment II ablation study does:

* ``cm``  -- Cognitive Modeling: estimates the user's level from the KV.
* ``kgd`` -- Knowledge Gap Detection: detects missing prerequisites.
* ``ag``  -- Adaptive Generation: renders the hierarchical, level-adapted
             response (prerequisite -> core -> advanced).

Disabling ``cm`` falls back to a static (intermediate) level estimate;
disabling ``kgd`` produces no prerequisite explanations; disabling ``ag`` falls
back to a generic single-passage answer.
"""

from typing import List, Optional

from .adaptive_generation import AdaptiveGenerator
from .cognitive_model import CognitiveModel
from .config import INTERMEDIATE, MISSING_THRESHOLD
from .gap_detection import KnowledgeGapDetector
from .knowledge import prerequisites_of
from .types import Response, Sample
from .vectorstore import KnowledgeBase


class CARAGPipeline:
    """End-to-end Cognitive-Adaptive RAG pipeline."""

    def __init__(self, kb: KnowledgeBase,
                 cm: Optional[CognitiveModel] = None,
                 kgd: Optional[KnowledgeGapDetector] = None,
                 ag: Optional[AdaptiveGenerator] = None):
        self.kb = kb
        self.cm = cm or CognitiveModel()
        self.kgd = kgd or KnowledgeGapDetector(missing_threshold=MISSING_THRESHOLD)
        self.ag = ag or AdaptiveGenerator(kb)

    # -- interactive entry point ------------------------------------------- #
    def answer(self, query: str, kv, target_concept: Optional[str] = None,
               cm: bool = True, kgd: bool = True, ag: bool = True) -> Response:
        """Answer ``query`` for a user whose Knowledge Vector is ``kv``.

        ``target_concept`` is optional; when omitted the pipeline asks the
        Knowledge Base's retriever to infer the most relevant concept from the
        query. The boolean flags implement the ablation switches.
        """
        target = target_concept or self._infer_target(query)
        return self._answer(target, kv, cm=cm, kgd=kgd, ag=ag)

    def answer_sample(self, sample: Sample, cm: bool = True, kgd: bool = True,
                      ag: bool = True) -> Response:
        """Answer an evaluation ``Sample`` (which already names the concept)."""
        return self._answer(sample.target_concept, sample.kv, cm=cm, kgd=kgd, ag=ag)

    # -- internal ----------------------------------------------------------- #
    def _answer(self, target_concept: str, kv, cm: bool, kgd: bool,
                ag: bool) -> Response:
        # Cognitive Modeling: estimate the user's level.
        if cm:
            level = self.cm.infer_level(kv)
        else:
            level = INTERMEDIATE

        # Knowledge Gap Detection: identify missing prerequisites.
        if kgd:
            missing_prereqs = self._missing_prereqs(target_concept, kv)
            instruction = self.kgd.build_instruction(kv, target_concept, level)
        else:
            missing_prereqs = []
            instruction = None

        # Adaptive Generation: render the hierarchical response.
        if ag:
            response = self.ag.generate(target_concept, level, missing_prereqs, kv)
        else:
            # Standard (non-adaptive) generation: a single unadapted passage.
            text = self.kb.passages_for_concept(target_concept, INTERMEDIATE).text
            response = Response(
                text=text,
                target_concept=target_concept,
                intended_level=INTERMEDIATE,
                covered_concepts=[target_concept],
            )

        response.addressed_gaps = missing_prereqs
        return response

    def _missing_prereqs(self, target_concept: str, kv) -> List[str]:
        missing = set(self.kgd.missing_concepts(kv))
        # Keep the prerequisite graph's ordering so foundational concepts are
        # explained before their dependents.
        closure = prerequisites_of(target_concept, closure=True)
        return [c for c in closure if c in missing]

    def _infer_target(self, query: str) -> str:
        """Infer the target concept for an unlabelled query via retrieval."""
        passages = self.kb.retrieve(query, k=1)
        if passages:
            return passages[0].concept
        return self.kb.concepts[0]

    def run_dialogue_turn(self, kv, query: str, target_concept: str,
                          signals) -> Response:
        """Update the KV from the turn's signals and then answer.

        This is the online feedback loop of Section 3.1: user interactions
        update the KV, which guides the next response.
        """
        self.cm.update(kv, signals)
        return self.answer(query, kv, target_concept=target_concept)
