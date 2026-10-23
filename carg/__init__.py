"""CA-RAG: Cognitive-Adaptive Retrieval-Augmented Generation.

A faithful, dependency-light implementation of the framework described in
"A Cognitively Informed Retrieval-Augmented Generation Framework for
Personalized Education".

The framework integrates three complementary modules:

* Cognitive Modeling (CM)      -- maintains a dynamic Knowledge Vector (KV)
* Knowledge Gap Detection (KGD) -- derives the Knowledge Gap Vector (KGV) and a
                                   Constraint Instruction
* Adaptive Generation (AG)     -- hierarchical prerequisite-aware generation

The core package is pure Python (standard library only) so that the cognitive
model, gap detection, metrics, and experiments run with no third-party
dependencies. LLM and vector-store backends are pluggable: a deterministic
``MockLLM`` / ``InMemoryVectorStore`` are provided for offline use, and real
LangChain / ObjectBox / Groq backends activate when the optional dependencies
are installed.
"""

from .config import (
    BEGINNER,
    INTERMEDIATE,
    EXPERT,
    LEVEL_NAMES,
    MAX_LEVEL_DIFF,
    MISSING_THRESHOLD,
    ADVANCED_READINESS_TAU,
    ALPHA,
    BETA,
    GAMMA,
    DWELL_THRESHOLD,
)
from .cognitive_model import KnowledgeVector, InteractionSignals, CognitiveModel
from .gap_detection import KnowledgeGapDetector, ConstraintInstruction
from .pipeline import CARAGPipeline

__all__ = [
    "BEGINNER",
    "INTERMEDIATE",
    "EXPERT",
    "LEVEL_NAMES",
    "MAX_LEVEL_DIFF",
    "MISSING_THRESHOLD",
    "ADVANCED_READINESS_TAU",
    "ALPHA",
    "BETA",
    "GAMMA",
    "DWELL_THRESHOLD",
    "KnowledgeVector",
    "InteractionSignals",
    "CognitiveModel",
    "KnowledgeGapDetector",
    "ConstraintInstruction",
    "CARAGPipeline",
]

__version__ = "0.1.0"
