"""Vector-store abstraction and knowledge base.

The framework retrieves evidence through a ``VectorStore`` interface. Two
backends are provided:

* ``InMemoryVectorStore`` -- a pure-Python retriever that ranks passages by
  lexical overlap (Jaccard similarity over character n-grams). It needs no
  external dependencies and is deterministic, making it suitable for the
  offline simulation experiments and unit tests.
* ``ObjectBoxVectorStore`` -- wraps ``langchain_objectbox.vectorstores.ObjectBox``
  for real dense retrieval. It is constructed lazily so the core package does
  not depend on the optional ObjectBox/LangChain stack.

The ``KnowledgeBase`` bundles the domain corpus (see ``carg.knowledge``) with a
retriever and exposes concept-level lookups used by the adaptive pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import BEGINNER, INTERMEDIATE, EXPERT
from .knowledge import EXPLANATIONS, CONCEPTS


@dataclass
class Passage:
    """A single corpus passage."""

    concept: str
    level: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.concept}:{self.level}"


def _char_ngrams(text: str, n: int = 3) -> set:
    text = " " + text.lower().strip() + " "
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _overlap_score(query: str, passage: str) -> float:
    q = _char_ngrams(query)
    p = _char_ngrams(passage)
    if not q or not p:
        return 0.0
    inter = len(q & p)
    return inter / len(q | p)


class VectorStore:
    """Minimal retriever interface."""

    def add_passages(self, passages: Sequence[Passage]) -> None:
        raise NotImplementedError

    def similarity_search(self, query: str, k: int = 4) -> List[Passage]:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Deterministic lexical-overlap retriever (offline stand-in for dense retrieval)."""

    def __init__(self):
        self._passages: List[Passage] = []

    def add_passages(self, passages: Sequence[Passage]) -> None:
        self._passages.extend(passages)

    def similarity_search(self, query: str, k: int = 4) -> List[Passage]:
        scored = [
            (_overlap_score(query, p.text), p) for p in self._passages
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [p for _, p in scored[:k]]


class ObjectBoxVectorStore(VectorStore):
    """Dense-retrieval backend backed by langchain-objectbox (optional)."""

    def __init__(self, embeddings=None, db_directory: str = "objectbox",
                 embedding_dimensions: int = 768):
        try:
            from langchain_objectbox.vectorstores import ObjectBox  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "langchain-objectbox is not installed. Install it or use "
                "InMemoryVectorStore."
            ) from exc

        self._embeddings = embeddings
        self._db_directory = db_directory
        self._embedding_dimensions = embedding_dimensions
        self._store = None

    def add_passages(self, passages: Sequence[Passage]) -> None:
        from langchain_core.documents import Document  # noqa: WPS433
        docs = [Document(page_content=p.text, metadata={"concept": p.concept,
                                                        "level": p.level})
                for p in passages]
        self._store = None  # rebuild on next search
        self._docs = docs

    def similarity_search(self, query: str, k: int = 4) -> List[Passage]:
        raise NotImplementedError(
            "ObjectBoxVectorStore.similarity_search requires the langchain "
            "ObjectBox retriever; construct it via build()."
        )


class KnowledgeBase:
    """Domain corpus plus a retriever.

    All concept-level passages are indexed for retrieval, and concept lookups
    are exposed for the adaptive generation stage.
    """

    def __init__(self, store: VectorStore):
        self.store = store
        self._passages = self._build_passages()
        self.store.add_passages(self._passages)
        # Index passages by (concept, level) for direct lookups.
        self._by_key: Dict[Tuple[str, int], Passage] = {
            (p.concept, p.level): p for p in self._passages
        }

    @staticmethod
    def _build_passages() -> List[Passage]:
        passages = []
        for concept, levels in EXPLANATIONS.items():
            for level, text in levels.items():
                passages.append(Passage(concept=concept, level=level, text=text))
        return passages

    def passages_for_concept(self, concept: str, level: int) -> Passage:
        return self._by_key[(concept, level)]

    def retrieve(self, query: str, k: int = 4) -> List[Passage]:
        return self.store.similarity_search(query, k=k)

    def all_passages(self) -> List[Passage]:
        return list(self._passages)

    @property
    def concepts(self) -> List[str]:
        return list(CONCEPTS)
