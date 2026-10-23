"""LLM backend abstraction.

The framework talks to a language model through a minimal ``LLM`` interface so
that the cognitive pipeline is independent of any specific provider. Two
implementations are provided:

* ``MockLLM``  -- a deterministic, offline model used for tests and the
                  reproducible simulation experiments. It calls no external
                  API.
* ``GroqLLM``  -- a thin wrapper over ``langchain_groq.ChatGroq``. Importing it
                  requires the optional ``langchain-groq`` dependency and a
                  ``GROQ_API_KEY`` environment variable.

The ``MockLLM`` also doubles as the "LLM-assisted" judge: it rates the
difficulty of a response and checks whether a concept is explained. In
production these judgments would come from prompting a real LLM with a rubric;
here they use deterministic, content-based rules so the experiments are
reproducible.
"""

import re
from typing import Optional

from .config import BEGINNER, INTERMEDIATE, EXPERT
from .knowledge import KEYWORDS, EXPLANATIONS


def _word_bigrams(text: str) -> set:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return set(zip(words, words[1:]))


class LLM:
    """Minimal language-model interface used by the CA-RAG pipeline."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def rate_difficulty(self, response: str) -> int:
        """Rate the difficulty of ``response`` on the 0/1/2 scale."""
        raise NotImplementedError

    def concept_explained(self, response: str, concept: str) -> bool:
        """Whether ``concept`` is explicitly mentioned and explained."""
        raise NotImplementedError


class MockLLM(LLM):
    """Deterministic, offline LLM used for tests and simulation experiments.

    Difficulty rating compares the response against the corpus passages at each
    difficulty level (by word-bigram overlap) and returns the best-matching
    level. This is the deterministic analogue of asking an LLM judge "is this
    explanation beginner, intermediate, or expert level?".
    """

    def generate(self, prompt: str) -> str:
        # MockLLM is not a general-purpose generator; the pipeline composes
        # responses structurally and only uses ``generate`` as a drop-in for
        # final assembly. Returning the prompt keeps it a valid fallback.
        return prompt

    def rate_difficulty(self, response: str) -> int:
        resp_bigrams = _word_bigrams(response)
        if not resp_bigrams:
            return INTERMEDIATE

        scores = {}
        for level in (BEGINNER, INTERMEDIATE, EXPERT):
            reference = set()
            for concept, levels in EXPLANATIONS.items():
                if level in levels:
                    reference |= _word_bigrams(levels[level])
            scores[level] = len(resp_bigrams & reference)

        return max(scores, key=scores.get)

    def concept_explained(self, response: str, concept: str) -> bool:
        keywords = KEYWORDS.get(concept, [concept])
        text = response.lower()
        return any(kw in text for kw in keywords)


class GroqLLM(LLM):
    """Real LLM backend backed by ``langchain_groq.ChatGroq``.

    Requires ``langchain-groq`` and ``GROQ_API_KEY``. Import is deferred so the
    core package stays importable without the dependency.
    """

    def __init__(self, model_name: str = "Llama3-8b-8192",
                 api_key: Optional[str] = None, temperature: float = 0.0):
        try:
            from langchain_groq import ChatGroq  # noqa: WPS433 (optional dep)
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "langchain-groq is not installed. Install it with "
                "`pip install langchain-groq` or use MockLLM."
            ) from exc

        import os
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")

        self._client = ChatGroq(groq_api_key=api_key, model_name=model_name,
                                temperature=temperature)

    def generate(self, prompt: str) -> str:
        result = self._client.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)

    def rate_difficulty(self, response: str) -> int:
        # Fall back to the deterministic heuristic; a real deployment would
        # prompt the model with a difficulty rubric instead.
        return MockLLM().rate_difficulty(response)

    def concept_explained(self, response: str, concept: str) -> bool:
        return MockLLM().concept_explained(response, concept)
