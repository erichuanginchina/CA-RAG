"""Shared data types for the CA-RAG framework."""

from dataclasses import dataclass, field
from typing import List, Tuple

from .config import INTERMEDIATE


@dataclass
class Response:
    """A generated response plus audit metadata.

    ``text`` is what is shown to the user and what the judge scores. The other
    fields are audit metadata recorded by the generator; the metrics always
    re-derive their inputs from ``text`` via the judge, never from this
    metadata, so evaluation stays faithful to the generated content.
    """

    text: str
    target_concept: str = ""
    intended_level: int = INTERMEDIATE
    covered_concepts: List[str] = field(default_factory=list)
    ordered_pairs: List[Tuple[str, str]] = field(default_factory=list)
    addressed_gaps: List[str] = field(default_factory=list)


@dataclass
class Sample:
    """A single evaluation sample.

    Represents one ``(user_state, query, response)`` triple described in
    Section 4.3, together with the ground-truth cognitive state and the
    task-required prerequisite set.
    """

    query: str
    target_concept: str
    user_level: int
    #: The user's true Knowledge Vector for this sample.
    kv: "object"
    #: Concepts that the task requires the user to understand (prerequisites
    #: of the target concept).
    required_prereqs: List[str] = field(default_factory=list)
    #: Concepts the user is missing (mastery below the missing threshold).
    missing_concepts: List[str] = field(default_factory=list)
