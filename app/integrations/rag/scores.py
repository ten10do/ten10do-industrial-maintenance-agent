"""Score semantics for upstream RAG retrieval scores.

The agent must never reinterpret a score it did not compute. The RAG system
returns different quantities depending on the retrieval mode, so every exposed
score carries an explicit label and a direction instead of being silently
treated as "relevance".

What the RAG retrieval core actually produces, read from
``light_rag_core`` / ``rag_core`` / ``retrieval.fusion``:

* ``lexical`` mode builds ``evidence_score`` as
  ``0.0 if bm25_score > 0 else 1.0``, a binary relevance flag. The graded value
  is the separate ``lexical_score``.
* ``vector`` mode builds ``evidence_score = 1.0 - cosine_similarity`` in the
  light backend, which is a cosine distance, and ``evidence_score = chroma
  distance`` in the full backend. Both are distances.
* ``hybrid`` mode runs RRF fusion, and the fused candidate inherits
  ``evidence_score`` from whichever source candidate populated it, with the
  vector candidate winning when a chunk appears on both sides. The fused hit's
  ``fusion_score`` (higher is better, RRF) is NOT what the result tuple carries.

Consequences encoded here:

1. Every score the RAG engine exposes to the agent is a lower-is-better
   quantity, so ``higher_is_better`` is ``False`` for all of them.
2. A distance is never converted into a similarity.
3. Because the semantics can differ between hits inside a single hybrid
   response, scores with different labels must not be ranked against each
   other. :func:`comparable` exists so callers can check that explicitly
   instead of assuming.
"""

from collections.abc import Iterable
from dataclasses import dataclass

LEXICAL_RELEVANCE_FLAG = "lexical_relevance_flag"
VECTOR_COSINE_DISTANCE = "vector_cosine_distance"
VECTOR_STORE_DISTANCE = "vector_store_distance"

SUPPORTED_MODES = frozenset({"lexical", "vector", "hybrid"})
SUPPORTED_BACKENDS = frozenset({"light", "full"})

# RRF in the RAG engine ranks with a higher-is-better fused score, but the
# retrieval result tuple carries the inherited ``evidence_score`` instead. The
# agent therefore never sees the fused score and must not order by it.
FUSION_SCORE_EXPOSED = False


@dataclass(frozen=True)
class ScoreSemantics:
    """Label and direction for one score value."""

    semantics: str | None
    higher_is_better: bool | None

    @property
    def annotated(self) -> bool:
        """Return ``True`` when the score's meaning was established."""
        return self.semantics is not None


UNKNOWN = ScoreSemantics(semantics=None, higher_is_better=None)


def vector_distance_name(backend: str) -> str:
    """Return the distance label for the configured RAG backend."""
    return VECTOR_STORE_DISTANCE if backend == "full" else VECTOR_COSINE_DISTANCE


def describe_score(
    retrieval_mode: str | None,
    *,
    has_vector_component: bool,
    backend: str = "light",
) -> ScoreSemantics:
    """Describe the score produced for one retrieved fragment.

    Args:
        retrieval_mode: ``retrieval_mode`` reported by the retrieval result.
        has_vector_component: whether this fragment came from the vector side of
            the retrieval. This matters only in ``hybrid`` mode, where the
            inherited score depends on which side populated the candidate.
        backend: ``light`` or ``full``; selects the vector distance label.

    Returns:
        A :class:`ScoreSemantics`. An unrecognised mode yields the unannotated
        :data:`UNKNOWN` rather than a guessed label.
    """
    if backend not in SUPPORTED_BACKENDS:
        return UNKNOWN

    mode = (retrieval_mode or "").strip().lower()
    if mode not in SUPPORTED_MODES:
        return UNKNOWN

    if mode == "lexical":
        return ScoreSemantics(LEXICAL_RELEVANCE_FLAG, False)
    if mode == "vector":
        return ScoreSemantics(vector_distance_name(backend), False)

    # hybrid: the fused candidate inherits evidence_score from one side.
    if has_vector_component:
        return ScoreSemantics(vector_distance_name(backend), False)
    return ScoreSemantics(LEXICAL_RELEVANCE_FLAG, False)


def has_vector_component(retrieval_source: str | None, vector_rank: int | None) -> bool:
    """Return ``True`` when a fragment came from the vector retriever.

    ``retrieval_source`` is ``lexical``, ``vector`` or ``hybrid``. A fused
    candidate is labelled ``hybrid`` only when it appeared on both sides, so
    ``hybrid`` implies a vector component. That matters because the RAG engine's
    fusion step overwrites the inherited score with the vector candidate's value
    for exactly those fragments.
    """
    if vector_rank is not None:
        return True
    value = (retrieval_source or "").lower()
    return "vector" in value or "hybrid" in value


def comparable(semantics: Iterable[str | None]) -> bool:
    """Return ``True`` when every labelled score shares one meaning.

    Callers that want to rank a result set should check this first. Mixing, for
    example, a cosine distance with a binary lexical flag produces an ordering
    that means nothing.
    """
    labels = {label for label in semantics if label is not None}
    return len(labels) <= 1
