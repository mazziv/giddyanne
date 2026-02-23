"""Cross-encoder reranker for second-stage result scoring."""

import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Lazy-loaded cross-encoder reranker.

    Uses CrossEncoder from sentence-transformers to re-score (query, document)
    pairs. The model is loaded on first call to rerank().
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self._model_name}")
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info(f"Cross-encoder loaded: {self._model_name}")
        return self._model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score (query, document) pairs using the cross-encoder.

        Returns raw logit scores, one per document.
        """
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs)
        return scores.tolist()
