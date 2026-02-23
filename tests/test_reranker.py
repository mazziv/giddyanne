"""Tests for reranker module."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.reranker import Reranker


class TestReranker:
    def test_lazy_load(self):
        """Model is not loaded until rerank() is called."""
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
        assert reranker._model is None

    def test_model_name(self):
        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L6-v2")
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L6-v2"

    def test_empty_documents(self):
        """Empty documents returns empty scores without loading model."""
        reranker = Reranker()
        scores = reranker.rerank("query", [])
        assert scores == []
        assert reranker._model is None  # Model not loaded

    @patch("src.reranker.CrossEncoder", create=True)
    def test_rerank_calls_predict(self, _mock_class):
        """rerank() calls CrossEncoder.predict with (query, doc) pairs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])

        reranker = Reranker("test-model")
        reranker._model = mock_model  # Skip lazy load

        docs = ["doc1", "doc2", "doc3"]
        scores = reranker.rerank("my query", docs)

        mock_model.predict.assert_called_once_with([
            ("my query", "doc1"),
            ("my query", "doc2"),
            ("my query", "doc3"),
        ])
        assert scores == [0.9, 0.1, 0.5]

    @patch("src.reranker.CrossEncoder", create=True)
    def test_rerank_single_document(self, _mock_class):
        """Works with a single document."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.75])

        reranker = Reranker()
        reranker._model = mock_model

        scores = reranker.rerank("query", ["single doc"])
        assert scores == [0.75]
