"""Embedding generation module using local sentence-transformers or Ollama."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx


@dataclass
class ChunkEmbedding:
    """Result of embedding a single chunk."""

    path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    description: str
    path_embedding: list[float]
    content_embedding: list[float] | None
    description_embedding: list[float] | None
    mtime: float


@dataclass
class TruncationStats:
    """Token truncation statistics from embedding a batch of texts."""

    total_texts: int
    truncated_texts: int
    total_tokens: int
    embedded_tokens: int  # capped at max_seq per text

if TYPE_CHECKING:
    from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Models that require a prefix on search queries.
# Key: model name suffix (matched against the end of model_name).
# Value: prefix string to prepend to queries.
QUERY_PREFIX_BY_MODEL: dict[str, str] = {
    "CodeRankEmbed": "Represent this query for searching relevant code: ",
    "nomic-embed-text-v1.5": "search_query: ",
}

# Models that require a prefix on document content at index time.
DOC_PREFIX_BY_MODEL: dict[str, str] = {
    "nomic-embed-text-v1.5": "search_document: ",
}

# Models that need trust_remote_code=True when loading.
TRUST_REMOTE_CODE_MODELS: set[str] = {
    "CodeRankEmbed",
    "nomic-embed-text-v1.5",
}


def _prefix_for(model_name: str, table: dict[str, str]) -> str:
    """Return prefix for a model from the given lookup table, or empty string."""
    for suffix, prefix in table.items():
        if model_name.endswith(suffix):
            return prefix
    return ""


def _needs_trust_remote_code(model_name: str) -> bool:
    """Check if a model needs trust_remote_code=True."""
    return any(model_name.endswith(s) for s in TRUST_REMOTE_CODE_MODELS)



class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    def query_prefix(self) -> str:
        """Prefix to prepend to search queries (empty for most models)."""
        return ""

    @property
    def doc_prefix(self) -> str:
        """Prefix to prepend to documents at index time (empty for most models)."""
        return ""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def max_seq_length(self) -> int:
        """Return maximum token sequence length for this model."""
        ...

    @abstractmethod
    def token_count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...

    @abstractmethod
    def count_truncated(self, texts: list[str]) -> TruncationStats:
        """Count how many texts exceed the model's token limit."""
        ...


class LocalEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._query_prefix = _prefix_for(model_name, QUERY_PREFIX_BY_MODEL)
        self._doc_prefix = _prefix_for(model_name, DOC_PREFIX_BY_MODEL)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def doc_prefix(self) -> str:
        return self._doc_prefix

    @property
    def max_seq_length(self) -> int:
        return self._load_model().max_seq_length

    def token_count(self, text: str) -> int:
        model = self._load_model()
        encoded = model.tokenizer(text, padding=False, truncation=False)
        return len(encoded["input_ids"])

    def _load_model(self):
        if self._model is None:
            logger.debug(f"Loading SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if _needs_trust_remote_code(self.model_name):
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
            logger.debug(f"Model loaded: {self.model_name}")
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    def count_truncated(self, texts: list[str]) -> TruncationStats:
        model = self._load_model()
        max_seq = model.max_seq_length
        encoded = model.tokenizer(texts, padding=False, truncation=False)
        total_tokens = 0
        embedded_tokens = 0
        truncated = 0
        for ids in encoded["input_ids"]:
            n = len(ids)
            total_tokens += n
            embedded_tokens += min(n, max_seq)
            if n > max_seq:
                truncated += 1
        return TruncationStats(
            total_texts=len(texts),
            truncated_texts=truncated,
            total_tokens=total_tokens,
            embedded_tokens=embedded_tokens,
        )


class OllamaEmbedding(EmbeddingProvider):
    """Embedding via Ollama's /api/embed endpoint (GPU-accelerated)."""

    def __init__(
        self,
        model_name: str = "all-minilm:l6-v2",
        base_url: str = "http://localhost:11434",
    ):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)
        self._dimension: int | None = None
        self._context_length: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_seq_length(self) -> int:
        return self._context_length or 512

    def token_count(self, text: str) -> int:
        return int(len(text) / 2.5)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Ollama rejects inputs exceeding the model's context window and also
        # limits total tokens across all inputs in a single /api/embed call.
        # Send one text per request — Ollama batches internally on the GPU,
        # so per-request overhead is minimal.
        max_chars = (self._context_length or 512) * 2
        all_embeddings: list[list[float]] = []
        for text in texts:
            truncated = text[:max_chars] if len(text) > max_chars else text
            resp = await self._client.post(
                "/api/embed",
                json={"model": self._model_name, "input": [truncated]},
            )
            if resp.status_code != 200:
                logger.error(
                    f"Ollama embed failed ({resp.status_code}): "
                    f"{resp.text} [{len(truncated)} chars]"
                )
                resp.raise_for_status()
            all_embeddings.append(resp.json()["embeddings"][0])
        return all_embeddings

    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Synchronous probes — called once at startup before the async loop
        with httpx.Client(base_url=self._base_url, timeout=120.0) as client:
            resp = client.post(
                "/api/embed",
                json={"model": self._model_name, "input": ["dimension probe"]},
            )
            resp.raise_for_status()
            self._dimension = len(resp.json()["embeddings"][0])

            # Fetch model context length for truncation
            resp = client.post(
                "/api/show", json={"name": self._model_name},
            )
            if resp.status_code == 200:
                info = resp.json().get("model_info", {})
                ctx = info.get("bert.context_length")
                if ctx:
                    self._context_length = int(ctx)
                    logger.debug(
                        f"Ollama model context length: {self._context_length}"
                    )

        return self._dimension

    def count_truncated(self, texts: list[str]) -> TruncationStats:
        # Approximate tokens from chars (~2.5 chars/token for code)
        chars_per_token = 2.5
        max_tokens = self._context_length or 512
        total_tokens = 0
        embedded_tokens = 0
        truncated = 0
        for text in texts:
            n = int(len(text) / chars_per_token)
            total_tokens += n
            embedded_tokens += min(n, max_tokens)
            if n > max_tokens:
                truncated += 1
        return TruncationStats(
            total_texts=len(texts),
            truncated_texts=truncated,
            total_tokens=total_tokens,
            embedded_tokens=embedded_tokens,
        )

    async def check_health(self) -> None:
        """Verify Ollama is reachable and the model is available.

        Raises RuntimeError with actionable messages on failure.
        """
        # Check if Ollama is reachable
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                "Ollama not running. Start it with: ollama serve"
            ) from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self._base_url}: {e}") from None

        # Check if model is available
        models = [m["name"] for m in resp.json().get("models", [])]
        # Ollama may list models with or without ":latest" suffix
        if not any(
            m == self._model_name or m.startswith(self._model_name + ":")
            for m in models
        ):
            raise RuntimeError(
                f"Model {self._model_name} not found. "
                f"Pull it with: ollama pull {self._model_name}"
            ) from None


class SharedEmbedding(EmbeddingProvider):
    """Embedding via the shared embed server over a Unix socket."""

    def __init__(self, socket_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._socket_path = socket_path
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=socket_path),
            timeout=120.0,
        )
        self._dimension: int | None = None
        self._max_seq: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_seq_length(self) -> int:
        if self._max_seq is not None:
            return self._max_seq
        self._fetch_model_info()
        return self._max_seq or 256

    def token_count(self, text: str) -> int:
        # Char-based approximation (same as OllamaEmbedding)
        return int(len(text) / 2.5)

    def _fetch_model_info(self) -> None:
        """Synchronous probe to get dimension and max_seq_length from /models."""
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=self._socket_path), timeout=30.0,
        ) as client:
            resp = client.get("http://localhost/models")
            if resp.status_code == 200:
                data = resp.json()
                model_info = data.get("models", {}).get(self._model_name)
                if model_info:
                    self._dimension = model_info["dimension"]
                    self._max_seq = model_info["max_seq_length"]
                    return

            # Fallback: embed a probe text to get dimension
            resp = client.post(
                "http://localhost/embed",
                json={"texts": ["dimension probe"], "model": self._model_name},
            )
            resp.raise_for_status()
            self._dimension = len(resp.json()["embeddings"][0])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            "http://localhost/embed",
            json={"texts": texts, "model": self._model_name},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        self._fetch_model_info()
        return self._dimension or 384

    def count_truncated(self, texts: list[str]) -> TruncationStats:
        # Char-based approximation (same as OllamaEmbedding)
        chars_per_token = 2.5
        max_tokens = self._max_seq or 256
        total_tokens = 0
        embedded_tokens = 0
        truncated = 0
        for text in texts:
            n = int(len(text) / chars_per_token)
            total_tokens += n
            embedded_tokens += min(n, max_tokens)
            if n > max_tokens:
                truncated += 1
        return TruncationStats(
            total_texts=len(texts),
            truncated_texts=truncated,
            total_tokens=total_tokens,
            embedded_tokens=embedded_tokens,
        )

    async def check_health(self) -> None:
        """Verify the embed server is reachable."""
        try:
            resp = await self._client.get("http://localhost/health")
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Embed server not reachable on {self._socket_path}. "
                "Start it with: giddy embed start"
            ) from None
        except httpx.HTTPError as e:
            raise RuntimeError(f"Embed server error: {e}") from None


class EmbeddingService:
    """High-level service for generating embeddings from files."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    async def embed_file(self, path: str, content: str, description: str = "") -> dict:
        """Generate embeddings for a file or chunk's path, content, and description."""
        prefix = self.provider.doc_prefix
        texts = [prefix + path]
        if content:
            texts.append(prefix + content)
        if description:
            texts.append(prefix + description)

        embeddings = await self.provider.embed(texts)

        result = {
            "path": path,
            "path_embedding": embeddings[0],
            "content_embedding": None,
            "content": content,
            "description": description,
            "description_embedding": None,
        }

        idx = 1
        if content:
            result["content_embedding"] = embeddings[idx]
            idx += 1
        if description:
            result["description_embedding"] = embeddings[idx]

        return result

    async def embed_chunks_batch(
        self,
        chunks: list[tuple[str, int, int, int, str, str, float]],
    ) -> tuple[list[ChunkEmbedding], TruncationStats]:
        """Embed multiple chunks in a single batch call.

        Args:
            chunks: List of tuples (path, chunk_index, start_line, end_line,
                    content, description, mtime).

        Returns:
            Tuple of (ChunkEmbedding results, TruncationStats).

        Groups all texts (paths, contents, descriptions) into one embed() call,
        then distributes results back to chunks.
        """
        if not chunks:
            return [], TruncationStats(0, 0, 0, 0)

        # Build text lists, tracking positions
        texts: list[str] = []
        positions: list[tuple[int, str]] = []  # (chunk_idx, field_type)

        prefix = self.provider.doc_prefix
        for i, (path, _idx, _start, _end, content, desc, _mtime) in enumerate(chunks):
            positions.append((i, "path"))
            texts.append(prefix + path)

            if content:
                positions.append((i, "content"))
                texts.append(prefix + content)

            if desc:
                positions.append((i, "description"))
                texts.append(prefix + desc)

        # Count truncation before embedding
        truncation = self.provider.count_truncated(texts)

        # Single batch embed call
        embeddings = await self.provider.embed(texts)

        # Distribute embeddings back to chunks
        embed_map: dict[int, dict[str, list[float]]] = {i: {} for i in range(len(chunks))}

        for (chunk_idx, field), embedding in zip(positions, embeddings):
            embed_map[chunk_idx][field] = embedding

        results: list[ChunkEmbedding] = []
        for i, (path, idx, start, end, content, desc, mtime) in enumerate(chunks):
            results.append(ChunkEmbedding(
                path=path,
                chunk_index=idx,
                start_line=start,
                end_line=end,
                content=content,
                description=desc,
                path_embedding=embed_map[i]["path"],
                content_embedding=embed_map[i].get("content"),
                description_embedding=embed_map[i].get("description"),
                mtime=mtime,
            ))

        return results, truncation

    async def embed_query(
        self, query: str, cache: VectorStore | None = None
    ) -> tuple[list[float], bool]:
        """Generate embedding for a search query.

        Applies the model's query prefix if one is configured (e.g. CodeRankEmbed
        requires a specific instruction prefix on queries but not on documents).

        Returns:
            Tuple of (embedding, cache_hit).
        """
        if cache:
            cached = await cache.get_cached_query(query)
            if cached is not None:
                return cached, True

        prefixed = self.provider.query_prefix + query
        embeddings = await self.provider.embed([prefixed])
        embedding = embeddings[0]

        if cache:
            await cache.cache_query(query, embedding)

        return embedding, False
