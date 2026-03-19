# ============================================================
# tools/vector_store.py
#
# FAISS vector store with:
# - sentence-transformers embeddings (all-MiniLM-L6-v2)
# - Add chunks, semantic search, persist to disk, load from disk
# - Duplicate prevention via chunk_id tracking
# - Typed result schema (RetrievalResult)
# - Full structured logging
# ============================================================

import asyncio
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from logger import get_logger, AgentCallLogger
from tools.pdf_loader import Chunk

logger = get_logger(__name__)


# ── Typed Schema ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""
    chunk_id: str
    text: str
    source: str
    score: float                        # Cosine similarity (0–1, higher = more relevant)
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return (
            f"RetrievalResult(score={self.score:.3f}, "
            f"source={self.source!r}, chunk={self.chunk_index})"
        )


@dataclass
class SearchResponse:
    """Response returned to the agent layer after retrieval."""
    query: str
    results: list[RetrievalResult] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

    def to_context_string(self) -> str:
        """
        Formats retrieved chunks into a single LLM-ready context string.
        Each result is numbered with source and content.
        """
        if not self.success:
            return f"Retrieval failed: {self.error}"
        if not self.results:
            return "No relevant documents found in the knowledge base."

        parts = []
        for i, r in enumerate(self.results, 1):
            parts.append(
                f"[{i}] Source: {r.source} (similarity: {r.score:.2f})\n"
                f"    {r.text.strip()}\n"
            )
        return "\n".join(parts)


# ── Vector Store ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed vector store for semantic chunk retrieval.

    Architecture:
    - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    - Index type: IndexFlatIP (Inner Product = cosine sim on normalized vectors)
    - Metadata: stored separately in a dict (FAISS only stores vectors)
    - Persistence: index + metadata saved to disk as .faiss + .pkl files

    WHY IndexFlatIP over IndexFlatL2 (interview answer):
    "We L2-normalize all vectors before adding them, so inner product
    equals cosine similarity. Cosine similarity is more appropriate for
    text because it's invariant to vector magnitude — a short chunk and
    a long chunk about the same topic will still score similarly."

    Usage:
        store = VectorStore()
        await store.add_chunks(chunks)
        response = await store.search("transformer attention mechanism", k=5)
        print(response.to_context_string())
    """

    # Files saved to disk
    FAISS_INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.pkl"
    STATS_FILE = "stats.json"

    def __init__(self, index_path: str = settings.faiss_index_path):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Load embedding model (downloads ~90MB on first run)
        logger.info(
            "loading_embedding_model",
            model=settings.embedding_model,
        )
        self._encoder = SentenceTransformer(settings.embedding_model)
        self._dim = self._encoder.get_sentence_embedding_dimension()

        # FAISS index — initialized empty or loaded from disk
        self._index: faiss.IndexFlatIP = None
        self._metadata: list[dict] = []          # Parallel list to FAISS vectors
        self._chunk_ids: set[str] = set()        # For duplicate prevention

        # Try to load existing index from disk
        if self._index_exists():
            self._load_from_disk()
            logger.info(
                "vector_store_loaded",
                chunks=len(self._metadata),
                index_path=str(self.index_path),
            )
        else:
            self._init_empty_index()
            logger.info(
                "vector_store_initialized_empty",
                dim=self._dim,
                index_path=str(self.index_path),
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embeds and adds chunks to the FAISS index.
        Skips duplicates based on chunk_id.
        Returns number of chunks actually added.
        """
        with AgentCallLogger(logger, "VectorStore.add_chunks",
                             f"{len(chunks)} chunks") as call_log:
            new_chunks = [c for c in chunks if c.chunk_id not in self._chunk_ids]

            if not new_chunks:
                logger.info("all_chunks_already_indexed", skipped=len(chunks))
                call_log.set_output("0 new chunks (all duplicates)")
                return 0

            # Embed in thread pool — CPU-bound operation
            texts = [c.text for c in new_chunks]
            embeddings = await asyncio.to_thread(self._embed, texts)

            # Add to FAISS
            self._index.add(embeddings)

            # Store metadata in parallel list
            for chunk in new_chunks:
                self._metadata.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                    "metadata": chunk.metadata,
                })
                self._chunk_ids.add(chunk.chunk_id)

            added = len(new_chunks)
            skipped = len(chunks) - added

            logger.info(
                "chunks_added",
                added=added,
                skipped=skipped,
                total_in_index=self._index.ntotal,
            )
            call_log.set_output(f"Added {added} chunks, skipped {skipped} duplicates")
            return added

    async def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> SearchResponse:
        """
        Semantic search over indexed chunks.

        Args:
            query:      Natural language query
            k:          Number of results to return
            min_score:  Minimum cosine similarity threshold (0–1)

        Returns:
            SearchResponse with ranked RetrievalResult list
        """
        with AgentCallLogger(logger, "VectorStore.search", query) as call_log:
            if self._index.ntotal == 0:
                return SearchResponse(
                    query=query,
                    results=[],
                    success=True,
                    error=None,
                )

            try:
                # Embed query in thread pool
                query_embedding = await asyncio.to_thread(self._embed, [query])

                # FAISS search — returns distances and indices
                actual_k = min(k, self._index.ntotal)
                scores, indices = await asyncio.to_thread(
                    self._index.search, query_embedding, actual_k
                )

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:           # FAISS returns -1 for empty slots
                        continue
                    if float(score) < min_score:
                        continue

                    meta = self._metadata[idx]
                    results.append(RetrievalResult(
                        chunk_id=meta["chunk_id"],
                        text=meta["text"],
                        source=meta["source"],
                        score=float(score),
                        chunk_index=meta["chunk_index"],
                        metadata=meta.get("metadata", {}),
                    ))

                call_log.set_output(f"{len(results)} results, top score={results[0].score:.3f}" if results else "0 results")
                return SearchResponse(query=query, results=results, success=True)

            except Exception as e:
                logger.error("vector_search_failed", query=query, error=str(e))
                return SearchResponse(
                    query=query,
                    success=False,
                    error=f"Search failed: {str(e)}",
                )

    async def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        await asyncio.to_thread(self._save_to_disk)
        logger.info(
            "vector_store_saved",
            path=str(self.index_path),
            total_chunks=len(self._metadata),
        )

    async def search_multiple(
        self,
        queries: list[str],
        k: int = 5,
    ) -> list[SearchResponse]:
        """Run multiple searches concurrently."""
        tasks = [self.search(query, k=k) for query in queries]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict:
        """Returns index statistics for logging and API endpoints."""
        return {
            "total_chunks": self._index.ntotal,
            "unique_sources": len({m["source"] for m in self._metadata}),
            "embedding_dim": self._dim,
            "embedding_model": settings.embedding_model,
            "index_path": str(self.index_path),
        }

    def clear(self) -> None:
        """Reset the index (useful for tests)."""
        self._init_empty_index()
        self._metadata = []
        self._chunk_ids = set()
        logger.info("vector_store_cleared")

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed texts using sentence-transformers.
        Normalizes vectors for cosine similarity via inner product.
        """
        embeddings = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        # L2 normalize — makes inner product == cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)

    def _init_empty_index(self) -> None:
        """Create a fresh empty FAISS IndexFlatIP."""
        self._index = faiss.IndexFlatIP(self._dim)

    def _index_exists(self) -> bool:
        """Check if a persisted index exists on disk."""
        return (
            (self.index_path / self.FAISS_INDEX_FILE).exists()
            and (self.index_path / self.METADATA_FILE).exists()
        )

    def _save_to_disk(self) -> None:
        """Save FAISS index + metadata to disk (synchronous)."""
        faiss.write_index(
            self._index,
            str(self.index_path / self.FAISS_INDEX_FILE),
        )
        with open(self.index_path / self.METADATA_FILE, "wb") as f:
            pickle.dump({
                "metadata": self._metadata,
                "chunk_ids": self._chunk_ids,
            }, f)
        with open(self.index_path / self.STATS_FILE, "w") as f:
            json.dump(self.get_stats(), f, indent=2)

    def _load_from_disk(self) -> None:
        """Load FAISS index + metadata from disk (synchronous)."""
        self._index = faiss.read_index(
            str(self.index_path / self.FAISS_INDEX_FILE)
        )
        with open(self.index_path / self.METADATA_FILE, "rb") as f:
            data = pickle.load(f)
            self._metadata = data["metadata"]
            self._chunk_ids = data["chunk_ids"]