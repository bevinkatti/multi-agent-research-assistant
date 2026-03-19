# ============================================================
# agents/reader_agent.py
#
# Reader Agent — RAG (Retrieval Augmented Generation) node that:
# 1. Ingests documents from PDF paths or URLs into FAISS
# 2. Retrieves semantically relevant chunks for a query
# 3. Uses Groq LLM to answer strictly from retrieved context
# 4. Returns structured AgentResult to the orchestrator
#
# Design principles:
# - Stateless: reads/writes LangGraph ResearchState only
# - Persistent: FAISS index survives across sessions
# - Grounded: LLM is explicitly instructed not to hallucinate
# - Retries: exponential backoff on LLM synthesis calls
# ============================================================

import asyncio
import time
from typing import Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from config import settings
from logger import get_logger, AgentCallLogger
from tools.pdf_loader import PDFLoaderTool
from tools.vector_store import VectorStore, SearchResponse as VectorSearchResponse
from agents.search_agent import (
    AgentResult,
    AgentStatus,
    ResearchState,
    get_llm,
)

logger = get_logger(__name__)


class ReaderAgent:
    """
    RAG Agent: ingest documents → retrieve relevant chunks → synthesize answer.

    Two modes of operation:
    1. INGEST mode:  Call ingest(sources) to load documents into FAISS
    2. QUERY mode:   Call run(state) as a LangGraph node to answer questions

    The vector store is shared and persistent — documents ingested in one
    session are available in the next (loaded from disk automatically).

    WHY RAG over pure LLM (interview answer):
    "A plain LLM answers from training data which may be stale or wrong
    for domain-specific content. RAG grounds the answer in the actual
    document, making it verifiable and reducing hallucination. The
    retrieved chunks also serve as evidence for the Critic Agent."
    """

    SYSTEM_PROMPT = """You are a precise document analyst. Your job is to answer questions using ONLY the provided document excerpts.

Rules:
- Answer ONLY from the provided context. Never use outside knowledge.
- If the context does not contain enough information, say exactly: "The provided documents do not contain sufficient information to answer this question."
- Cite sources inline as [Doc 1], [Doc 2], etc.
- Be concise but complete. Aim for 150-250 words.
- Do not speculate or infer beyond what is explicitly stated.
- End with a "Referenced chunks:" section listing chunk IDs used."""

    NO_DOCS_RESPONSE = (
        "No documents have been ingested yet. "
        "Please provide PDF paths or URLs to ingest first."
    )

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Args:
            vector_store: Optional shared VectorStore instance.
                         If None, creates its own (loads from disk if exists).
                         Pass a shared instance when using with orchestrator
                         to avoid loading the embedding model twice.
        """
        self._loader = PDFLoaderTool()
        self._store = vector_store or VectorStore()
        self._llm = get_llm(temperature=0.0)   # Zero temp — max factual grounding

        logger.info(
            "reader_agent_initialized",
            provider=settings.default_llm_provider,
            model=settings.get_llm_model_name(),
            index_chunks=self._store.get_stats()["total_chunks"],
        )

    # ── Public: Ingest documents ───────────────────────────────────────────────

    async def ingest(self, sources: list[str]) -> dict:
        """
        Load documents from file paths or URLs into the FAISS index.
        Can be called before or during a research session.

        Args:
            sources: List of PDF file paths or HTTP/HTTPS URLs

        Returns:
            Summary dict with per-source results and total chunks added
        """
        logger.info("reader_agent_ingesting", source_count=len(sources))

        # Load all documents concurrently
        load_results = await self._loader.load_multiple(sources)

        total_added = 0
        summary = []

        for result in load_results:
            if not result.success:
                logger.warning(
                    "ingest_source_failed",
                    source=result.source,
                    error=result.error,
                )
                summary.append({
                    "source": result.source,
                    "success": False,
                    "error": result.error,
                    "chunks_added": 0,
                })
                continue

            added = await self._store.add_chunks(result.chunks)
            total_added += added
            summary.append({
                "source": result.source,
                "success": True,
                "chunks_loaded": result.chunk_count,
                "chunks_added": added,  # May be less if duplicates skipped
            })

        # Persist updated index to disk
        await self._store.save()

        ingest_result = {
            "total_sources": len(sources),
            "successful_sources": sum(1 for s in summary if s["success"]),
            "total_chunks_added": total_added,
            "index_size": self._store.get_stats()["total_chunks"],
            "details": summary,
        }

        logger.info(
            "ingest_complete",
            total_added=total_added,
            index_size=ingest_result["index_size"],
        )
        return ingest_result

    # ── LangGraph Node Entry Point ─────────────────────────────────────────────

    async def run(self, state: ResearchState) -> ResearchState:
        """
        LangGraph node function.

        Reads:  state["query"], state["sources"] (for context)
        Writes: state["reader_result"], state["errors"]
        """
        query = state.get("query", "")
        if not query:
            return {
                **state,
                "reader_result": AgentResult(
                    agent_name="ReaderAgent",
                    status=AgentStatus.FAILED,
                    output="",
                    error="Empty query provided",
                ),
            }

        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._execute(query),
                timeout=settings.agent_timeout_seconds,
            )
        except asyncio.TimeoutError:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "reader_agent_timeout",
                query=query,
                timeout=settings.agent_timeout_seconds,
            )
            result = AgentResult(
                agent_name="ReaderAgent",
                status=AgentStatus.TIMEOUT,
                output="",
                error=f"Reader timed out after {settings.agent_timeout_seconds}s",
                latency_ms=latency_ms,
            )

        current_errors = state.get("errors", [])
        if result.error:
            current_errors = current_errors + [f"ReaderAgent: {result.error}"]

        return {
            **state,
            "reader_result": result,
            "errors": current_errors,
        }

    # ── Core Execution ─────────────────────────────────────────────────────────

    async def _execute(self, query: str) -> AgentResult:
        """Retrieve relevant chunks → synthesize grounded answer."""
        start = time.perf_counter()

        with AgentCallLogger(logger, "ReaderAgent", query) as call_log:

            # Check if index has any documents
            stats = self._store.get_stats()
            if stats["total_chunks"] == 0:
                return AgentResult(
                    agent_name="ReaderAgent",
                    status=AgentStatus.FAILED,
                    output=self.NO_DOCS_RESPONSE,
                    error="No documents ingested",
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

            # Step 1: Semantic retrieval from FAISS
            retrieval: VectorSearchResponse = await self._store.search(
                query=query,
                k=5,
                min_score=0.3,   # Filter out low-relevance chunks
            )

            if not retrieval.success:
                return AgentResult(
                    agent_name="ReaderAgent",
                    status=AgentStatus.FAILED,
                    output="",
                    error=f"Retrieval failed: {retrieval.error}",
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

            if not retrieval.results:
                return AgentResult(
                    agent_name="ReaderAgent",
                    status=AgentStatus.SUCCESS,
                    output=(
                        "No relevant content found in the ingested documents "
                        f"for query: '{query}'. Try ingesting more relevant sources."
                    ),
                    raw_data=retrieval,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    metadata={"chunks_retrieved": 0},
                )

            # Step 2: LLM synthesis over retrieved chunks
            synthesized = await self._synthesize(query, retrieval)

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            call_log.set_output(synthesized[:200])

            return AgentResult(
                agent_name="ReaderAgent",
                status=AgentStatus.SUCCESS,
                output=synthesized,
                raw_data=retrieval,
                latency_ms=latency_ms,
                metadata={
                    "chunks_retrieved": len(retrieval.results),
                    "top_score": retrieval.results[0].score if retrieval.results else 0,
                    "sources_used": list({r.source for r in retrieval.results}),
                    "index_size": stats["total_chunks"],
                },
            )

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.retry_base_delay,
            min=1,
            max=10,
        ),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _synthesize(
        self,
        query: str,
        retrieval: VectorSearchResponse,
    ) -> str:
        """
        Synthesize a grounded answer from retrieved chunks.
        Temperature=0 ensures maximum factual consistency.
        """
        # Build numbered context from retrieved chunks
        context_parts = []
        for i, result in enumerate(retrieval.results, 1):
            context_parts.append(
                f"[Doc {i}] Source: {result.source}\n"
                f"Chunk ID: {result.chunk_id} | Similarity: {result.score:.3f}\n"
                f"{result.text.strip()}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""Question: {query}

Document Excerpts:
{context}

Answer the question using ONLY the above excerpts."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self._llm.ainvoke(messages)
        return response.content.strip()

    # ── Convenience methods for testing ───────────────────────────────────────

    async def ask(self, query: str) -> AgentResult:
        """Direct query without LangGraph state. For testing."""
        state: ResearchState = {"query": query, "sources": [], "errors": []}
        new_state = await self.run(state)
        return new_state["reader_result"]

    def get_index_stats(self) -> dict:
        """Returns current FAISS index statistics."""
        return self._store.get_stats()