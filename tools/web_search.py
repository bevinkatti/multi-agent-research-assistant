# ============================================================
# tools/web_search.py
#
# Tavily-backed web search tool with:
# - Async execution (non-blocking for FastAPI)
# - Exponential backoff retry (max 3 attempts)
# - Typed result schema (SearchResult, SearchResponse)
# - Full structured logging with latency
# - Clean separation from agent layer
# ============================================================

import asyncio
from typing import Optional
from dataclasses import dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from tavily import TavilyClient, MissingAPIKeyError
import logging

from config import settings
from logger import get_logger, AgentCallLogger

logger = get_logger(__name__)


# ── Typed Result Schema ────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result from Tavily."""
    title: str
    url: str
    content: str                        # Snippet / summary of the page
    score: float = 0.0                  # Tavily relevance score (0–1)
    raw_content: Optional[str] = None   # Full page content if requested


@dataclass
class SearchResponse:
    """Normalized response returned to the agent layer."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    answer: Optional[str] = None        # Tavily's synthesized answer (if available)
    success: bool = True
    error: Optional[str] = None

    def to_context_string(self) -> str:
        """
        Formats results into a single string for LLM context injection.
        Each result is numbered with title, URL, and content snippet.
        """
        if not self.success:
            return f"Search failed: {self.error}"

        parts = []
        if self.answer:
            parts.append(f"Quick Answer: {self.answer}\n")

        for i, r in enumerate(self.results, 1):
            parts.append(
                f"[{i}] {r.title}\n"
                f"    URL: {r.url}\n"
                f"    {r.content}\n"
            )

        return "\n".join(parts) if parts else "No results found."


# ── Core Search Tool ───────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Async wrapper around the Tavily search API.

    Usage:
        tool = WebSearchTool()
        response = await tool.search("latest LLM benchmarks 2025")
        context = response.to_context_string()
    """

    def __init__(self):
        if not settings.tavily_api_key:
            raise MissingAPIKeyError(
                "TAVILY_API_KEY is not set in your .env file. "
                "Get a free key at https://app.tavily.com"
            )
        # TavilyClient is synchronous — we'll run it in a thread pool
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        logger.info("web_search_tool_initialized", model="tavily")

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            multiplier=settings.retry_base_delay,
            min=1,
            max=10,
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def _search_sync(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        include_answer: bool,
    ) -> dict:
        """
        Synchronous Tavily call — wrapped by tenacity for retry logic.
        Runs inside asyncio.to_thread() to avoid blocking the event loop.

        WHY SYNC HERE:
        Tavily's Python SDK is synchronous. Rather than fighting it,
        we offload it to a thread pool via asyncio.to_thread(). This is
        the correct pattern for blocking I/O in async applications.
        """
        return self._client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,      # "basic" (fast) or "advanced" (thorough)
            include_answer=include_answer,  # Tavily's AI-synthesized answer
            include_raw_content=False,      # Raw HTML — skip for speed
        )

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> SearchResponse:
        """
        Async search — safe to call from FastAPI endpoints and LangGraph nodes.

        Args:
            query:          The search query string
            max_results:    Number of results to return (1–10)
            search_depth:   "basic" for speed, "advanced" for thoroughness
            include_answer: Whether to include Tavily's synthesized answer

        Returns:
            SearchResponse with typed results and context-ready string method
        """
        with AgentCallLogger(logger, "WebSearchTool", query):
            try:
                # Run blocking SDK call in thread pool — non-blocking for event loop
                raw = await asyncio.to_thread(
                    self._search_sync,
                    query,
                    max_results,
                    search_depth,
                    include_answer,
                )
                return self._parse_response(query, raw)

            except MissingAPIKeyError as e:
                logger.error("tavily_api_key_missing", error=str(e))
                return SearchResponse(
                    query=query,
                    success=False,
                    error="Tavily API key missing or invalid.",
                )
            except Exception as e:
                logger.error(
                    "web_search_failed",
                    query=query,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return SearchResponse(
                    query=query,
                    success=False,
                    error=f"Search failed after {settings.max_retries} retries: {str(e)}",
                )

    def _parse_response(self, query: str, raw: dict) -> SearchResponse:
        """
        Normalizes raw Tavily API response into our typed SearchResponse.
        Defensive: handles missing fields gracefully instead of crashing.
        """
        results = []
        for item in raw.get("results", []):
            results.append(SearchResult(
                title=item.get("title", "No title"),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
                raw_content=item.get("raw_content"),
            ))

        # Sort by Tavily relevance score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return SearchResponse(
            query=query,
            results=results,
            answer=raw.get("answer"),       # None if include_answer=False
            success=True,
            error=None,
        )

    async def search_multiple(
        self,
        queries: list[str],
        max_results: int = 3,
    ) -> list[SearchResponse]:
        """
        Run multiple searches concurrently using asyncio.gather().
        Used by the orchestrator when it needs to fan out searches.

        WHY THIS MATTERS:
        Sequential searches for 3 queries = 3x latency.
        Concurrent searches = ~1x latency. This is a real performance win
        that interviewers notice when you mention it.
        """
        tasks = [
            self.search(query, max_results=max_results)
            for query in queries
        ]
        return await asyncio.gather(*tasks)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Instantiated once and reused — avoids re-reading API key on every call.
# Import with: from tools.web_search import web_search_tool

def get_web_search_tool() -> WebSearchTool:
    """
    Factory function for WebSearchTool.
    Use this in FastAPI dependency injection.
    """
    return WebSearchTool()