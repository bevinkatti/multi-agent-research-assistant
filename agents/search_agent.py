# ============================================================
# agents/search_agent.py
#
# Search Agent — LangGraph node that:
# 1. Accepts a research query from graph state
# 2. Calls WebSearchTool (Tavily) for real-time results
# 3. Uses LLM (Groq/Anthropic/Google) to synthesize results into a cited answer
# 4. Returns structured AgentResult back to the orchestrator
#
# Design principles:
# - Stateless: reads from and writes to LangGraph state only
# - Multi-provider with fallback: Groq → Gemini
# - Retries: tenacity exponential backoff on LLM calls
# - Timeout-aware: respects AGENT_TIMEOUT_SECONDS from config
# - Fully logged: every call tracked with latency
# ============================================================

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
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
from tools.web_search import WebSearchTool, SearchResponse

logger = get_logger(__name__)


# ── Shared State Schema (used by all agents + orchestrator) ───────────────────

class AgentStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    TIMEOUT   = "timeout"


@dataclass
class AgentResult:
    """
    Standardized result returned by every agent.
    The orchestrator reads this to decide next steps.
    """
    agent_name: str
    status: AgentStatus
    output: str                             # Main text output for LLM context
    raw_data: Any = None                    # Original typed response (SearchResponse etc.)
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ── LangGraph State Schema ─────────────────────────────────────────────────────
# This TypedDict defines the shared state that flows through the graph.
# Every node reads from it and writes back to it.

from typing import TypedDict

class ResearchState(TypedDict, total=False):
    """
    Shared state for the multi-agent research graph.
    total=False means all keys are optional (agents add their own keys).
    """
    query: str                              # Original user query
    search_result: Optional[AgentResult]   # Output from SearchAgent
    reader_result: Optional[AgentResult]   # Output from ReaderAgent
    critic_result: Optional[AgentResult]   # Output from CriticAgent
    final_answer: Optional[str]            # Synthesized final response
    sources: list[str]                     # All URLs/files used
    errors: list[str]                      # Accumulated error messages
    metadata: dict                         # Timestamps, latencies, etc.


# ── LLM Factory ───────────────────────────────────────────────────────────────

class _LLMFallback:
    """Wrapper that tries multiple LLM providers sequentially on every ainvoke call."""

    def __init__(self, provider_factories: list[tuple[str, Any]]):
        self._provider_factories = provider_factories
        self._instances: list[Optional[Any]] = [None] * len(provider_factories)

    async def ainvoke(self, messages: list[Any]) -> Any:
        last_error: Optional[Exception] = None
        for idx, (name, factory) in enumerate(self._provider_factories):
            try:
                if self._instances[idx] is None:
                    self._instances[idx] = factory()
                return await self._instances[idx].ainvoke(messages)
            except Exception as e:
                logger.warning(
                    "%s provider failed during ainvoke: %s. Falling back to next provider.",
                    name,
                    str(e),
                )
                last_error = e
        if last_error is None:
            raise ValueError("No LLM providers were configured.")
        raise last_error


def get_llm(temperature: float = 0.1) -> Any:
    """
    Returns an LLM instance with multi-provider fallback support.
    Tries providers in this order: Groq → Gemini.

    The wrapper catches runtime errors during `ainvoke` and retries the next
    available provider, which handles token limit or availability failures.

    WHY temperature=0.1:
    Research synthesis needs factual consistency, not creativity.
    Low temperature = deterministic, grounded responses.
    """
    provider_factories: list[tuple[str, Any]] = []

    if settings.groq_api_key:
        provider_factories.append(
            (
                "Groq",
                lambda: ChatGroq(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model_name,
                    temperature=temperature,
                    max_tokens=1024,
                ),
            )
        )

    if settings.google_api_key:
        provider_factories.append(
            (
                "Gemini",
                lambda: ChatGoogleGenerativeAI(
                    api_key=settings.google_api_key,
                    model=settings.google_model_name,
                    temperature=temperature,
                    max_tokens=1024,
                ),
            )
        )

    if not provider_factories:
        raise ValueError(
            "No LLM provider available. Set GROQ_API_KEY or GOOGLE_API_KEY in .env"
        )

    return _LLMFallback(provider_factories)


# ── Search Agent ───────────────────────────────────────────────────────────────

class SearchAgent:
    """
    Researches a query using real-time web search + LLM synthesis.

    Flow:
        query → WebSearchTool → raw results → LLM synthesis → AgentResult

    The LLM step is crucial: raw search snippets are noisy and
    disconnected. The LLM synthesizes them into a coherent answer
    while citing sources — much more useful for downstream agents.
    """

    SYSTEM_PROMPT = """You are a precise research assistant. Your job is to synthesize web search results into a clear, factual answer.

Rules:
- Answer ONLY based on the provided search results. Do not use prior knowledge.
- Cite sources inline using [1], [2], etc. matching the numbered results.
- If results are insufficient, say so clearly.
- Be concise but complete. Aim for 150-300 words.
- Structure: direct answer first, then supporting details.
- End with a "Sources:" section listing the URLs used."""

    def __init__(self):
        self._search_tool = WebSearchTool()
        self._llm = get_llm(temperature=0.1)
        logger.info(
            "search_agent_initialized",
            provider=settings.default_llm_provider,
            model=settings.get_llm_model_name(),
        )

    # ── LangGraph Node Entry Point ─────────────────────────────────────────────

    async def run(self, state: ResearchState) -> ResearchState:
        """
        LangGraph node function. Signature must be:
            async def run(self, state: StateType) -> StateType

        Reads: state["query"]
        Writes: state["search_result"], state["sources"], state["errors"]
        """
        query = state.get("query", "")
        if not query:
            logger.warning("search_agent_empty_query")
            return {
                **state,
                "search_result": AgentResult(
                    agent_name="SearchAgent",
                    status=AgentStatus.FAILED,
                    output="",
                    error="Empty query provided",
                ),
            }

        start = time.perf_counter()

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._execute(query),
                timeout=settings.agent_timeout_seconds,
            )
        except asyncio.TimeoutError:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "search_agent_timeout",
                query=query,
                timeout=settings.agent_timeout_seconds,
                latency_ms=latency_ms,
            )
            result = AgentResult(
                agent_name="SearchAgent",
                status=AgentStatus.TIMEOUT,
                output="",
                error=f"Search timed out after {settings.agent_timeout_seconds}s",
                latency_ms=latency_ms,
            )

        # Update state
        current_sources = state.get("sources", [])
        current_errors = state.get("errors", [])

        if result.status == AgentStatus.SUCCESS and result.raw_data:
            new_sources = [r.url for r in result.raw_data.results if r.url]
            current_sources = list(set(current_sources + new_sources))

        if result.error:
            current_errors = current_errors + [f"SearchAgent: {result.error}"]

        return {
            **state,
            "search_result": result,
            "sources": current_sources,
            "errors": current_errors,
        }

    # ── Core Execution ─────────────────────────────────────────────────────────

    async def _execute(self, query: str) -> AgentResult:
        """Search → synthesize → return AgentResult."""
        start = time.perf_counter()

        with AgentCallLogger(logger, "SearchAgent", query) as call_log:
            # Step 1: Web search
            search_response: SearchResponse = await self._search_tool.search(
                query=query,
                max_results=5,
                search_depth="basic",
                include_answer=True,
            )

            if not search_response.success:
                return AgentResult(
                    agent_name="SearchAgent",
                    status=AgentStatus.FAILED,
                    output="",
                    error=search_response.error,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                )

            # Step 2: LLM synthesis
            synthesized = await self._synthesize(query, search_response)

            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            call_log.set_output(synthesized[:200])

            return AgentResult(
                agent_name="SearchAgent",
                status=AgentStatus.SUCCESS,
                output=synthesized,
                raw_data=search_response,
                latency_ms=latency_ms,
                metadata={
                    "result_count": len(search_response.results),
                    "had_quick_answer": search_response.answer is not None,
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
    async def _synthesize(self, query: str, search_response: SearchResponse) -> str:
        """
        Uses LLM to synthesize raw search results into a coherent answer.
        Retried up to MAX_RETRIES times on failure.
        """
        context = search_response.to_context_string()

        user_prompt = f"""Research query: {query}

Search Results:
{context}

Please synthesize these results into a clear, cited answer."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # LLM call — async invoke
        response = await self._llm.ainvoke(messages)
        return response.content.strip()

    # ── Direct call interface (for testing without LangGraph) ─────────────────

    async def search(self, query: str) -> AgentResult:
        """
        Convenience method for direct testing outside LangGraph.
        Usage: result = await agent.search("your query")
        """
        state: ResearchState = {"query": query, "sources": [], "errors": []}
        new_state = await self.run(state)
        return new_state["search_result"]