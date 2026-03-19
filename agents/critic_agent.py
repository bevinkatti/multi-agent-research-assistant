# ============================================================
# agents/critic_agent.py
#
# Full Critic Agent with:
# - 5-dimension scoring rubric (faithfulness, relevance,
#   groundedness, contradiction, hallucination risk)
# - Structured JSON scorecard output
# - Per-dimension explanations for transparency
# - Retry logic on LLM calls
# - Used by orchestrator in Week 2 full pipeline
# ============================================================

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
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
from agents.search_agent import (
    AgentResult,
    AgentStatus,
    ResearchState,
    get_llm,
)

logger = get_logger(__name__)


# ── Scorecard Schema ───────────────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    score: float          # 0.0 – 1.0
    explanation: str      # One sentence rationale
    passed: bool          # True if score >= threshold

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "explanation": self.explanation,
            "passed": self.passed,
        }


@dataclass
class CriticScorecard:
    """
    Complete scorecard returned by the CriticAgent.

    Dimensions:
    - faithfulness:    Are claims supported by cited sources?
    - relevance:       Does the answer address the actual query?
    - groundedness:    Is evidence specific (not vague/generic)?
    - consistency:     Do search and reader answers agree?
    - hallucination:   Inverse risk of made-up facts (1.0 = safe)

    Overall = weighted average of all dimensions.
    """
    query: str
    faithfulness: DimensionScore
    relevance: DimensionScore
    groundedness: DimensionScore
    consistency: DimensionScore
    hallucination_safety: DimensionScore
    overall_score: float
    recommendation: str      # "pass" | "retry" | "escalate"
    summary: str             # Human-readable critique summary

    # Thresholds
    PASS_THRESHOLD: float = 0.6
    RETRY_THRESHOLD: float = 0.4

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "overall_score": round(self.overall_score, 3),
            "recommendation": self.recommendation,
            "summary": self.summary,
            "dimensions": {
                "faithfulness": self.faithfulness.to_dict(),
                "relevance": self.relevance.to_dict(),
                "groundedness": self.groundedness.to_dict(),
                "consistency": self.consistency.to_dict(),
                "hallucination_safety": self.hallucination_safety.to_dict(),
            },
        }

    def to_formatted_report(self) -> str:
        """Human-readable text version for LLM context injection."""
        lines = [
            f"CRITIC SCORECARD",
            f"Query: {self.query}",
            f"Overall Score: {self.overall_score:.2f} | Recommendation: {self.recommendation.upper()}",
            f"",
            f"Dimension Scores:",
            f"  Faithfulness:        {self.faithfulness.score:.2f} — {self.faithfulness.explanation}",
            f"  Relevance:           {self.relevance.score:.2f} — {self.relevance.explanation}",
            f"  Groundedness:        {self.groundedness.score:.2f} — {self.groundedness.explanation}",
            f"  Consistency:         {self.consistency.score:.2f} — {self.consistency.explanation}",
            f"  Hallucination Safety:{self.hallucination_safety.score:.2f} — {self.hallucination_safety.explanation}",
            f"",
            f"Summary: {self.summary}",
        ]
        return "\n".join(lines)


# ── Critic Agent ───────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Full Critic Agent that scores Search + Reader outputs
    across five evaluation dimensions.

    The scorecard's "recommendation" field tells the orchestrator:
    - "pass":     Quality is sufficient, proceed to synthesis
    - "retry":    Quality is borderline, retry the weakest agent
    - "escalate": Quality is too low, flag for human review

    WHY THIS MATTERS (interview answer):
    "Without a critic, the system has no self-awareness of output
    quality. The critic creates a feedback signal that the orchestrator
    can use to decide whether to trust the answer or try again —
    turning a one-shot pipeline into a self-correcting system."
    """

    # Dimension weights for overall score calculation
    WEIGHTS = {
        "faithfulness": 0.30,
        "relevance": 0.25,
        "groundedness": 0.20,
        "consistency": 0.15,
        "hallucination_safety": 0.10,
    }

    SYSTEM_PROMPT = """You are an expert AI output evaluator. Score two research answers across 5 dimensions.

For each dimension, provide:
- A score from 0.0 to 1.0 (use decimals like 0.7, not just 0 or 1)
- A one-sentence explanation

Respond ONLY with valid JSON in exactly this format (no markdown, no extra text):
{
  "faithfulness": {
    "score": 0.0,
    "explanation": "one sentence"
  },
  "relevance": {
    "score": 0.0,
    "explanation": "one sentence"
  },
  "groundedness": {
    "score": 0.0,
    "explanation": "one sentence"
  },
  "consistency": {
    "score": 0.0,
    "explanation": "one sentence"
  },
  "hallucination_safety": {
    "score": 0.0,
    "explanation": "one sentence"
  },
  "summary": "2-3 sentence overall critique"
}

Scoring rubrics:
- faithfulness (0-1): Are all claims backed by cited sources? 1.0 = every claim has a citation. 0.0 = no citations at all.
- relevance (0-1): Does the answer actually address the query? 1.0 = directly answers. 0.0 = completely off-topic.
- groundedness (0-1): Is evidence specific and verifiable? 1.0 = concrete facts with sources. 0.0 = vague generalities.
- consistency (0-1): Do search and reader answers agree? 1.0 = fully consistent. 0.0 = direct contradiction.
- hallucination_safety (0-1): How safe from fabricated facts? 1.0 = no hallucination risk. 0.0 = clear fabrication."""

    def __init__(self, llm: Optional[Any] = None):
        self._llm = llm or get_llm(temperature=0.0)
        logger.info(
            "critic_agent_initialized",
            provider=settings.default_llm_provider,
            model=settings.get_llm_model_name(),
        )

    # ── LangGraph Node Entry Point ─────────────────────────────────────────────

    async def run(self, state: ResearchState) -> ResearchState:
        """
        LangGraph node function.
        Reads:  state["query"], state["search_result"], state["reader_result"]
        Writes: state["critic_result"], state["errors"]
        """
        query = state.get("query", "")
        search_result: Optional[AgentResult] = state.get("search_result")
        reader_result: Optional[AgentResult] = state.get("reader_result")

        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._execute(query, search_result, reader_result),
                timeout=settings.agent_timeout_seconds,
            )
        except asyncio.TimeoutError:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error("critic_agent_timeout", query=query)
            result = AgentResult(
                agent_name="CriticAgent",
                status=AgentStatus.TIMEOUT,
                output="Critique timed out — proceeding without validation.",
                latency_ms=latency_ms,
                error=f"Timed out after {settings.agent_timeout_seconds}s",
            )

        current_errors = state.get("errors", [])
        if result.error:
            current_errors = current_errors + [f"CriticAgent: {result.error}"]

        return {**state, "critic_result": result, "errors": current_errors}

    # ── Core Execution ─────────────────────────────────────────────────────────

    async def _execute(
        self,
        query: str,
        search_result: Optional[AgentResult],
        reader_result: Optional[AgentResult],
    ) -> AgentResult:
        """Score both agent outputs and produce a CriticScorecard."""
        start = time.perf_counter()

        with AgentCallLogger(logger, "CriticAgent", query) as call_log:
            try:
                scorecard = await self._score(query, search_result, reader_result)

                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                report = scorecard.to_formatted_report()
                call_log.set_output(
                    f"overall={scorecard.overall_score:.2f} "
                    f"recommendation={scorecard.recommendation}"
                )

                logger.info(
                    "critic_scored",
                    overall=scorecard.overall_score,
                    recommendation=scorecard.recommendation,
                    faithfulness=scorecard.faithfulness.score,
                    relevance=scorecard.relevance.score,
                    groundedness=scorecard.groundedness.score,
                    consistency=scorecard.consistency.score,
                    hallucination_safety=scorecard.hallucination_safety.score,
                )

                return AgentResult(
                    agent_name="CriticAgent",
                    status=AgentStatus.SUCCESS,
                    output=report,
                    raw_data=scorecard,
                    latency_ms=latency_ms,
                    metadata=scorecard.to_dict(),
                )

            except Exception as e:
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.error("critic_execution_failed", error=str(e))
                return AgentResult(
                    agent_name="CriticAgent",
                    status=AgentStatus.FAILED,
                    output="Critique failed — proceeding without validation.",
                    latency_ms=latency_ms,
                    error=str(e),
                )

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=settings.retry_base_delay, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    async def _score(
        self,
        query: str,
        search_result: Optional[AgentResult],
        reader_result: Optional[AgentResult],
    ) -> CriticScorecard:
        """
        Call LLM to score both outputs, parse JSON response into scorecard.
        Retried on failure — JSON parsing errors trigger a retry.
        """
        search_text = (
            search_result.output
            if search_result and search_result.output
            else "No search result available."
        )
        reader_text = (
            reader_result.output
            if reader_result and reader_result.output
            else "No document result available."
        )

        user_prompt = f"""Query: {query}

Search Agent Answer:
{search_text}

Reader Agent Answer (from documents):
{reader_text}

Score both answers across all 5 dimensions. Return ONLY valid JSON."""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self._llm.ainvoke(messages)
        raw_text = response.content.strip()

        # Parse JSON — strip markdown fences if present
        parsed = self._parse_json(raw_text)
        return self._build_scorecard(query, parsed)

    def _parse_json(self, raw_text: str) -> dict:
        """
        Robustly parse LLM JSON response.
        Handles markdown fences, trailing commas, and minor formatting issues.
        """
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        cleaned = cleaned.rstrip("`").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract just the JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(
                f"Could not parse critic JSON response: {cleaned[:200]}"
            )

    def _build_scorecard(self, query: str, data: dict) -> CriticScorecard:
        """Convert parsed JSON dict into a typed CriticScorecard."""

        def make_dimension(key: str, threshold: float = 0.5) -> DimensionScore:
            dim = data.get(key, {})
            score = float(dim.get("score", 0.5))
            return DimensionScore(
                score=score,
                explanation=dim.get("explanation", "No explanation provided."),
                passed=score >= threshold,
            )

        faithfulness       = make_dimension("faithfulness", threshold=0.5)
        relevance          = make_dimension("relevance", threshold=0.6)
        groundedness       = make_dimension("groundedness", threshold=0.5)
        consistency        = make_dimension("consistency", threshold=0.4)
        hallucination_safety = make_dimension("hallucination_safety", threshold=0.5)

        # Weighted overall score
        overall = (
            faithfulness.score       * self.WEIGHTS["faithfulness"]
            + relevance.score        * self.WEIGHTS["relevance"]
            + groundedness.score     * self.WEIGHTS["groundedness"]
            + consistency.score      * self.WEIGHTS["consistency"]
            + hallucination_safety.score * self.WEIGHTS["hallucination_safety"]
        )

        # Recommendation logic
        if overall >= CriticScorecard.PASS_THRESHOLD:
            recommendation = "pass"
        elif overall >= CriticScorecard.RETRY_THRESHOLD:
            recommendation = "retry"
        else:
            recommendation = "escalate"

        return CriticScorecard(
            query=query,
            faithfulness=faithfulness,
            relevance=relevance,
            groundedness=groundedness,
            consistency=consistency,
            hallucination_safety=hallucination_safety,
            overall_score=overall,
            recommendation=recommendation,
            summary=data.get("summary", "No summary provided."),
        )

    # ── Convenience method for testing ────────────────────────────────────────

    async def critique(
        self,
        query: str,
        search_output: str,
        reader_output: str,
    ) -> CriticScorecard:
        """
        Direct scoring without LangGraph state. For testing.

        Usage:
            scorecard = await critic.critique(query, search_text, reader_text)
            print(scorecard.to_formatted_report())
        """
        search_result = AgentResult(
            agent_name="SearchAgent",
            status=AgentStatus.SUCCESS,
            output=search_output,
        )
        reader_result = AgentResult(
            agent_name="ReaderAgent",
            status=AgentStatus.SUCCESS,
            output=reader_output,
        )
        result = await self._execute(query, search_result, reader_result)

        if result.raw_data:
            return result.raw_data
        raise ValueError(f"Critique failed: {result.error}")