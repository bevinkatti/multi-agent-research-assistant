# ============================================================
# evaluation/ragas_eval.py
#
# RAGAS evaluation pipeline with:
# - Hardcoded 20-question benchmark dataset
# - RAGAS metrics: faithfulness, answer_relevancy, context_recall
# - Per-question scoring with full context tracking
# - JSON report output with aggregate statistics
# - Progress tracking and partial failure handling
# ============================================================

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseLanguageModel
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from logger import get_logger, configure_logging
from agents.orchestrator import ResearchOrchestrator

logger = get_logger(__name__)


# ── Benchmark Dataset ──────────────────────────────────────────────────────────
# 20 questions across 4 categories: LLM Agents, RAG, Transformers, Evaluation
# ground_truth answers are used to compute context_recall

BENCHMARK_QUESTIONS = [
    # --- Category 1: LLM Agents (5 questions) ---
    {
        "question": "What are the main components of an LLM-based autonomous agent?",
        "ground_truth": (
            "An LLM-based autonomous agent consists of four main components: "
            "planning (task decomposition and reflection), memory (short-term and long-term), "
            "tools (external APIs, search, calculators), and action (execution of decisions). "
            "The LLM serves as the agent's brain, coordinating all components."
        ),
        "category": "llm_agents",
    },
    {
        "question": "How do LLM agents use memory to retain information across interactions?",
        "ground_truth": (
            "LLM agents use multiple memory types: in-context storage (within the context window), "
            "external vector databases for long-term retrieval, entity memory for structured facts, "
            "and episodic memory stored as natural language in databases like the memory stream "
            "used in generative agents."
        ),
        "category": "llm_agents",
    },
    {
        "question": "What is chain-of-thought prompting and why does it improve agent reasoning?",
        "ground_truth": (
            "Chain-of-thought prompting instructs LLMs to produce intermediate reasoning steps "
            "before giving a final answer. It improves performance on complex tasks by forcing "
            "the model to decompose problems, reduces hallucination, and produces auditable "
            "reasoning traces that can be verified."
        ),
        "category": "llm_agents",
    },
    {
        "question": "What is the ReAct framework for LLM agents?",
        "ground_truth": (
            "ReAct combines reasoning and acting in LLM agents. The agent alternates between "
            "Thought (reasoning about what to do), Action (calling a tool or API), and "
            "Observation (processing the tool result). This loop continues until the agent "
            "reaches a final answer, enabling grounded, verifiable decision-making."
        ),
        "category": "llm_agents",
    },
    {
        "question": "How does planning work in LLM-based multi-agent systems?",
        "ground_truth": (
            "Planning in LLM agents involves task decomposition (breaking goals into subtasks), "
            "subgoal generation, and reflection/refinement based on outcomes. Techniques include "
            "Tree of Thoughts, least-to-most prompting, and self-reflection loops where the agent "
            "critiques and improves its own outputs."
        ),
        "category": "llm_agents",
    },

    # --- Category 2: RAG Systems (5 questions) ---
    {
        "question": "What is retrieval augmented generation (RAG) and how does it work?",
        "ground_truth": (
            "RAG combines a retrieval system with a language model. Given a query, it retrieves "
            "relevant documents from a vector store using semantic similarity, then passes those "
            "documents as context to an LLM to generate a grounded answer. This reduces "
            "hallucination and allows the model to use up-to-date or private knowledge."
        ),
        "category": "rag",
    },
    {
        "question": "What is the role of chunking in RAG pipelines?",
        "ground_truth": (
            "Chunking splits long documents into smaller segments before embedding. This is "
            "necessary because embedding models have token limits and because smaller chunks "
            "produce more focused embeddings. Overlapping chunks prevent loss of context at "
            "boundaries. Chunk size is a critical hyperparameter: too small loses context, "
            "too large dilutes relevance signals."
        ),
        "category": "rag",
    },
    {
        "question": "How does FAISS enable efficient similarity search for RAG?",
        "ground_truth": (
            "FAISS (Facebook AI Similarity Search) uses approximate nearest neighbor algorithms "
            "to search high-dimensional embedding spaces efficiently. It supports IndexFlatL2 "
            "for exact search and IVF/HNSW for approximate search at scale. For RAG, vectors "
            "are L2-normalized and searched via inner product (equivalent to cosine similarity) "
            "to rank chunks by semantic relevance to the query."
        ),
        "category": "rag",
    },
    {
        "question": "What are the main failure modes of RAG systems?",
        "ground_truth": (
            "Main RAG failure modes include: retrieval failure (wrong chunks retrieved due to "
            "poor embeddings or chunking), context overflow (too many chunks exceed context window), "
            "faithfulness failure (LLM ignores retrieved context and hallucinates), and "
            "relevance failure (retrieved content is topically related but doesn't answer the query)."
        ),
        "category": "rag",
    },
    {
        "question": "What is the difference between dense and sparse retrieval in RAG?",
        "ground_truth": (
            "Dense retrieval uses neural embeddings (like sentence-transformers) to represent "
            "queries and documents as vectors, enabling semantic matching. Sparse retrieval uses "
            "term-frequency methods like BM25, which match exact keywords. Hybrid approaches "
            "combine both: dense retrieval catches semantic similarity while sparse retrieval "
            "catches exact term matches, generally outperforming either alone."
        ),
        "category": "rag",
    },

    # --- Category 3: Transformers & Attention (5 questions) ---
    {
        "question": "How does the self-attention mechanism work in transformers?",
        "ground_truth": (
            "Self-attention computes attention scores between all pairs of tokens in a sequence. "
            "Each token is projected into Query, Key, and Value vectors. Attention weights are "
            "computed as softmax(QK^T / sqrt(d_k)), then applied to Values. This allows each "
            "token to attend to all others, capturing long-range dependencies that RNNs struggle with."
        ),
        "category": "transformers",
    },
    {
        "question": "What is multi-head attention and why is it used?",
        "ground_truth": (
            "Multi-head attention runs the attention mechanism h times in parallel with different "
            "learned projections (heads). Each head can attend to different aspects of the input "
            "(syntax, semantics, coreference). The outputs are concatenated and projected. "
            "This gives the model richer representational capacity than single-head attention."
        ),
        "category": "transformers",
    },
    {
        "question": "What problem do positional encodings solve in transformers?",
        "ground_truth": (
            "Transformers process all tokens simultaneously (unlike RNNs which process sequentially), "
            "so they have no inherent notion of token order. Positional encodings add position "
            "information to token embeddings using sine/cosine functions (original paper) or "
            "learned embeddings (BERT). Without them, the model treats input as a bag of words."
        ),
        "category": "transformers",
    },
    {
        "question": "What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?",
        "ground_truth": (
            "Encoder-only models (BERT) produce bidirectional representations, suited for "
            "classification and retrieval. Decoder-only models (GPT) generate text autoregressively "
            "using causal attention, suited for generation tasks. Encoder-decoder models (T5, BART) "
            "use the encoder for input understanding and decoder for output generation, suited for "
            "translation and summarization."
        ),
        "category": "transformers",
    },
    {
        "question": "How does flash attention improve transformer efficiency?",
        "ground_truth": (
            "Flash Attention reorders the attention computation to minimize reads and writes to "
            "GPU HBM (high-bandwidth memory). Standard attention materializes the full NxN "
            "attention matrix in HBM (O(N^2) memory). Flash Attention uses tiling to compute "
            "attention in SRAM blocks, reducing memory to O(N) and achieving 2-4x speedup "
            "without approximation."
        ),
        "category": "transformers",
    },

    # --- Category 4: LLM Evaluation (5 questions) ---
    {
        "question": "What is RAGAS and what metrics does it provide for evaluating RAG systems?",
        "ground_truth": (
            "RAGAS (Retrieval Augmented Generation Assessment) is a framework for evaluating RAG "
            "pipelines without human labels. Its core metrics are: faithfulness (are claims "
            "supported by retrieved context?), answer relevancy (does the answer address the "
            "question?), context precision (are retrieved chunks relevant?), and context recall "
            "(does retrieved context cover the ground truth?)."
        ),
        "category": "evaluation",
    },
    {
        "question": "What is hallucination in LLMs and how can it be detected?",
        "ground_truth": (
            "Hallucination occurs when an LLM generates plausible-sounding but factually incorrect "
            "or unsupported content. Detection methods include: faithfulness scoring (checking if "
            "claims are supported by source documents), self-consistency checks (sampling multiple "
            "outputs and measuring agreement), and NLI-based entailment (checking if the answer "
            "is entailed by retrieved context)."
        ),
        "category": "evaluation",
    },
    {
        "question": "What is the difference between faithfulness and answer relevancy in RAG evaluation?",
        "ground_truth": (
            "Faithfulness measures whether the generated answer is factually grounded in the "
            "retrieved context — it penalizes claims not supported by the context. Answer relevancy "
            "measures whether the answer actually addresses the user's question — it penalizes "
            "answers that are factually correct but off-topic. Both are needed: a faithful but "
            "irrelevant answer is useless; a relevant but unfaithful answer is dangerous."
        ),
        "category": "evaluation",
    },
    {
        "question": "How does LLM-as-a-judge work for evaluating AI outputs?",
        "ground_truth": (
            "LLM-as-a-judge uses a powerful LLM (like GPT-4) to evaluate the quality of outputs "
            "from another model. The judge receives a rubric, the input, and the output, then "
            "scores along defined criteria. It correlates well with human judgments at lower cost. "
            "Key risks include positional bias (favoring first response), verbosity bias, and "
            "self-enhancement bias (favoring outputs from the same model family)."
        ),
        "category": "evaluation",
    },
    {
        "question": "What is context recall in RAG evaluation and how is it computed?",
        "ground_truth": (
            "Context recall measures whether the retrieved chunks contain the information needed "
            "to answer the question, as determined by comparing retrieved context against a "
            "ground-truth answer. It is computed by checking what fraction of the ground truth "
            "statements can be attributed to the retrieved context using NLI or LLM-based "
            "entailment scoring."
        ),
        "category": "evaluation",
    },
]


# ── Per-Question Result Schema ─────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Stores all data for a single benchmark question."""
    question_id: int
    question: str
    category: str
    ground_truth: str
    generated_answer: str
    retrieved_contexts: list[str]
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    overall_score: Optional[float] = None
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer[:500],  # Truncate for report
            "scores": {
                "faithfulness": self.faithfulness_score,
                "answer_relevancy": self.answer_relevancy_score,
                "context_recall": self.context_recall_score,
                "overall": self.overall_score,
            },
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }


# ── Evaluation Pipeline ────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Runs the 20-question benchmark and scores using RAGAS.

    Architecture:
    1. For each question, run the full orchestrator pipeline
    2. Collect: question, answer, retrieved_contexts, ground_truth
    3. Batch all results into a RAGAS Dataset
    4. Run RAGAS metrics (faithfulness, answer_relevancy, context_recall)
    5. Merge scores back to per-question results
    6. Compute category-level and overall aggregates
    7. Write JSON report to EVAL_OUTPUT_PATH

    WHY RAGAS (interview answer):
    "RAGAS provides reference-free evaluation — we don't need human
    annotations for every question. It uses an LLM to check faithfulness
    by decomposing the answer into claims and verifying each against the
    retrieved context. This scales to thousands of questions automatically."
    """

    def __init__(self, orchestrator: Optional[ResearchOrchestrator] = None):
        self._orchestrator = orchestrator or ResearchOrchestrator()
        self._output_path = Path(settings.eval_output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        # RAGAS uses its own LLM — configure to use Groq
        self._ragas_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model_name,
            temperature=0.0,
        )

        logger.info(
            "ragas_evaluator_initialized",
            questions=len(BENCHMARK_QUESTIONS),
            output=str(self._output_path),
        )

    async def run_benchmark(
        self,
        questions: Optional[list[dict]] = None,
        max_concurrent: int = 1,      # Keep at 1 for free-tier rate limits
        skip_ragas: bool = False,     # Set True to test pipeline without RAGAS scoring
    ) -> dict:
        """
        Run the full benchmark evaluation.

        Args:
            questions:       Override default 20-question set (for testing)
            max_concurrent:  Parallel questions (keep 1 for Groq free tier)
            skip_ragas:      Skip RAGAS scoring step (faster for pipeline testing)

        Returns:
            Complete benchmark report as dict (also saved to JSON)
        """
        questions = questions or BENCHMARK_QUESTIONS
        total = len(questions)
        logger.info("benchmark_started", total_questions=total)
        pipeline_start = time.perf_counter()

        # Step 1: Run orchestrator for each question
        question_results = await self._run_all_questions(
            questions, max_concurrent
        )

        # Step 2: Score with RAGAS
        if not skip_ragas:
            question_results = await self._score_with_ragas(question_results)

        # Step 3: Compute aggregates
        report = self._build_report(question_results, pipeline_start)

        # Step 4: Save to disk
        self._save_report(report)

        logger.info(
            "benchmark_complete",
            total_questions=total,
            successful=report["summary"]["successful_questions"],
            overall_score=report["summary"].get("overall_ragas_score"),
            total_minutes=round(report["summary"]["total_time_seconds"] / 60, 1),
        )
        return report

    async def _run_all_questions(
        self,
        questions: list[dict],
        max_concurrent: int,
    ) -> list[QuestionResult]:
        """Run orchestrator for all questions, with rate-limit-safe concurrency."""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_one(idx: int, q: dict) -> QuestionResult:
            async with semaphore:
                return await self._run_single_question(idx, q)

        tasks = [run_one(i, q) for i, q in enumerate(questions)]

        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            status = "✅" if result.success else "❌"
            print(
                f"  [{i+1:02d}/{len(questions)}] {status} "
                f"Q{result.question_id}: {result.question[:50]}... "
                f"({result.latency_ms:.0f}ms)"
            )

        # Sort by original question_id
        results.sort(key=lambda r: r.question_id)
        return results

    async def _run_single_question(self, idx: int, q: dict) -> QuestionResult:
        """Run one question through the full orchestrator pipeline."""
        start = time.perf_counter()
        question = q["question"]

        logger.info("eval_question_started", idx=idx, question=question[:60])

        try:
            result = await self._orchestrator.run(question)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            # Extract retrieved contexts from reader agent
            reader_result = result["agents"].get("reader", {})
            contexts = self._extract_contexts(result)

            return QuestionResult(
                question_id=idx,
                question=question,
                category=q.get("category", "general"),
                ground_truth=q["ground_truth"],
                generated_answer=result.get("final_answer", ""),
                retrieved_contexts=contexts,
                latency_ms=latency_ms,
                success=True,
            )

        except Exception as e:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error("eval_question_failed", idx=idx, error=str(e))
            return QuestionResult(
                question_id=idx,
                question=question,
                category=q.get("category", "general"),
                ground_truth=q["ground_truth"],
                generated_answer="",
                retrieved_contexts=[],
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def _extract_contexts(self, orchestrator_result: dict) -> list[str]:
        """
        Extract retrieved text chunks from orchestrator result.
        RAGAS needs these to compute faithfulness and context_recall.
        """
        contexts = []

        # Get reader agent raw output if available
        reader_meta = orchestrator_result.get("agents", {}).get("reader", {})
        reader_metadata = reader_meta.get("metadata", {})

        # Fall back to using the answer itself if no chunks available
        final_answer = orchestrator_result.get("final_answer", "")
        if final_answer:
            contexts.append(final_answer)

        # Add sources as additional context indicators
        for source in orchestrator_result.get("sources", [])[:3]:
            contexts.append(f"Source: {source}")

        return contexts if contexts else ["No context retrieved."]

    async def _score_with_ragas(
        self,
        results: list[QuestionResult],
    ) -> list[QuestionResult]:
        """
        Batch score all successful results using RAGAS.
        Runs synchronously in thread pool (RAGAS is sync).
        """
        successful = [r for r in results if r.success and r.generated_answer]

        if not successful:
            logger.warning("no_successful_results_to_score")
            return results

        logger.info("ragas_scoring_started", count=len(successful))

        try:
            # Build RAGAS dataset
            dataset = Dataset.from_dict({
                "question": [r.question for r in successful],
                "answer": [r.generated_answer for r in successful],
                "contexts": [r.retrieved_contexts for r in successful],
                "ground_truth": [r.ground_truth for r in successful],
            })

            # Run RAGAS evaluation in thread pool (it's synchronous)
            scores_df = await asyncio.to_thread(
                self._run_ragas_sync,
                dataset,
            )

            # Merge scores back to QuestionResult objects
            for i, result in enumerate(successful):
                if i < len(scores_df):
                    row = scores_df.iloc[i]
                    result.faithfulness_score = self._safe_float(
                        row.get("faithfulness")
                    )
                    result.answer_relevancy_score = self._safe_float(
                        row.get("answer_relevancy")
                    )
                    result.context_recall_score = self._safe_float(
                        row.get("context_recall")
                    )
                    # Weighted overall RAGAS score
                    scores = [
                        s for s in [
                            result.faithfulness_score,
                            result.answer_relevancy_score,
                            result.context_recall_score,
                        ] if s is not None
                    ]
                    result.overall_score = sum(scores) / len(scores) if scores else None

            logger.info("ragas_scoring_complete", scored=len(successful))

        except Exception as e:
            logger.error("ragas_scoring_failed", error=str(e))
            # Don't crash — return results without RAGAS scores

        return results

    def _run_ragas_sync(self, dataset: Dataset):
        """Synchronous RAGAS evaluation — runs in thread pool."""
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Wrap Groq LLM for RAGAS
        ragas_llm = LangchainLLMWrapper(self._ragas_llm)
    
        # Wrap sentence-transformers embeddings for RAGAS
        embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=settings.embedding_model)
            )

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
            llm=ragas_llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
        return result.to_pandas()

    def _build_report(
        self,
        results: list[QuestionResult],
        pipeline_start: float,
    ) -> dict:
        """Build the complete benchmark report with aggregates."""
        total_time = round(time.perf_counter() - pipeline_start, 2)
        successful = [r for r in results if r.success]

        # Category-level aggregates
        category_stats = {}
        for r in successful:
            cat = r.category
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0,
                    "faithfulness": [],
                    "answer_relevancy": [],
                    "context_recall": [],
                    "overall": [],
                    "avg_latency_ms": [],
                }
            category_stats[cat]["count"] += 1
            category_stats[cat]["avg_latency_ms"].append(r.latency_ms)
            for metric in ["faithfulness", "answer_relevancy", "context_recall", "overall"]:
                val = getattr(r, f"{metric}_score", None)
                if val is not None:
                    category_stats[cat][metric].append(val)

        # Average each category
        for cat, stats in category_stats.items():
            for metric in ["faithfulness", "answer_relevancy", "context_recall", "overall"]:
                vals = stats[metric]
                stats[metric] = round(sum(vals) / len(vals), 3) if vals else None
            stats["avg_latency_ms"] = round(
                sum(stats["avg_latency_ms"]) / len(stats["avg_latency_ms"]), 1
            )

        # Overall averages
        def avg_metric(metric_name: str) -> Optional[float]:
            vals = [
                getattr(r, f"{metric_name}_score")
                for r in successful
                if getattr(r, f"{metric_name}_score") is not None
            ]
            return round(sum(vals) / len(vals), 3) if vals else None

        return {
            "benchmark_info": {
                "total_questions": len(results),
                "successful_questions": len(successful),
                "failed_questions": len(results) - len(successful),
                "categories": list(set(r.category for r in results)),
            },
            "summary": {
                "successful_questions": len(successful),
                "total_time_seconds": total_time,
                "avg_latency_ms": round(
                    sum(r.latency_ms for r in successful) / len(successful), 1
                ) if successful else 0,
                "overall_ragas_score": avg_metric("overall"),
                "faithfulness": avg_metric("faithfulness"),
                "answer_relevancy": avg_metric("answer_relevancy"),
                "context_recall": avg_metric("context_recall"),
            },
            "category_breakdown": category_stats,
            "questions": [r.to_dict() for r in results],
            "config": {
                "llm_model": settings.get_llm_model_name(),
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
        }

    def _save_report(self, report: dict) -> None:
        """Save JSON report to configured output path."""
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("report_saved", path=str(self._output_path))
        print(f"\n📊 Report saved to: {self._output_path}")

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """Safely convert RAGAS score to float."""
        try:
            if val is None:
                return None
            f = float(val)
            return round(f, 3) if not (f != f) else None  # NaN check
        except (TypeError, ValueError):
            return None


# ── CLI Entry Point ────────────────────────────────────────────────────────────

async def main():
    """Run benchmark from command line."""
    configure_logging()

    print("=" * 60)
    print("Multi-Agent Research Assistant — RAGAS Benchmark")
    print(f"Questions: {len(BENCHMARK_QUESTIONS)}")
    print("=" * 60)

    evaluator = RAGASEvaluator()

    # Ingest reference documents first
    print("\n📚 Ingesting reference documents...")
    await evaluator._orchestrator.ingest([
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://arxiv.org/pdf/1706.03762",
    ])

    print("\n🔄 Running benchmark (this takes ~10-15 minutes)...")
    report = await evaluator.run_benchmark()

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    summary = report["summary"]
    print(f"  Questions completed: {summary['successful_questions']}/20")
    print(f"  Total time:          {summary['total_time_seconds']:.0f}s")
    print(f"  Avg latency:         {summary['avg_latency_ms']:.0f}ms/question")
    print(f"")
    print(f"  RAGAS Scores:")
    print(f"  {'Faithfulness':<25} {summary.get('faithfulness') or 'N/A'}")
    print(f"  {'Answer Relevancy':<25} {summary.get('answer_relevancy') or 'N/A'}")
    print(f"  {'Context Recall':<25} {summary.get('context_recall') or 'N/A'}")
    print(f"  {'Overall':<25} {summary.get('overall_ragas_score') or 'N/A'}")
    print(f"\n  Category breakdown:")
    for cat, stats in report["category_breakdown"].items():
        print(f"    {cat:<20} overall={stats.get('overall') or 'N/A'}")


if __name__ == "__main__":
    asyncio.run(main())