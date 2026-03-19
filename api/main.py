# ============================================================
# api/main.py
#
# FastAPI backend with:
# - Async endpoints for research, ingest, eval, health
# - Server-Sent Events (SSE) for real-time agent activity
# - Pydantic request/response schemas
# - Lifespan context manager (startup/shutdown)
# - CORS for Streamlit frontend
# - Global error handling
# ============================================================

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger, configure_logging
from agents.orchestrator import ResearchOrchestrator
from tools.vector_store import VectorStore

configure_logging(settings.log_level)
logger = get_logger(__name__)

# ── Shared State ───────────────────────────────────────────────────────────────
# Single orchestrator instance reused across all requests.
# VectorStore is shared to avoid loading embedding model multiple times.

_vector_store: Optional[VectorStore] = None
_orchestrator: Optional[ResearchOrchestrator] = None

# Active SSE streams: request_id → asyncio.Queue of events
_event_streams: dict[str, asyncio.Queue] = {}


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Initializes the orchestrator once — expensive (loads embedding model).
    On shutdown, saves the FAISS index to disk.

    WHY LIFESPAN (interview answer):
    "Lifespan replaces deprecated @app.on_event handlers. It uses an
    async context manager so initialization and cleanup are colocated,
    and the yielded app state is available to all endpoints via
    app.state — no globals needed."
    """
    global _vector_store, _orchestrator

    logger.info("api_startup_begin")
    start = time.perf_counter()

    _vector_store = VectorStore()
    _orchestrator = ResearchOrchestrator(vector_store=_vector_store)

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info("api_startup_complete", elapsed_ms=elapsed)

    yield   # Application runs here

    # Shutdown: persist index
    logger.info("api_shutdown_begin")
    if _vector_store:
        await _vector_store.save()
    logger.info("api_shutdown_complete")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description=(
        "A production-grade multi-agent research system with "
        "web search, RAG, critic scoring, and RAGAS evaluation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500,
                       description="Research question to investigate")
    stream: bool = Field(default=False,
                         description="If True, returns SSE stream instead of JSON")


class IngestRequest(BaseModel):
    sources: list[str] = Field(
        ..., min_length=1,
        description="List of PDF file paths or HTTP/HTTPS URLs to ingest"
    )


class EvalRequest(BaseModel):
    num_questions: int = Field(
        default=5, ge=1, le=20,
        description="Number of benchmark questions to evaluate (1-20)"
    )
    skip_ragas: bool = Field(
        default=False,
        description="Skip RAGAS scoring (faster, for pipeline testing)"
    )


class AgentSummary(BaseModel):
    status: str
    latency_ms: float
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class ResearchResponse(BaseModel):
    request_id: str
    query: str
    final_answer: str
    sources: list[str]
    errors: list[str]
    total_latency_ms: float
    agents: dict[str, AgentSummary]
    critic_scorecard: Optional[dict] = None


class IngestResponse(BaseModel):
    request_id: str
    total_sources: int
    successful_sources: int
    total_chunks_added: int
    index_size: int
    details: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    index_size: int
    uptime_seconds: float


# ── Helpers ────────────────────────────────────────────────────────────────────

_startup_time = time.time()


def get_orchestrator() -> ResearchOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return _orchestrator


async def emit_event(request_id: str, event: str, data: dict) -> None:
    """Push an SSE event to the queue for a given request_id."""
    if request_id in _event_streams:
        await _event_streams[request_id].put({
            "event": event,
            "data": data,
        })


async def sse_generator(request_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator for Server-Sent Events.
    Yields formatted SSE strings consumed by the frontend.
    """
    queue = _event_streams.get(request_id)
    if queue is None:
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                event_type = event.get("event", "message")
                data = json.dumps(event.get("data", {}))
                yield f"event: {event_type}\ndata: {data}\n\n"

                # Terminal event — close the stream
                if event_type in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping
                yield f"event: ping\ndata: {{}}\n\n"
    finally:
        _event_streams.pop(request_id, None)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status, model info, and index size.
    Used by Docker healthcheck and monitoring.
    """
    orchestrator = get_orchestrator()
    stats = orchestrator.get_index_stats()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=settings.get_llm_model_name(),
        index_size=stats["total_chunks"],
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.post("/research", tags=["Research"])
async def research(request: ResearchRequest):
    """
    Main research endpoint. Runs the full multi-agent pipeline.

    - stream=False: Returns complete JSON response (default)
    - stream=True:  Returns SSE stream with real-time agent updates

    The SSE stream emits these events:
    - agent_start:    {agent: "SearchAgent", query: "..."}
    - agent_complete: {agent: "SearchAgent", latency_ms: 2400}
    - complete:       {final_answer: "...", sources: [...]}
    - error:          {message: "..."}
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info("research_request", request_id=request_id, query=request.query)

    orchestrator = get_orchestrator()

    if request.stream:
        # Set up SSE queue for this request
        _event_streams[request_id] = asyncio.Queue()

        # Run pipeline in background, emitting events
        asyncio.create_task(
            _run_research_with_events(request_id, request.query, orchestrator)
        )

        return StreamingResponse(
            sse_generator(request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    # Non-streaming: run and return complete result
    try:
        result = await asyncio.wait_for(
            orchestrator.run(request.query),
            timeout=settings.agent_timeout_seconds * 4,  # Full pipeline timeout
        )

        return ResearchResponse(
            request_id=request_id,
            query=result["query"],
            final_answer=result["final_answer"],
            sources=result["sources"],
            errors=result["errors"],
            total_latency_ms=result["total_latency_ms"],
            agents={
                k: AgentSummary(**v)
                for k, v in result["agents"].items()
            },
            critic_scorecard=result.get("critic_scorecard"),
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {settings.agent_timeout_seconds * 4}s",
        )
    except Exception as e:
        logger.error("research_endpoint_failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_research_with_events(
    request_id: str,
    query: str,
    orchestrator: ResearchOrchestrator,
) -> None:
    """
    Runs the research pipeline and emits SSE events at each stage.
    Runs as a background task alongside the SSE stream.
    """
    try:
        await emit_event(request_id, "agent_start", {
            "agent": "SearchAgent + ReaderAgent",
            "query": query,
            "message": "Starting parallel research...",
        })

        result = await orchestrator.run(query)

        await emit_event(request_id, "agent_complete", {
            "agent": "SearchAgent",
            "latency_ms": result["agents"]["search"]["latency_ms"],
            "status": result["agents"]["search"]["status"],
        })
        await emit_event(request_id, "agent_complete", {
            "agent": "ReaderAgent",
            "latency_ms": result["agents"]["reader"]["latency_ms"],
            "status": result["agents"]["reader"]["status"],
        })
        await emit_event(request_id, "agent_complete", {
            "agent": "CriticAgent",
            "latency_ms": result["agents"]["critic"]["latency_ms"],
            "status": result["agents"]["critic"]["status"],
            "scorecard": result.get("critic_scorecard"),
        })
        await emit_event(request_id, "complete", {
            "final_answer": result["final_answer"],
            "sources": result["sources"],
            "errors": result["errors"],
            "total_latency_ms": result["total_latency_ms"],
            "critic_scorecard": result.get("critic_scorecard"),
        })

    except Exception as e:
        logger.error("sse_pipeline_failed", request_id=request_id, error=str(e))
        await emit_event(request_id, "error", {"message": str(e)})


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest PDFs or URLs into the FAISS vector store.
    Accepts local file paths and HTTP/HTTPS URLs.
    Duplicate chunks are automatically skipped.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(
        "ingest_request",
        request_id=request_id,
        sources=len(request.sources),
    )

    orchestrator = get_orchestrator()

    try:
        result = await orchestrator.ingest(request.sources)
        return IngestResponse(
            request_id=request_id,
            **result,
        )
    except Exception as e:
        logger.error("ingest_failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/stats", tags=["Documents"])
async def index_stats():
    """Returns current FAISS index statistics."""
    orchestrator = get_orchestrator()
    return orchestrator.get_index_stats()


@app.post("/evaluate", tags=["Evaluation"])
async def run_evaluation(
    request: EvalRequest,
    background_tasks: BackgroundTasks,
):
    """
    Runs the RAGAS benchmark evaluation.
    Runs as a background task — returns immediately with a job ID.
    Poll /evaluate/{job_id} to check status.

    NOTE: Full 20-question benchmark takes ~15 minutes on Groq free tier.
    """
    job_id = str(uuid.uuid4())[:8]
    logger.info("eval_request", job_id=job_id, questions=request.num_questions)

    background_tasks.add_task(
        _run_evaluation_background,
        job_id,
        request.num_questions,
        request.skip_ragas,
    )

    return {
        "job_id": job_id,
        "status": "started",
        "message": (
            f"Evaluation started for {request.num_questions} questions. "
            f"Results will be saved to {settings.eval_output_path}"
        ),
    }


# Tracks background eval jobs: job_id → status dict
_eval_jobs: dict[str, dict] = {}


async def _run_evaluation_background(
    job_id: str,
    num_questions: int,
    skip_ragas: bool,
) -> None:
    """Background task for evaluation — doesn't block the API."""
    from evaluation.ragas_eval import RAGASEvaluator, BENCHMARK_QUESTIONS

    _eval_jobs[job_id] = {"status": "running", "started_at": time.time()}

    try:
        evaluator = RAGASEvaluator(orchestrator=get_orchestrator())
        report = await evaluator.run_benchmark(
            questions=BENCHMARK_QUESTIONS[:num_questions],
            skip_ragas=skip_ragas,
        )
        _eval_jobs[job_id] = {
            "status": "complete",
            "overall_score": report["summary"].get("overall_ragas_score"),
            "report_path": settings.eval_output_path,
            "finished_at": time.time(),
        }
    except Exception as e:
        logger.error("eval_background_failed", job_id=job_id, error=str(e))
        _eval_jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "finished_at": time.time(),
        }


@app.get("/evaluate/{job_id}", tags=["Evaluation"])
async def get_eval_status(job_id: str):
    """Poll evaluation job status."""
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _eval_jobs[job_id]


@app.get("/evaluate/results/latest", tags=["Evaluation"])
async def get_latest_results():
    """Returns the most recent saved benchmark report."""
    import json
    from pathlib import Path
    report_path = Path(settings.eval_output_path)
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark report found. Run /evaluate first."
        )
    with open(report_path) as f:
        return json.load(f)


# ── Global Error Handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )