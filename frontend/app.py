# ============================================================
# frontend/app.py
#
# Streamlit frontend with:
# - Real-time agent activity feed (polling FastAPI)
# - Live agent status indicators (thinking / done / failed)
# - Critic scorecard visualization
# - Document ingestion panel
# - Clean dark research terminal aesthetic
# ============================================================

import time
import requests
import streamlit as st
import os

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Assistant",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Root theme */
:root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --border:   #1f2433;
    --accent:   #4fffb0;
    --accent2:  #7b61ff;
    --warn:     #ff6b6b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Input fields */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,255,176,0.15) !important;
}

/* Buttons */
[data-testid="stButton"] button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    padding: 10px 24px !important;
    transition: opacity 0.15s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Agent status row */
.agent-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 8px;
    font-family: var(--mono);
    font-size: 13px;
}
.agent-row.running  { border-color: var(--accent2); }
.agent-row.success  { border-color: var(--accent); }
.agent-row.failed   { border-color: var(--warn); }
.agent-row.pending  { opacity: 0.45; }

.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot.pending  { background: var(--muted); }
.dot.running  { background: var(--accent2);
                animation: pulse 1s ease-in-out infinite; }
.dot.success  { background: var(--accent); }
.dot.failed   { background: var(--warn); }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

.agent-name  { color: var(--text); font-weight: 700; min-width: 130px; }
.agent-time  { color: var(--muted); font-size: 11px; margin-left: auto; }
.agent-score { color: var(--accent); font-size: 11px; }

/* Score bars */
.score-bar-wrap { margin-bottom: 10px; }
.score-label {
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 11px;
    color: var(--muted); margin-bottom: 4px;
}
.score-label span:last-child { color: var(--accent); }
.score-track {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden;
}
.score-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.6s ease;
}

/* Answer box */
.answer-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 20px 24px;
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
}

/* Source chip */
.source-chip {
    display: inline-block;
    background: var(--border);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    margin: 3px 4px 3px 0;
    word-break: break-all;
}

/* Section headers */
.section-header {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Logo / title */
.logo-row {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 8px;
}
.logo-hex {
    font-size: 28px; color: var(--accent);
    font-family: var(--mono);
}
.logo-title {
    font-family: var(--mono); font-size: 18px;
    font-weight: 700; color: var(--text);
    letter-spacing: -0.02em;
}
.logo-sub {
    font-family: var(--sans); font-size: 12px;
    color: var(--muted); margin-top: 2px;
}

/* Recommendation badge */
.badge {
    display: inline-block; border-radius: 4px;
    padding: 2px 10px; font-family: var(--mono);
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-pass     { background: rgba(79,255,176,0.15); color: var(--accent); }
.badge-retry    { background: rgba(255,107,107,0.15); color: var(--warn); }
.badge-escalate { background: rgba(255,107,107,0.25); color: var(--warn); }

/* Error box */
.error-box {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 6px; padding: 12px 16px;
    font-family: var(--mono); font-size: 12px;
    color: var(--warn);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 5) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, body: dict, timeout: int = 120) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The pipeline is still running."}
    except Exception as e:
        return {"error": str(e)}


def score_color(score: float) -> str:
    if score >= 0.8:
        return "#4fffb0"
    elif score >= 0.6:
        return "#ffd166"
    return "#ff6b6b"


def render_agent_row(name: str, status: str, latency_ms: float = 0, extra: str = ""):
    icons = {"pending": "○", "running": "◉", "success": "●", "failed": "✕", "timeout": "⊘"}
    icon = icons.get(status, "○")
    time_str = f"{latency_ms:.0f}ms" if latency_ms > 0 else ""
    st.markdown(f"""
    <div class="agent-row {status}">
        <span class="dot {status}"></span>
        <span class="agent-name">{icon} {name}</span>
        <span style="color:var(--muted);font-size:12px;flex:1">{extra}</span>
        <span class="agent-time">{time_str}</span>
    </div>
    """, unsafe_allow_html=True)


def render_score_bar(label: str, score: float | None):
    if score is None:
        return
    pct = int(score * 100)
    color = score_color(score)
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-label">
            <span>{label}</span>
            <span style="color:{color}">{score:.3f}</span>
        </div>
        <div class="score-track">
            <div class="score-fill" style="width:{pct}%;background:{color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def check_api_health() -> tuple[bool, dict | None]:
    health = api_get("/health", timeout=3)
    return health is not None, health


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="logo-row">
        <span class="logo-hex">⬡</span>
        <div>
            <div class="logo-title">RESEARCH<br>ASSISTANT</div>
        </div>
    </div>
    <div class="logo-sub">Multi-Agent · RAG · Critic · RAGAS</div>
    <hr style="border-color:var(--border);margin:16px 0">
    """, unsafe_allow_html=True)

    # API Status
    api_ok, health = check_api_health()
    if api_ok:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
            <span style="color:#4fffb0;font-size:10px">●</span>
            <span style="font-family:var(--mono);font-size:12px;color:#64748b">
                API ONLINE · {health.get('index_size',0)} chunks indexed
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
            <span style="color:#ff6b6b;font-size:10px">●</span>
            <span style="font-family:var(--mono);font-size:12px;color:#ff6b6b">
                API OFFLINE — start uvicorn
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        label="page",
        options=["🔍 Research", "📚 Ingest Documents", "📊 Benchmark"],
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:var(--border);margin:16px 0">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:var(--mono);font-size:10px;color:var(--muted)">
        MODEL: {health.get('model','—') if api_ok else '—'}<br>
        UPTIME: {health.get('uptime_seconds',0):.0f}s
    </div>
    """, unsafe_allow_html=True)


# ── Page: Research ─────────────────────────────────────────────────────────────

if page == "🔍 Research":
    st.markdown("""
                <div style="margin-bottom:24px">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
                        <span style="font-family:var(--mono);font-size:28px;color:var(--accent)">⬡</span>
                        <div>
                            <div style="font-family:var(--mono);font-size:22px;font-weight:700;
                                        color:var(--text);letter-spacing:-0.02em">
                                RESEARCH ASSISTANT
                            </div>
                            <div style="font-family:var(--sans);font-size:12px;color:var(--muted);
                                        margin-top:2px">
                                Multi-Agent · RAG · Critic · RAGAS
                            </div>
                        </div>
                    </div>
                    <hr style="border-color:var(--border);margin-top:12px">
                </div>
                <div class="section-header">Research Query</div>
                """, unsafe_allow_html=True)

    query = st.text_area(
        label="query",
        placeholder="Ask a research question — e.g. 'How does the attention mechanism work in transformers?'",
        height=100,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("⬡  RUN", disabled=not api_ok, use_container_width=True)

    if run_btn and query.strip():
        st.markdown('<hr style="border-color:var(--border);margin:16px 0">', unsafe_allow_html=True)

        # Agent activity panel
        st.markdown('<div class="section-header">Agent Pipeline</div>', unsafe_allow_html=True)

        agent_placeholder = st.empty()
        answer_placeholder = st.empty()
        critic_placeholder = st.empty()
        sources_placeholder = st.empty()

        # Show "running" state
        with agent_placeholder.container():
            render_agent_row("SearchAgent", "running", extra="querying web...")
            render_agent_row("ReaderAgent", "running", extra="searching FAISS...")
            render_agent_row("CriticAgent", "pending")
            render_agent_row("Synthesizer", "pending")

        start_time = time.time()

        with st.spinner(""):
            result = api_post("/research", {"query": query, "stream": False}, timeout=180)

        elapsed = time.time() - start_time

        if result and "error" not in result:
            agents = result.get("agents", {})
            scorecard = result.get("critic_scorecard", {})

            # Update agent rows with real results
            with agent_placeholder.container():
                for name, key in [  
                    ("SearchAgent", "search"),
                    ("ReaderAgent", "reader"),
                    ("CriticAgent", "critic"),
                ]:
                    agent_data = agents.get(key, {})
                    status = agent_data.get("status", "pending")
                    latency = agent_data.get("latency_ms", 0)
                    err = agent_data.get("error") or ""

                    # ReaderAgent: show friendly message instead of red error
                    # when no documents are ingested yet
                    if key == "reader" and status == "failed":
                        no_docs = (
                            "No documents" in err
                            or "not ingested" in err.lower()
                            or "no documents" in err.lower()
                        )
                        if no_docs:
                            status = "success"
                            err = "no docs yet — use Ingest page to add PDFs"

                    render_agent_row(name, status, latency, err)

                # Synthesizer
                render_agent_row(
                    "Synthesizer", "success",
                    latency_ms=result.get("total_latency_ms", 0),
                    extra="final answer ready",
                )

            # Final answer
            answer_placeholder.markdown(f"""
            <div class="section-header" style="margin-top:24px">Final Answer
                <span style="font-size:10px;margin-left:12px;color:var(--accent)">
                    {result.get('total_latency_ms',0):.0f}ms total
                </span>
            </div>
            <div class="answer-box">{result.get('final_answer','')}</div>
            """, unsafe_allow_html=True)

            # Critic scorecard
            if scorecard:
                rec = scorecard.get("recommendation", "pass")
                badge_class = f"badge-{rec}"
                overall = scorecard.get("overall_score", 0)
                dims = scorecard.get("dimensions", {})

                critic_placeholder.markdown(f"""
                <div style="margin-top:20px">
                <div class="section-header">Critic Scorecard
                    <span class="badge {badge_class}" style="margin-left:10px">{rec}</span>
                    <span style="margin-left:10px;color:{score_color(overall)};
                           font-family:var(--mono);font-size:11px">
                        {overall:.3f} overall
                    </span>
                </div>
                </div>
                """, unsafe_allow_html=True)

                with critic_placeholder.container():
                    st.markdown(f"""
                    <div class="section-header">Critic Scorecard
                        <span class="badge {badge_class}" style="margin-left:10px">{rec}</span>
                        <span style="margin-left:10px;color:{score_color(overall)};
                               font-family:var(--mono);font-size:11px">
                            {overall:.3f} overall
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    dim_list = list(dims.items())
                    for i, (dim, val) in enumerate(dim_list):
                        with col_a if i % 2 == 0 else col_b:
                            render_score_bar(dim.replace("_", " ").title(), val.get("score"))

                    summary = scorecard.get("summary", "")
                    if summary:
                        st.markdown(f"""
                        <div style="font-size:13px;color:var(--muted);
                                    font-style:italic;margin-top:8px;
                                    padding:12px;background:var(--bg);
                                    border-radius:6px;border:1px solid var(--border)">
                            {summary}
                        </div>
                        """, unsafe_allow_html=True)

            # Sources
            sources = result.get("sources", [])
            if sources:
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
                sources_placeholder.markdown(f"""
                <div style="margin-top:16px">
                <div class="section-header">Sources ({len(sources)})</div>
                {chips}
                </div>
                """, unsafe_allow_html=True)

            # Errors
            errors = result.get("errors", [])
            if errors:
                for err in errors:
                    st.markdown(f'<div class="error-box">⚠ {err}</div>',
                                unsafe_allow_html=True)

        else:
            err_msg = result.get("error", "Unknown error") if result else "API returned no response"
            with agent_placeholder.container():
                render_agent_row("SearchAgent", "failed", extra=err_msg)
                render_agent_row("ReaderAgent", "failed")
                render_agent_row("CriticAgent", "failed")
                render_agent_row("Synthesizer", "failed")
            st.markdown(f'<div class="error-box">⚠ {err_msg}</div>',
                        unsafe_allow_html=True)

    elif run_btn and not query.strip():
        st.warning("Please enter a research question.")


# ── Page: Ingest ───────────────────────────────────────────────────────────────

elif page == "📚 Ingest Documents":
    st.markdown('<div class="section-header">Ingest Documents</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:13px;color:var(--muted);margin-bottom:16px">
        Add PDFs or web articles to the knowledge base.
        Paste one URL or file path per line.
    </div>
    """, unsafe_allow_html=True)

    sources_input = st.text_area(
        label="sources",
        placeholder="https://arxiv.org/pdf/1706.03762\nhttps://lilianweng.github.io/posts/2023-06-23-agent/",
        height=150,
        label_visibility="collapsed",
    )

    if st.button("⬡  INGEST", disabled=not api_ok, use_container_width=False):
        sources = [s.strip() for s in sources_input.strip().splitlines() if s.strip()]
        if not sources:
            st.warning("Please enter at least one URL or file path.")
        else:
            with st.spinner(f"Ingesting {len(sources)} source(s)..."):
                result = api_post("/ingest", {"sources": sources}, timeout=120)

            if result and "error" not in result:
                st.markdown(f"""
                <div class="card" style="border-color:var(--accent)">
                    <div style="font-family:var(--mono);font-size:13px">
                        <span style="color:var(--accent)">✓ Ingest complete</span><br><br>
                        <span style="color:var(--muted)">Sources processed:</span>
                        <span style="color:var(--text)"> {result.get('successful_sources',0)}/{result.get('total_sources',0)}</span><br>
                        <span style="color:var(--muted)">Chunks added:</span>
                        <span style="color:var(--text)"> {result.get('total_chunks_added',0)}</span><br>
                        <span style="color:var(--muted)">Index size:</span>
                        <span style="color:var(--text)"> {result.get('index_size',0)} chunks</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                for detail in result.get("details", []):
                    status_color = "#4fffb0" if detail.get("success") else "#ff6b6b"
                    status_text = "✓" if detail.get("success") else "✗"
                    st.markdown(f"""
                    <div class="agent-row {'success' if detail.get('success') else 'failed'}">
                        <span class="dot {'success' if detail.get('success') else 'failed'}"></span>
                        <span style="font-size:12px;word-break:break-all;color:var(--muted)">
                            {status_text} {detail.get('source','')[:80]}
                        </span>
                        <span class="agent-time">
                            {detail.get('chunks_added',0)} added
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                err = result.get("error", "Unknown error") if result else "No response"
                st.markdown(f'<div class="error-box">⚠ {err}</div>',
                            unsafe_allow_html=True)

    # Current index stats
    st.markdown('<hr style="border-color:var(--border);margin:24px 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Index Status</div>', unsafe_allow_html=True)

    stats = api_get("/index/stats")
    if stats:
        col1, col2, col3 = st.columns(3)
        for col, label, val in [
            (col1, "Total Chunks", stats.get("total_chunks", 0)),
            (col2, "Unique Sources", stats.get("unique_sources", 0)),
            (col3, "Embedding Dim", stats.get("embedding_dim", 0)),
        ]:
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center">
                    <div style="font-family:var(--mono);font-size:24px;
                                color:var(--accent);font-weight:700">{val}</div>
                    <div style="font-size:12px;color:var(--muted);margin-top:4px">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Page: Benchmark ────────────────────────────────────────────────────────────

elif page == "📊 Benchmark":
    st.markdown('<div class="section-header">RAGAS Benchmark</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        n_questions = st.slider("Questions to evaluate", 1, 20, 5)
    with col2:
        skip_ragas = st.checkbox("Skip RAGAS scoring", value=False)

    if st.button("⬡  RUN BENCHMARK", disabled=not api_ok):
        with st.spinner("Starting benchmark job..."):
            result = api_post("/evaluate", {
                "num_questions": n_questions,
                "skip_ragas": skip_ragas,
            }, timeout=30)

        if result and "job_id" in result:
            job_id = result["job_id"]
            st.markdown(f"""
            <div class="card" style="border-color:var(--accent2)">
                <div style="font-family:var(--mono);font-size:13px">
                    <span style="color:var(--accent2)">⬡ Job started</span><br><br>
                    <span style="color:var(--muted)">Job ID:</span>
                    <span style="color:var(--text)"> {job_id}</span><br>
                    <span style="color:var(--muted)">Questions:</span>
                    <span style="color:var(--text)"> {n_questions}</span><br>
                    <span style="color:var(--muted)">Est. time:</span>
                    <span style="color:var(--text)"> ~{n_questions * 45}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state["last_job_id"] = job_id

    # Poll job status
    if "last_job_id" in st.session_state:
        job_id = st.session_state["last_job_id"]
        status = api_get(f"/evaluate/{job_id}")
        if status:
            s = status.get("status", "unknown")
            if s == "complete":
                sc = status.get("overall_score")
                st.success(f"✓ Complete — Overall RAGAS score: {sc:.3f}" if sc else "✓ Complete")
            elif s == "running":
                st.info("⬡ Running...")
            elif s == "failed":
                st.error(f"✗ Failed: {status.get('error')}")

    # Latest results
    st.markdown('<hr style="border-color:var(--border);margin:24px 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Latest Results</div>', unsafe_allow_html=True)

    if st.button("⬡  LOAD RESULTS"):
        report = api_get("/evaluate/results/latest", timeout=10)
        if report:
            summary = report.get("summary", {})
            col1, col2, col3, col4 = st.columns(4)
            for col, label, key in [
                (col1, "Faithfulness",     "faithfulness"),
                (col2, "Ans. Relevancy",   "answer_relevancy"),
                (col3, "Context Recall",   "context_recall"),
                (col4, "Overall",          "overall_ragas_score"),
            ]:
                with col:
                    val = summary.get(key)
                    display = f"{val:.3f}" if val is not None else "N/A"
                    color = score_color(val) if val else "#64748b"
                    st.markdown(f"""
                    <div class="card" style="text-align:center">
                        <div style="font-family:var(--mono);font-size:24px;
                                    color:{color};font-weight:700">{display}</div>
                        <div style="font-size:11px;color:var(--muted);margin-top:4px">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Category breakdown
            breakdown = report.get("category_breakdown", {})
            if breakdown:
                st.markdown('<div class="section-header" style="margin-top:16px">Category Breakdown</div>',
                            unsafe_allow_html=True)
                for cat, stats in breakdown.items():
                    render_score_bar(
                        f"{cat.replace('_',' ').upper()} ({stats.get('count',0)} q)",
                        stats.get("overall"),
                    )
        else:
            st.info("No results yet. Run a benchmark first.")