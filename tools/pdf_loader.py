# ============================================================
# tools/pdf_loader.py
#
# Document ingestion pipeline with:
# - PDF loading from local file path or remote URL
# - Plain URL content scraping (for web articles)
# - Text chunking with configurable size and overlap
# - Typed document schema (Document, ChunkResult)
# - Full structured logging with latency
# ============================================================

import asyncio
import hashlib
import io
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
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

logger = get_logger(__name__)


# ── Typed Schema ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    A single loaded document before chunking.
    source: file path or URL string
    """
    content: str
    source: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        # Deterministic ID based on source — prevents duplicate ingestion
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.source.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    text: str
    source: str
    chunk_index: int
    doc_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_chunk_{self.chunk_index}"


@dataclass
class LoadResult:
    """Result returned to the agent layer after loading + chunking."""
    source: str
    chunks: list[Chunk] = field(default_factory=list)
    total_chars: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ── Text Chunking ──────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits raw text into overlapping chunks.

    WHY OVERLAP MATTERS (interview answer):
    "If a key fact sits at the boundary of two chunks, a model
    seeing only one chunk may miss it entirely. Overlap ensures
    boundary content appears in both adjacent chunks, so retrieval
    is robust to where the splitter cuts."
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, source: str, doc_id: str) -> list[Chunk]:
        """
        Splits text into overlapping chunks using a sliding window.

        Strategy:
        1. Clean the text (normalize whitespace)
        2. Try to split on sentence boundaries first
        3. Fall back to character sliding window
        """
        text = self._clean_text(text)
        if not text.strip():
            return []

        raw_chunks = self._sliding_window_split(text)

        return [
            Chunk(
                text=chunk,
                source=source,
                chunk_index=i,
                doc_id=doc_id,
                metadata={"source": source, "chunk_index": i, "doc_id": doc_id},
            )
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()   # Skip empty chunks
        ]

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and remove junk characters."""
        # Collapse multiple newlines into two
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Remove non-printable characters (except newlines/tabs)
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', '', text)
        return text.strip()

    def _sliding_window_split(self, text: str) -> list[str]:
        """
        Core sliding window algorithm.
        Window moves forward by (chunk_size - chunk_overlap) each step.
        """
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to end at a sentence boundary (. ! ?) to avoid mid-sentence cuts
            if end < len(text):
                last_punct = max(
                    chunk.rfind('. '),
                    chunk.rfind('! '),
                    chunk.rfind('? '),
                    chunk.rfind('\n'),
                )
                if last_punct > self.chunk_size // 2:
                    # Only snap to boundary if it's in the second half of the chunk
                    chunk = chunk[:last_punct + 1]

            chunks.append(chunk)
            start += step

        return chunks


# ── PDF & URL Loaders ──────────────────────────────────────────────────────────

class PDFLoaderTool:
    """
    Loads and chunks documents from:
    - Local PDF file paths
    - Remote PDF URLs
    - Web article URLs (scrapes visible text via BeautifulSoup)

    Usage:
        loader = PDFLoaderTool()
        result = await loader.load("path/to/paper.pdf")
        result = await loader.load("https://arxiv.org/pdf/2301.00234")
        result = await loader.load("https://example.com/article")
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; ResearchAssistant/1.0; "
                "+https://github.com/your-repo)"
            )
        })
        logger.info(
            "pdf_loader_initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def load(self, source: str) -> LoadResult:
        """
        Main entry point. Detects source type and routes accordingly.
        Always async-safe — blocking I/O runs in thread pool.
        """
        with AgentCallLogger(logger, "PDFLoaderTool", source) as call_log:
            try:
                if self._is_url(source):
                    doc = await asyncio.to_thread(self._load_url, source)
                else:
                    doc = await asyncio.to_thread(self._load_local_pdf, source)

                chunks = self.chunker.split(doc.content, doc.source, doc.doc_id)

                result = LoadResult(
                    source=source,
                    chunks=chunks,
                    total_chars=len(doc.content),
                    success=True,
                )
                call_log.set_output(
                    f"Loaded {result.chunk_count} chunks from {result.total_chars} chars"
                )
                logger.info(
                    "document_loaded",
                    source=source,
                    chunks=result.chunk_count,
                    total_chars=result.total_chars,
                )
                return result

            except FileNotFoundError:
                error = f"File not found: {source}"
                logger.error("pdf_load_failed", source=source, error=error)
                return LoadResult(source=source, success=False, error=error)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                logger.error("pdf_load_failed", source=source, error=error)
                return LoadResult(source=source, success=False, error=error)

    async def load_multiple(self, sources: list[str]) -> list[LoadResult]:
        """Load multiple documents concurrently."""
        tasks = [self.load(source) for source in sources]
        return await asyncio.gather(*tasks)

    # ── Private loaders ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    def _load_url(self, url: str) -> Document:
        """
        Loads content from a URL. Routes to PDF or HTML parser
        based on Content-Type header.
        """
        response = self._session.get(url, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return self._parse_pdf_bytes(response.content, source=url)
        else:
            return self._parse_html(response.text, source=url)

    def _load_local_pdf(self, path: str) -> Document:
        """Loads a PDF from local filesystem."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected .pdf file, got: {file_path.suffix}")

        with open(file_path, "rb") as f:
            return self._parse_pdf_bytes(f.read(), source=str(file_path))

    def _parse_pdf_bytes(self, pdf_bytes: bytes, source: str) -> Document:
        """Extract text from PDF bytes using pypdf."""
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text}")

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            raise ValueError(
                "PDF appears to be scanned/image-only. "
                "OCR is required but not supported in this version."
            )

        return Document(
            content=full_text,
            source=source,
            metadata={
                "type": "pdf",
                "pages": len(reader.pages),
                "source": source,
            },
        )

    def _parse_html(self, html: str, source: str) -> Document:
        """
        Extracts readable text from HTML using BeautifulSoup.
        Removes nav, footer, scripts, and ads — keeps main content.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "advertisement", "noscript"]):
            tag.decompose()

        # Try to find the main content area first
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.body
        )

        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        return Document(
            content=text,
            source=source,
            metadata={"type": "html", "source": source},
        )

    @staticmethod
    def _is_url(source: str) -> bool:
        """Returns True if source looks like an HTTP/HTTPS URL."""
        try:
            result = urlparse(source)
            return result.scheme in ("http", "https")
        except ValueError:
            return False