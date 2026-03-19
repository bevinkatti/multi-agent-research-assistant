# ============================================================
# logger.py — Structured JSON logging for all agents
#
# WHY THIS MATTERS (interview answer):
# "Every agent call is logged with its name, input hash, output
# summary, latency in ms, and success/failure. This lets us
# replay failures, measure per-agent P99 latency, and audit
# what the system actually did — critical for debugging
# multi-agent systems where failures cascade."
# ============================================================

import structlog
import logging
import sys
import time
from typing import Any


def configure_logging(log_level: str = "INFO") -> None:
    """
    Call once at application startup (in FastAPI lifespan or __main__).
    Sets up structlog with JSON output for production,
    colored console output for development.
    """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # Standard library logging baseline
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_int,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # JSON in production; pretty console in dev
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger. Use at module level: logger = get_logger(__name__)"""
    return structlog.get_logger(name)


class AgentCallLogger:
    """
    Context manager that automatically logs every agent call with:
    - agent_name, input_preview, output_preview
    - latency_ms (measured wall clock)
    - success / error details

    Usage:
        with AgentCallLogger(logger, "SearchAgent", query) as call_log:
            result = await search(query)
            call_log.set_output(result)
    """

    def __init__(
        self,
        logger: Any,
        agent_name: str,
        input_data: Any,
        extra: dict | None = None,
    ):
        self.logger = logger
        self.agent_name = agent_name
        self.input_preview = str(input_data)[:200]   # Truncate for log safety
        self.extra = extra or {}
        self._output_preview: str = ""
        self._start: float = 0.0

    def set_output(self, output: Any) -> None:
        self._output_preview = str(output)[:300]

    def __enter__(self) -> "AgentCallLogger":
        self._start = time.perf_counter()
        self.logger.info(
            "agent_call_started",
            agent=self.agent_name,
            input_preview=self.input_preview,
            **self.extra,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        latency_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type is None:
            self.logger.info(
                "agent_call_success",
                agent=self.agent_name,
                latency_ms=latency_ms,
                output_preview=self._output_preview,
                **self.extra,
            )
        else:
            self.logger.error(
                "agent_call_failed",
                agent=self.agent_name,
                latency_ms=latency_ms,
                error_type=exc_type.__name__,
                error=str(exc_val),
                **self.extra,
            )
        return False  # Do not suppress exceptions
