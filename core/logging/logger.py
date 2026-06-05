"""AegisMem structured logging with JSON output and trace IDs."""
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

# python-json-logger gives structured JSON logs in production but is optional;
# fall back to a plain formatter so the service still runs with minimal deps.
try:
    from pythonjsonlogger import jsonlogger

    _JSON_LOGGING = True
except Exception:  # pragma: no cover - dependency guard
    jsonlogger = None
    _JSON_LOGGING = False

# Context variables for trace propagation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
agent_id_var: ContextVar[str] = ContextVar("agent_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


if _JSON_LOGGING:

    class AegisJsonFormatter(jsonlogger.JsonFormatter):
        """Custom JSON formatter that injects trace context."""

        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            super().add_fields(log_record, record, message_dict)
            log_record["service"] = "aegismem"
            log_record["request_id"] = request_id_var.get() or str(uuid.uuid4())
            log_record["session_id"] = session_id_var.get()
            log_record["agent_id"] = agent_id_var.get()
            log_record["user_id"] = user_id_var.get()
            log_record["level"] = record.levelname
            log_record["logger"] = record.name


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Uses JSON output when ``python-json-logger`` is available, otherwise a
    plain text formatter so the zero-infra path has no hard dependency.
    """
    handler = logging.StreamHandler(sys.stdout)
    if _JSON_LOGGING:
        formatter: logging.Formatter = AegisJsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "neo4j", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def new_request_id() -> str:
    rid = str(uuid.uuid4())
    request_id_var.set(rid)
    return rid


def set_context(
    session_id: str = "",
    agent_id: str = "",
    user_id: str = "",
) -> None:
    if session_id:
        session_id_var.set(session_id)
    if agent_id:
        agent_id_var.set(agent_id)
    if user_id:
        user_id_var.set(user_id)
