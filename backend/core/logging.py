"""Structured, redaction-aware logging.

Logs are emitted as JSON lines (OpenTelemetry-compatible field names) and every
record passes through the sensitive-data redactor before it is formatted, so a
secret that accidentally reaches a log call is still masked.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from typing import Any

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("log_context", default={})

_REDACTOR = None  # lazily bound to avoid import cycles with governance.redaction


def bind_log_context(**fields: Any) -> contextvars.Token:
    merged = {**_context.get(), **{k: v for k, v in fields.items() if v is not None}}
    return _context.set(merged)


def reset_log_context(token: contextvars.Token) -> None:
    _context.reset(token)


def current_log_context() -> dict[str, Any]:
    return dict(_context.get())


def _redact(value: Any) -> Any:
    global _REDACTOR
    if _REDACTOR is None:
        from backend.agents.governance.redaction import redact_value

        _REDACTOR = redact_value
    return _REDACTOR(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(current_log_context())
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{time.strftime('%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        fields = {**current_log_context(), **(getattr(record, "fields", None) or {})}
        if fields:
            base += " " + json.dumps(_redact(fields), default=str)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return str(_redact(base))


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter accepting keyword fields: ``log.info("msg", step_id=...)``."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        fields = {k: kwargs.pop(k) for k in list(kwargs) if k not in ("exc_info", "stack_info", "stacklevel", "extra")}
        extra = kwargs.setdefault("extra", {})
        extra["fields"] = {**extra.get("fields", {}), **fields}
        return msg, kwargs


_configured = False


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    global _configured
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "asyncio", "mcp", "urllib3", "anthropic", "httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name), {})
