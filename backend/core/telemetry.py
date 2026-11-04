"""OpenTelemetry tracing helpers.

Spans are created for graph nodes, agent invocations, model calls and every MCP
tool call. Without an exporter configured the SDK keeps spans in-process (and
the platform mirrors the important ones into ``agent_traces``), so tracing has
no external dependency.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

_provider: TracerProvider | None = None


class _InMemoryExporter(SpanExporter):
    """Keeps the most recent spans for diagnostics endpoints and tests."""

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.spans: list[dict[str, Any]] = []

    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            self.spans.append(
                {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "duration_ms": round((span.end_time - span.start_time) / 1e6, 2),
                    "attributes": dict(span.attributes or {}),
                    "status": span.status.status_code.name,
                }
            )
        if len(self.spans) > self.capacity:
            del self.spans[: len(self.spans) - self.capacity]
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        self.spans.clear()


memory_exporter = _InMemoryExporter()


def configure_tracing(service_name: str) -> None:
    global _provider
    if _provider is not None:
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    _provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    trace.set_tracer_provider(_provider)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    tracer = get_tracer("aitest")
    clean = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in attributes.items() if v is not None}
    start = time.perf_counter()
    with tracer.start_as_current_span(name, attributes=clean) as current:
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            current.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)[:200]))
            raise
        finally:
            current.set_attribute("duration_ms", round((time.perf_counter() - start) * 1000, 2))
