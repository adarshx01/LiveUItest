"""Per-execution budgets: LLM calls/tokens/cost, tool calls, wall-clock deadline.

The tracker is bound to the running execution through a context variable so
deep call sites (model gateway, MCP adapter) enforce limits without plumbing.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

from backend.core.config import ExecutionBudget, get_settings
from backend.core.exceptions import BudgetExceeded, ExecutionCancelled


@dataclass
class BudgetTracker:
    execution_id: str
    limits: ExecutionBudget
    started_at: float = field(default_factory=time.monotonic)
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0
    model_latency_ms: float = 0.0
    tool_calls: int = 0
    tool_latency_ms: float = 0.0
    cancelled: bool = False
    paused_seconds: float = 0.0
    _pause_started: float | None = None
    agent_calls: dict[str, int] = field(default_factory=dict)
    llm_cached_tokens: int = 0
    llm_reasoning_tokens: int = 0
    llm_failed_calls: int = 0
    # usage by agent and by model: calls, input/output/cached/reasoning tokens, cost, latency
    llm_by_agent: dict[str, dict[str, float]] = field(default_factory=dict)
    llm_by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    # ---------------------------------------------------------------- checks
    def elapsed(self) -> float:
        paused = self.paused_seconds + ((time.monotonic() - self._pause_started) if self._pause_started else 0.0)
        return time.monotonic() - self.started_at - paused

    def remaining_seconds(self) -> float:
        return self.limits.execution_deadline_seconds - self.elapsed()

    def check_alive(self) -> None:
        if self.cancelled:
            raise ExecutionCancelled("Execution was cancelled", details={"execution_id": self.execution_id})
        if self.remaining_seconds() <= 0:
            raise BudgetExceeded("Execution deadline exceeded",
                                 details={"deadline_seconds": self.limits.execution_deadline_seconds})

    def check_llm(self, estimated_tokens: int = 0) -> None:
        self.check_alive()
        if self.llm_calls >= self.limits.max_llm_calls:
            raise BudgetExceeded("LLM call budget exhausted", details={"max_llm_calls": self.limits.max_llm_calls})
        if self.llm_input_tokens + self.llm_output_tokens + estimated_tokens > self.limits.max_llm_tokens:
            raise BudgetExceeded("LLM token budget exhausted", details={"max_llm_tokens": self.limits.max_llm_tokens})

    def check_tool(self) -> None:
        self.check_alive()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("Tool call budget exhausted", details={"max_tool_calls": self.limits.max_tool_calls})

    # ---------------------------------------------------------------- records
    def record_llm(self, input_tokens: int, output_tokens: int, latency_ms: float, cost_usd: float, *,
                   agent: str = "unknown", model: str = "unknown", tier: str = "", cached_tokens: int = 0,
                   reasoning_tokens: int = 0) -> None:
        self.llm_calls += 1
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.model_latency_ms += latency_ms
        self.llm_cost_usd += cost_usd
        self.llm_cached_tokens += cached_tokens
        self.llm_reasoning_tokens += reasoning_tokens
        for table, key in ((self.llm_by_agent, agent), (self.llm_by_model, model)):
            row = table.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                                         "reasoning_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0, "failed": 0})
            row["calls"] += 1
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["cached_tokens"] += cached_tokens
            row["reasoning_tokens"] += reasoning_tokens
            row["cost_usd"] = round(row["cost_usd"] + cost_usd, 6)
            row["latency_ms"] = round(row["latency_ms"] + latency_ms, 1)

    def record_llm_failure(self, *, agent: str = "unknown", model: str = "unknown") -> None:
        """A failed attempt costs a call slot (it may have been billed) but adds no tokens."""

        self.llm_calls += 1
        self.llm_failed_calls += 1
        for table, key in ((self.llm_by_agent, agent), (self.llm_by_model, model)):
            row = table.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                                         "reasoning_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0, "failed": 0})
            row["failed"] += 1

    def record_tool(self, latency_ms: float) -> None:
        self.tool_calls += 1
        self.tool_latency_ms += latency_ms

    def record_agent(self, name: str) -> None:
        self.agent_calls[name] = self.agent_calls.get(name, 0) + 1

    def pause(self) -> None:
        if self._pause_started is None:
            self._pause_started = time.monotonic()

    def resume(self) -> None:
        if self._pause_started is not None:
            self.paused_seconds += time.monotonic() - self._pause_started
            self._pause_started = None

    def snapshot(self) -> dict[str, float | int]:
        return {
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
            "llm_cached_tokens": self.llm_cached_tokens,
            "llm_reasoning_tokens": self.llm_reasoning_tokens,
            "llm_failed_calls": self.llm_failed_calls,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed(), 2),
            "remaining_seconds": round(self.remaining_seconds(), 2),
            "model_latency_ms": round(self.model_latency_ms, 1),
            "tool_latency_ms": round(self.tool_latency_ms, 1),
        }


def token_usage(tracker: BudgetTracker) -> dict[str, object]:
    """Run-level model usage for reports and the CLI summary."""

    return {"calls": tracker.llm_calls, "failed_calls": tracker.llm_failed_calls,
            "input_tokens": tracker.llm_input_tokens, "output_tokens": tracker.llm_output_tokens,
            "cached_tokens": tracker.llm_cached_tokens, "reasoning_tokens": tracker.llm_reasoning_tokens,
            "total_tokens": tracker.llm_input_tokens + tracker.llm_output_tokens,
            "cost_usd": round(tracker.llm_cost_usd, 6), "model_latency_ms": round(tracker.model_latency_ms, 1),
            "by_agent": {k: dict(v) for k, v in sorted(tracker.llm_by_agent.items())},
            "by_model": {k: dict(v) for k, v in sorted(tracker.llm_by_model.items())}}


_current: contextvars.ContextVar[BudgetTracker | None] = contextvars.ContextVar("budget_tracker", default=None)


def new_tracker(execution_id: str, limits: ExecutionBudget | None = None) -> BudgetTracker:
    return BudgetTracker(execution_id=execution_id, limits=limits or get_settings().budget)


def bind_tracker(tracker: BudgetTracker | None) -> contextvars.Token:
    return _current.set(tracker)


def current_tracker() -> BudgetTracker | None:
    return _current.get()
