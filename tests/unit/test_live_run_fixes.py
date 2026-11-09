"""Regressions from long live runs (HealthLink, Quintiles): crashes, budgets, token accounting."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.core.config import ExecutionBudget, Settings
from backend.core.exceptions import ModelUnavailable
from backend.execution.runtime import scaled_budget
from backend.harness.budget_manager import BudgetTracker, _current, bind_tracker, token_usage
from backend.harness.model_gateway import ModelGateway, ModelTier, ModelUsage, _token_details
from backend.memory.memory_manager import confidence_from
from backend.models.entities import Base, Project


async def test_timestamps_come_back_timezone_aware_from_sqlite(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'tz.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine)() as session:
        session.add(Project(id="p1", organization_id="o1", name="P"))
        await session.commit()
    async with async_sessionmaker(engine)() as session:
        row = (await session.execute(select(Project))).scalar_one()
        assert row.created_at.tzinfo is not None
        # the subtraction that crashed resolve_target_element mid-run
        assert (dt.datetime.now(dt.UTC) - row.created_at).total_seconds() >= 0
    await engine.dispose()


def test_memory_confidence_accepts_naive_and_aware_timestamps():
    naive, aware = dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    assert confidence_from(4, 1, naive) == confidence_from(4, 1, aware)
    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=3)  # clock skew between machines
    assert confidence_from(4, 1, future) == confidence_from(4, 1, None)


def test_budgets_grow_with_the_test():
    base = ExecutionBudget()
    short = scaled_budget(base, 10)
    assert short.max_tool_calls == base.max_tool_calls and short.max_llm_calls == base.max_llm_calls
    long = scaled_budget(base, 191)  # the Quintiles registration form
    assert long.max_tool_calls >= 191 * 30 > 1500
    assert long.execution_deadline_seconds >= 191 * 25
    assert long.max_llm_calls >= 191 * 6 and long.max_steps >= 241


class Echo(BaseModel):
    ok: bool


class FakeProvider:
    name = "fake"
    available = True

    async def structured(self, **_: object) -> tuple[Echo, ModelUsage]:
        return Echo(ok=True), ModelUsage("fake", "m1", 1000, 200, 12.0, 0.01, cached_input_tokens=600,
                                         reasoning_tokens=50)


async def test_model_budget_exhaustion_degrades_instead_of_aborting():
    settings = Settings(llm_provider="openai_compatible", openai_base_url="http://x.test")
    gateway = ModelGateway(settings, providers={"openai_compatible": FakeProvider()})
    tracker = BudgetTracker("exec_x", ExecutionBudget(max_llm_calls=1))
    token = bind_tracker(tracker)
    try:
        await gateway.structured(tier=ModelTier.REASONING, system="s", prompt="p", output_model=Echo,
                                 agent="grounder")
        with pytest.raises(ModelUnavailable, match="continuing without the model"):
            await gateway.structured(tier=ModelTier.REASONING, system="s", prompt="p", output_model=Echo,
                                     agent="grounder")
    finally:
        _current.reset(token)
    usage = token_usage(tracker)
    assert usage["calls"] == 1 and usage["cached_tokens"] == 600 and usage["reasoning_tokens"] == 50
    assert usage["by_agent"]["grounder"]["input_tokens"] == 1000


def test_cached_and_reasoning_tokens_are_read_from_every_sdk_shape():
    responses = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4},
                 "output_tokens_details": {"reasoning_tokens": 3}}
    chat = {"prompt_tokens_details": {"cached_tokens": 5}, "completion_tokens_details": {"reasoning_tokens": 2}}
    assert _token_details(responses) == (4, 3)
    assert _token_details(chat) == (5, 2)
    assert _token_details({"cache_read_input_tokens": 7}) == (7, 0)
    assert _token_details(None) == (0, 0)


# --------------------------------------------------------------------------- spatial grounding

from backend.agents.grounding.element_grounder import ElementGrounder, GroundingContext  # noqa: E402
from backend.agents.planner.nl_parser import ParseContext, StepParser  # noqa: E402
from backend.dsl.observation import BoundingBox, InventoryElement, Observation  # noqa: E402


def _el(eid: str, role: str, name: str, x: float, y: float, w: float = 80, h: float = 22, **kw: object) -> InventoryElement:
    return InventoryElement(element_id=eid, ref=eid, role=role, name=name, visible=True,
                            bbox=BoundingBox(x=x, y=y, width=w, height=h), **kw)


def _ground(line: str, elements: list[InventoryElement]) -> str | None:
    step = StepParser().parse(line, ParseContext()).steps[0]
    decision = ElementGrounder().ground(step, Observation(observation_id="o", elements=elements), GroundingContext())
    return decision.element_id


def test_link_next_to_a_label_beats_the_same_link_elsewhere():
    page = [_el("lbl_docs", "generic", "Upload Documents", 40, 300, 160), _el("add_docs", "link", "Add more", 220, 300),
            _el("lbl_notes", "generic", "Notes", 40, 500, 160), _el("add_notes", "link", "Add more", 220, 500)]
    assert _ground('Click on the "Add more" link next to "Upload Documents"', page) == "add_docs"
    assert _ground('Click on the "Add more" link next to "Notes"', page) == "add_notes"


def test_first_row_under_a_column_header():
    page = [_el("h_cycle", "columnheader", "Cycle No", 40, 200, 120), _el("h_date", "columnheader", "Date", 200, 200, 120),
            _el("r1c1", "cell", "1", 40, 230, 120, clickable_hint=True), _el("r1c2", "cell", "02 Jul 2026", 200, 230, 120),
            _el("r2c1", "cell", "2", 40, 260, 120, clickable_hint=True)]
    assert _ground('Click on the first row under "Cycle No" in the listing', page) == "r1c1"
    assert _ground('Click on the second row under "Cycle No"', page) == "r2c1"


def test_second_of_a_from_to_pair_of_dropdowns():
    page = [_el("lbl", "generic", "Data period", 20, 100, 90),
            _el("from", "combobox", "Data period", 120, 100, 90, tag="select"),
            _el("sep", "generic", "～", 215, 100, 14),
            _el("to", "combobox", "", 235, 100, 90, tag="select"),
            _el("other", "combobox", "Master Year Month", 120, 60, 90, tag="select")]
    assert _ground('Click the first "Data period" dropdown', page) == "from"
    assert _ground('Click the second "Data period" dropdown', page) == "to"


def test_icon_to_the_right_of_a_name_and_three_dots():
    page = [_el("n1", "cell", "John Smith", 40, 100, 160), _el("e1", "button", "edit", 400, 100, 24),
            _el("n2", "cell", "Mary Jones", 40, 140, 160), _el("e2", "button", "edit", 400, 140, 24),
            _el("dots", "button", "More options", 600, 20, 24), _el("del", "button", "Delete", 640, 20, 24)]
    assert _ground('Click the edit button to the right of "Mary Jones"', page) == "e2"
    assert _ground("Click on the three dots icon", page) == "dots"


def test_button_below_a_field():
    page = [_el("search", "textbox", "Search", 100, 100, 300, editable=True), _el("go", "button", "Go", 120, 140),
            _el("reset", "button", "Reset", 600, 400)]
    assert _ground('Click on the button below the "Search" field', page) == "go"


# --------------------------------------------------------------------------- search not submitted


def _runtime_with_search(lines: list[str]):
    from backend.dsl.models import TestSpec
    from backend.execution.runtime import ExecutionRequest, ExecutionRuntime

    ctx, steps = ParseContext(), []
    for line in lines:  # the analyzer feeds the parser one line at a time
        steps.extend(StepParser().parse(line, ctx).steps)
    for number, step in enumerate(steps, start=1):
        step.index, step.id = number, f"step_{number:03d}"
    spec = TestSpec(test_name="t", base_url="http://demo.test", steps=steps)
    request = ExecutionRequest(execution_id="exec_s", organization_id="o", project_id="p", application_id="a",
                               test_case_id="tc", test_case_version=1, spec=spec)
    return ExecutionRuntime(request, settings=Settings()), steps


def _not_found():
    from backend.agents.recovery.failure_classifier import FailureAnalysis
    from backend.dsl.results import FailureCategory

    return FailureAnalysis(category=FailureCategory.ELEMENT_NOT_FOUND, root_cause="no match", healable=True,
                           product_failure=False, blocker=False)


async def test_click_after_an_unsubmitted_search_tries_enter_first():
    from backend.dsl.results import StepResult, StepStatus

    runtime, steps = _runtime_with_search(['Enter "Thomas Kutty" in the "Search patient" field',
                                           'Click on "Thomas Kutty"'])
    search, click = steps
    runtime.ensure_work(search).element = _el("q", "textbox", "Search patient", 100, 100, 300, editable=True)
    runtime.results[search.id] = StepResult(step_id=search.id, index=search.index, original_text=search.original_text,
                                            action=search.action.value)
    await runtime.finish_step(search, StepStatus.PASSED)
    plan = runtime.plan_recovery(click, _not_found())
    assert plan is not None and plan["strategy"] == "wait_for_stability"
    runtime.ensure_work(click).tried_strategies.append("wait_for_stability")
    plan = runtime.plan_recovery(click, _not_found())
    assert plan is not None and plan["strategy"] == "submit_search"


def test_no_search_submit_without_a_search_fill_and_model_hint_goes_first():
    runtime, steps = _runtime_with_search(['Click on "Load older records"'])
    work = runtime.ensure_work(steps[0])
    work.tried_strategies.append("wait_for_stability")
    plan = runtime.plan_recovery(steps[0], _not_found())
    assert plan is not None and plan["strategy"] == "reground_without_memory"
    work.model_hint = "needs_scroll"
    plan = runtime.plan_recovery(steps[0], _not_found())
    assert plan is not None and plan["strategy"] == "scroll_to_reveal"
