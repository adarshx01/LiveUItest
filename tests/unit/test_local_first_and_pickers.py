"""Local-first infrastructure, secret handling and the custom date / time picker logic."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import Column, MetaData, String, Table, inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.agents.governance.redaction import MASK, redact_value
from backend.agents.grounding.element_grounder import name_score
from backend.agents.planner.nl_parser import _looks_like_data, build_target
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.planner.test_ingestion import ingest
from backend.cli import read_test_file
from backend.core.exceptions import ConfigurationError
from backend.core.security import _validated
from backend.dsl.models import ActionType, AssertionType
from backend.dsl.observation import InventoryElement
from backend.execution.pickers import date_format_of, looks_like_date, looks_like_time, value_matches
from backend.services.platform_service import _mask_strings

# --------------------------------------------------------------------------- picker helpers


@pytest.mark.parametrize("text,expected", [("09 June 2026", dt.date(2026, 6, 9)), ("2026-06-09", dt.date(2026, 6, 9)),
                                           ("Tue Jun 09 2026", dt.date(2026, 6, 9)), ("Initial Call", None),
                                           ("Outbound", None), ("12", None)])
def test_looks_like_date(text, expected):
    assert looks_like_date(text) == expected


@pytest.mark.parametrize("text,expected", [("23:15", "23:15"), ("11:15 pm", "23:15"), ("9.30", "09:30"),
                                           ("1 pm", "13:00"), ("23", None), ("Room 23:15b", None)])
def test_looks_like_time(text, expected):
    assert looks_like_time(text) == expected


def test_placeholder_formats():
    assert date_format_of("DD/MM/YYYY") == "%d/%m/%Y"
    assert date_format_of("dd MMM yyyy") == "%d %b %Y"
    assert date_format_of("mm-dd-yy") == "%m-%d-%y"
    assert date_format_of("Select a date") is None
    assert dt.date(2026, 6, 9).strftime(date_format_of("DD/MM/YYYY")) == "09/06/2026"


@pytest.mark.parametrize("shown", ["09/06/2026", "09 Jun 2026", "June 9, 2026", "2026-06-09T00:00", "Tue Jun 09 2026"])
def test_date_read_back_matches(shown):
    assert value_matches("date", shown, "2026-06-09", placeholder="DD/MM/YYYY")


def test_date_read_back_respects_placeholder_order():
    # 06/09/2026 is 6 September in a DD/MM field: not the requested 9 June
    assert not value_matches("date", "06/09/2026", "2026-06-09", placeholder="DD/MM/YYYY")
    assert value_matches("date", "06/09/2026", "2026-06-09", placeholder="MM/DD/YYYY")
    assert not value_matches("date", "", "2026-06-09")


@pytest.mark.parametrize("shown,ok", [("23:15", True), ("11:15 PM", True), ("23 : 15", True), ("23:15:00", True),
                                      ("11:15", False), ("", False), ("HH:MM", False)])
def test_time_read_back(shown, ok):
    assert value_matches("time", shown, "23:15") is ok


# --------------------------------------------------------------------------- parsing a CRM-style test case

HEALTH_FLOW = """http://portal.example.test/sign-in
Enter care.user@example.com as Email
Enter {{care_password}} as Password
Click on Login button
Click on Patient Support Program
Select the Patient named Abc Xy
Click on Interaction tab
Click on Record Interaction to open sidebar
Click on Theme Box and select Initial Call
Click on Status box and select Not reachable
Click on Call Direction and select Outbound
Click on Start Date Selection and select 09 June 2026
Select Start Time and set the first field as 23 HH and second field as 15 MM
Select Record Interaction
Check whether new Interaction for patient Abc is added on the Interaction Logs tab
"""


class Parsed:
    def __init__(self, spec):
        self.steps = spec.flat_steps()
        self.ambiguities = spec.ambiguities


def _parse(content: str):
    import asyncio

    case = ingest(content)[0]
    spec = TestCompiler().compile(asyncio.run(RequirementAnalyzer().analyze(case, secret_names={"care_password"})))
    return case, Parsed(spec)


def test_crm_flow_parses_into_the_intended_steps():
    case, parsed = _parse(HEALTH_FLOW)
    steps = parsed.steps
    assert case.base_url == "http://portal.example.test/sign-in"
    assert not parsed.ambiguities
    email, password = steps[1], steps[2]
    assert email.action == ActionType.FILL and email.target.intent == "Email"
    assert email.value.value == "care.user@example.com"  # "Enter X as Email": X is the value
    assert password.value.source == "secret" and password.value.sensitive
    patient = steps[5]
    assert patient.action == ActionType.CLICK and patient.target.exact_name == "Abc Xy"
    record = steps[7]
    assert record.target.intent == "Record Interaction"  # "to open sidebar" is a purpose, not the name
    theme, status, direction, start = steps[8:12]
    assert (theme.action, theme.target.intent, theme.target.option_text) == (ActionType.SELECT, "Theme", "Initial Call")
    assert status.target.option_text == "Not reachable" and direction.target.option_text == "Outbound"
    assert start.target.intent == "Start Date" and start.value.value == "09 June 2026"
    time = steps[12]
    assert time.action == ActionType.FILL and time.value.value == "23:15" and "time_parts" in time.notes
    check = steps[-1]
    assert check.action == ActionType.ASSERT
    assert steps[-2].target.intent == "Interaction Logs"
    assertion = check.assertion
    assert assertion.type == AssertionType.ALL_OF
    assert assertion.children[0].type == AssertionType.TEXT_VISIBLE and assertion.children[0].expected == "Abc"


def test_variables_are_values_even_when_named_like_fields():
    assert _looks_like_data("{{care_email}}") and _looks_like_data("'Email'")
    assert not _looks_like_data("Email")
    _, parsed = _parse("http://x.test\nEnter {{care_email}} as Email\n")
    fill = parsed.steps[1]
    assert fill.target.intent == "Email" and fill.value.reference == "care_email"


def test_control_nouns_are_not_part_of_the_label():
    assert build_target("Theme Box", "select").semantic_hints[0] == "Theme"
    _, parsed = _parse("http://x.test\nOpen the Country dropdown and choose India\n")
    assert parsed.steps[1].target.intent == "Country"


def test_prose_sections_are_not_steps():
    case = ingest("# Test: t\n## About\nSign in and do things.\n## Steps\nhttp://x.test\nClick Save\n")[0]
    assert [line.text for line in case.lines] == ["Click Save"]
    assert "Sign in" in case.description


# --------------------------------------------------------------------------- grounding names


def _el(name: str, role: str = "link") -> InventoryElement:
    return InventoryElement(element_id=name, ref="e1", role=role, name=name)


def test_card_named_by_its_leading_title():
    target = build_target("Patient Support Program", "click")
    card, _ = name_score(target, _el("Patient Support Program Enrolled patients, interactions and follow-ups"))
    other, _ = name_score(target, _el("Programs"))
    assert card >= 0.85 and card - other > 0.2


def test_exact_names_are_never_typo_repaired():
    target = build_target("Abc Xy", "click")
    target.exact_name, target.literal = "Abc Xy", True
    exact, _ = name_score(target, _el("Abc Xy"))
    near, _ = name_score(target, _el("Abcd Xy"))
    assert exact == 1.0 and near < 0.8


# --------------------------------------------------------------------------- files, secrets, redaction


def test_test_files_decode_from_any_editor(tmp_path: Path):
    text = "Enter 山田太郎 as Name\nClick Envoyer ✓\n"
    for name, data in {"utf8.txt": text.encode("utf-8"), "bom.txt": "﻿".encode() + text.encode("utf-8"),
                       "utf16.txt": text.encode("utf-16")}.items():
        (tmp_path / name).write_bytes(data)
        assert read_test_file(tmp_path / name) == text
    (tmp_path / "legacy.txt").write_bytes("Café crème\n".encode("cp1252"))
    assert read_test_file(tmp_path / "legacy.txt") == "Café crème\n"


def test_invalid_secret_key_is_explained():
    key = Fernet.generate_key()
    assert _validated(key.decode(), "env") == key
    with pytest.raises(ConfigurationError) as info:
        _validated("not-a-fernet-key", "AITEST_SECRET_KEY")
    assert "AITEST_SECRET_KEY" in str(info.value.message) + str(info.value.details or "")


def test_vaulted_literals_are_masked_everywhere():
    spec = {"steps": [{"original_text": "Enter Pa$$w0rd as Password", "notes": ["typed Pa$$w0rd"]}], "n": 3}
    masked = _mask_strings(spec, ["Pa$$w0rd"])
    assert masked == {"steps": [{"original_text": f"Enter {MASK} as Password", "notes": [f"typed {MASK}"]}], "n": 3}


def test_platform_ids_stay_readable_but_tokens_do_not():
    out = redact_value({"session_id": "bs_1f2e3d4c5b6a7980", "token": "eyJhbGciOi.secret.value"})
    assert out == {"session_id": "bs_1f2e3d4c5b6a7980", "token": MASK}
    cookie = redact_value({"session_id": "s%3Aabcdef0123456789.Xy"})  # a real session cookie stays masked
    assert cookie == {"session_id": MASK}


# --------------------------------------------------------------------------- SQLite schema maintenance


async def test_sqlite_auto_migration_and_clear(tmp_path: Path):
    from backend.models import database
    from backend.models.entities import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    old = MetaData()
    Table("projects", old, Column("id", String, primary_key=True), Column("organization_id", String))
    async with engine.begin() as conn:
        await conn.run_sync(old.create_all)
        await conn.execute(text("INSERT INTO projects (id, organization_id) VALUES ('p1', 'o1')"))
        await conn.run_sync(lambda c: Base.metadata.create_all(c, checkfirst=True))
        added = await conn.run_sync(database._add_missing_columns)
        columns = await conn.run_sync(lambda c: {col["name"] for col in inspect(c).get_columns("projects")})
    assert added and {c.name for c in Base.metadata.tables["projects"].columns} <= columns
    from sqlalchemy.ext.asyncio import async_sessionmaker

    async with async_sessionmaker(engine)() as session:
        await database.clear_tables(session, ["projects"])
        await session.commit()
        assert (await session.execute(text("SELECT count(*) FROM projects"))).scalar() == 0
    await engine.dispose()


def test_integer_primary_keys_autoincrement_on_sqlite():
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.schema import CreateTable

    from backend.models.entities import Base

    ddl = str(CreateTable(Base.metadata.tables["test_steps"]).compile(dialect=sqlite.dialect()))
    assert "id INTEGER NOT NULL" in ddl and "PRIMARY KEY (id)" in ddl  # INTEGER PRIMARY KEY = rowid autoincrement
