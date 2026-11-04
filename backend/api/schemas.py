"""Typed request/response schemas for the public API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ApplicationCreate(BaseModel):
    project_id: str
    base_url: str
    name: str | None = None


class TestCaseCreate(BaseModel):
    project_id: str
    content: str | dict[str, Any] | list[Any]
    format: str | None = Field(default=None, description="text | markdown | json | csv (auto-detected)")
    base_url: str | None = None
    name: str | None = None
    tags: list[str] = Field(default_factory=list)
    environment_id: str | None = None


class TestCasePatch(BaseModel):
    content: str | dict[str, Any] | None = None
    format: str | None = None
    base_url: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    archived: bool | None = None


class SuiteCreate(BaseModel):
    project_id: str
    name: str
    test_case_ids: list[str]


class ExecutionCreate(BaseModel):
    test_case_id: str
    version: int | None = None
    variables: dict[str, str] = Field(default_factory=dict)
    datasets: list[dict[str, str]] = Field(default_factory=list, description="data-driven: one execution per set")
    environment_id: str | None = None
    allowed_domains: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    step_ids: list[str] = Field(default_factory=list, description="run only these steps (partial run)")
    with_prerequisites: bool = True


class StepRerun(BaseModel):
    step_ids: list[str] = Field(min_length=1, description="step ids from GET /executions/{id}/steps")
    with_prerequisites: bool = Field(True, description="also run the earlier steps that build the page state; "
                                                       "false runs only the entry navigation and the selection")


class SuiteRun(BaseModel):
    variables: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class SecretCreate(BaseModel):
    name: str
    value: str
    project_id: str | None = None
    environment_id: str | None = None


class ApprovalDecision(BaseModel):
    note: str | None = None


class HumanBrowserAction(BaseModel):
    type: str = Field(description="click | type | press")
    x: float | None = None
    y: float | None = None
    text: str | None = None
    key: str | None = None


class ResumeRequest(BaseModel):
    approved: bool = True
    note: str | None = None
