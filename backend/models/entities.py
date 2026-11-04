"""PostgreSQL system of record (spec section 23).

Every tenant-owned row carries ``organization_id``; repositories always filter on
it. Embeddings use pgvector. Screenshots, videos and traces live in object
storage; the ``artifacts`` table indexes them by kind.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 256
# SQLite only auto-increments INTEGER PRIMARY KEY columns
BigIntPK = BigInteger().with_variant(Integer, "sqlite")
JSONType = JSON().with_variant(JSONB(), "postgresql")
# pgvector on PostgreSQL (indexed cosine search); a plain JSON array elsewhere (SQLite), where the
# memory manager ranks candidates in Python. Same values either way.
EmbeddingType = JSON().with_variant(Vector(EMBED_DIM), "postgresql")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONType, list[Any]: JSONType}


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())


# --------------------------------------------------------------------------- tenancy
class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    name: Mapped[str] = mapped_column(String(200), default="")
    role: Mapped[str] = mapped_column(String(32), default="engineer")  # admin | engineer | viewer | approver


class Project(Base, TimestampMixin):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")


class Application(Base, TimestampMixin):
    __tablename__ = "applications"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    base_url: Mapped[str] = mapped_column(String(2048))
    host: Mapped[str] = mapped_column(String(255), index=True)
    known_routes: Mapped[list[Any]] = mapped_column(default=list)
    auth_behavior: Mapped[dict[str, Any]] = mapped_column(default=dict)
    fingerprint: Mapped[dict[str, Any]] = mapped_column(default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ApplicationEnvironment(Base, TimestampMixin):
    __tablename__ = "application_environments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    base_url: Mapped[str] = mapped_column(String(2048))
    environment_type: Mapped[str] = mapped_column(String(32), default="test")
    allowed_domains: Mapped[list[Any]] = mapped_column(default=list)
    policy: Mapped[dict[str, Any]] = mapped_column(default=dict)
    variables: Mapped[dict[str, Any]] = mapped_column(default=dict)
    browser: Mapped[dict[str, Any]] = mapped_column(default=dict)  # viewport, locale, timezone, device


class EnvironmentCredential(Base, TimestampMixin):
    __tablename__ = "environment_credentials"
    __table_args__ = (UniqueConstraint("organization_id", "project_id", "environment_id", "name"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(String(64), default="*")
    environment_id: Mapped[str] = mapped_column(String(64), default="*")
    name: Mapped[str] = mapped_column(String(128))
    encrypted_value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


# --------------------------------------------------------------------------- tests
class TestSuite(Base, TimestampMixin):
    __tablename__ = "test_suites"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    test_case_ids: Mapped[list[Any]] = mapped_column(default=list)


class TestCase(Base, TimestampMixin):
    __tablename__ = "test_cases"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    name: Mapped[str] = mapped_column(String(300))
    tags: Mapped[list[Any]] = mapped_column(default=list)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class TestCaseVersion(Base, TimestampMixin):
    __tablename__ = "test_case_versions"
    __table_args__ = (UniqueConstraint("test_case_id", "version"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    test_case_id: Mapped[str] = mapped_column(ForeignKey("test_cases.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)  # sensitive literals are vaulted and masked
    source_format: Mapped[str] = mapped_column(String(32))
    spec: Mapped[dict[str, Any]] = mapped_column()
    spec_hash: Mapped[str] = mapped_column(String(64), index=True)
    issues: Mapped[list[Any]] = mapped_column(default=list)


class TestStep(Base):
    __tablename__ = "test_steps"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    test_case_version_id: Mapped[str] = mapped_column(ForeignKey("test_case_versions.id"), index=True)
    step_id: Mapped[str] = mapped_column(String(32))
    index: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(32))
    original_text: Mapped[str] = mapped_column(Text)
    spec: Mapped[dict[str, Any]] = mapped_column()


class TestDataSet(Base, TimestampMixin):
    __tablename__ = "test_data_sets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    test_case_id: Mapped[str] = mapped_column(ForeignKey("test_cases.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    variables: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Schedule(Base, TimestampMixin):
    __tablename__ = "execution_schedules"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(16))  # test_case | suite
    target_id: Mapped[str] = mapped_column(String(64))
    interval_minutes: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)


# --------------------------------------------------------------------------- executions
class ExecutionRun(Base, TimestampMixin):
    __tablename__ = "execution_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), index=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    environment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    test_case_id: Mapped[str] = mapped_column(String(64), index=True)
    test_case_version: Mapped[int] = mapped_column(Integer, default=1)
    dataset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    suite_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    retry_of: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    runner: Mapped[str | None] = mapped_column(String(32), nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(default=dict)
    report_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    report_json_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    video_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    flaky: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExecutionStep(Base):
    __tablename__ = "execution_steps"
    __table_args__ = (UniqueConstraint("execution_id", "step_id"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("execution_runs.id"), index=True)
    step_id: Mapped[str] = mapped_column(String(32))
    index: Mapped[int] = mapped_column(Integer)
    original_text: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text, default="")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    recovered: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_used: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0)
    result: Mapped[dict[str, Any]] = mapped_column(default=dict)


class BrowserSessionRecord(Base):
    __tablename__ = "browser_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    execution_id: Mapped[str] = mapped_column(ForeignKey("execution_runs.id"), index=True)
    server_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BrowserSessionEvent(Base):
    """MCP tool-call audit log (redacted)."""

    __tablename__ = "browser_session_events"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[float] = mapped_column(Float)
    tool: Mapped[str] = mapped_column(String(64))
    ok: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)
    args: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)


class ExecutionEvent(Base):
    """Execution event stream (also published live over WebSocket)."""

    __tablename__ = "agent_events"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    ts: Mapped[float] = mapped_column(Float)
    event: Mapped[str] = mapped_column(String(64))
    step_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)


class AgentTrace(Base):
    __tablename__ = "agent_traces"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[float] = mapped_column(Float)
    agent: Mapped[str] = mapped_column(String(64))
    node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    step_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    duration_ms: Mapped[float] = mapped_column(Float, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)


class AssertionResultRecord(Base):
    __tablename__ = "assertion_results"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    step_id: Mapped[str] = mapped_column(String(32))
    assertion_id: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(16))
    expected: Mapped[dict[str, Any]] = mapped_column(default=dict)
    actual: Mapped[dict[str, Any]] = mapped_column(default=dict)
    message: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[Any]] = mapped_column(default=list)


class RecoveryAttemptRecord(Base):
    __tablename__ = "recovery_attempts"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    step_id: Mapped[str] = mapped_column(String(32))
    attempt: Mapped[int] = mapped_column(Integer)
    failure_category: Mapped[str] = mapped_column(String(64))
    strategy: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))
    rationale: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)


class Approval(Base, TimestampMixin):
    __tablename__ = "approval_events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    step_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))  # action | captcha | guidance
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending | approved | rejected | expired
    request: Mapped[dict[str, Any]] = mapped_column(default=dict)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Artifact(Base, TimestampMixin):
    """Screenshots, videos, traces, snapshots and reports (kind column)."""

    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    step_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # screenshot | video | trace | snapshot | report
    name: Mapped[str] = mapped_column(String(255))
    uri: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    size: Mapped[int] = mapped_column(BigInteger, default=0)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str] = mapped_column(String(64))
    resource_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)


class ExecutionMetric(Base):
    __tablename__ = "execution_metrics"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------- memory
class MemoryScopeMixin:
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    environment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class PageFingerprintRecord(Base, MemoryScopeMixin):
    __tablename__ = "page_fingerprints"
    __table_args__ = (UniqueConstraint("application_id", "structure_hash"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    route_pattern: Mapped[str] = mapped_column(String(512), index=True)
    structure_hash: Mapped[str] = mapped_column(String(32), index=True)
    title_pattern: Mapped[str] = mapped_column(String(512), default="")
    signature: Mapped[list[Any]] = mapped_column(default=list)
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApplicationMemory(Base, MemoryScopeMixin):
    __tablename__ = "application_memory"
    __table_args__ = (UniqueConstraint("application_id", "key"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[dict[str, Any]] = mapped_column(default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class PageMemory(Base, MemoryScopeMixin):
    __tablename__ = "page_memory"
    __table_args__ = (UniqueConstraint("application_id", "route_pattern", "structure_hash"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    route_pattern: Mapped[str] = mapped_column(String(512), index=True)
    structure_hash: Mapped[str] = mapped_column(String(32), index=True)
    page_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    title_pattern: Mapped[str] = mapped_column(String(512), default="")
    forms: Mapped[dict[str, Any]] = mapped_column(default=dict)
    signature: Mapped[list[Any]] = mapped_column(default=list)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType, nullable=True)
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    last_verified_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class LocatorMemory(Base, MemoryScopeMixin):
    __tablename__ = "locator_memory"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    test_case_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    route_pattern: Mapped[str] = mapped_column(String(512), index=True)
    page_fingerprint: Mapped[str] = mapped_column(String(32), index=True)
    action_signature: Mapped[str] = mapped_column(String(256), index=True)
    intent_text: Mapped[str] = mapped_column(Text, default="")
    target_strategy: Mapped[dict[str, Any]] = mapped_column(default=dict)
    successful_execution_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_execution_count: Mapped[int] = mapped_column(Integer, default=0)
    validated_run_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, index=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)  # active|superseded|invalidated
    memory_version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    supersedes: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType, nullable=True)
    last_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_verified_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


Index("ix_locator_memory_lookup", LocatorMemory.application_id, LocatorMemory.route_pattern,
      LocatorMemory.action_signature, LocatorMemory.status)


class TestExecutionMemory(Base, MemoryScopeMixin):
    __tablename__ = "test_execution_memory"
    __table_args__ = (UniqueConstraint("test_case_id", "test_case_version"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    test_case_id: Mapped[str] = mapped_column(String(64), index=True)
    test_case_version: Mapped[int] = mapped_column(Integer)
    trajectory: Mapped[list[Any]] = mapped_column(default=list)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    last_execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_verified_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InteractionPattern(Base):
    __tablename__ = "interaction_patterns"
    __table_args__ = (UniqueConstraint("organization_id", "application_id", "pattern_type", "signature"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    application_id: Mapped[str] = mapped_column(String(64), index=True)
    pattern_type: Mapped[str] = mapped_column(String(64))  # custom_dropdown | autocomplete | login_flow | ...
    signature: Mapped[str] = mapped_column(String(256))
    details: Mapped[dict[str, Any]] = mapped_column(default=dict)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class FailureMemory(Base, MemoryScopeMixin):
    __tablename__ = "failure_memory"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(String(64), index=True)
    route_pattern: Mapped[str] = mapped_column(String(512), index=True)
    action_signature: Mapped[str] = mapped_column(String(256), index=True)
    failure_category: Mapped[str] = mapped_column(String(64), index=True)
    failure_signature: Mapped[str] = mapped_column(String(64), index=True)
    context: Mapped[dict[str, Any]] = mapped_column(default=dict)
    failed_strategies: Mapped[list[Any]] = mapped_column(default=list)
    successful_alternative: Mapped[dict[str, Any] | None] = mapped_column(nullable=True)
    root_cause: Mapped[str] = mapped_column(Text, default="")
    recovery_attempts: Mapped[int] = mapped_column(Integer, default=0)
    evidence: Mapped[list[Any]] = mapped_column(default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MemoryVersion(Base):
    __tablename__ = "memory_versions"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    memory_id: Mapped[str] = mapped_column(String(64), index=True)
    memory_type: Mapped[str] = mapped_column(String(32))
    version: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, Any]] = mapped_column(default=dict)
    change_reason: Mapped[str] = mapped_column(Text, default="")
    execution_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
