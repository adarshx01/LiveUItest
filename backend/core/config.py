"""Central, typed configuration for the platform.

Every tunable that affects safety, budgets, browser isolation or model usage lives
here so it can be audited and overridden per deployment through environment
variables (prefix ``AITEST_``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfidenceThresholds(BaseModel):
    """Initial, configurable execution thresholds (spec section 9.5)."""

    execute: float = 0.95
    enhanced_verification: float = 0.85
    refine: float = 0.65


class RecoveryBudget(BaseModel):
    """Bounded retry policy (spec section 17.3)."""

    max_action_attempts: int = 2
    max_locator_healing_attempts: int = 3
    max_recovery_cycles_per_step: int = 2
    max_test_recovery_cycles: int = 5
    max_recovery_seconds: float = 60.0


class ExecutionBudget(BaseModel):
    max_llm_calls: int = 60
    max_llm_tokens: int = 400_000
    max_tool_calls: int = 1500
    max_steps: int = 200
    action_timeout_ms: int = 8_000
    navigation_timeout_ms: int = 30_000
    execution_deadline_seconds: float = 900.0
    approval_wait_seconds: float = 1800.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AITEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    service_name: str = "ai-testing-platform"
    log_level: str = "INFO"
    log_json: bool = True

    # ---- persistence -------------------------------------------------------
    database_url: str = "postgresql+asyncpg://aitest:aitest@localhost:5432/aitest"
    # LangGraph's Postgres checkpointer uses psycopg, which wants a plain URL.
    checkpoint_database_url: str = "postgresql://aitest:aitest@localhost:5432/aitest"
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = True
    embedding_dimensions: int = 256

    # ---- artifacts -----------------------------------------------------------
    artifact_backend: Literal["local", "s3"] = "local"
    artifact_dir: Path = REPO_ROOT / "data" / "artifacts"
    s3_endpoint_url: str | None = None
    s3_bucket: str = "aitest-artifacts"
    s3_region: str = "us-east-1"
    artifact_retention_days: int = 30

    # ---- browser / Playwright MCP -------------------------------------------
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_node_command: str = "node"
    mcp_server_script: Path = (
        REPO_ROOT / "browser" / "playwright-mcp" / "node_modules" / "@playwright" / "mcp" / "cli.js"
    )
    mcp_http_url: str | None = None  # e.g. http://browser-worker:8931/mcp
    mcp_expected_version_prefix: str = "1.65.0"
    browser_name: Literal["chromium", "firefox", "webkit"] = "chromium"
    browser_executable_path: str | None = "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"
    browser_headless: bool = True
    browser_no_sandbox: bool = True
    browser_viewport: str = "1280x800"
    browser_vision_enabled: bool = True
    browser_record_video: bool = True
    browser_record_trace: bool = False
    browser_output_root: Path = REPO_ROOT / "data" / "browser"
    browser_pool_size: int = 4
    browser_session_idle_ttl_seconds: float = 1800.0
    test_id_attribute: str = "data-testid"

    # ---- models ----------------------------------------------------------------
    llm_provider: Literal["none", "anthropic", "openai_compatible"] = "none"
    llm_reasoning_model: str = "claude-opus-5-5"
    llm_classification_model: str = "claude-haiku-5-5"
    llm_vision_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 16_000
    llm_request_timeout_seconds: float = 120.0
    llm_enable_refusal_fallback: bool = True
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    # USD per million tokens, used for agent cost reporting
    llm_price_input_per_mtok: float = 4.0
    llm_price_output_per_mtok: float = 20.0

    # ---- safety / governance ------------------------------------------------
    allowed_domains: list[str] = Field(default_factory=list)
    always_allowed_domains: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    allow_destructive_actions: bool = False
    allow_financial_actions: bool = False
    allow_messaging_actions: bool = False
    auto_approve_in_test_environments: bool = False
    secret_key: str | None = None  # Fernet key; generated for development if absent
    secret_key_file: Path = REPO_ROOT / "data" / "secret.key"

    # ---- execution ------------------------------------------------------------
    thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    recovery: RecoveryBudget = Field(default_factory=RecoveryBudget)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    max_concurrent_executions: int = 4
    screenshot_policy: Literal["every_action", "on_failure", "off"] = "every_action"
    memory_min_successes_for_replay: int = 1
    memory_min_confidence_for_replay: float = 0.75

    # ---- API ----------------------------------------------------------------------
    api_keys: list[str] = Field(default_factory=lambda: ["dev-api-key"])
    default_organization_id: str = "org_default"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    run_worker_in_api: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    def viewport_size(self) -> tuple[int, int]:
        width, height = self.browser_viewport.lower().split("x")
        return int(width), int(height)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def override_settings(**kwargs: object) -> Settings:
    """Replace the process-wide settings (used by tests and the CLI)."""

    global _settings
    _settings = Settings(**kwargs)  # type: ignore[arg-type]
    return _settings
