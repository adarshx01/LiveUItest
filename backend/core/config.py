"""Central, typed configuration for the platform.

Every tunable that affects safety, budgets, browser isolation or model usage lives
here so it can be audited and overridden per deployment through environment
variables (prefix ``AITEST_``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
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
    # Local-first: one SQLite file under data/ needs no server, Docker or admin rights. Point these at
    # PostgreSQL (postgresql+asyncpg://... and postgresql://...) for production / multi-worker setups.
    database_url: str = f"sqlite+aiosqlite:///{(REPO_ROOT / 'data' / 'aitest.db').as_posix()}"
    # LangGraph checkpoints: "auto" keeps them next to the database (SQLite file or the same PostgreSQL).
    checkpoint_database_url: str = "auto"
    # Redis is optional (multi-process event fan-out and worker queues). Empty = everything in-process.
    redis_url: str = ""
    use_redis: bool = False
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
    # Empty = the Chromium installed by `npx playwright install chromium` (recommended, works on Windows).
    browser_executable_path: str | None = None
    browser_headless: bool = True
    browser_no_sandbox: bool = True  # containers need it; harmless on a desktop
    browser_viewport: str = "1280x800"
    browser_vision_enabled: bool = True
    browser_record_video: bool = True
    browser_record_trace: bool = False
    browser_output_root: Path = REPO_ROOT / "data" / "browser"
    # Optional PLAYWRIGHT_BROWSERS_PATH for the MCP subprocess (see infrastructure/scripts/link_browsers.sh)
    playwright_browsers_path: Path | None = None
    browser_pool_size: int = 4
    browser_session_idle_ttl_seconds: float = 1800.0
    test_id_attribute: str = "data-testid"

    # ---- models ----------------------------------------------------------------
    llm_provider: Literal["none", "anthropic", "openai_compatible", "azure_openai"] = "none"
    llm_reasoning_model: str = "claude-opus-5-5"
    llm_classification_model: str = "claude-haiku-5-5"
    llm_vision_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 16_000
    llm_request_timeout_seconds: float = 120.0
    llm_enable_refusal_fallback: bool = True
    # read from .env as well as the process environment (the SDK alone only sees the process environment)
    anthropic_api_key: str | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "AITEST_ANTHROPIC_API_KEY"))
    openai_base_url: str | None = None
    openai_api_key: str | None = None
    llm_max_retries: int = 2  # SDK-level retries with backoff (rate limits, 5xx, timeouts)

    # Azure OpenAI (AITEST_LLM_PROVIDER=azure_openai). The standard AZURE_OPENAI_* names are read as-is.
    azure_openai_endpoint: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_ENDPOINT", "AITEST_AZURE_OPENAI_ENDPOINT"))
    azure_openai_api_key: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_API_KEY", "AITEST_AZURE_OPENAI_API_KEY"))
    azure_openai_api_version: str = Field(
        "2025-03-01-preview",
        validation_alias=AliasChoices("AZURE_OPENAI_API_VERSION", "AITEST_AZURE_OPENAI_API_VERSION"))
    # Default deployment for every tier; the per-tier deployments below override it when set.
    azure_openai_deployment: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT", "AITEST_AZURE_OPENAI_DEPLOYMENT"))
    azure_openai_reasoning_deployment: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_REASONING_DEPLOYMENT",
                                            "AITEST_AZURE_OPENAI_REASONING_DEPLOYMENT"))
    azure_openai_classification_deployment: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_CLASSIFICATION_DEPLOYMENT",
                                            "AITEST_AZURE_OPENAI_CLASSIFICATION_DEPLOYMENT"))
    azure_openai_vision_deployment: str | None = Field(
        None, validation_alias=AliasChoices("AZURE_OPENAI_VISION_DEPLOYMENT", "AITEST_AZURE_OPENAI_VISION_DEPLOYMENT"))
    # "auto" tries the Responses API and falls back to Chat Completions (json_schema) when the
    # resource/API version does not offer it; force one with "responses" or "chat".
    azure_openai_api: Literal["auto", "responses", "chat"] = Field(
        "auto", validation_alias=AliasChoices("AZURE_OPENAI_API", "AITEST_AZURE_OPENAI_API"))
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
    secret_key: str | None = None  # Fernet key (URL-safe base64, 32 bytes); see security.py for the lookup order
    secret_key_file: Path = REPO_ROOT / "data" / "secret.key"
    # where a generated key is kept: "auto" = OS keyring (Windows Credential Manager, macOS Keychain,
    # Secret Service) when available, else the key file; "keyring" or "file" to force one
    secret_key_store: Literal["auto", "keyring", "file"] = "auto"

    # ---- execution ------------------------------------------------------------
    thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    recovery: RecoveryBudget = Field(default_factory=RecoveryBudget)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    max_concurrent_executions: int = 4
    screenshot_policy: Literal["every_action", "on_failure", "off"] = "every_action"
    memory_min_successes_for_replay: int = 1
    # after a failed step: "assertions" keeps running when the failed step was a check (it changed nothing,
    # later steps stay valid); "never" stops at the first failure. Failed actions always block dependents.
    continue_on_failure: Literal["assertions", "never"] = "assertions"
    # model-guided recovery: after the deterministic strategies, let the configured model choose one more
    # strategy from the closed list (never free-form actions); 0 disables it
    llm_recovery_attempts: int = 1
    memory_min_confidence_for_replay: float = 0.75

    # ---- API ----------------------------------------------------------------------
    api_keys: list[str] = Field(default_factory=lambda: ["dev-api-key"])
    default_organization_id: str = "org_default"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    run_worker_in_api: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def resolved_checkpoint_url(self) -> str:
        """Where LangGraph checkpoints live; "auto" follows the main database."""

        if self.checkpoint_database_url and self.checkpoint_database_url != "auto":
            return self.checkpoint_database_url
        if self.is_sqlite:
            db_path = self.database_url.split("///", 1)[1]
            return "sqlite:///" + str(Path(db_path).with_name("checkpoints.db").as_posix())
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    @property
    def redis_enabled(self) -> bool:
        return bool(self.use_redis and self.redis_url)

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
