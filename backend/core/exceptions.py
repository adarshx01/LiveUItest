"""Typed exception hierarchy shared across the platform."""

from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base class for all platform errors."""

    code: str = "PLATFORM_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(PlatformError):
    code = "CONFIGURATION_ERROR"


class ValidationFailed(PlatformError):
    code = "VALIDATION_FAILED"


class NotFound(PlatformError):
    code = "NOT_FOUND"


class PermissionDenied(PlatformError):
    code = "PERMISSION_DENIED"


class PolicyViolation(PlatformError):
    """Raised by the deterministic policy layer. The model cannot override it."""

    code = "POLICY_VIOLATION"


class ApprovalRequired(PlatformError):
    code = "APPROVAL_REQUIRED"


class BudgetExceeded(PlatformError):
    code = "BUDGET_EXCEEDED"


class ExecutionCancelled(PlatformError):
    code = "EXECUTION_CANCELLED"


class ExecutionTimeout(PlatformError):
    code = "EXECUTION_TIMEOUT"


class BrowserError(PlatformError):
    code = "BROWSER_ERROR"


class BrowserDisconnected(BrowserError):
    code = "BROWSER_DISCONNECTED"


class ToolNotAllowed(BrowserError):
    code = "TOOL_NOT_ALLOWED"


class ToolCallFailed(BrowserError):
    code = "TOOL_CALL_FAILED"

    def __init__(self, message: str, *, tool: str, raw_error: str = "", details: dict[str, Any] | None = None):
        super().__init__(message, details={"tool": tool, **(details or {})})
        self.tool = tool
        self.raw_error = raw_error


class ModelError(PlatformError):
    code = "MODEL_ERROR"


class ModelUnavailable(ModelError):
    code = "MODEL_UNAVAILABLE"


class AgentOutputInvalid(PlatformError):
    code = "AGENT_OUTPUT_INVALID"


class SecretNotFound(PlatformError):
    code = "SECRET_NOT_FOUND"


class ParseError(PlatformError):
    code = "PARSE_ERROR"
