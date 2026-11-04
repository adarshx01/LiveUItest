"""Secret store and resolver (spec section 16.4).

Secrets live encrypted (Fernet) in ``environment_credentials``; CI pipelines may
also inject ``AITEST_SECRET_<NAME>`` environment variables. A secret is only
decrypted when an approved action needs it, the plaintext is registered with the
redactor immediately, and it is never written to memory, events or reports.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from backend.agents.governance.redaction import is_sensitive_name, register_sensitive
from backend.core.exceptions import SecretNotFound
from backend.core.security import get_cipher


class SecretBackend(Protocol):
    async def names(self, scope: SecretScope) -> set[str]: ...

    async def get_encrypted(self, scope: SecretScope, name: str) -> str | None: ...

    async def put_encrypted(self, scope: SecretScope, name: str, token: str) -> None: ...


class SecretScope:
    __slots__ = ("organization_id", "project_id", "environment_id")

    def __init__(self, organization_id: str, project_id: str | None = None, environment_id: str | None = None):
        self.organization_id = organization_id
        self.project_id = project_id
        self.environment_id = environment_id

    def key(self) -> tuple[str, str, str]:
        return self.organization_id, self.project_id or "*", self.environment_id or "*"


class InMemorySecretBackend:
    """Process-local backend used by the CLI and tests (values still stored encrypted)."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str], dict[str, str]] = {}

    async def names(self, scope: SecretScope) -> set[str]:
        names: set[str] = set()
        for key in _scope_chain(scope):
            names |= set(self._data.get(key, {}))
        return names

    async def get_encrypted(self, scope: SecretScope, name: str) -> str | None:
        for key in _scope_chain(scope):
            token = self._data.get(key, {}).get(name)
            if token:
                return token
        return None

    async def put_encrypted(self, scope: SecretScope, name: str, token: str) -> None:
        self._data.setdefault(scope.key(), {})[name] = token


def _scope_chain(scope: SecretScope) -> list[tuple[str, str, str]]:
    org, project, env = scope.key()
    return [(org, project, env), (org, project, "*"), (org, "*", "*")]


class SecretManager:
    ENV_PREFIX = "AITEST_SECRET_"

    def __init__(self, backend: SecretBackend | None = None) -> None:
        self.backend: SecretBackend = backend or InMemorySecretBackend()

    async def available_names(self, scope: SecretScope) -> set[str]:
        env_names = {k[len(self.ENV_PREFIX):].lower() for k in os.environ if k.startswith(self.ENV_PREFIX)}
        return env_names | await self.backend.names(scope)

    async def store(self, scope: SecretScope, name: str, value: str) -> None:
        register_sensitive(value)
        await self.backend.put_encrypted(scope, name, get_cipher().encrypt(value))

    async def resolve(self, scope: SecretScope, name: str) -> str:
        token = await self.backend.get_encrypted(scope, name)
        if token is not None:
            value = get_cipher().decrypt(token)
        else:
            env_value = os.environ.get(self.ENV_PREFIX + name.upper())
            if env_value is None:
                raise SecretNotFound(f"Secret {name!r} is not configured", details={"name": name})
            value = env_value
        register_sensitive(value)
        return value


_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    global _manager
    if _manager is None:
        _manager = SecretManager()
    return _manager


def set_secret_manager(manager: SecretManager) -> None:
    global _manager
    _manager = manager


def seal_variables(variables: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Split run variables into plain test data and a sealed credential bundle.

    Graph state is checkpointed to PostgreSQL, queued through Redis and stored on the run record, so a
    credential passed as an ordinary variable (``{"password": ...}``) would otherwise be persisted in
    clear text. Credential-named values are Fernet-encrypted with the platform key, the same protection
    as the credential store, registered with the redactor, and only decrypted when the browser session
    starts. Only their names stay readable (for plan validation).
    """

    plain: dict[str, str] = {}
    sensitive: dict[str, str] = {}
    for name, value in (variables or {}).items():
        (sensitive if is_sensitive_name(name) else plain)[name] = "" if value is None else str(value)
    if not sensitive:
        return plain, None
    for value in sensitive.values():
        register_sensitive(value)
    return plain, {"names": sorted(sensitive), "token": get_cipher().encrypt(json.dumps(sensitive))}


def unseal_variables(sealed: dict[str, Any] | None) -> dict[str, str]:
    if not sealed or not sealed.get("token"):
        return {}
    values: dict[str, str] = json.loads(get_cipher().decrypt(sealed["token"]))
    for value in values.values():
        register_sensitive(value)
    return values
