"""Database-backed encrypted secret backend (``environment_credentials``)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.agents.governance.secret_manager import SecretScope
from backend.core.ids import new_id
from backend.models.entities import EnvironmentCredential, Organization


class DatabaseSecretBackend:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    @staticmethod
    def _chain(scope: SecretScope) -> list[tuple[str, str]]:
        _, project, env = scope.key()
        return [(project, env), (project, "*"), ("*", "*")]

    async def names(self, scope: SecretScope) -> set[str]:
        async with self.sessionmaker() as s:
            rows = (await s.execute(select(EnvironmentCredential.name, EnvironmentCredential.project_id,
                                           EnvironmentCredential.environment_id).where(
                EnvironmentCredential.organization_id == scope.organization_id))).all()
        allowed = set(self._chain(scope))
        return {name for name, project, env in rows if (project, env) in allowed}

    async def get_encrypted(self, scope: SecretScope, name: str) -> str | None:
        async with self.sessionmaker() as s:
            rows = (await s.execute(select(EnvironmentCredential).where(
                EnvironmentCredential.organization_id == scope.organization_id,
                EnvironmentCredential.name == name))).scalars().all()
        by_scope = {(r.project_id, r.environment_id): r.encrypted_value for r in rows}
        for key in self._chain(scope):
            if key in by_scope:
                return by_scope[key]
        return None

    async def put_encrypted(self, scope: SecretScope, name: str, token: str) -> None:
        _, project, env = scope.key()
        async with self.sessionmaker() as s:
            if await s.get(Organization, scope.organization_id) is None:
                s.add(Organization(id=scope.organization_id, name=scope.organization_id))
                await s.flush()
            row = (await s.execute(select(EnvironmentCredential).where(
                EnvironmentCredential.organization_id == scope.organization_id,
                EnvironmentCredential.project_id == project, EnvironmentCredential.environment_id == env,
                EnvironmentCredential.name == name))).scalar_one_or_none()
            if row is None:
                s.add(EnvironmentCredential(id=new_id("cred"), organization_id=scope.organization_id,
                                            project_id=project, environment_id=env, name=name, encrypted_value=token))
            else:
                row.encrypted_value = token
            await s.commit()
