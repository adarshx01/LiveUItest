"""API dependencies: authentication, tenant resolution, platform access."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from backend.bootstrap import Platform
from backend.core.config import get_settings
from backend.core.security import verify_api_key


@dataclass
class Principal:
    organization_id: str
    actor: str


async def principal(x_api_key: str | None = Header(default=None), x_organization_id: str | None = Header(default=None),
                    api_key: str | None = None) -> Principal:
    key = x_api_key or api_key
    if not verify_api_key(key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
    return Principal(organization_id=x_organization_id or get_settings().default_organization_id,
                     actor=f"api-key:{(key or '')[:4]}…")


def platform(request: Request) -> Platform:
    value = getattr(request.app.state, "platform", None)
    if value is None:
        raise HTTPException(status_code=503, detail="platform is starting")
    return value


PrincipalDep = Depends(principal)
PlatformDep = Depends(platform)
