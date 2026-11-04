"""Artifact storage (screenshots, snapshots, videos, traces, reports).

Large evidence never enters graph state; state holds ``artifact://`` references.
Backends: local filesystem (default) and S3-compatible object storage.
"""

from __future__ import annotations

import asyncio
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.core.config import Settings, get_settings
from backend.core.exceptions import NotFound

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ArtifactRef:
    uri: str  # artifact://{execution_id}/{kind}/{name}
    execution_id: str
    kind: str
    name: str
    content_type: str
    size: int

    @property
    def api_path(self) -> str:
        return f"/api/v1/artifacts/{self.execution_id}/{self.kind}/{self.name}"


def parse_uri(uri: str) -> tuple[str, str, str]:
    if not uri.startswith("artifact://"):
        raise NotFound("Not an artifact reference", details={"uri": uri})
    parts = uri[len("artifact://"):].split("/", 2)
    if len(parts) != 3:
        raise NotFound("Malformed artifact reference", details={"uri": uri})
    return parts[0], parts[1], parts[2]


class ArtifactBackend(Protocol):
    async def write(self, key: str, data: bytes, content_type: str) -> None: ...

    async def read(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...

    def local_path(self, key: str) -> Path | None: ...


class LocalArtifactBackend:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise NotFound("Invalid artifact key")
        return path

    async def write(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def read(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise NotFound("Artifact not found", details={"key": key})
        return await asyncio.to_thread(path.read_bytes)

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def local_path(self, key: str) -> Path | None:
        return self._path(key)


class S3ArtifactBackend:  # pragma: no cover - exercised only with an S3 endpoint
    def __init__(self, settings: Settings) -> None:
        import boto3

        self.bucket = settings.s3_bucket
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url, region_name=settings.s3_region)

    async def write(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=key, Body=data,
                                ContentType=content_type, ServerSideEncryption="AES256")

    async def read(self, key: str) -> bytes:
        response = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=key)
        return await asyncio.to_thread(response["Body"].read)

    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self.client.head_object, Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def local_path(self, key: str) -> Path | None:
        return None


class ArtifactService:
    def __init__(self, settings: Settings | None = None, backend: ArtifactBackend | None = None) -> None:
        self.settings = settings or get_settings()
        if backend is not None:
            self.backend = backend
        elif self.settings.artifact_backend == "s3":
            self.backend = S3ArtifactBackend(self.settings)
        else:
            self.backend = LocalArtifactBackend(self.settings.artifact_dir)

    @staticmethod
    def _key(execution_id: str, kind: str, name: str) -> str:
        return f"{SAFE_NAME.sub('_', execution_id)}/{SAFE_NAME.sub('_', kind)}/{SAFE_NAME.sub('_', name)}"

    async def put_bytes(self, execution_id: str, kind: str, name: str, data: bytes,
                        content_type: str | None = None) -> ArtifactRef:
        content_type = content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        key = self._key(execution_id, kind, name)
        await self.backend.write(key, data, content_type)
        return ArtifactRef(uri=f"artifact://{key}", execution_id=execution_id, kind=kind,
                           name=key.rsplit("/", 1)[-1], content_type=content_type, size=len(data))

    async def put_file(self, execution_id: str, kind: str, path: Path, name: str | None = None) -> ArtifactRef:
        data = await asyncio.to_thread(path.read_bytes)
        return await self.put_bytes(execution_id, kind, name or path.name, data)

    async def get_bytes(self, uri: str) -> bytes:
        execution_id, kind, name = parse_uri(uri)
        return await self.backend.read(self._key(execution_id, kind, name))

    def local_path(self, uri: str) -> Path | None:
        execution_id, kind, name = parse_uri(uri)
        return self.backend.local_path(self._key(execution_id, kind, name))


_service: ArtifactService | None = None


def get_artifact_service() -> ArtifactService:
    global _service
    if _service is None:
        _service = ArtifactService()
    return _service


def set_artifact_service(service: ArtifactService | None) -> None:
    global _service
    _service = service
