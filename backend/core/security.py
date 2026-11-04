"""Cryptographic primitives: secret encryption and API-key verification."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from backend.core.config import Settings, get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger

log = get_logger(__name__)


class SecretCipher:
    """Symmetric authenticated encryption (Fernet: AES-128-CBC + HMAC-SHA256)."""

    def __init__(self, key: bytes) -> None:
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:  # pragma: no cover - defensive
            raise ConfigurationError("Secret could not be decrypted with the configured key") from exc


def _load_or_create_key(settings: Settings) -> bytes:
    if settings.secret_key:
        return settings.secret_key.encode("ascii")
    if settings.is_production:
        raise ConfigurationError("AITEST_SECRET_KEY must be configured in production")
    path: Path = settings.secret_key_file
    if path.exists():
        return path.read_bytes().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # create with restrictive permissions before writing the key material
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    log.warning("Generated a development secret key; configure AITEST_SECRET_KEY for shared deployments",
                key_file=str(path))
    return key


_cipher: SecretCipher | None = None


def get_cipher() -> SecretCipher:
    global _cipher
    if _cipher is None:
        _cipher = SecretCipher(_load_or_create_key(get_settings()))
    return _cipher


def reset_cipher() -> None:
    global _cipher
    _cipher = None


def verify_api_key(candidate: str | None, settings: Settings | None = None) -> bool:
    if not candidate:
        return False
    settings = settings or get_settings()
    return any(hmac.compare_digest(candidate, key) for key in settings.api_keys)
