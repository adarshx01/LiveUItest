"""Cryptographic primitives: secret encryption and API-key verification."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

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


KEYRING_SERVICE = "ai-testing-platform"
KEYRING_USER = "secret-store-fernet-key"
GENERATE_HINT = ('python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
_key_source = "not loaded"


def _validated(raw: str | bytes, source: str) -> bytes:
    key = raw.strip().encode("ascii") if isinstance(raw, str) else raw.strip()
    try:
        Fernet(key)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"{source} is not a valid Fernet key (it must be 44 URL-safe base64 characters "
                                 f"encoding 32 bytes). Generate one with: {GENERATE_HINT}") from exc
    return key


def _keyring() -> Any:
    """The OS credential store (Windows Credential Manager, macOS Keychain, Secret Service), or None."""

    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring

        backend = keyring.get_keyring()
        if isinstance(backend, FailKeyring) or "fail" in type(backend).__module__ or "null" in type(backend).__module__:
            return None
        return keyring
    except Exception:  # noqa: BLE001 - missing package or no usable backend
        return None


def _restrict_file(path: Path) -> None:
    """Owner-only access: POSIX mode 600 at creation; on Windows drop inherited ACLs (no admin needed)."""

    if os.name != "nt":
        return
    import subprocess

    user = os.environ.get("USERNAME")
    if not user:
        return
    try:
        subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"], check=False,
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - best effort hardening
        pass


def _load_or_create_key(settings: Settings) -> bytes:
    """Lookup order: AITEST_SECRET_KEY (validated) -> existing key file -> OS keyring -> new key.

    A new key goes to the OS keyring when one is available (``AITEST_SECRET_KEY_STORE=auto``), otherwise
    to an owner-only key file. An existing key file always wins so previously stored secrets stay readable.
    """

    global _key_source
    if settings.secret_key:
        _key_source = "AITEST_SECRET_KEY"
        return _validated(settings.secret_key, "AITEST_SECRET_KEY")
    if settings.is_production:
        raise ConfigurationError(f"AITEST_SECRET_KEY must be configured in production ({GENERATE_HINT})")
    path: Path = settings.secret_key_file
    if path.exists() and settings.secret_key_store != "keyring":
        _key_source = f"key file {path}"
        return _validated(path.read_bytes(), f"key file {path}")
    if settings.secret_key_store in ("auto", "keyring"):
        ring = _keyring()
        if ring is not None:
            try:
                stored = ring.get_password(KEYRING_SERVICE, KEYRING_USER)
                if stored:
                    _key_source = "OS keyring"
                    return _validated(stored, "the OS keyring entry")
                key = Fernet.generate_key()
                ring.set_password(KEYRING_SERVICE, KEYRING_USER, key.decode("ascii"))
                _key_source = "OS keyring (generated)"
                log.info("generated the secret-store key and saved it in the OS keyring",
                         service=KEYRING_SERVICE)
                return key
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - locked/unavailable keyring: fall back to the file
                log.warning("OS keyring unavailable; using a key file", error=str(exc)[:120])
        elif settings.secret_key_store == "keyring":
            raise ConfigurationError("AITEST_SECRET_KEY_STORE=keyring but no OS keyring backend is available")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # create with restrictive permissions before writing the key material
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    _restrict_file(path)
    _key_source = f"key file {path} (generated)"
    log.warning("Generated a development secret key file; set AITEST_SECRET_KEY for shared deployments",
                key_file=str(path))
    return key


def describe_key_source() -> str:
    """Where the secret-store key comes from (never the key itself); loads it if needed."""

    get_cipher()
    return _key_source


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
