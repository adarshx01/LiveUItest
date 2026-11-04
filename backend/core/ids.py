"""Identifier helpers. IDs are prefixed so they are self-describing in logs."""

from __future__ import annotations

import hashlib
import secrets
import time


def new_id(prefix: str) -> str:
    # time-ordered prefix keeps IDs roughly sortable; random suffix avoids collisions
    stamp = format(int(time.time() * 1000), "x")[-8:]
    return f"{prefix}_{stamp}{secrets.token_hex(6)}"


def stable_hash(*parts: object, length: int = 16) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return digest[:length]
