"""Deterministic local text embeddings for pgvector retrieval.

A signed feature-hashing embedder over word unigrams/bigrams and character
trigrams. It needs no network, is stable across processes and captures lexical
similarity of step intents ("enter username", "fill the username field").
Similarity only *proposes* memories; the grounder must re-validate them live.
"""

from __future__ import annotations

import hashlib
import math

from backend.agents.planner.lexicon import normalize_text, synonyms_for

DIM = 256


def _h(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % DIM, 1.0 if (value >> 63) & 1 else -1.0


def embed(text: str, *, purpose: str | None = None) -> list[float]:
    norm = normalize_text(text)
    words = norm.split()
    if purpose:
        words += normalize_text(" ".join(synonyms_for(purpose)[:4])).split()
    vec = [0.0] * DIM
    feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
    feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
    for w in words:
        padded = f"#{w}#"
        feats += [(f"c:{padded[i:i + 3]}", 0.35) for i in range(max(0, len(padded) - 2))]
    for token, weight in feats:
        idx, sign = _h(token)
        vec[idx] += sign * weight
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / n for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
