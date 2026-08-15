"""A files-only vector index.

All embeddings live in one JSON file (vault/index/embeddings.json) keyed
by id ("idea:<slug>", "paper:<pmid>", "topic:<run>"). Search is brute
force cosine in pure Python — the vault grows lazily around researched
topics, so this stays comfortably fast into the tens of thousands of
vectors, and there is no database to run.
"""

import json
import math
from pathlib import Path


class EmbeddingIndex:
    def __init__(self, vault: Path):
        self.path = vault / "index" / "embeddings.json"
        self.vectors: dict[str, list[float]] = {}
        if self.path.exists():
            self.vectors = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.vectors))

    def add(self, key: str, vector: list[float]) -> None:
        self.vectors[key] = vector

    def get(self, key: str) -> list[float] | None:
        return self.vectors.get(key)

    def search(
        self, vector: list[float], k: int = 5, prefix: str = "", exclude: str | None = None
    ) -> list[tuple[str, float]]:
        """Top-k cosine matches, optionally restricted to a key prefix."""
        scores = []
        for key, other in self.vectors.items():
            if prefix and not key.startswith(prefix):
                continue
            if key == exclude:
                continue
            scores.append((key, cosine(vector, other)))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:k]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
