"""The frontier: what to explore next, and the breadth/depth knob.

Breadth and depth are not two agents — they are one scoring function:

    score = alpha * relevance + beta * domain_novelty

relevance       cosine similarity between the item and the run topic
domain_novelty  how unfamiliar the item's domains are to this run so far

Depth mode weights alpha; breadth mode weights beta. The frontier is
persisted to frontier.json on every change so a run can be inspected
mid-flight and resumed after a crash.
"""

import json
from collections import Counter
from pathlib import Path


class Frontier:
    def __init__(self, path: Path, alpha: float, beta: float):
        self.path = path
        self.alpha = alpha
        self.beta = beta
        self.items: list[dict] = []
        self.visited: set[str] = set()
        self.domain_counts: Counter = Counter()
        if path.exists():
            state = json.loads(path.read_text())
            self.items = state["items"]
            self.visited = set(state["visited"])
            self.domain_counts = Counter(state["domain_counts"])

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "items": self.items,
                    "visited": sorted(self.visited),
                    "domain_counts": dict(self.domain_counts),
                },
                indent=2,
            )
            + "\n"
        )

    # ------------------------------------------------------------ scoring

    def domain_novelty(self, domains: list[str]) -> float:
        """1.0 when every domain is unseen this run, 0.0 when all are familiar."""
        if not domains:
            return 0.5
        total = sum(self.domain_counts.values())
        if total == 0:
            return 1.0
        seen = sum(1 for d in domains if self.domain_counts[d.lower()] > 0)
        return 1.0 - (seen / len(domains))

    def score(self, relevance: float, domains: list[str]) -> float:
        return self.alpha * relevance + self.beta * self.domain_novelty(domains)

    # -------------------------------------------------------------- queue

    def push(self, kind: str, ident: str, relevance: float, domains: list[str], reason: str) -> None:
        """kind is one of: paper, idea, query."""
        key = f"{kind}:{ident}"
        if key in self.visited or any(i["key"] == key for i in self.items):
            return
        self.items.append(
            {
                "key": key,
                "kind": kind,
                "id": ident,
                "score": round(self.score(relevance, domains), 4),
                "relevance": round(relevance, 4),
                "domains": domains,
                "reason": reason,
            }
        )
        self.items.sort(key=lambda i: i["score"], reverse=True)
        self.save()

    def pop(self) -> dict | None:
        while self.items:
            item = self.items.pop(0)
            if item["key"] in self.visited:
                continue
            self.visited.add(item["key"])
            for domain in item.get("domains", []):
                self.domain_counts[domain.lower()] += 1
            self.save()
            return item
        return None

    def __len__(self) -> int:
        return len(self.items)
