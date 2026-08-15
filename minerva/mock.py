"""A deterministic mock model: run the whole pipeline with no LLM.

Set MINERVA_LLM_CHAT_MODEL=mock (in the environment or `.env`; the embed
model is then mocked too) to dry-run research sessions, tree builds, and ingestion —
useful for testing the plumbing, the vault structure, and the file
network before spending real model time.

Behavior is intentionally simple and reproducible:
- summaries are the leading sentences of the content,
- ideas are the first substantial sentence of the content,
- embeddings are hashed bag-of-words vectors (similar text -> similar
  vector, identical text -> identical vector),
- merge adjudication merges only exact statement matches,
- reflection immediately reports done.
"""

import hashlib
import json
import math
import re


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class MockLLM:
    # -------------------------------------------------------------- chat

    def chat(self, system: str, user: str, response_format: dict | None = None,
             model: str | None = None, extra_body: dict | None = None) -> str:
        if "condensing one node" in system:
            return json.dumps(self._summarize(user))
        if "Extract the CORE IDEAS" in system:
            return json.dumps(self._extract(user))
        if "knowledge graph of scientific ideas" in system:
            return json.dumps(self._merge(user))
        if "planning half" in system:
            return json.dumps({"assessment": "mock: coverage looks sufficient",
                               "gaps": [], "queries": [], "done": True})
        if "final report" in system:
            findings = [line for line in user.splitlines() if line.startswith("- ")]
            return ("# Research report (mock model)\n\n"
                    + "\n".join(findings)
                    + "\n\n## Open questions\n\n(mock)\n\n## Sources\n(see notebook)\n")
        return "mock reply"

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        return json.loads(self.chat(system, user))

    # -------------------------------------------------------- embeddings

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * 128
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = int.from_bytes(hashlib.sha1(token.encode()).digest()[:4], "big")
            vector[digest % 128] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    # ----------------------------------------------------------- helpers

    def _body(self, user: str) -> str:
        # Drop "Content (...):" / "Title:" style headers, keep the payload.
        return re.sub(r"^(Content \([^)]*\):|Title:.*|Journal:.*|MeSH terms:.*|Abstract:)\s*",
                      "", user, flags=re.MULTILINE).strip()

    def _idea_from(self, text: str) -> list[dict]:
        for sentence in _sentences(text):
            if len(sentence) >= 50:
                return [{"statement": sentence.rstrip("."), "type": "finding",
                         "relation_to_paper": "supports", "domain": "mock",
                         "entities": re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", sentence)[:5]}]
        return []

    def _summarize(self, user: str) -> dict:
        body = self._body(user)
        sentences = _sentences(body)
        summary = " ".join(sentences[:2])[:400] or body[:400]
        return {"summary": summary,
                "key_finding": (sentences[0][:200] if sentences else ""),
                "ideas": self._idea_from(body)}

    def _extract(self, user: str) -> dict:
        body = self._body(user)
        sentences = _sentences(body)
        return {"key_finding": (sentences[0][:200] if sentences else ""),
                "ideas": self._idea_from(body)}

    def _merge(self, user: str) -> dict:
        new_match = re.search(r"statement: (.+)", user)
        new_statement = new_match.group(1).strip().lower() if new_match else ""
        level_match = re.search(r"level: (\d+)", user)
        new_level = int(level_match.group(1)) if level_match else 0

        # (slug, statement, eligibility, level) for each ranked candidate.
        candidates = [
            (m.group(1), m.group(3).strip().lower(), m.group(2), int(m.group(4)))
            for m in re.finditer(
                r"- slug: (\S+) \(similarity [0-9.]+, (merge-eligible|link-only)\)\s*\n"
                r"\s*statement: (.+)\n\s*type:.*?level: (\d+)",
                user,
            )
        ]
        # Same wording and close enough to merge → merge.
        for slug, statement, eligibility, _ in candidates:
            if statement == new_statement and eligibility == "merge-eligible":
                return {"decision": "merge", "target": slug, "relation": None, "note": ""}
        # A candidate at a different abstraction level → part_of link (cross-level).
        for slug, _, _, cand_level in candidates:
            if cand_level != new_level:
                return {"decision": "link", "target": slug, "relation": "part_of",
                        "note": "cross-level (mock)"}
        if candidates:
            return {"decision": "link", "target": candidates[0][0],
                    "relation": "related_to", "note": "mock"}
        return {"decision": "new", "target": None, "relation": None, "note": ""}


def make_llm(config: dict):
    """LLM factory: the real client, or the mock when chat_model is 'mock'."""
    if config["llm"]["chat_model"] == "mock":
        return MockLLM()
    from .llm import LLM
    return LLM(config)
