"""A deterministic mock model: run the whole pipeline with no LLM.

Set `"chat_model": "mock"` in minerva.config.json (embed model is then
mocked too) to dry-run research sessions, tree builds, and ingestion —
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
    chat_model = "mock"
    embed_model = "mock"

    # -------------------------------------------------------------- chat

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
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

    def chat_json(self, system: str, user: str) -> dict:
        return json.loads(self.chat(system, user, json_mode=True))

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
        slug, statement = None, None
        for match in re.finditer(r"slug: (\S+).*?\n\s*statement: (.+)", user):
            if match.group(2).strip().lower() == new_statement:
                slug = match.group(1)
                break
        if slug:
            return {"decision": "merge", "target": slug, "relation": None, "note": ""}
        return {"decision": "new", "target": None, "relation": None, "note": ""}


def make_llm(config: dict):
    """LLM factory: the real client, or the mock when chat_model is 'mock'."""
    if config["llm"]["chat_model"] == "mock":
        return MockLLM()
    from .llm import LLM
    return LLM(config)
