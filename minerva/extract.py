"""Turn an abstract into core ideas, then canonicalize them into the vault.

Canonicalization is what makes this a graph instead of a pile of
extractions: each new idea is embedded, compared against existing ideas,
and an LLM adjudicates merge / link / new. "PD-1 blockade" in the 40th
paper becomes the 40th edge on one node, not a 40th node.
"""

from pathlib import Path

from .embeddings import EmbeddingIndex
from .llm import LLM
from .store import IDEA_RELATIONS, PAPER_RELATIONS, Vault

PROMPTS = Path(__file__).parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text()


def extract_ideas(llm: LLM, paper: dict) -> dict:
    """Run the extraction prompt on one paper. Returns {key_finding, ideas}."""
    user = (
        f"Title: {paper['title']}\n"
        f"Journal: {paper.get('journal', '')} ({paper.get('year', '')})\n"
        f"MeSH terms: {', '.join(paper.get('mesh', []))}\n\n"
        f"Abstract:\n{paper['abstract']}"
    )
    result = llm.chat_json(_prompt("extract"), user)
    ideas = []
    for raw in result.get("ideas", []):
        statement = (raw.get("statement") or "").strip()
        if not statement:
            continue
        ideas.append(
            {
                "statement": statement,
                "type": raw.get("type") if raw.get("type") in
                ("mechanism", "method", "finding", "problem") else "finding",
                "relation_to_paper": raw.get("relation_to_paper")
                if raw.get("relation_to_paper") in PAPER_RELATIONS else "supports",
                "domain": (raw.get("domain") or "general").strip().lower(),
                "entities": [e for e in (raw.get("entities") or []) if isinstance(e, str)][:5],
            }
        )
    return {"key_finding": (result.get("key_finding") or "").strip(), "ideas": ideas}


def canonicalize(
    llm: LLM,
    vault: Vault,
    index: EmbeddingIndex,
    pmid: str,
    extracted: dict,
    merge_threshold: float,
) -> list[str]:
    """Merge extracted ideas into the vault. Returns slugs touched (new or merged)."""
    touched = []
    for raw in extracted["ideas"]:
        vector = llm.embed(raw["statement"])
        candidates = [
            (key.removeprefix("idea:"), score)
            for key, score in index.search(vector, k=5, prefix="idea:")
            if score >= merge_threshold
        ]
        decision = {"decision": "new"}
        if candidates:
            decision = _adjudicate(llm, vault, raw, candidates)

        if decision["decision"] == "merge" and vault.has_idea(decision.get("target") or ""):
            slug = decision["target"]
        else:
            idea = vault.create_idea(
                raw["statement"], raw["type"], raw["domain"], raw["entities"]
            )
            slug = idea["slug"]
            index.add(f"idea:{slug}", vector)
            if decision["decision"] == "link" and vault.has_idea(decision.get("target") or ""):
                relation = decision.get("relation")
                if relation not in IDEA_RELATIONS:
                    relation = "related_to"
                vault.link_ideas(slug, decision["target"], relation, decision.get("note", ""))

        vault.link_paper_idea(pmid, slug, raw["relation_to_paper"])
        touched.append(slug)
    index.save()
    return touched


def _adjudicate(
    llm: LLM, vault: Vault, raw: dict, candidates: list[tuple[str, float]]
) -> dict:
    lines = [
        "NEW idea:",
        f"  statement: {raw['statement']}",
        f"  type: {raw['type']}  domain: {raw['domain']}",
        "",
        "SIMILAR existing ideas:",
    ]
    for slug, score in candidates:
        existing = vault.load_idea(slug)
        lines.append(
            f"  - slug: {slug} (similarity {score:.2f})\n"
            f"    statement: {existing['statement']}\n"
            f"    type: {existing['type']}  domain: {existing['domain']}"
        )
    result = llm.chat_json(_prompt("merge"), "\n".join(lines))
    if result.get("decision") not in ("merge", "link", "new"):
        result["decision"] = "new"
    return result
