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

# JSON schemas mirror the reply contracts in prompts/*.md; chat_json sends
# them as response_format so capable servers constrain decoding to them.
IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "type": {"type": "string",
                 "enum": ["mechanism", "method", "finding", "problem"]},
        "relation_to_paper": {"type": "string", "enum": list(PAPER_RELATIONS)},
        "domain": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["statement", "type", "relation_to_paper", "domain", "entities"],
    "additionalProperties": False,
}

EXTRACT_SCHEMA = {
    "title": "extraction",
    "type": "object",
    "properties": {
        "key_finding": {"type": "string"},
        "ideas": {"type": "array", "items": IDEA_SCHEMA},
    },
    "required": ["key_finding", "ideas"],
    "additionalProperties": False,
}

MERGE_SCHEMA = {
    "title": "merge_decision",
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["merge", "link", "new"]},
        "target": {"type": ["string", "null"]},
        "relation": {"type": ["string", "null"]},
        "note": {"type": "string"},
    },
    "required": ["decision", "target", "relation", "note"],
    "additionalProperties": False,
}


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
    result = llm.chat_json(_prompt("extract"), user, EXTRACT_SCHEMA)
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
    link_threshold: float | None = None,
    level: int = 0,
) -> list[str]:
    """Merge extracted ideas into the vault. Returns slugs touched (new or merged).

    Two-tier candidate net so ideas at DIFFERENT abstraction levels connect:
    - candidates at/above `merge_threshold` may be merged (same idea).
    - candidates between `link_threshold` and `merge_threshold` are shown as
      link-only. This lower gate is what lets a specific claim in one paper
      reach a broader, differently-worded idea in another — cross-level pairs
      never hit the high merge bar, so without it they'd never even be seen.
    """
    if link_threshold is None:  # sensible default: well below the merge bar
        link_threshold = max(0.5, round(merge_threshold - 0.18, 2))

    touched = []
    for raw in extracted["ideas"]:
        vector = llm.embed(raw["statement"])
        ranked = [
            (key.removeprefix("idea:"), score)
            for key, score in index.search(vector, k=8, prefix="idea:")
            if score >= link_threshold
        ]
        decision = {"decision": "new"}
        if ranked:
            decision = _adjudicate(llm, vault, raw, level, ranked, merge_threshold)

        target = decision.get("target") or ""
        if decision["decision"] == "merge" and vault.has_idea(target):
            slug = target
        else:
            idea = vault.create_idea(
                raw["statement"], raw["type"], raw["domain"], raw["entities"], level
            )
            slug = idea["slug"]
            index.add(f"idea:{slug}", vector)
            if decision["decision"] == "link" and vault.has_idea(target):
                relation = decision.get("relation")
                if relation not in IDEA_RELATIONS:
                    relation = "related_to"
                source, dest = _orient(vault, slug, level, target, relation)
                vault.link_ideas(source, dest, relation, decision.get("note", ""))

        vault.link_paper_idea(pmid, slug, raw["relation_to_paper"])
        touched.append(slug)
    index.save()
    return touched


def _orient(vault: Vault, new_slug: str, new_level: int, target: str,
            relation: str) -> tuple[str, str]:
    """Point a part_of edge from the more specific idea to the broader one.

    `part_of` is directional (small is part of large). Whichever idea sits at
    the lower tree level is the specific one, regardless of which paper's
    canonicalization created the edge. Symmetric relations keep new -> target.
    """
    if relation != "part_of":
        return new_slug, target
    target_level = vault.load_idea(target).get("level", 0)
    if target_level < new_level:  # the existing idea is the more specific one
        return target, new_slug
    return new_slug, target


def _adjudicate(
    llm: LLM, vault: Vault, raw: dict, new_level: int,
    ranked: list[tuple[str, float]], merge_threshold: float,
) -> dict:
    lines = [
        "NEW idea:",
        f"  statement: {raw['statement']}",
        f"  type: {raw['type']}  domain: {raw['domain']}  level: {new_level}",
        "",
        "EXISTING ideas ranked by embedding similarity:",
    ]
    for slug, score in ranked:
        existing = vault.load_idea(slug)
        eligibility = "merge-eligible" if score >= merge_threshold else "link-only"
        lines.append(
            f"  - slug: {slug} (similarity {score:.2f}, {eligibility})\n"
            f"    statement: {existing['statement']}\n"
            f"    type: {existing['type']}  domain: {existing['domain']}  "
            f"level: {existing.get('level', 0)}"
        )
    result = llm.chat_json(_prompt("merge"), "\n".join(lines), MERGE_SCHEMA)
    if result.get("decision") not in ("merge", "link", "new"):
        result["decision"] = "new"
    # Guard: a link-only candidate can never be merged, whatever the model says.
    if result["decision"] == "merge":
        target = result.get("target")
        eligible = {s for s, sc in ranked if sc >= merge_threshold}
        if target not in eligible:
            result["decision"] = "link" if target in {s for s, _ in ranked} else "new"
    return result
