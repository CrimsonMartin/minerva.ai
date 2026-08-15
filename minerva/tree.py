"""Recursive paper trees: arbitrary-length documents as bounded LLM calls.

A paper of any length is decomposed bottom-up:

    paragraphs (leaves, no LLM)
      -> groups of consecutive leaves        one summarize+ideas call each
        -> groups of summaries               one call each
          -> ... -> root                     the paper's "topic level"

Every LLM call sees at most `group_chars` of content plus the prompt, so
context is bounded no matter how long the paper is; total calls grow
linearly (~ n_paragraphs / branching factor).

Ideas are collected during the build but canonicalized TOP-DOWN after it
finishes: the root's topic-level ideas anchor into the global graph
first, then each level below. Child-node ideas get a `part_of` edge to
their parent node's primary idea, projecting the paper's structure into
the idea network. The tree is stored as tree.json (source of truth) and
tree.md (readable outline with links) in the paper's folder.
"""

from pathlib import Path

from .embeddings import EmbeddingIndex
from .extract import canonicalize, _prompt
from .llm import LLM
from .store import PAPER_RELATIONS, Vault


# -------------------------------------------------------------- paragraphs

def split_paragraphs(text: str, leaf_chars: int = 1200) -> list[str]:
    """Split into leaf-sized paragraphs: merge tiny ones, hard-split huge ones."""
    leaves: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        while len(paragraph) > leaf_chars:
            if current:
                leaves.append("\n\n".join(current))
                current, size = [], 0
            cut = paragraph.rfind(" ", 0, leaf_chars)
            cut = cut if cut > leaf_chars // 2 else leaf_chars
            leaves.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if not paragraph:
            continue
        if size + len(paragraph) > leaf_chars and current:
            leaves.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        leaves.append("\n\n".join(current))
    return leaves


# -------------------------------------------------------------- tree build

def build_paper_tree(
    llm: LLM,
    vault: Vault,
    index: EmbeddingIndex,
    paper_id: str,
    paragraphs: list[str],
    merge_threshold: float,
    link_threshold: float | None = None,
    group_chars: int = 3500,
    max_paragraphs: int = 500,
) -> dict:
    """Build the tree, canonicalize ideas root-first, write tree.json/tree.md.

    Returns {"summary", "key_finding", "slugs" (root-first), "n_nodes",
    "n_leaves", "depth", "truncated"}.
    """
    truncated = max(0, len(paragraphs) - max_paragraphs)
    paragraphs = paragraphs[:max_paragraphs]

    nodes: dict[str, dict] = {}
    for i, text in enumerate(paragraphs):
        nodes[f"L{i}"] = {"id": f"L{i}", "level": 0, "text": text,
                          "children": [], "summary": "", "ideas": []}

    current = [f"L{i}" for i in range(len(paragraphs))]
    level, counter = 1, 0
    key_finding = ""
    # Recurse upward until a single root remains. A single-leaf document
    # still gets one pass so its root has a summary and ideas.
    while len(current) > 1 or (level == 1 and current and not nodes[current[0]]["summary"]):
        next_level = []
        for group in _group(current, nodes, group_chars):
            node_id = f"n{counter}"
            counter += 1
            content = "\n\n".join(_content(nodes[child]) for child in group)
            result = _summarize(llm, content, level)
            nodes[node_id] = {
                "id": node_id, "level": level, "children": group,
                "summary": result["summary"], "ideas": result["ideas"], "text": "",
            }
            key_finding = result["key_finding"] or key_finding
            next_level.append(node_id)
        current = next_level
        level += 1
    root_id = current[0] if current else None
    if root_id is None:
        return {"summary": "", "key_finding": "", "slugs": [],
                "n_nodes": 0, "n_leaves": 0, "depth": 0, "truncated": truncated}

    # Canonicalize top-down: the root's topic-level ideas enter the global
    # graph first, then each level beneath anchors under them.
    ordered = _levels_root_first(nodes, root_id)
    all_slugs: list[str] = []
    for node_id in ordered:
        node = nodes[node_id]
        if not node["ideas"]:
            continue
        slugs = canonicalize(
            llm, vault, index, paper_id,
            {"key_finding": "", "ideas": node["ideas"]}, merge_threshold,
            link_threshold=link_threshold, level=node["level"],
        )
        node["slugs"] = slugs
        all_slugs.extend(s for s in slugs if s not in all_slugs)

    # Project the tree into the idea graph: child ideas are part_of the
    # parent node's primary idea (unless they merged into the same node).
    for node_id in ordered:
        node = nodes[node_id]
        parent_slugs = node.get("slugs", [])
        if not parent_slugs:
            continue
        primary = parent_slugs[0]
        for child_id in node["children"]:
            for slug in nodes[child_id].get("slugs", []):
                if slug not in parent_slugs:
                    vault.link_ideas(slug, primary, "part_of",
                                     f"paper {paper_id} structure")

    root = nodes[root_id]
    tree = {"root": root_id, "paper": paper_id, "nodes": nodes}
    directory = vault.papers_dir / paper_id
    directory.mkdir(parents=True, exist_ok=True)
    import json
    (directory / "tree.json").write_text(json.dumps(tree, indent=2) + "\n")
    (directory / "tree.md").write_text(_render(vault, nodes, root_id, paper_id))

    return {
        "summary": root["summary"], "key_finding": key_finding,
        "slugs": all_slugs, "n_nodes": len(nodes), "n_leaves": len(paragraphs),
        "depth": root["level"], "truncated": truncated,
    }


def _group(ids: list[str], nodes: dict, group_chars: int) -> list[list[str]]:
    """Group consecutive nodes so each group's content fits one call."""
    groups, current, size = [], [], 0
    for node_id in ids:
        length = len(_content(nodes[node_id]))
        if current and size + length > group_chars:
            groups.append(current)
            current, size = [], 0
        current.append(node_id)
        size += length + 2
    if current:
        groups.append(current)
    return groups


def _content(node: dict) -> str:
    return node["text"] if node["level"] == 0 else node["summary"]


def _summarize(llm: LLM, content: str, level: int) -> dict:
    kind = "raw paragraphs from the paper" if level == 1 else "summaries of child sections"
    result = llm.chat_json(_prompt("summarize"), f"Content ({kind}):\n\n{content}")
    ideas = []
    for raw in result.get("ideas", []) or []:
        statement = (raw.get("statement") or "").strip()
        if not statement:
            continue
        ideas.append({
            "statement": statement,
            "type": raw.get("type") if raw.get("type") in
            ("mechanism", "method", "finding", "problem") else "finding",
            "relation_to_paper": raw.get("relation_to_paper")
            if raw.get("relation_to_paper") in PAPER_RELATIONS else "supports",
            "domain": (raw.get("domain") or "general").strip().lower(),
            "entities": [e for e in (raw.get("entities") or []) if isinstance(e, str)][:5],
        })
    return {"summary": (result.get("summary") or "").strip(),
            "key_finding": (result.get("key_finding") or "").strip(),
            "ideas": ideas}


def _levels_root_first(nodes: dict, root_id: str) -> list[str]:
    order, queue = [], [root_id]
    while queue:
        node_id = queue.pop(0)
        node = nodes[node_id]
        if node["level"] > 0:
            order.append(node_id)
            queue.extend(node["children"])
    return order


def _render(vault: Vault, nodes: dict, root_id: str, paper_id: str) -> str:
    lines = [f"# Paper tree — {paper_id}\n",
             "Root first; each node condenses everything beneath it.\n"]

    def walk(node_id: str, indent: int) -> None:
        node = nodes[node_id]
        pad = "  " * indent
        if node["level"] == 0:
            excerpt = node["text"][:110].replace("\n", " ")
            lines.append(f"{pad}- ¶ *{excerpt}…*")
            return
        lines.append(f"{pad}- **[{node_id}]** {node['summary']}")
        for slug in node.get("slugs", []):
            if vault.has_idea(slug):
                statement = vault.load_idea(slug)["statement"]
                lines.append(f"{pad}  - 💡 [{statement}](../../ideas/{slug}/idea.md)")
        for child in node["children"]:
            walk(child, indent + 1)

    walk(root_id, 0)
    lines.append("")
    return "\n".join(lines)
