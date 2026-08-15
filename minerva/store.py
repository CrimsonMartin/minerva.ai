"""The vault: a files-only knowledge graph you can read.

Layout — the folder structure IS the network:

    vault/
      papers/<pmid>/paper.json     source of truth for one paper
      papers/<pmid>/paper.md       rendered view, links to its ideas
      ideas/<slug>/idea.json       source of truth for one core idea
      ideas/<slug>/idea.md         rendered view, links to papers + related ideas
      index/embeddings.json        vector index (see embeddings.py)
      runs/<run>/                  per-research-run state, notebook, report

Every .md is regenerated from its .json on write, and cross-references
are relative markdown links, so the whole vault is browsable in any
editor (and Obsidian's graph view will draw the idea network).
"""

import hashlib
import json
import re
from pathlib import Path

PAPER_RELATIONS = ("introduces", "uses", "supports", "contradicts", "extends")
IDEA_RELATIONS = ("part_of", "causes", "enables", "analogous_to", "related_to")


def slugify(statement: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", statement.lower()).split()
    stem = "-".join(words[:6]) or "idea"
    digest = hashlib.sha1(statement.encode()).hexdigest()[:6]
    return f"{stem}-{digest}"


class Vault:
    def __init__(self, root: Path):
        self.root = root
        self.papers_dir = root / "papers"
        self.ideas_dir = root / "ideas"
        self.runs_dir = root / "runs"
        for directory in (self.papers_dir, self.ideas_dir, self.runs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ papers

    def has_paper(self, pmid: str) -> bool:
        return (self.papers_dir / pmid / "paper.json").exists()

    def load_paper(self, pmid: str) -> dict:
        return json.loads((self.papers_dir / pmid / "paper.json").read_text())

    def save_paper(self, paper: dict) -> None:
        paper.setdefault("ideas", [])  # [{"slug", "relation"}]
        directory = self.papers_dir / paper["pmid"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "paper.json").write_text(json.dumps(paper, indent=2) + "\n")
        (directory / "paper.md").write_text(self._render_paper(paper))

    def list_papers(self) -> list[str]:
        return sorted(p.name for p in self.papers_dir.iterdir() if p.is_dir())

    # ------------------------------------------------------------- ideas

    def has_idea(self, slug: str) -> bool:
        return (self.ideas_dir / slug / "idea.json").exists()

    def load_idea(self, slug: str) -> dict:
        return json.loads((self.ideas_dir / slug / "idea.json").read_text())

    def save_idea(self, idea: dict) -> None:
        idea.setdefault("papers", [])  # [{"pmid", "relation"}]
        idea.setdefault("edges", [])  # [{"relation", "target", "note"}]
        directory = self.ideas_dir / idea["slug"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "idea.json").write_text(json.dumps(idea, indent=2) + "\n")
        (directory / "idea.md").write_text(self._render_idea(idea))

    def list_ideas(self) -> list[str]:
        return sorted(p.name for p in self.ideas_dir.iterdir() if p.is_dir())

    def create_idea(self, statement: str, idea_type: str, domain: str, entities: list[str]) -> dict:
        idea = {
            "slug": slugify(statement),
            "statement": statement,
            "type": idea_type,
            "domain": domain,
            "entities": entities,
            "papers": [],
            "edges": [],
        }
        self.save_idea(idea)
        return idea

    # ------------------------------------------------------------- edges

    def link_paper_idea(self, pmid: str, slug: str, relation: str) -> None:
        paper = self.load_paper(pmid)
        if not any(link["slug"] == slug for link in paper["ideas"]):
            paper["ideas"].append({"slug": slug, "relation": relation})
            self.save_paper(paper)
        idea = self.load_idea(slug)
        if not any(link["pmid"] == pmid for link in idea["papers"]):
            idea["papers"].append({"pmid": pmid, "relation": relation})
            self.save_idea(idea)

    def link_ideas(self, source: str, target: str, relation: str, note: str = "") -> None:
        if source == target:
            return
        idea = self.load_idea(source)
        if not any(e["target"] == target and e["relation"] == relation for e in idea["edges"]):
            idea["edges"].append({"relation": relation, "target": target, "note": note})
            self.save_idea(idea)
        # Mirror on the target so the network is walkable from both ends.
        mirror = {"part_of": "related_to", "causes": "related_to", "enables": "related_to"}
        back = mirror.get(relation, relation)
        other = self.load_idea(target)
        if not any(e["target"] == source for e in other["edges"]):
            other["edges"].append({"relation": back, "target": source, "note": note})
            self.save_idea(other)

    # --------------------------------------------------------- rendering

    def _render_paper(self, paper: dict) -> str:
        lines = [f"# {paper['title']}\n"]
        meta = [f"**PMID:** [{paper['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/)"]
        if paper.get("journal"):
            meta.append(f"**Journal:** {paper['journal']}")
        if paper.get("year"):
            meta.append(f"**Year:** {paper['year']}")
        lines.append(" · ".join(meta) + "\n")
        if paper.get("mesh"):
            lines.append(f"**MeSH:** {', '.join(paper['mesh'])}\n")
        if paper.get("abstract"):
            lines.append("## Abstract\n")
            lines.append(paper["abstract"] + "\n")
        if paper["ideas"]:
            lines.append("## Core ideas\n")
            for link in paper["ideas"]:
                if self.has_idea(link["slug"]):
                    statement = self.load_idea(link["slug"])["statement"]
                    lines.append(
                        f"- [{statement}](../../ideas/{link['slug']}/idea.md) "
                        f"*({link['relation']})*"
                    )
            lines.append("")
        return "\n".join(lines)

    def _render_idea(self, idea: dict) -> str:
        lines = [f"# {idea['statement']}\n"]
        lines.append(f"**Type:** {idea['type']} · **Domain:** {idea['domain']}\n")
        if idea.get("entities"):
            lines.append(f"**Entities:** {', '.join(idea['entities'])}\n")
        if idea["papers"]:
            lines.append("## Papers\n")
            for link in idea["papers"]:
                title = link["pmid"]
                if self.has_paper(link["pmid"]):
                    title = self.load_paper(link["pmid"])["title"]
                lines.append(
                    f"- [{title}](../../papers/{link['pmid']}/paper.md) "
                    f"*({link['relation']})*"
                )
            lines.append("")
        if idea["edges"]:
            lines.append("## Related ideas\n")
            for edge in idea["edges"]:
                if self.has_idea(edge["target"]):
                    statement = self.load_idea(edge["target"])["statement"]
                    note = f" — {edge['note']}" if edge.get("note") else ""
                    lines.append(
                        f"- **{edge['relation']}** → "
                        f"[{statement}](../{edge['target']}/idea.md){note}"
                    )
            lines.append("")
        return "\n".join(lines)
