"""Render the vault's idea network as one self-contained HTML file.

The output is a static page — canvas force layout, domain legend, search,
and a detail panel — with the graph data embedded, so it can be opened
from disk, mailed, or hosted anywhere with no server or dependencies.
"""

import json
from pathlib import Path

from .store import Vault

TEMPLATE = Path(__file__).parent / "templates" / "graph.html"


def graph_data(vault: Vault) -> dict:
    """Nodes + edges for every idea in the vault, as plain dicts.

    Papers are carried once in a lookup keyed by id (nodes reference ids), so
    a title shared by many ideas is not repeated through the payload.
    """
    nodes, edges = [], []
    papers: dict[str, dict] = {}
    for slug in vault.list_ideas():
        idea = vault.load_idea(slug)
        cites = []
        for link in idea["papers"]:
            pmid = link["pmid"]
            cites.append({"id": pmid, "relation": link["relation"]})
            if pmid not in papers and vault.has_paper(pmid):
                paper = vault.load_paper(pmid)
                papers[pmid] = {
                    "title": paper.get("title") or pmid,
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    # local ingests have no PubMed record to link out to
                    "local": bool(paper.get("source")),
                }
        nodes.append({
            "id": idea["slug"],
            "statement": idea["statement"],
            "type": idea["type"],
            "domain": idea["domain"],
            "level": idea.get("level", 0),
            "papers": len(idea["papers"]),
            "cites": cites,
        })
        for edge in idea["edges"]:
            edges.append({"source": idea["slug"], "target": edge["target"],
                          "rel": edge["relation"]})
    return {"nodes": nodes, "edges": edges, "paperCount": len(vault.list_papers()),
            "paperMeta": papers}


def render_graph_html(vault: Vault, out_path: Path, title: str = "Idea Network") -> Path:
    data = graph_data(vault)
    html = TEMPLATE.read_text()
    html = html.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
