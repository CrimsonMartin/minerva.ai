"""Render the vault's idea network as one self-contained HTML file.

The output is a static page — canvas force layout, domain legend, search,
and a detail panel — with the graph data embedded, so it can be opened
from disk, mailed, or hosted anywhere with no server or dependencies.
"""

import base64
import json
import math
from pathlib import Path

from .embeddings import EmbeddingIndex
from .store import Vault

TEMPLATE = Path(__file__).parent / "templates" / "graph.html"


def _quantized_vectors(vault: Vault, slugs: list[str]) -> dict | None:
    """Idea vectors as one base64 int8 blob, in `slugs` order.

    Vectors are unit-normalized first, so a dot product of the int8 values
    (over 127²) approximates cosine — accurate enough for ranking at an
    eighth the size of float32.
    """
    index = EmbeddingIndex(vault.root)
    vectors = [index.get(f"idea:{slug}") for slug in slugs]
    present = [v for v in vectors if v]
    if not present:
        return None
    dims = len(present[0])
    if any(len(v) != dims for v in present):
        return None  # mixed embedding spaces: not comparable, skip the feature
    buf = bytearray()
    for vector in vectors:
        if not vector:
            buf.extend(b"\x00" * dims)
            continue
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        for x in vector:
            q = round(x / norm * 127)
            buf.append(max(-127, min(127, q)) & 0xFF)
    return {"dims": dims, "model": index.model,
            "data": base64.b64encode(bytes(buf)).decode()}


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


def render_graph_html(vault: Vault, out_path: Path, title: str = "Idea Network",
                      embed_search: dict | None = None) -> Path:
    """`embed_search` is {"url", "model", "apiKey"} for the OpenAI-compatible
    endpoint the page should use to embed search queries. The page falls back
    to text matching whenever it is absent or unreachable."""
    data = graph_data(vault)
    vectors = _quantized_vectors(vault, [n["id"] for n in data["nodes"]])
    if vectors:
        data["vectors"] = vectors
    if embed_search:
        data["embedSearch"] = embed_search
    html = TEMPLATE.read_text()
    html = html.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
