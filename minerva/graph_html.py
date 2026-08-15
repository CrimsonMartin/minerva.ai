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


# The page embeds search queries in the browser, so its vectors come from a
# model small enough to run there — not from the vault's (larger) model. The
# two need only agree with each other; the vault's own embeddings are
# untouched, since they drive idea canonicalization.
SEARCH_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The same weights packaged as ONNX, which is what transformers.js loads in
# the browser. Both sides must stay the same model or the query vector lands
# in a different space than the vectors shipped in the page.
SEARCH_MODEL_ONNX = "Xenova/all-MiniLM-L6-v2"


def _search_vectors(statements: list[str]) -> dict | None:
    """Embed idea statements with the in-browser search model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = SentenceTransformer(SEARCH_MODEL)
    vectors = model.encode(statements, normalize_embeddings=True,
                           batch_size=32, show_progress_bar=False)
    return _quantize([v.tolist() for v in vectors], SEARCH_MODEL)


def _quantize(vectors: list[list[float]], model: str) -> dict | None:
    """Vectors as one base64 int8 blob.

    Unit-normalized first, so a dot product of the int8 values (over 127²)
    approximates cosine — accurate enough for ranking at an eighth the size
    of float32.
    """
    present = [v for v in vectors if v]
    if not present:
        return None
    dims = len(present[0])
    if any(len(v) != dims for v in present):
        return None
    buf = bytearray()
    for vector in vectors:
        if not vector:
            buf.extend(b"\x00" * dims)
            continue
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        for x in vector:
            q = round(x / norm * 127)
            buf.append(max(-127, min(127, q)) & 0xFF)
    return {"dims": dims, "model": model,
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


def render_graph_html(vault: Vault, out_path: Path,
                      title: str = "Idea Network") -> Path:
    """Render the network, with vectors for in-browser semantic search."""
    data = graph_data(vault)
    vectors = _search_vectors([n["statement"] for n in data["nodes"]])
    if vectors:
        data["vectors"] = vectors
        data["searchModel"] = SEARCH_MODEL_ONNX
    html = TEMPLATE.read_text()
    html = html.replace("__TITLE__", title)
    html = html.replace("__DATA__", json.dumps(data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
