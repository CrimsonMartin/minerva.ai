"""Ingest local documents (.pdf, .docx, .txt, .md) as research base.

A local file becomes a vault paper with id "local-<hash>", so the rest
of the pipeline (idea extraction, canonicalization, linking, citations)
treats it exactly like a PubMed paper. Text extraction ladder for PDFs:

  1. embedded text layer via pypdf (digital PDFs — fast, no models)
  2. PaddleOCR fallback when the text layer is missing/thin (scanned PDFs)

PaddleOCR is an optional heavy dependency; it is imported lazily and
only when actually needed, with a clear install hint otherwise.

DOCX needs no dependency at all: it is a zip of XML, read with stdlib.
"""

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .embeddings import EmbeddingIndex
from .llm import LLM
from .store import Vault

_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class IngestError(RuntimeError):
    pass


# ------------------------------------------------------------ text extraction

def extract_text(path: Path, min_chars_per_page: int = 200) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(path, min_chars_per_page)
    if suffix == ".docx":
        return _docx_text(path)
    if suffix in (".txt", ".md"):
        return path.read_text(errors="replace")
    raise IngestError(f"unsupported file type {suffix!r} (need .pdf, .docx, .txt, .md)")


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{_DOCX_NS}p"):
        text = "".join(node.text or "" for node in para.iter(f"{_DOCX_NS}t"))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)


def _pdf_text(path: Path, min_chars_per_page: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError("pypdf is required for PDF input: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    # Scanned PDFs have no (or a junk) text layer — fall back to OCR.
    if len(text) < min_chars_per_page * max(len(pages), 1):
        ocr_text = _ocr_pdf(path)
        if len(ocr_text) > len(text):
            return ocr_text
    return text


def _ocr_pdf(path: Path) -> str:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise IngestError(
            "this PDF has no usable text layer, so OCR is needed: "
            "pip install paddlepaddle paddleocr"
        ) from exc
    lines: list[str] = []
    try:  # PaddleOCR 3.x
        engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False)
        for page in engine.predict(str(path)):
            result = page if isinstance(page, dict) else getattr(page, "res", {})
            lines.extend(result.get("rec_texts", []))
    except (TypeError, AttributeError):  # PaddleOCR 2.x
        engine = PaddleOCR(use_angle_cls=True, show_log=False)
        for page in engine.ocr(str(path)) or []:
            for entry in page or []:
                lines.append(entry[1][0])
    return "\n".join(lines)


# ------------------------------------------------------------------- ingest

def ingest_file(
    llm: LLM,
    vault: Vault,
    index: EmbeddingIndex,
    path: Path,
    merge_threshold: float,
    tree_config: dict | None = None,
    min_chars_per_page: int = 200,
    link_threshold: float | None = None,
) -> dict:
    """Ingest one local file via a recursive paper tree.

    Returns {"id", "slugs" (root-level first), "findings", "cached",
    "summary", "tree"}.
    """
    from .tree import build_paper_tree, split_paragraphs

    tree_config = tree_config or {}
    path = path.expanduser().resolve()
    if not path.exists():
        raise IngestError(f"input file not found: {path}")
    doc_id = "local-" + hashlib.sha1(path.read_bytes()).hexdigest()[:10]

    if vault.has_paper(doc_id):  # same bytes already ingested — reuse the graph
        paper = vault.load_paper(doc_id)
        return {"id": doc_id, "slugs": [l["slug"] for l in paper["ideas"]],
                "findings": [], "cached": True, "summary": paper.get("summary", ""),
                "tree": None}

    text = extract_text(path, min_chars_per_page).strip()
    if not text:
        raise IngestError(f"no text could be extracted from {path.name}")

    vault.save_paper(
        {"pmid": doc_id, "title": path.stem, "abstract": "",
         "source": str(path), "journal": "", "year": "", "mesh": []}
    )
    (vault.papers_dir / doc_id / "fulltext.md").write_text(f"# {path.stem}\n\n{text}\n")

    paragraphs = split_paragraphs(text, tree_config.get("leaf_chars", 1200))
    result = build_paper_tree(
        llm, vault, index, doc_id, paragraphs, merge_threshold,
        link_threshold=link_threshold,
        group_chars=tree_config.get("group_chars", 3500),
        max_paragraphs=tree_config.get("max_paragraphs", 500),
    )

    paper = vault.load_paper(doc_id)
    paper["summary"] = result["summary"]
    paper["abstract"] = result["summary"]  # condensed stand-in for downstream reads
    vault.save_paper(paper)

    findings = [result["key_finding"]] if result["key_finding"] else []
    return {"id": doc_id, "slugs": result["slugs"], "findings": findings,
            "cached": False, "summary": result["summary"], "tree": result}
