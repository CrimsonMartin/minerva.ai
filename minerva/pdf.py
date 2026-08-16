"""Optional markdown → PDF rendering.

Markdown stays the vault's source of truth (greppable, Obsidian-friendly);
a PDF is rendered *next to* it when the `markdown-pdf` package is
installed (`pip install markdown-pdf`). That package pulls markdown-it-py
(pure Python) and PyMuPDF — a large but self-contained wheel with no
system dependencies. It stays an optional extra, not a requirement,
because PyMuPDF is AGPL-licensed and most runs only need the .md.
"""

from pathlib import Path


def render_pdf(md_path: Path) -> Path | None:
    """Render a markdown file to a PDF beside it (report.md → report.pdf).

    Returns the PDF path, or None when markdown-pdf is not installed.
    """
    try:
        from markdown_pdf import MarkdownPdf, Section
    except ImportError:
        return None
    pdf = MarkdownPdf(toc_level=2)  # bookmarks from #/## headings
    pdf.add_section(Section(md_path.read_text()))
    out = md_path.with_suffix(".pdf")
    pdf.save(str(out))
    return out
